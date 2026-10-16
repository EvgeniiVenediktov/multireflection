"""
agent.py - Policy network: frame encoder, causal transformer over the last K steps, heads
(PLAN phase 6: P1-P6, D6)

    frame      CNNEncoder (encoders.py, spatial-softmax keypoints): uint8 (1, 64, 64) -> z (256),
               optionally initialized from a pretrain_encoder.py checkpoint (encoder only);
               channels_last weights and inputs, bf16 autocast, torch.compile (compile());
               with grad (unfrozen encoder) in activation-checkpointed chunks of enc_chunk
               images (enc_checkpoint) or in one pass that keeps the activations
    token      Linear(concat(LayerNorm(z_t), a_{t-1})) -> d_model; a_{t-1} is the applied
               action that led to frame t (env obs["action"], delta / largest ladder step,
               in [-1, 1]; zero at an episode start)
    core       causal pre-LN transformer over the window of the last K tokens (oldest
               first, the current step last): n_layers blocks, n_heads heads, a learned
               relative-position bias per layer and head over the distances 0..K-1, keys
               before the episode start masked (each query may always see itself, so a
               padded query never has an empty row); the last block computes only the
               current position; h_t = LayerNorm(output at the last position)
    heads      (2-layer MLPs on h_t)
               policy  two factorized categoricals over the n_bins ladder steps per axis
                       (joint (0, 0) = not move); last layer scaled to ~0 (uniform start)
               value   in the trainer's normalized units (ppo.ValueNorm)
               path    P_t / P_ref of the current frame (path head, P6)
               tilt    current controlled tilt / tilt_scale: mean and log-variance (2 + 2)
               scenario  (separation, laser pitch, laser yaw), each mapped to [-1, 1] over
                       the atlas range (scen_lo, scen_hi): mean and log-variance (3 + 3),
                       Gaussian NLL (stage 2; replaces the 4-way setup head)

Two call paths compute the same function of the same inputs:
    act()               rollout: per env a Ring of the last K (z, a, valid, frame); only
                        the newest frame is encoded per step
    evaluate_windows()  PPO update: tokens of a [K-1 context + T] sequence per env computed
                        once, the T windows unfolded from it, the core applied per window
    final_windows()     truncation bootstrap: the window of a final (pre-reset) frame
Window mask from the sequence: position q is visible to the window ending at p if q holds
a frame (real) and no episode starts in (q, p] (first flags; cumulative segment ids).

init_from(path): weights of a previous PPO checkpoint (ppo.py latest.pt / best.pt) where
names and shapes match; the rest keeps its fresh initialization and is reported.
"""

import functools
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from encoders import CNNEncoder, EncoderConfig


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Agent settings."""

    window: int = 16  # K: tokens per window (current step included)
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    head_hidden: int = 256
    n_bins: int = 11  # Ladder steps per axis (env.n_bins)
    policy_init_scale: float = 0.01  # Last policy layer weights x this, bias 0
    tilt_scale: float = 2.0  # deg; tilt labels are divided by this (the mount limit)
    scen_lo: Tuple[float, float, float] = (50.0, 0.0, 0.0)  # Atlas range: mm, deg, deg
    scen_hi: Tuple[float, float, float] = (200.0, 8.0, 8.0)
    logvar_min: float = -12.0
    logvar_max: float = 4.0
    amp: bool = True  # bf16 autocast for encoder, token and core (heads in fp32)
    channels_last: bool = True  # Encoder weights and inputs in NHWC (tensor-core convs)
    enc_chunk: int = 1024  # Images per checkpointed chunk when the encoder trains
    enc_checkpoint: bool = True  # False: keep the encoder activations (no recompute)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AgentConfig":
        d = dict(d)
        enc = d.pop("encoder", {})
        enc = EncoderConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in enc.items()})
        names = {f.name for f in fields(AgentConfig)}
        return AgentConfig(encoder=enc, **{k: tuple(v) if isinstance(v, list) else v
                                           for k, v in d.items() if k in names})


class Ring(NamedTuple):
    """Per-env rollout window, oldest first, the current step at index K - 1."""

    z: torch.Tensor  # (B, K, Dz) float32 frame embeddings
    a: torch.Tensor  # (B, K, 2) float32 applied action that led to each frame
    valid: torch.Tensor  # (B, K) bool, same episode as the newest entry
    frame: torch.Tensor  # (B, K, 1, H, W) uint8 (re-encoding when the encoder trains)


# ── Action distribution (factorized categorical) ────────────────────────────


def log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Joint log-probability (...,) of actions (..., 2) under logits (..., 2, n_bins)."""
    lp = torch.log_softmax(logits.float(), -1)
    return lp.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1).sum(-1)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Joint entropy (...,) = sum of the two factors' entropies."""
    lp = torch.log_softmax(logits.float(), -1)
    return -(lp.exp() * lp).sum(-1).sum(-1)


def sample(logits: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """(..., 2) int64 actions, Gumbel-max per factor (no host sync)."""
    u = torch.rand(logits.shape, generator=gen, device=logits.device, dtype=torch.float32)
    g = -torch.log(-torch.log(u.clamp(1e-20, 1.0 - 1e-7)))
    return (torch.log_softmax(logits.float(), -1) + g).argmax(-1)


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """(..., 2) int64 argmax per factor (deterministic policy)."""
    return logits.argmax(-1)


# ── Window masks ────────────────────────────────────────────────────────────


def push_ring(ring: Ring, z: torch.Tensor, a: torch.Tensor, first: torch.Tensor,
              frame: torch.Tensor) -> Ring:
    """Ring with (z (B, Dz), a (B, 2), frame) appended as the newest entry; first (B,)
    True starts a new episode (older entries become invalid). No mutation, no sync."""
    keep = ~first
    valid = torch.cat((ring.valid[:, 1:] & keep[:, None], torch.ones_like(keep)[:, None]), 1)
    return Ring(
        z=torch.cat((ring.z[:, 1:], z.float().unsqueeze(1)), 1),
        a=torch.cat((ring.a[:, 1:], a.float().unsqueeze(1)), 1),
        valid=valid,
        frame=torch.cat((ring.frame[:, 1:], frame.unsqueeze(1)), 1),
    )


def window_masks(real: torch.Tensor, first: torch.Tensor, K: int) -> torch.Tensor:
    """(Bm, T, K) bool masks of the windows ending at positions K-1 .. L-1 of sequences
    real / first (Bm, L): entry q visible iff real[q] and no episode start in (q, p]."""
    seg = torch.cumsum(first.long(), 1)
    seg_w = seg.unfold(1, K, 1)  # (Bm, T, K)
    return real.unfold(1, K, 1) & (seg_w == seg_w[..., -1:])


# ── Core ────────────────────────────────────────────────────────────────────


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))


class CausalBlock(nn.Module):
    """Pre-LN block: masked multi-head self-attention with an additive bias, MLP."""

    def __init__(self, d: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        hid = int(d * mlp_ratio)
        self.fc1 = nn.Linear(d, hid)
        self.fc2 = nn.Linear(hid, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """x (N, K, d); bias (N, H, K, K) additive, -inf where masked. last_only: return
        (N, 1, d), the current position only."""
        N, K, D = x.shape
        H = self.heads
        h = self.n1(x)
        hq = h[:, -1:] if last_only else h
        Kq = hq.shape[1]
        q = self.q(hq).view(N, Kq, H, D // H).transpose(1, 2)  # (N, H, Kq, dh)
        k, v = self.kv(h).view(N, K, 2, H, D // H).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)).float() * (1.0 / math.sqrt(D // H))
        s = s + (bias[:, :, -1:] if last_only else bias)
        p = torch.softmax(s, -1).to(v.dtype)
        o = (p @ v).transpose(1, 2).reshape(N, Kq, D)
        x = (x[:, -1:] if last_only else x) + self.proj(o)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class Core(nn.Module):
    """Causal transformer over (N, K, d) windows -> h (N, d) of the last position."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.K = cfg.window
        self.blocks = nn.ModuleList(
            CausalBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_layers))
        self.rel_bias = nn.Parameter(torch.zeros(cfg.n_layers, cfg.n_heads, cfg.window))
        self.norm = nn.LayerNorm(cfg.d_model)
        i = torch.arange(cfg.window)
        dist = i[:, None] - i[None, :]  # Query minus key position
        self.register_buffer("dist", dist.clamp(min=0), persistent=False)
        self.register_buffer("causal", dist >= 0, persistent=False)
        self.register_buffer("eye", torch.eye(cfg.window, dtype=torch.bool), persistent=False)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x (N, K, d) tokens, valid (N, K) bool key mask (the last entry is the current
        step, always valid)."""
        allowed = self.causal & (valid[:, None, :] | self.eye)  # (N, K, K)
        L = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            rb = self.rel_bias[i][:, self.dist]  # (H, K, K)
            bias = torch.where(allowed[:, None], rb[None].float(), float("-inf"))
            x = blk(x, bias, last_only=i == L - 1)
        return self.norm(x[:, -1])


# ── Agent ───────────────────────────────────────────────────────────────────


class Agent(nn.Module):
    """Encoder, token projection, core and heads; see the module docstring."""

    def __init__(self, cfg: Optional[AgentConfig] = None):
        super().__init__()
        self.cfg = c = cfg or AgentConfig()
        assert c.encoder.arch == "cnn", "the agent uses the CNN encoder (D6)"
        self.encoder = CNNEncoder(c.encoder)
        dz = c.encoder.z_dim
        self.z_norm = nn.LayerNorm(dz)
        self.token = nn.Linear(dz + 2, c.d_model)
        self.core = Core(c)
        hid = c.head_hidden
        self.pi = _mlp(c.d_model, hid, 2 * c.n_bins)
        self.v = _mlp(c.d_model, hid, 1)
        self.path = _mlp(c.d_model, hid, 1)
        self.tilt = _mlp(c.d_model, hid, 4)
        self.scenario = _mlp(c.d_model, hid, 6)
        lo = torch.tensor(c.scen_lo, dtype=torch.float32)
        hi = torch.tensor(c.scen_hi, dtype=torch.float32)
        self.register_buffer("scen_mid", (hi + lo) / 2, persistent=False)
        self.register_buffer("scen_half", (hi - lo) / 2, persistent=False)
        self._init_weights()
        if c.channels_last:
            self.encoder.to(memory_format=torch.channels_last)
        self._compiled: Dict[str, nn.Module] = {}  # Not registered: state_dict unchanged

    def _init_weights(self):
        c = self.cfg
        for m in self.core.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        for blk in self.core.blocks:  # Residual projections scaled by depth (GPT-2)
            for m in (blk.proj, blk.fc2):
                nn.init.trunc_normal_(m.weight, std=0.02 / math.sqrt(2 * c.n_layers))
        last = self.pi[-1]
        with torch.no_grad():
            last.weight.mul_(c.policy_init_scale)
            last.bias.zero_()

    # ── Setup ──

    def load_encoder(self, path: str) -> dict:
        """Encoder weights from a pretrain_encoder.py checkpoint (model.pt); returns its
        info (path, run name, step, number of setups it was trained on)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        mine = asdict(self.cfg.encoder)
        for k in ("arch", "z_dim", "img_res", "cnn_widths", "cnn_blocks", "n_keypoints",
                  "gn_groups", "mlp_hidden"):
            v = tuple(cfg[k]) if isinstance(cfg[k], list) else cfg[k]
            assert v == mine[k], f"encoder {k}: checkpoint {v} != agent {mine[k]}"
        sd = {k[len("encoder."):]: v for k, v in ck["model"].items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(sd)
        name = ck.get("name") or ""
        if "data_setups" in ck:
            n_set = len(ck["data_setups"])
        else:  # Older checkpoints: the run name carries _set<S> when S > 1
            m = re.search(r"_set(\d+)", name)
            n_set = int(m.group(1)) if m else 1
        return {"path": os.path.abspath(path), "name": name, "step": ck.get("step"),
                "n_setups": n_set}

    def init_from(self, path: str) -> dict:
        """Weights of a ppo.py checkpoint where names and shapes match (encoder, token
        projection, core, policy, value, path, tilt heads); the rest (e.g. the old setup
        head, the new scenario head) keeps its initialization. Returns the info: path,
        loaded / skipped (checkpoint tensors not used, with the reason) / missing (agent
        tensors not in the checkpoint), and the checkpoint's name, iteration, env steps,
        encoder info and value-norm state."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        src = ck["agent"]
        own = self.state_dict()
        loaded, skipped = [], {}
        sd = {}
        for k, v in src.items():
            if k not in own:
                skipped[k] = "not in the agent"
            elif tuple(own[k].shape) != tuple(v.shape):
                skipped[k] = f"shape {tuple(v.shape)} != {tuple(own[k].shape)}"
            else:
                sd[k] = v
                loaded.append(k)
        missing = sorted(k for k in own if k not in sd)
        self.load_state_dict({**own, **sd})
        return {"path": os.path.abspath(path), "name": ck.get("name"),
                "iteration": ck.get("iteration"), "env_steps": ck.get("env_steps"),
                "encoder_info": ck.get("encoder_info"), "vnorm": ck.get("vnorm"),
                "loaded": loaded, "skipped": skipped, "missing": missing}

    def compile(self, **kw):
        """torch.compile the encoder and the core (used by every call path). The
        checkpointed encoder (unfrozen, enc_checkpoint) is compiled as one function around
        checkpoint(): checkpoint() around the compiled encoder fails in the backward
        recompute with channels_last activations (stride assertion, torch 2.14)."""
        self._compiled["encoder"] = torch.compile(self.encoder, **kw)
        self._compiled["encoder_ckpt"] = torch.compile(
            functools.partial(checkpoint, self.encoder, use_reentrant=False), **kw)
        self._compiled["core"] = torch.compile(self.core, **kw)
        return self

    def _amp(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.amp)

    def init_ring(self, B: int, device: torch.device) -> Ring:
        K, dz, r = self.cfg.window, self.cfg.encoder.z_dim, self.cfg.encoder.img_res
        return Ring(
            z=torch.zeros(B, K, dz, device=device),
            a=torch.zeros(B, K, 2, device=device),
            valid=torch.zeros(B, K, dtype=torch.bool, device=device),
            frame=torch.zeros(B, K, 1, r, r, dtype=torch.uint8, device=device),
        )

    # ── Pieces ──

    def encode(self, images: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """uint8 (N, 1, H, W) -> z (N, Dz) float32. grad (unfrozen encoder): with
        enc_checkpoint, activation checkpointing in chunks of enc_chunk images (only z is
        kept between forward and backward; the backward recomputes the forward), else one
        pass that keeps the activations (no recompute, memory ~ N)."""
        if self.cfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        with self._amp():
            if not grad or not self.cfg.enc_checkpoint:
                return self._compiled.get("encoder", self.encoder)(images).float()
            enc = self._compiled.get("encoder_ckpt") or functools.partial(
                checkpoint, self.encoder, use_reentrant=False)
            n, ch = images.shape[0], self.cfg.enc_chunk
            return torch.cat([enc(images[i:i + ch]) for i in range(0, n, ch)]).float()

    def tokens(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(..., Dz), (..., 2) -> (..., d_model)."""
        with self._amp():
            return self.token(torch.cat((self.z_norm(z), a.to(z.dtype)), -1))

    def run_core(self, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        core = self._compiled.get("core", self.core)
        with self._amp():
            return core(tok, valid).float()

    def heads(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Head outputs (fp32) of h (..., d)."""
        c = self.cfg
        t = self.tilt(h)
        sc = self.scenario(h)
        return {
            "logits": self.pi(h).unflatten(-1, (2, c.n_bins)),
            "value": self.v(h).squeeze(-1),
            "path": self.path(h).squeeze(-1),
            "tilt_mu": t[..., :2],
            "tilt_logvar": t[..., 2:].clamp(c.logvar_min, c.logvar_max),
            "scen_mu": sc[..., :3],
            "scen_logvar": sc[..., 3:].clamp(c.logvar_min, c.logvar_max),
        }

    def scen_norm(self, params: torch.Tensor) -> torch.Tensor:
        """(..., 3) scenario parameters -> scenario-head units ([-1, 1] over the range)."""
        return (params - self.scen_mid) / self.scen_half

    def scen_denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scen_half + self.scen_mid

    # ── Rollout path ──

    def act_z(self, z: torch.Tensor, a: torch.Tensor, first: torch.Tensor, ring: Ring,
              frame: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Ring]:
        """Push an already encoded frame and run the window: (outputs (B, ...), ring)."""
        ring = push_ring(ring, z, a, first, frame)
        h = self.run_core(self.tokens(ring.z, ring.a), ring.valid)
        return self.heads(h), ring

    def act(self, image: torch.Tensor, prev_action: torch.Tensor, first: torch.Tensor,
            ring: Ring) -> Tuple[Dict[str, torch.Tensor], Ring]:
        """Rollout step: image (B, 1, H, W) uint8 of the current frame, prev_action (B, 2)
        the applied action that led to it, first (B,) episode start. Encodes only this
        frame. Returns (outputs incl. z, new ring); the input ring is not modified."""
        z = self.encode(image)
        out, ring = self.act_z(z, prev_action, first, ring, image)
        out["z"] = z
        return out, ring

    def reencode_ring(self, ring: Ring) -> Ring:
        """Ring with z recomputed from its frames by the current encoder (after an
        encoder update); invalid entries get z of their (stale or zero) frames."""
        B, K = ring.valid.shape
        z = self.encode(ring.frame.flatten(0, 1)).view(B, K, -1)
        return ring._replace(z=z)

    # ── Update path ──

    def evaluate_windows(self, z: torch.Tensor, a: torch.Tensor, real: torch.Tensor,
                         first: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Outputs (Bm, T, ...) of the T = L - K + 1 windows of sequences z (Bm, L, Dz),
        a (Bm, L, 2), real / first (Bm, L) (first K - 1 positions: context)."""
        K = self.cfg.window
        Bm, L, _ = z.shape
        T = L - K + 1
        tok = self.tokens(z, a)  # (Bm, L, d), once per position
        win = tok.unfold(1, K, 1).permute(0, 1, 3, 2).reshape(Bm * T, K, -1)
        mask = window_masks(real, first, K).reshape(Bm * T, K)
        out = self.heads(self.run_core(win, mask))
        return {k: v.unflatten(0, (Bm, T)) for k, v in out.items()}

    def final_windows(self, z: torch.Tensor, a: torch.Tensor, real: torch.Tensor,
                      first: torch.Tensor, t: torch.Tensor, z_fin: torch.Tensor,
                      a_fin: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Outputs (Bm, S, ...) for final (pre-reset) frames: the window of step t
        (Bm, S) shifted by one with the token (z_fin (Bm, S, Dz), a_fin (Bm, S, 2))
        appended, as act_z() would push it without a reset. Sequences as in
        evaluate_windows(); t must lie in [0, T)."""
        K = self.cfg.window
        Bm, S = t.shape
        tok = self.tokens(z, a)  # (Bm, L, d)
        d = tok.shape[-1]
        idx = t[..., None] + 1 + torch.arange(K - 1, device=t.device)  # (Bm, S, K-1)
        ctx = tok.gather(1, idx.flatten(1)[..., None].expand(-1, -1, d)).view(Bm, S, K - 1, d)
        seg = torch.cumsum(first.long(), 1)
        seg_c = seg.gather(1, idx.flatten(1)).view(Bm, S, K - 1)
        seg_p = seg.gather(1, t + K - 1)  # Step t's frame
        m = real.gather(1, idx.flatten(1)).view(Bm, S, K - 1) & (seg_c == seg_p[..., None])
        tok_f = self.tokens(z_fin, a_fin).unsqueeze(2)  # (Bm, S, 1, d)
        win = torch.cat((ctx, tok_f.to(ctx.dtype)), 2).view(Bm * S, K, d)
        mask = torch.cat((m, torch.ones_like(m[..., :1])), 2).view(Bm * S, K)
        out = self.heads(self.run_core(win, mask))
        return {k: v.unflatten(0, (Bm, S)) for k, v in out.items()}


# ── Parameter groups ────────────────────────────────────────────────────────


def param_groups(agent: Agent, lr: float, weight_decay: float, encoder_lr_scale: float):
    """AdamW groups: decay on weight matrices / conv kernels only (not biases, norms,
    the relative-position bias, scalars); encoder groups at lr * encoder_lr_scale.
    Group names: decay, no_decay, enc_decay, enc_no_decay."""
    groups = {k: [] for k in ("decay", "no_decay", "enc_decay", "enc_no_decay")}
    for name, p in agent.named_parameters():
        enc = name.startswith("encoder.")
        nd = p.ndim < 2 or "rel_bias" in name
        groups[("enc_" if enc else "") + ("no_decay" if nd else "decay")].append(p)
    out = []
    for k, ps in groups.items():
        enc = k.startswith("enc_")
        out.append({"params": ps, "name": k, "lr": lr * (encoder_lr_scale if enc else 1.0),
                    "lr_scale": encoder_lr_scale if enc else 1.0,
                    "weight_decay": 0.0 if k.endswith("no_decay") else weight_decay})
    return out
