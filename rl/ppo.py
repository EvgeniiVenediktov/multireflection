"""
ppo.py - PPO trainer for the alignment agent (PLAN phase 6: P1-P6, T1-T6)

Run from rl/:
    uv run python ppo.py --encoder-init runs/encoder/<run>/model.pt          # defaults below
    uv run python ppo.py --num-envs 64 --total-env-steps 100000 --no-wandb   # smoke
    uv run python ppo.py --run-dir <dir> --resume                            # continue latest.pt
Cluster: cluster/ppo_l40s.slurm.

    env        AlignEnv: the four geometry.SETUPS (--setups all) or the measured cell,
               P_ref = per-setup P_max, auto-reset, history_len 0 (the agent keeps the
               windows), max_steps 64, step_cost 0.01, 64 px; envs persist across
               iterations (episodes span rollouts)
    rollout    T steps per env with agent.act(): one frame encoded per env and step,
               actions sampled per factor (Gumbel-max). Stored per env as a sequence of
               K - 1 context positions (the ring of the previous rollout) + T steps:
               frames (uint8), z, applied actions, first / real flags; labels of the
               frame at each step (path_norm, tilt, setup of the obs the agent saw)
    bootstrap  terminated (not move): none; truncated: V of the final (pre-reset) frame
               and its window. The final frames go into ceil(T / max_steps) slots per env
               during the rollout (no sync); their values are computed after the last
               step with the same parameters (agent.final_windows), as act() would
               give them. The value after the last step: act() on the current obs
               without committing it to the ring
    GAE        gamma 0.99, lambda 0.95; advantages normalized over the rollout batch;
               returns normalized by a running mean / std (ValueNorm, EMA with bias
               correction) for the value loss; values are de-normalized for GAE
    update     epochs x minibatches by env (all T steps of B / n_minibatches envs),
               windows recomputed by agent.evaluate_windows; clipped PPO + value +
               entropy (sum over the two factors) + auxiliary losses (path MSE, tilt
               Gaussian NLL, setup CE); AdamW, grad clip, target-KL early stop after
               each epoch
    encoder    --encoder-freeze (default): the update reuses the rollout's z (no encoder
               backward). --unfreeze-at ITER: from that iteration the encoder trains at
               lr x --encoder-lr-scale; z is recomputed with grad from the stored frames
               (checkpointed chunks), and the ring is re-encoded at each rollout start
    eval       every --eval-every iterations and at the end: evaluation.evaluate on the
               seeded set of (initial tilt, setup) pairs, separate env without
               auto-reset, argmax policy; oracle reference (baselines.plan_oracle)
               computed once and cached in the run dir
    logging    train.csv / eval.csv in the run dir; W&B (project herriott-rl) unless
               --no-wandb, display name built at start from the config
    checkpoint latest.pt (every --ckpt-minutes and at exit), best.pt (best eval mean
               return): agent, optimizer, value norm, counters, RNG, env and ring state,
               config. SIGTERM / SIGUSR1: abort the current rollout, write latest.pt,
               exit 0
Host syncs: none inside a rollout step; one per epoch (KL early stop) and a few per
iteration for logging; evaluation syncs per step (evaluation.evaluate).
"""

import argparse
import csv
import json
import math
import os
import signal
import time
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from agent import Agent, AgentConfig, Ring, entropy, greedy, log_prob, param_groups, sample
from env import AlignEnv, EnvConfig
from evaluation import eval_initial_tilts, eval_setups, evaluate
from geometry import SETUPS, CellGeometry

RL_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(RL_DIR, "runs", "ppo")


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    """Trainer settings (CLI arguments with the same names)."""

    # ── Env ──
    setups: str = "all"  # all = geometry.SETUPS, measured = the measured cell only
    num_envs: int = 1024
    rollout_len: int = 128
    max_steps: int = 64
    step_cost: float = 0.01
    img_res: int = 64
    path_ref: str = "max"
    init_center: Tuple[float, float] = (0.0, 0.0)
    init_radius: float = 2.0
    seed: int = 0

    # ── Agent ──
    history_len: int = 16  # Window K of the agent (the env's own stack is off)
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    encoder_init: str = ""
    encoder_freeze: bool = True
    unfreeze_at: int = -1  # Iteration at which the encoder starts training; -1 = never
    encoder_lr_scale: float = 0.1
    enc_chunk: int = 1024
    amp: bool = True
    compile: bool = True

    # ── PPO ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    path_coef: float = 0.5
    tilt_coef: float = 0.1
    setup_coef: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    value_norm_beta: float = 0.99

    # ── Run ──
    total_env_steps: int = 200_000_000
    eval_every: int = 200  # ~15 min on an L40S at B = 1024 (estimate, see the slurm header)
    eval_n: int = 4096
    eval_envs: int = 2048
    eval_seed: int = 0
    ckpt_minutes: float = 15.0
    wandb: bool = True
    wandb_project: str = "herriott-rl"
    name_tag: str = ""


def env_config(c: PPOConfig, num_envs: int, auto_reset: bool, seed: int) -> EnvConfig:
    return EnvConfig(
        num_envs=num_envs, seed=seed, setups=SETUPS if c.setups == "all" else (CellGeometry(),),
        path_ref=c.path_ref, max_steps=c.max_steps, step_cost=c.step_cost, auto_reset=auto_reset,
        init_center=tuple(c.init_center), init_radius=c.init_radius, img_res=c.img_res,
        history_len=0,
    )


def agent_config(c: PPOConfig, n_bins: int, n_setups: int) -> AgentConfig:
    return AgentConfig(window=c.history_len, d_model=c.d_model, n_layers=c.n_layers,
                       n_heads=c.n_heads, mlp_ratio=c.mlp_ratio, n_bins=n_bins,
                       n_setups=n_setups, amp=c.amp, enc_chunk=c.enc_chunk)


# ── Value normalization ─────────────────────────────────────────────────────


class ValueNorm:
    """Running mean / std of the returns: EMA (beta per update) with bias correction.
    Before the first update: mean 0, std 1."""

    def __init__(self, beta: float, device: torch.device, var_min: float = 1e-4):
        self.beta = beta
        self.var_min = var_min
        self.m1 = torch.zeros((), dtype=torch.float64, device=device)
        self.m2 = torch.zeros((), dtype=torch.float64, device=device)
        self.w = torch.zeros((), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.double()
        b = self.beta
        self.m1 = b * self.m1 + (1 - b) * x.mean()
        self.m2 = b * self.m2 + (1 - b) * (x * x).mean()
        self.w = b * self.w + (1 - b)

    def stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(mean, std) float32 scalars."""
        ok = self.w > 1e-12
        w = torch.where(ok, self.w, 1.0)
        mean = torch.where(ok, self.m1 / w, 0.0)
        var = torch.where(ok, (self.m2 / w - mean * mean).clamp(min=self.var_min), 1.0)
        return mean.float(), var.sqrt().float()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats()
        return y * std + mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"m1": self.m1, "m2": self.m2, "w": self.w}

    def load_state_dict(self, d: Dict[str, torch.Tensor]):
        dev = self.m1.device
        self.m1, self.m2, self.w = (d[k].to(dev) for k in ("m1", "m2", "w"))


# ── GAE ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def gae(rewards: torch.Tensor, values: torch.Tensor, terminated: torch.Tensor,
        truncated: torch.Tensor, v_final: torch.Tensor, v_last: torch.Tensor,
        gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Advantages and returns (B, T). values (B, T): V of the frame at each step;
    v_final (B, T): V of the final frame where truncated; v_last (B,): V of the obs after
    the last step. Terminated: no bootstrap; truncated: bootstrap from v_final; the trace
    is cut at both."""
    T = rewards.shape[1]
    done = terminated | truncated
    adv = torch.zeros_like(rewards)
    run = torch.zeros_like(v_last)
    next_v = v_last
    for t in reversed(range(T)):
        nv = torch.where(done[:, t], torch.where(truncated[:, t], v_final[:, t], 0.0), next_v)
        delta = rewards[:, t] + gamma * nv - values[:, t]
        run = delta + gamma * lam * (~done[:, t]).float() * run
        adv[:, t] = run
        next_v = values[:, t]
    return adv, adv + values


# ── Rollout storage ─────────────────────────────────────────────────────────


class Rollout:
    """Preallocated per-iteration buffers (B envs, C = K - 1 context + T steps)."""

    def __init__(self, B: int, T: int, K: int, dz: int, res: int, n_slots: int,
                 device: torch.device):
        L = K - 1 + T
        z = lambda *s, **kw: torch.zeros(*s, device=device, **kw)
        self.B, self.T, self.K, self.C, self.S = B, T, K, K - 1, n_slots
        self.frames = z(B, L, 1, res, res, dtype=torch.uint8)
        self.z = z(B, L, dz)
        self.a_in = z(B, L, 2)
        self.real = z(B, L, dtype=torch.bool)
        self.first = z(B, L, dtype=torch.bool)
        self.actions = z(B, T, 2, dtype=torch.long)
        self.logp = z(B, T)
        self.values = z(B, T)
        self.rewards = z(B, T)
        self.term = z(B, T, dtype=torch.bool)
        self.trunc = z(B, T, dtype=torch.bool)
        self.path = z(B, T)  # Labels of the frame at each step
        self.tilt = z(B, T, 2)
        self.setup = z(B, T, dtype=torch.long)
        self.slot_t = z(B, n_slots, dtype=torch.long)  # Truncation step per slot, -1 = none
        self.slot_frame = z(B, n_slots, 1, res, res, dtype=torch.uint8)
        self.slot_a = z(B, n_slots, 2)
        self.v_final = z(B, T)
        self.v_last = z(B)
        self.adv = z(B, T)
        self.ret = z(B, T)


# ── Trainer ─────────────────────────────────────────────────────────────────


class EvalPolicy:
    """evaluation.Policy adapter: argmax actions through the agent's act() path; the
    state is the agent's Ring. Accumulates head diagnostics over active steps."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.sums: Dict[str, torch.Tensor] = {}

    def _acc(self, k: str, v: torch.Tensor):
        self.sums[k] = self.sums.get(k, 0.0) + v.double().sum()

    @torch.no_grad()
    def __call__(self, obs, info, state):
        ag = self.agent
        if state is None:
            state = ag.init_ring(obs["image"].shape[0], obs["image"].device)
        out, state = ag.act(obs["image"], obs["action"], info["reset"], state)
        m = info["active"].float()
        pn = info["path_mm"] / info["path_ref"]
        tilt_err = (out["tilt_mu"] * ag.cfg.tilt_scale - info["tilt"]).abs().mean(-1)
        seen = (info["bounces"] >= 3).float() * m
        self._acc("n", m)
        self._acc("path_mae", (out["path"] - pn).abs() * m)
        self._acc("setup_acc", (out["setup_logits"].argmax(-1) == info["setup"]).float() * m)
        self._acc("tilt_mae_deg", tilt_err * m)
        self._acc("n_seen", seen)
        self._acc("tilt_mae_deg_ge3", tilt_err * seen)
        self._acc("tilt_std_deg", (0.5 * out["tilt_logvar"]).exp().mean(-1) * ag.cfg.tilt_scale * m)
        return greedy(out["logits"]), state

    def diagnostics(self) -> Dict[str, float]:
        s = {k: float(v) for k, v in self.sums.items()}
        n, ns = max(s.get("n", 0.0), 1.0), max(s.get("n_seen", 0.0), 1.0)
        return {"head_path_mae": s.get("path_mae", 0) / n, "head_setup_acc": s.get("setup_acc", 0) / n,
                "head_tilt_mae_deg": s.get("tilt_mae_deg", 0) / n,
                "head_tilt_mae_deg_ge3": s.get("tilt_mae_deg_ge3", 0) / ns,
                "head_tilt_std_deg": s.get("tilt_std_deg", 0) / n}


class Trainer:
    """PPO trainer, see the module docstring."""

    def __init__(self, cfg: PPOConfig, run_dir: str = "", device: Optional[torch.device] = None,
                 build_eval: bool = True, load_encoder: bool = True):
        """run_dir: default runs/ppo/<name>. load_encoder False: skip cfg.encoder_init
        (resume: the weights come from the checkpoint)."""
        self.cfg = c = cfg
        self.device = dev = torch.device(device or "cuda")
        torch.manual_seed(c.seed)

        # ── Env and agent ──
        self.env = AlignEnv(env_config(c, c.num_envs, True, c.seed))
        self.names = [g.name for g in self.env.setups]
        self.agent = Agent(agent_config(c, self.env.n_bins, self.env.n_setups)).to(dev)
        self.encoder_info = (self.agent.load_encoder(c.encoder_init)
                             if c.encoder_init and load_encoder else None)
        if c.compile:
            self.agent.compile()
        self.opt = torch.optim.AdamW(
            param_groups(self.agent, c.lr, c.weight_decay, c.encoder_lr_scale),
            lr=c.lr, betas=(0.9, 0.999), eps=1e-5)
        self.vnorm = ValueNorm(c.value_norm_beta, dev)
        self.gen = torch.Generator(device=dev)  # Action sampling
        self.gen.manual_seed(c.seed * 1_000_003 + 17)
        self.gen_mb = torch.Generator(device=dev)  # Minibatch permutations
        self.gen_mb.manual_seed(c.seed * 1_000_003 + 29)

        # ── Persistent rollout state ──
        B, K = c.num_envs, c.history_len
        self.ring: Ring = self.agent.init_ring(B, dev)
        self.next_first = torch.ones(B, dtype=torch.bool, device=dev)
        self.n_slots = math.ceil(c.rollout_len / c.max_steps)
        self.buf = Rollout(B, c.rollout_len, K, self.agent.cfg.encoder.z_dim, c.img_res,
                           self.n_slots, dev)
        self.iteration = 0
        self.env_steps = 0
        self.best_eval = -math.inf
        self.wandb_id: Optional[str] = None
        self.name = run_name(c, self.encoder_info)
        self.run_dir = run_dir or os.path.join(RUNS_DIR, self.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.stop_signal: Optional[int] = None
        self.eval_env: Optional[AlignEnv] = None
        self.oracle: Optional[Dict[str, Any]] = None
        self._build_eval = build_eval

    # ── Helpers ──

    def encoder_trains(self, iteration: Optional[int] = None) -> bool:
        c = self.cfg
        it = self.iteration if iteration is None else iteration
        if not c.encoder_freeze:
            return True
        return c.unfreeze_at >= 0 and it >= c.unfreeze_at

    def _signal(self, signum, frame):
        self.stop_signal = signum
        print(f"signal {signum}: writing latest.pt after the current step", flush=True)

    # ── Rollout ──

    @torch.no_grad()
    def rollout(self) -> Optional[Dict[str, torch.Tensor]]:
        """Collect T steps into self.buf; returns episode stats (device tensors), or None
        if a stop signal arrived (the partial rollout is dropped; env and ring stay
        consistent)."""
        c, ag, env, b = self.cfg, self.agent, self.env, self.buf
        C, T = b.C, b.T
        if self.encoder_trains():
            self.ring = ag.reencode_ring(self.ring)
        r = self.ring
        b.frames[:, :C] = r.frame[:, 1:]
        b.z[:, :C] = r.z[:, 1:]
        b.a_in[:, :C] = r.a[:, 1:]
        b.real[:, :C] = r.valid[:, 1:]
        b.first[:, :C] = False
        b.slot_t.fill_(-1)
        slot_n = torch.zeros(c.num_envs, dtype=torch.long, device=self.device)
        st = {k: torch.zeros((), dtype=torch.float64, device=self.device)
              for k in ("n", "ret", "len", "stop", "frac", "reward")}
        image, prev = env.image, env.prev_action
        for t in range(T):
            if self.stop_signal is not None:
                return None
            first = self.next_first
            lab = env.state_info()
            out, ring = ag.act(image, prev, first, self.ring)
            action = sample(out["logits"], self.gen)
            p = C + t
            b.frames[:, p] = image
            b.z[:, p] = out["z"]
            b.a_in[:, p] = prev
            b.real[:, p] = True
            b.first[:, p] = first
            b.actions[:, t] = action
            b.logp[:, t] = log_prob(out["logits"], action)
            b.values[:, t] = self.vnorm.denormalize(out["value"])
            b.path[:, t] = lab["path_norm"]
            b.tilt[:, t] = lab["tilt"]
            b.setup[:, t] = lab["setup"]
            self.ring = ring

            obs, rew, term, trunc, info = env.step(action)
            b.rewards[:, t] = rew
            b.term[:, t] = term
            b.trunc[:, t] = trunc
            for s in range(self.n_slots):
                w = trunc & (slot_n == s)
                b.slot_t[:, s] = torch.where(w, t, b.slot_t[:, s])
                b.slot_frame[:, s] = torch.where(w.view(-1, 1, 1, 1), info["final_image"],
                                                 b.slot_frame[:, s])
                b.slot_a[:, s] = torch.where(w[:, None], info["applied_action"], b.slot_a[:, s])
            slot_n = slot_n + trunc.long()
            done = term | trunc
            d = done.double()
            st["n"] += d.sum()
            st["ret"] += (info["ep_return"].double() * d).sum()
            st["len"] += (info["ep_length"].double() * d).sum()
            st["stop"] += (term.double()).sum()
            st["frac"] += ((info["ep_final_path_mm"] / info["path_ref"]).double() * d).sum()
            st["reward"] += rew.double().sum()
            self.next_first = done
            image, prev = obs["image"], obs["action"]

        # Bootstrap values: obs after the last step (not committed) and final frames
        out, _ = ag.act(image, prev, self.next_first, self.ring)
        b.v_last.copy_(self.vnorm.denormalize(out["value"]))
        B, S = b.B, b.S
        zf = ag.encode(b.slot_frame.flatten(0, 1)).view(B, S, -1)
        ok = b.slot_t >= 0
        tc = b.slot_t.clamp(min=0)
        fo = ag.final_windows(b.z, b.a_in, b.real, b.first, tc, zf, b.slot_a)
        vf = torch.where(ok, self.vnorm.denormalize(fo["value"]), 0.0)
        b.v_final.zero_().scatter_add_(1, tc, vf)
        st["slot_overflow"] = (slot_n > S).double().sum()
        return st

    # ── Update ──

    def update(self) -> Dict[str, torch.Tensor]:
        """GAE, value-norm update and the PPO epochs on self.buf; returns stats."""
        c, ag, b = self.cfg, self.agent, self.buf
        adv, ret = gae(b.rewards, b.values, b.term, b.trunc, b.v_final, b.v_last,
                       c.gamma, c.gae_lambda)
        b.adv.copy_(adv)
        b.ret.copy_(ret)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.vnorm.update(ret)
        ret_n = self.vnorm.normalize(ret)
        trains_enc = self.encoder_trains()
        B = b.B
        Bm = B // c.minibatches
        assert Bm * c.minibatches == B, "num_envs must be divisible by minibatches"
        params = [p for p in ag.parameters() if p.requires_grad]
        keys = ("loss", "pg", "vf", "ent", "path", "tilt", "setup", "kl", "clipfrac", "gn",
                "ratio_dev")
        acc = {k: torch.zeros((), device=self.device) for k in keys}
        n_mb = 0
        epochs = 0
        for epoch in range(c.epochs):
            perm = torch.randperm(B, generator=self.gen_mb, device=self.device)
            ekl = torch.zeros((), device=self.device)
            for i in range(c.minibatches):
                if self.stop_signal is not None:
                    break
                idx = perm[i * Bm:(i + 1) * Bm]
                if trains_enc:
                    z = ag.encode(b.frames[idx].flatten(0, 1), grad=True).view(Bm, -1, b.z.shape[-1])
                else:
                    z = b.z[idx]
                out = ag.evaluate_windows(z, b.a_in[idx], b.real[idx], b.first[idx])
                lp = log_prob(out["logits"], b.actions[idx])
                logr = lp - b.logp[idx]
                ratio = logr.exp()
                A = adv_n[idx]
                pg = -torch.minimum(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).mean()
                vf = ((out["value"] - ret_n[idx]) ** 2).mean()
                ent = entropy(out["logits"]).mean()
                l_path = ((out["path"] - b.path[idx]) ** 2).mean()
                tt = b.tilt[idx] / ag.cfg.tilt_scale
                lv = out["tilt_logvar"]
                l_tilt = (0.5 * (lv + (tt - out["tilt_mu"]) ** 2 * torch.exp(-lv))).mean()
                l_setup = F.cross_entropy(out["setup_logits"].flatten(0, 1), b.setup[idx].flatten())
                loss = (pg + c.vf_coef * vf - c.ent_coef * ent + c.path_coef * l_path
                        + c.tilt_coef * l_tilt + c.setup_coef * l_setup)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - logr).mean()
                    ekl += kl
                    if epoch == 0 and i == 0:
                        acc["ratio_dev"] += (ratio - 1).abs().max()
                    for k, v in (("loss", loss), ("pg", pg), ("vf", vf), ("ent", ent),
                                 ("path", l_path), ("tilt", l_tilt), ("setup", l_setup),
                                 ("kl", kl), ("gn", gn)):
                        acc[k] += v.detach().float()
                    acc["clipfrac"] += ((ratio - 1).abs() > c.clip).float().mean()
                n_mb += 1
            epochs += 1
            if self.stop_signal is not None:
                break
            if float(ekl) / c.minibatches > c.target_kl:  # Host sync, once per epoch
                break
        out = {k: v / max(n_mb, 1) for k, v in acc.items() if k != "ratio_dev"}
        out["ratio_dev_first_mb"] = acc["ratio_dev"]
        out["epochs"] = torch.tensor(float(epochs))
        with torch.no_grad():
            vr = ret.var()
            out["explained_var"] = 1 - (ret - b.values).var() / vr.clamp(min=1e-12)
            mean, std = self.vnorm.stats()
            out["vnorm_mean"], out["vnorm_std"] = mean, std
            out["adv_mean"], out["adv_std"] = adv.mean(), adv.std()
            out["ret_mean"] = ret.mean()
            out["value_mean"] = b.values.mean()
        return out

    # ── Evaluation ──

    def _eval_set(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Seeded (initial tilt, setup) pairs; tilts mapped into the init box (the
        default box is the full +-limit box, identical to eval_initial_tilts)."""
        c, env = self.cfg, self.eval_env
        L = env.limit
        u = eval_initial_tilts(c.eval_n, c.eval_seed, L)
        center = torch.tensor(c.init_center, dtype=torch.float32)
        tilts = (center + (c.init_radius / L) * u).clamp(-L, L)
        setups = (eval_setups(c.eval_n, env.n_setups, c.eval_seed) if env.n_setups > 1
                  else torch.zeros(c.eval_n, dtype=torch.long))
        return tilts.to(self.device), setups.to(self.device)

    def setup_eval(self):
        """Eval env, eval set and the oracle reference (cached in the run dir)."""
        from baselines import plan_oracle

        c = self.cfg
        self.eval_env = AlignEnv(env_config(c, c.eval_envs, False, c.eval_seed + 12345))
        tilts, setups = self._eval_set()
        path = os.path.join(self.run_dir, "oracle_eval.pt")
        key = {"n": c.eval_n, "seed": c.eval_seed, "setups": c.setups,
               "center": list(c.init_center), "radius": c.init_radius,
               "step_cost": c.step_cost, "path_ref": c.path_ref}
        oracle = None
        if os.path.exists(path):
            d = torch.load(path, map_location=self.device)
            if d.get("key") == key:
                oracle = d
        if oracle is None:
            t0 = time.time()
            plan = plan_oracle(self.eval_env, tilts, chunk=8, setups=setups)
            oracle = {"key": key, "ret": plan.ret.float(), "target_path_mm": plan.target_path_mm,
                      "plan_s": time.time() - t0}
            torch.save(oracle, path)
            print(f"oracle plan: {oracle['plan_s']:.1f} s, mean return {float(plan.ret.mean()):.4f}",
                  flush=True)
        self.oracle = oracle
        self.eval_tilts, self.eval_setups = tilts, setups

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Deterministic eval; flat metrics dict (eval/<split>/<metric>)."""
        if self.eval_env is None:
            self.setup_eval()
        pol = EvalPolicy(self.agent)
        ep, met = evaluate(self.eval_env, pol, self.eval_tilts, self.oracle["target_path_mm"],
                           self.oracle["ret"], init_setups=self.eval_setups)
        ref = self.oracle["ret"].double()
        blind = ep["init_bounces"] <= 2
        masks = {"all": torch.ones_like(blind), "blind": blind, "informative": ~blind}
        for i, nm in enumerate(self.names if len(self.names) > 1 else []):
            masks[f"setup:{nm}"] = ep["setup"] == i
        flat: Dict[str, float] = {}
        for split, m in masks.items():
            s = met[split]
            o = float(ref[m].mean()) if bool(m.any()) else float("nan")
            key = split.replace("setup:", "")
            for name, val in (("return", s["return_mean"]), ("oracle_return", o),
                              ("pct_oracle", 100.0 * s["return_mean"] / o if s["return_mean"] is not None and o else None),
                              ("regret", s.get("regret_mean")), ("path_frac", s.get("path_frac_pref_mean")),
                              ("success", s.get("success")), ("stop_rate", s["terminated"]),
                              ("length", s["length_mean"]), ("n", s["n"])):
                flat[f"eval/{key}/{name}"] = float("nan") if val is None else float(val)
        flat.update({f"eval/{k}": v for k, v in pol.diagnostics().items()})
        flat["eval/wall_s"] = met["wall_s"]
        return flat

    # ── Checkpoints ──

    def state(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.state_dict(),
            "opt": self.opt.state_dict(),
            "vnorm": self.vnorm.state_dict(),
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "best_eval": self.best_eval,
            "cfg": asdict(self.cfg),
            "agent_cfg": self.agent.cfg.to_dict(),
            "encoder_info": self.encoder_info,
            "name": self.name,
            "wandb_id": self.wandb_id,
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device),
                    "gen": self.gen.get_state(), "gen_mb": self.gen_mb.get_state()},
            "env": self.env.get_state(),
            "ring": self.ring._asdict(),
            "next_first": self.next_first,
        }

    def save(self, name: str):
        path = os.path.join(self.run_dir, name)
        tmp = path + ".tmp"
        torch.save(self.state(), tmp)
        os.replace(tmp, path)

    def load(self, path: str):
        d = torch.load(path, map_location="cpu", weights_only=False)  # Adam steps stay on CPU
        self.agent.load_state_dict(d["agent"])
        self.opt.load_state_dict(d["opt"])
        self.vnorm.load_state_dict(d["vnorm"])
        self.iteration, self.env_steps = d["iteration"], d["env_steps"]
        self.best_eval = d["best_eval"]
        self.encoder_info = d.get("encoder_info", self.encoder_info)
        self.name, self.wandb_id = d["name"], d.get("wandb_id")
        torch.set_rng_state(d["rng"]["torch"].cpu())
        torch.cuda.set_rng_state(d["rng"]["cuda"].cpu(), self.device)
        self.gen.set_state(d["rng"]["gen"].cpu())
        self.gen_mb.set_state(d["rng"]["gen_mb"].cpu())
        self.env.set_state(d["env"])
        self.ring = Ring(**{k: v.to(self.device) for k, v in d["ring"].items()})
        self.next_first = d["next_first"].to(self.device)

    # ── Main loop ──

    def train(self):
        c = self.cfg
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGUSR1, self._signal)
        log = Logger(self.run_dir, self.name, asdict(c), self.encoder_info, c.wandb,
                     c.wandb_project, self.wandb_id)
        self.wandb_id = log.wandb_id
        if self._build_eval:
            self.setup_eval()
        sync = lambda: torch.cuda.synchronize(self.device)
        steps_per_iter = c.num_envs * c.rollout_len
        last_ckpt = t_start = time.time()
        print(f"{self.name}: {count(self.agent)} -> {self.run_dir}", flush=True)
        while self.env_steps < c.total_env_steps and self.stop_signal is None:
            if self.encoder_trains() and not self.encoder_trains(self.iteration - 1) \
                    and self.iteration > 0:
                print(f"iteration {self.iteration}: encoder unfrozen "
                      f"(lr x {c.encoder_lr_scale})", flush=True)
            sync()
            t0 = time.time()
            st = self.rollout()
            if st is None:
                break
            sync()
            t1 = time.time()
            us = self.update()
            sync()
            t2 = time.time()
            self.iteration += 1
            self.env_steps += steps_per_iter
            row = {"iteration": self.iteration, "env_steps": self.env_steps,
                   "time_rollout_s": t1 - t0, "time_update_s": t2 - t1, "time_eval_s": 0.0,
                   "fps": steps_per_iter / (t2 - t0),
                   "fps_rollout": steps_per_iter / (t1 - t0),
                   "fps_update": steps_per_iter / (t2 - t1),
                   "encoder_trains": int(self.encoder_trains(self.iteration - 1))}
            n = float(st["n"])
            row.update({"episodes": n,
                        "ep_return": float(st["ret"]) / n if n else float("nan"),
                        "ep_length": float(st["len"]) / n if n else float("nan"),
                        "ep_stop_rate": float(st["stop"]) / n if n else float("nan"),
                        "ep_final_path_frac": float(st["frac"]) / n if n else float("nan"),
                        "reward_mean": float(st["reward"]) / steps_per_iter,
                        "slot_overflow": float(st["slot_overflow"])})
            row.update({f"loss/{k}" if k in ("loss", "pg", "vf", "ent", "path", "tilt", "setup")
                        else f"ppo/{k}": float(v) for k, v in us.items()})
            row["lr"] = self.opt.param_groups[0]["lr"]
            last = self.env_steps >= c.total_env_steps
            ev = None
            if c.eval_every > 0 and (self.iteration % c.eval_every == 0 or last) \
                    and self.stop_signal is None:
                sync()
                te = time.time()
                ev = self.evaluate()
                row["time_eval_s"] = time.time() - te
                if ev["eval/all/return"] > self.best_eval:
                    self.best_eval = ev["eval/all/return"]
                    self.save("best.pt")
                ev["iteration"], ev["env_steps"] = self.iteration, self.env_steps
            row["wall_min"] = (time.time() - t_start) / 60
            log.train(row)
            if ev is not None:
                log.eval(ev)
                print(f"  eval: return {ev['eval/all/return']:+.4f} ({ev['eval/all/pct_oracle']:.1f} % "
                      f"of oracle), P/Pmax {ev['eval/all/path_frac']:.3f}, success "
                      f"{ev['eval/all/success']:.3f}, stop {ev['eval/all/stop_rate']:.3f}, len "
                      f"{ev['eval/all/length']:.1f}", flush=True)
            if time.time() - last_ckpt > 60 * c.ckpt_minutes:
                self.save("latest.pt")
                last_ckpt = time.time()
        self.save("latest.pt")
        log.close()
        why = f"signal {self.stop_signal}" if self.stop_signal is not None else "done"
        print(f"stopped ({why}) at iteration {self.iteration}, {self.env_steps} env steps; "
              f"latest.pt written", flush=True)


# ── Logging ─────────────────────────────────────────────────────────────────


def count(agent: Agent) -> str:
    n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    return (f"encoder {n(agent.encoder):.2f} M, core {n(agent.core):.2f} M, total "
            f"{n(agent):.2f} M params")


class Logger:
    """train.csv / eval.csv (appended on resume) and W&B."""

    def __init__(self, run_dir: str, name: str, config: dict, encoder_info: Optional[dict],
                 use_wandb: bool, project: str, wandb_id: Optional[str]):
        self.run_dir = run_dir
        self.files: Dict[str, Any] = {}
        self.writers: Dict[str, csv.DictWriter] = {}
        self.wb = None
        self.wandb_id = wandb_id
        if use_wandb:
            import wandb

            self.wandb_id = wandb_id or uuid.uuid4().hex[:12]  # Kept in the checkpoint
            self.wb = wandb.init(project=project, name=name, id=self.wandb_id, resume="allow",
                                 config={**config, "encoder_info": encoder_info},
                                 dir=os.environ.get("WANDB_DIR", run_dir))
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"name": name, "config": config, "encoder_info": encoder_info}, f, indent=1)

    def _write(self, kind: str, row: Dict[str, Any]):
        if kind not in self.writers:
            path = os.path.join(self.run_dir, f"{kind}.csv")
            header = None
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path) as f:
                    header = next(csv.reader(f))
            f = open(path, "a", newline="")
            w = csv.DictWriter(f, fieldnames=header or list(row), extrasaction="ignore")
            if header is None:
                w.writeheader()
            self.files[kind], self.writers[kind] = f, w
        self.writers[kind].writerow(row)
        self.files[kind].flush()

    def train(self, row: Dict[str, Any]):
        self._write("train", row)
        if self.wb is not None:
            self.wb.log({k: v for k, v in row.items() if k != "env_steps"}, step=row["env_steps"])
        print(f"it {row['iteration']:5d} {row['env_steps'] / 1e6:8.2f} M  ret {row['ep_return']:+.3f} "
              f"len {row['ep_length']:5.1f} stop {row['ep_stop_rate']:.2f} frac "
              f"{row['ep_final_path_frac']:.3f} | pg {row['loss/pg']:+.4f} vf {row['loss/vf']:.4f} "
              f"ent {row['loss/ent']:.3f} path {row['loss/path']:.4f} tilt {row['loss/tilt']:+.3f} "
              f"setup {row['loss/setup']:.3f} kl {row['ppo/kl']:.4f} ev {row['ppo/explained_var']:+.3f} "
              f"ep {int(row['ppo/epochs'])} | {row['fps']:.0f} fps (roll {row['time_rollout_s']:.1f} s, "
              f"upd {row['time_update_s']:.1f} s)", flush=True)

    def eval(self, row: Dict[str, Any]):
        self._write("eval", row)
        if self.wb is not None:
            self.wb.log({k: v for k, v in row.items() if k not in ("env_steps", "iteration")},
                        step=row["env_steps"])

    def close(self):
        for f in self.files.values():
            f.close()
        if self.wb is not None:
            self.wb.finish()


def run_name(c: PPOConfig, encoder_info: Optional[dict]) -> str:
    """Display / directory name from the config, built at start:
    ppo_cnn[-pre<S>]_k<K>_set<S>_b<B>_s<seed>[_uf<it>|_e2e][_<tag>]_<jobid>; -pre<S>: the
    encoder was initialized from a checkpoint pretrained on S setups."""
    enc = "cnn"
    if encoder_info is not None:
        enc += f"-pre{encoder_info.get('n_setups', 1)}"
    n_set = len(SETUPS) if c.setups == "all" else 1
    name = f"ppo_{enc}_k{c.history_len}_set{n_set}_b{c.num_envs}_s{c.seed}"
    if not c.encoder_freeze:
        name += "_e2e"
    elif c.unfreeze_at >= 0:
        name += f"_uf{c.unfreeze_at}"
    if c.name_tag:
        name += f"_{c.name_tag}"
    return name + "_" + os.environ.get("SLURM_JOB_ID", "local")


# ── CLI ─────────────────────────────────────────────────────────────────────


RESUME_OVERRIDES = ("total_env_steps", "eval_every", "ckpt_minutes", "wandb", "wandb_project",
                    "unfreeze_at", "encoder_lr_scale", "compile", "target_kl")


def build_parser(suppress: bool = False) -> argparse.ArgumentParser:
    """CLI: one flag per PPOConfig field (defaults from PPOConfig); suppress: no defaults
    (the namespace then holds only the flags given)."""
    d = PPOConfig()
    dflt = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    ap = argparse.ArgumentParser(description="PPO trainer for the Herriott alignment agent")
    ap.add_argument("--run-dir", default=dflt(""), help="Default runs/ppo/<name>")
    ap.add_argument("--resume", action="store_true", default=dflt(False),
                    help="Continue from <run-dir>/latest.pt; the config comes from the "
                         "checkpoint, only " + ", ".join(RESUME_OVERRIDES) + " follow the flags given")
    for f in fields(PPOConfig):
        flag = "--" + f.name.replace("_", "-")
        v = getattr(d, f.name)
        if f.name in ("encoder_freeze", "wandb", "amp", "compile"):
            ap.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=dflt(v))
        elif f.name == "init_center":
            ap.add_argument(flag, type=float, nargs=2, default=dflt(list(v)))
        elif f.name == "setups":
            ap.add_argument(flag, choices=("all", "measured"), default=dflt(v))
        else:
            ap.add_argument(flag, type=type(v), default=dflt(v))
    return ap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cli = {f.name: getattr(args, f.name) for f in fields(PPOConfig)}
    cli["init_center"] = tuple(cli["init_center"])
    if args.resume:
        assert args.run_dir, "--resume needs --run-dir"
        path = os.path.join(args.run_dir, "latest.pt")
        saved = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
        saved["init_center"] = tuple(saved["init_center"])
        given = vars(build_parser(suppress=True).parse_args(argv))
        for k in RESUME_OVERRIDES:  # Run control from the flags given; the rest as saved
            if k in given:
                saved[k] = cli[k]
        ignored = sorted(k for k in given if k not in RESUME_OVERRIDES + ("run_dir", "resume"))
        if ignored:
            print(f"resume: ignoring {ignored} (taken from the checkpoint)", flush=True)
        cfg = PPOConfig(**saved)
        trainer = Trainer(cfg, args.run_dir, load_encoder=False)
        trainer.load(path)
        print(f"resumed {trainer.name} at iteration {trainer.iteration}, "
              f"{trainer.env_steps} env steps", flush=True)
    else:
        cfg = PPOConfig(**cli)
        if cfg.encoder_init:
            cfg.encoder_init = os.path.abspath(cfg.encoder_init)
        trainer = Trainer(cfg, args.run_dir)
    trainer.train()


if __name__ == "__main__":
    main()
