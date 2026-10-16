"""
ppo.py - PPO trainer for the alignment agent (PLAN phase 6: P1-P6, T1-T6; stage 2)

Run from rl/:
    uv run python ppo.py --encoder-init runs/encoder/<run>/model.pt          # defaults below
    uv run python ppo.py --setups all --num-envs 64 --total-env-steps 100000 --no-wandb  # smoke
    uv run python ppo.py --run-dir <dir> --resume                            # continue latest.pt
Cluster: cluster/ppo_l40s.slurm.

    env        AlignEnv, auto-reset, history_len 0 (the agent keeps the windows), max_steps
               64, step_cost 0.01, 64 px; envs persist across iterations (episodes span
               rollouts). --setups full (default, stage 2): the scenario atlas
               (scenarios.py; --atlas, built there if missing), P_ref per scenario =
               max(P_max, P_floor), scenarios drawn per episode by the regret-prioritized
               ScenarioSampler (--plr; --no-plr: uniform over the good set; 10 %
               impossible either way). --setups all: the four geometry.SETUPS, P_ref =
               per-setup P_max; measured: the measured cell
    init       --encoder-init: encoder weights of a pretrain_encoder.py checkpoint.
               --init-from: a previous ppo.py checkpoint, every tensor whose name and
               shape match (encoder, token, core, policy / value / path / tilt heads; the
               skipped and missing ones are printed and kept in config.json), its
               value-norm statistics; fresh optimizer, counters, entropy coefficient
    rollout    T steps per env with agent.act(): one frame encoded per env and step,
               actions sampled per factor (Gumbel-max). Stored per env as a sequence of
               K - 1 context positions (the ring of the previous rollout) + T steps:
               frames (uint8), z, applied actions, first / real flags; labels of the
               frame at each step (path_norm, tilt, scenario parameters of the obs the
               agent saw). Episode ends update the sampler's scores (scatter ops)
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
               entropy (sum over the two factors) + auxiliary losses (path MSE, tilt and
               scenario Gaussian NLL); AdamW, grad clip, target-KL early stop after each
               epoch
    lr         linear warmup over --lr-warmup iterations (from lr / warmup to the peak
               --lr), then cosine decay to lr x --lr-final-frac at --total-env-steps
               (lr_at); encoder groups at lr x --encoder-lr-scale
    entropy    --ent-adaptive (default): coefficient alpha with a target entropy that
               decays linearly from --ent-target-start to --ent-target-end nats (joint
               entropy of the two factors) over the first --ent-target-frac of the env
               steps, then stays; per minibatch log alpha -= --ent-lr x (H - target)
               (dual gradient), clamped to [--ent-coef-min, --ent-coef-max], initial
               --ent-coef. --no-ent-adaptive: fixed --ent-coef
    encoder    --encoder-freeze (default): the update reuses the rollout's z (no encoder
               backward). --unfreeze-at ITER: from that iteration the encoder trains at
               lr x --encoder-lr-scale; z is recomputed with grad from the stored frames
               (bf16, channels_last, compiled; activation-checkpointed chunks, or one pass
               without recompute with --no-enc-checkpoint), once per frame and epoch; the
               ring is re-encoded at each rollout start
    eval       every --eval-every iterations and at the end, separate envs without
               auto-reset, deterministic (argmax) and sampled (one seeded sample per
               episode and step, --eval-sampled) policies; oracle reference
               (baselines.plan_oracle) computed once per set and cached in the run dir.
               Sets: "full" (--setups full): --eval-n (initial tilt, scenario) pairs,
               scenarios 90 % uniform over the good set and 10 % over the impossible set
               (scenarios.eval_scenarios), splits all / blind / informative / good /
               impossible / separation bins (good scenarios); "set4": the four setups'
               set of run 4398136 (with --setups full only with --eval-set4); "measured"
               with --setups measured. Prefixes: eval_full/, eval_full_samp/ and, for the
               four-setup set, eval/ (the keys of run 4398136) and eval_samp/
    logging    train.csv / eval.csv in the run dir; W&B (project herriott-rl) unless
               --no-wandb, display name built at start from the config
    checkpoint latest.pt (every --ckpt-minutes and at exit), best.pt (best deterministic
               % of the oracle: on the full set with --setups full, else on the set of
               the run): agent, optimizer, value norm, entropy coefficient, sampler,
               counters, RNG, env and ring state, config. SIGTERM / SIGUSR1: abort the
               current rollout, write latest.pt, exit 0
Host syncs: none inside a rollout step; one per epoch (KL early stop) and a few per
iteration for logging (sampler statistics); evaluation syncs per step
(evaluation.evaluate).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import time
import uuid
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from agent import Agent, AgentConfig, Ring, entropy, greedy, log_prob, param_groups, sample
from env import AlignEnv, EnvConfig
from evaluation import BLIND_MAX_BOUNCES, eval_initial_tilts, eval_setups, evaluate, split_metrics
from geometry import SETUPS, CellGeometry

RL_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(RL_DIR, "runs", "ppo")
SETUP_CHOICES = ("full", "all", "measured")


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    """Trainer settings (CLI arguments with the same names)."""

    # ── Env ──
    setups: str = "full"  # full = scenario atlas, all = geometry.SETUPS, measured = one cell
    atlas: str = ""  # --setups full: atlas file ("" = scenarios.DEFAULT_ATLAS), built if missing
    num_envs: int = 1024
    rollout_len: int = 128
    max_steps: int = 64
    step_cost: float = 0.01
    img_res: int = 64
    path_ref: str = "max"  # --setups all / measured; full: the atlas P_ref
    init_center: Tuple[float, float] = (0.0, 0.0)
    init_radius: float = 2.0
    seed: int = 0

    # ── Scenario sampler (--setups full) ──
    plr: bool = True  # Regret-prioritized replay; False: good scenarios uniform
    plr_impossible: float = 0.1  # Share of draws from the impossible set
    plr_alpha: float = 0.3  # Score EMA
    plr_p_fresh: float = 0.3  # Fresh uniform share of the good draws (1 until min_seen)
    plr_min_seen: int = 2048
    plr_beta: float = 1.0  # Rank temperature (0.3 put ~half of all episodes on one scenario)
    plr_rho: float = 0.1  # Staleness share

    # ── Agent ──
    history_len: int = 16  # Window K of the agent (the env's own stack is off)
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mlp_ratio: float = 4.0
    encoder_init: str = ""
    init_from: str = ""  # Previous ppo.py checkpoint (matching tensors, value norm)
    encoder_freeze: bool = True
    unfreeze_at: int = -1  # Iteration at which the encoder starts training; -1 = never
    encoder_lr_scale: float = 0.1
    enc_chunk: int = 1024
    enc_checkpoint: bool = True  # Unfrozen encoder: checkpointed chunks (False: no recompute)
    amp: bool = True
    compile: bool = True

    # ── PPO ──
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01  # Initial (adaptive) or fixed entropy coefficient
    ent_adaptive: bool = True
    ent_target_start: float = 2.0  # nats, joint entropy of the two factors
    ent_target_end: float = 0.7
    ent_target_frac: float = 0.8  # Share of total_env_steps over which the target decays
    ent_lr: float = 1e-3  # Dual-gradient step on log alpha per minibatch
    ent_coef_min: float = 1e-4
    ent_coef_max: float = 0.1
    path_coef: float = 0.5
    tilt_coef: float = 0.1
    scen_coef: float = 0.1
    lr: float = 3e-4  # Peak
    lr_warmup: int = 20  # Iterations of linear warmup
    lr_final_frac: float = 0.1  # Cosine decay to lr x this at total_env_steps
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    value_norm_beta: float = 0.99

    # ── Run ──
    total_env_steps: int = 0  # 0 = 1e9 with --setups full, else 2e8
    eval_every: int = 200  # ~15 min on an L40S at B = 1024 with the frozen encoder
    eval_n: int = 4096
    eval_envs: int = 2048
    eval_seed: int = 0
    eval_impossible: float = 0.1  # Impossible share of the full eval set
    eval_set4: bool = True  # --setups full: also the four-setup set
    eval_sampled: bool = True  # Also the sampled policy
    ckpt_minutes: float = 15.0
    wandb: bool = True
    wandb_project: str = "herriott-rl"
    name_tag: str = ""


def default_total_steps(c: PPOConfig) -> int:
    return 1_000_000_000 if c.setups == "full" else 200_000_000


def env_config(c: PPOConfig, num_envs: int, auto_reset: bool, seed: int,
               setups: Optional[str] = None) -> EnvConfig:
    """EnvConfig of the trainer; setups overrides c.setups (full: no setups, the
    scenario table is passed to AlignEnv)."""
    s = setups or c.setups
    geoms = () if s == "full" else (SETUPS if s == "all" else (CellGeometry(),))
    return EnvConfig(
        num_envs=num_envs, seed=seed, setups=geoms, path_ref=c.path_ref, max_steps=c.max_steps,
        step_cost=c.step_cost, auto_reset=auto_reset, init_center=tuple(c.init_center),
        init_radius=c.init_radius, img_res=c.img_res, history_len=0,
    )


def agent_config(c: PPOConfig, n_bins: int) -> AgentConfig:
    return AgentConfig(window=c.history_len, d_model=c.d_model, n_layers=c.n_layers,
                       n_heads=c.n_heads, mlp_ratio=c.mlp_ratio, n_bins=n_bins, amp=c.amp,
                       enc_chunk=c.enc_chunk, enc_checkpoint=c.enc_checkpoint)


# ── Schedules ───────────────────────────────────────────────────────────────


def lr_at(c: PPOConfig, iteration: int, env_steps: int) -> float:
    """Learning rate of the update at this iteration (env_steps before its rollout):
    peak c.lr x (iteration + 1) / lr_warmup during the warmup, then cosine from the
    peak at the end of the warmup to lr x lr_final_frac at total_env_steps (constant
    after)."""
    if iteration < c.lr_warmup:
        return c.lr * (iteration + 1) / c.lr_warmup
    s0 = c.lr_warmup * c.num_envs * c.rollout_len
    prog = min(max((env_steps - s0) / max(c.total_env_steps - s0, 1), 0.0), 1.0)
    lo = c.lr * c.lr_final_frac
    return lo + (c.lr - lo) * 0.5 * (1.0 + math.cos(math.pi * prog))


def ent_target_at(c: PPOConfig, env_steps: int) -> float:
    """Target entropy: linear from ent_target_start to ent_target_end over the first
    ent_target_frac of total_env_steps, then constant."""
    f = min(env_steps / max(c.ent_target_frac * c.total_env_steps, 1.0), 1.0)
    return c.ent_target_start + (c.ent_target_end - c.ent_target_start) * f


def ent_coef_step(log_alpha: torch.Tensor, ent: torch.Tensor, target: float,
                  c: PPOConfig) -> torch.Tensor:
    """Dual-gradient step on log alpha: down when the entropy is above the target, up
    when below; clamped to [log ent_coef_min, log ent_coef_max] (no host sync)."""
    la = log_alpha - c.ent_lr * (ent.detach().float() - target)
    return la.clamp(math.log(c.ent_coef_min), math.log(c.ent_coef_max))


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
        self.scen = z(B, T, 3)  # Scenario parameters in scenario-head units
        self.slot_t = z(B, n_slots, dtype=torch.long)  # Truncation step per slot, -1 = none
        self.slot_frame = z(B, n_slots, 1, res, res, dtype=torch.uint8)
        self.slot_a = z(B, n_slots, 2)
        self.v_final = z(B, T)
        self.v_last = z(B)
        self.adv = z(B, T)
        self.ret = z(B, T)


# ── Evaluation ──────────────────────────────────────────────────────────────


class EvalPolicy:
    """evaluation.Policy adapter through the agent's act() path; the state is the
    agent's Ring. gen: sampled actions (Gumbel-max from this generator), else argmax.
    Accumulates head diagnostics over active steps."""

    def __init__(self, agent: Agent, gen: Optional[torch.Generator] = None):
        self.agent = agent
        self.gen = gen
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
        scen_err = (ag.scen_denorm(out["scen_mu"]) - info["scenario_params"]).abs()
        seen = (info["bounces"] >= 3).float() * m
        self._acc("n", m)
        self._acc("path_mae", (out["path"] - pn).abs() * m)
        self._acc("tilt_mae_deg", tilt_err * m)
        self._acc("n_seen", seen)
        self._acc("tilt_mae_deg_ge3", tilt_err * seen)
        self._acc("tilt_std_deg", (0.5 * out["tilt_logvar"]).exp().mean(-1) * ag.cfg.tilt_scale * m)
        for j, k in enumerate(("sep_mae_mm", "laser_pitch_mae_deg", "laser_yaw_mae_deg")):
            self._acc(k, scen_err[:, j] * m)
        act = greedy(out["logits"]) if self.gen is None else sample(out["logits"], self.gen)
        return act, state

    def diagnostics(self) -> Dict[str, float]:
        s = {k: float(v) for k, v in self.sums.items()}
        n, ns = max(s.get("n", 0.0), 1.0), max(s.get("n_seen", 0.0), 1.0)
        out = {"head_path_mae": s.get("path_mae", 0) / n,
               "head_tilt_mae_deg": s.get("tilt_mae_deg", 0) / n,
               "head_tilt_mae_deg_ge3": s.get("tilt_mae_deg_ge3", 0) / ns,
               "head_tilt_std_deg": s.get("tilt_std_deg", 0) / n}
        for k in ("sep_mae_mm", "laser_pitch_mae_deg", "laser_yaw_mae_deg"):
            out[f"head_{k}"] = s.get(k, 0) / n
        return out


@dataclass
class EvalSet:
    """A fixed evaluation set: env, (initial tilt, setup / scenario) pairs, oracle."""

    name: str  # full / set4 / measured
    env: AlignEnv
    tilts: torch.Tensor
    setups: torch.Tensor
    oracle: Dict[str, Any]
    prefix: str  # Deterministic policy
    prefix_samp: str  # Sampled policy
    splits: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]


def _digest(*ts: torch.Tensor) -> str:
    h = hashlib.sha1()
    for t in ts:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ── Trainer ─────────────────────────────────────────────────────────────────


class Trainer:
    """PPO trainer, see the module docstring."""

    def __init__(self, cfg: PPOConfig, run_dir: str = "", device: Optional[torch.device] = None,
                 build_eval: bool = True, load_encoder: bool = True):
        """run_dir: default runs/ppo/<name>. load_encoder False: skip cfg.encoder_init
        and cfg.init_from (resume: the weights come from the checkpoint)."""
        self.cfg = c = cfg
        if c.total_env_steps <= 0:
            c.total_env_steps = default_total_steps(c)
        assert c.setups in SETUP_CHOICES, f"setups {c.setups!r} not in {SETUP_CHOICES}"
        self.device = dev = torch.device(device or "cuda")
        torch.manual_seed(c.seed)

        # ── Atlas and sampler (--setups full) ──
        self.atlas = self.sampler = None
        scen_kw: Dict[str, Any] = {}
        if c.setups == "full":
            from scenarios import DEFAULT_ATLAS, SamplerConfig, ScenarioSampler, load_or_build

            c.atlas = os.path.abspath(c.atlas or DEFAULT_ATLAS)
            t0 = time.time()
            self.atlas = load_or_build(c.atlas, dev, log=lambda s: print(s, flush=True))
            ec = EnvConfig()
            self.atlas.check(ec.max_bounces, ec.geometry.m2_tilt_limit, ec.step_unit)
            print(f"{self.atlas.summary()}\natlas ready in {time.time() - t0:.1f} s", flush=True)
            self.sampler = ScenarioSampler(self.atlas, SamplerConfig(
                impossible_share=c.plr_impossible, alpha=c.plr_alpha, p_fresh=c.plr_p_fresh,
                min_seen=c.plr_min_seen, beta=c.plr_beta, rho=c.plr_rho, replay=c.plr))
            scen_kw = {"scenarios": self.atlas.table(), "sampler": self.sampler}

        # ── Env and agent ──
        self.env = AlignEnv(env_config(c, c.num_envs, True, c.seed), **scen_kw)
        self.names = [g.name for g in self.env.setups]
        self.agent = Agent(agent_config(c, self.env.n_bins)).to(dev)
        self.vnorm = ValueNorm(c.value_norm_beta, dev)
        # init_from first (all matching tensors), then encoder_init, so a freshly pretrained
        # encoder can replace the one in the PPO checkpoint
        self.encoder_info = None
        self.init_info: Optional[Dict[str, Any]] = None
        if c.init_from and load_encoder:
            info = self.agent.init_from(c.init_from)
            if info.get("vnorm") is not None:
                self.vnorm.load_state_dict(info["vnorm"])
            if info.get("encoder_info") is not None:
                self.encoder_info = info["encoder_info"]
            self.init_info = {k: v for k, v in info.items() if k != "vnorm"}
            other = sorted(k for k in info["skipped"] if not k.startswith("setup."))
            print(f"init from {info['path']} ({info['name']}, iteration {info['iteration']}): "
                  f"{len(info['loaded'])} tensors loaded; skipped {sorted(info['skipped'])}; "
                  f"fresh {info['missing']}; value norm from the checkpoint", flush=True)
            if other:
                print(f"WARNING: init-from skipped tensors beyond the setup head: {other}",
                      flush=True)
        if c.encoder_init and load_encoder:
            self.encoder_info = self.agent.load_encoder(c.encoder_init)
            if c.init_from:
                print(f"encoder replaced by {c.encoder_init} after init-from", flush=True)
        if c.compile:
            self.agent.compile()
        self.opt = torch.optim.AdamW(
            param_groups(self.agent, c.lr, c.weight_decay, c.encoder_lr_scale),
            lr=c.lr, betas=(0.9, 0.999), eps=1e-5)
        self.log_alpha = torch.tensor(math.log(c.ent_coef), device=dev)
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
        self.best_metric = "eval_full/all/pct_oracle" if c.setups == "full" else "eval/all/pct_oracle"
        self.wandb_id: Optional[str] = None
        self.name = run_name(c, self.encoder_info)
        self.run_dir = run_dir or os.path.join(RUNS_DIR, self.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.stop_signal: Optional[int] = None
        self.eval_sets: Optional[List[EvalSet]] = None
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
            b.scen[:, t] = ag.scen_norm(lab["scenario_params"])
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
            if self.sampler is not None:
                self.sampler.observe(info["setup"], info["ep_final_path_mm"], done,
                                     info["scenario_src"])
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
        """GAE, value-norm update and the PPO epochs on self.buf; returns stats. Sets
        the learning rate of this iteration (lr_at) and, with ent_adaptive, steps the
        entropy coefficient after every minibatch."""
        c, ag, b = self.cfg, self.agent, self.buf
        lr = lr_at(c, self.iteration, self.env_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        target = ent_target_at(c, self.env_steps)
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
        keys = ("loss", "pg", "vf", "ent", "path", "tilt", "scen", "kl", "clipfrac", "gn",
                "ent_coef", "ratio_dev")
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
                lvs = out["scen_logvar"]
                l_scen = (0.5 * (lvs + (b.scen[idx] - out["scen_mu"]) ** 2 * torch.exp(-lvs))).mean()
                alpha = self.log_alpha.exp() if c.ent_adaptive else c.ent_coef
                loss = (pg + c.vf_coef * vf - alpha * ent + c.path_coef * l_path
                        + c.tilt_coef * l_tilt + c.scen_coef * l_scen)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    if c.ent_adaptive:
                        self.log_alpha = ent_coef_step(self.log_alpha, ent, target, c)
                    kl = ((ratio - 1) - logr).mean()
                    ekl += kl
                    if epoch == 0 and i == 0:
                        acc["ratio_dev"] += (ratio - 1).abs().max()
                    for k, v in (("loss", loss), ("pg", pg), ("vf", vf), ("ent", ent),
                                 ("path", l_path), ("tilt", l_tilt), ("scen", l_scen),
                                 ("kl", kl), ("gn", gn)):
                        acc[k] += v.detach().float()
                    acc["ent_coef"] += torch.as_tensor(alpha, dtype=torch.float32,
                                                       device=self.device)
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
        out["lr"] = torch.tensor(lr)
        out["ent_target"] = torch.tensor(target)
        out["ent_coef_now"] = self.log_alpha.exp() if c.ent_adaptive else torch.tensor(c.ent_coef)
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

    def _init_tilts(self, n: int, limit: float) -> torch.Tensor:
        """Seeded initial tilts mapped into the init box (the default box is the full
        +-limit box, identical to eval_initial_tilts)."""
        c = self.cfg
        u = eval_initial_tilts(n, c.eval_seed, limit)
        center = torch.tensor(c.init_center, dtype=torch.float32)
        return (center + (c.init_radius / limit) * u).clamp(-limit, limit).to(self.device)

    def _oracle(self, env: AlignEnv, tilts: torch.Tensor, setups: torch.Tensor, file: str,
                key: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Oracle plan of the set (baselines.plan_oracle), cached in the run dir by key."""
        from baselines import plan_oracle

        path = os.path.join(self.run_dir, file)
        if os.path.exists(path):
            d = torch.load(path, map_location=self.device)
            if d.get("key") == key:
                return d
        t0 = time.time()
        plan = plan_oracle(env, tilts, chunk=8, setups=setups)
        oracle = {"key": key, "ret": plan.ret.float(), "target_path_mm": plan.target_path_mm,
                  "plan_s": time.time() - t0}
        torch.save(oracle, path)
        print(f"oracle plan ({label}, {tilts.shape[0]} episodes): {oracle['plan_s']:.1f} s, "
              f"mean return {float(plan.ret.mean()):.4f}", flush=True)
        return oracle

    def _setup_set(self, kind: str) -> EvalSet:
        """The four-setup (kind "all") or measured-cell set, as in run 4398136."""
        c = self.cfg
        env = AlignEnv(env_config(c, c.eval_envs, False, c.eval_seed + 12345, setups=kind))
        tilts = self._init_tilts(c.eval_n, env.limit)
        setups = (eval_setups(c.eval_n, env.n_setups, c.eval_seed) if env.n_setups > 1
                  else torch.zeros(c.eval_n, dtype=torch.long)).to(self.device)
        key = {"n": c.eval_n, "seed": c.eval_seed, "setups": kind,
               "center": list(c.init_center), "radius": c.init_radius,
               "step_cost": c.step_cost, "path_ref": c.path_ref}
        oracle = self._oracle(env, tilts, setups, "oracle_eval.pt", key, kind)
        names = [g.name for g in env.setups]

        def splits(ep):
            blind = ep["init_bounces"] <= BLIND_MAX_BOUNCES
            m = {"all": torch.ones_like(blind), "blind": blind, "informative": ~blind}
            for i, nm in enumerate(names if len(names) > 1 else []):
                m[nm] = ep["setup"] == i
            return m

        name = "set4" if kind == "all" else kind
        return EvalSet(name, env, tilts, setups, oracle, "eval", "eval_samp", splits)

    def _full_set(self) -> EvalSet:
        """The full-range set: (initial tilt, scenario) pairs over the atlas."""
        from scenarios import eval_scenarios, sep_bin_masks

        c, at = self.cfg, self.atlas
        env = AlignEnv(env_config(c, c.eval_envs, False, c.eval_seed + 12345),
                       scenarios=at.table())
        tilts = self._init_tilts(c.eval_n, env.limit)
        scen = eval_scenarios(at, c.eval_n, c.eval_seed, c.eval_impossible).to(self.device)
        key = {"n": c.eval_n, "seed": c.eval_seed, "impossible": c.eval_impossible,
               "atlas": _digest(at.params, at.path_ref), "center": list(c.init_center),
               "radius": c.init_radius, "step_cost": c.step_cost}
        oracle = self._oracle(env, tilts, scen, "oracle_full.pt", key, "full")
        good_t, params = at.good, at.params

        def splits(ep):
            blind = ep["init_bounces"] <= BLIND_MAX_BOUNCES
            good = good_t[ep["setup"]]
            m = {"all": torch.ones_like(blind), "blind": blind, "informative": ~blind,
                 "good": good, "impossible": ~good}
            for nm, sm in sep_bin_masks(params[ep["setup"], 0]).items():
                m[nm] = sm & good
            return m

        return EvalSet("full", env, tilts, scen, oracle, "eval_full", "eval_full_samp", splits)

    def setup_eval(self):
        """Eval envs, sets and oracle references (cached in the run dir)."""
        c = self.cfg
        sets = []
        if c.setups == "full":
            sets.append(self._full_set())
            if c.eval_set4:
                sets.append(self._setup_set("all"))
        else:
            sets.append(self._setup_set(c.setups))
        self.eval_sets = sets

    @torch.no_grad()
    def _run_eval(self, es: EvalSet, sampled: bool) -> Dict[str, float]:
        """One evaluation of a set; flat metrics <prefix>/<split>/<metric>."""
        c = self.cfg
        gen = None
        if sampled:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(c.eval_seed * 1_000_003 + 4099)  # Same noise at every eval
        pol = EvalPolicy(self.agent, gen)
        ep, met = evaluate(es.env, pol, es.tilts, es.oracle["target_path_mm"], es.oracle["ret"],
                           init_setups=es.setups)
        pre = es.prefix_samp if sampled else es.prefix
        ref = es.oracle["ret"].double()
        cap = es.env.cfg.max_bounces
        flat: Dict[str, float] = {}
        names = ("return", "oracle_return", "pct_oracle", "regret", "path_frac", "success",
                 "stop_rate", "stop_first", "length", "n")
        for split, m in es.splits(ep).items():
            if not bool(m.any()):  # Keys always present (fixed CSV header)
                flat.update({f"{pre}/{split}/{k}": float("nan") for k in names})
                continue
            s = split_metrics(ep, m, cap)
            o = float(ref[m].mean())
            r = s["return_mean"]
            pct = 100.0 * r / o if r is not None and abs(o) > 1e-3 else None
            vals = (r, o, pct, s.get("regret_mean"), s.get("path_frac_pref_mean"),
                    s.get("success"), s["terminated"], s["stop_first"], s["length_mean"], s["n"])
            for name, val in zip(names, vals):
                flat[f"{pre}/{split}/{name}"] = float("nan") if val is None else float(val)
        flat.update({f"{pre}/{k}": v for k, v in pol.diagnostics().items()})
        flat[f"{pre}/wall_s"] = met["wall_s"]
        return flat

    def evaluate(self) -> Dict[str, float]:
        """All eval sets, deterministic and (eval_sampled) sampled; flat metrics dict."""
        if self.eval_sets is None:
            self.setup_eval()
        flat: Dict[str, float] = {}
        for es in self.eval_sets:
            flat.update(self._run_eval(es, False))
            if self.cfg.eval_sampled:
                flat.update(self._run_eval(es, True))
        return flat

    # ── Checkpoints ──

    def state(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.state_dict(),
            "opt": self.opt.state_dict(),
            "vnorm": self.vnorm.state_dict(),
            "log_alpha": self.log_alpha,
            "sampler": self.sampler.state_dict() if self.sampler is not None else None,
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "best_eval": self.best_eval,
            "best_metric": self.best_metric,
            "cfg": asdict(self.cfg),
            "agent_cfg": self.agent.cfg.to_dict(),
            "encoder_info": self.encoder_info,
            "init_info": self.init_info,
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
        if d.get("log_alpha") is not None:
            self.log_alpha = d["log_alpha"].to(self.device)
        if self.sampler is not None and d.get("sampler") is not None:
            self.sampler.load_state_dict(d["sampler"])
        self.iteration, self.env_steps = d["iteration"], d["env_steps"]
        same = d.get("best_metric") == self.best_metric
        self.best_eval = d["best_eval"] if same else -math.inf
        self.encoder_info = d.get("encoder_info", self.encoder_info)
        self.init_info = d.get("init_info", self.init_info)
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
                     c.wandb_project, self.wandb_id, self.init_info)
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
            if self.sampler is not None:
                self.sampler.refresh(self.iteration)
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
            for k, v in us.items():
                if k in ("loss", "pg", "vf", "ent", "path", "tilt", "scen"):
                    row[f"loss/{k}"] = float(v)
                elif k in ("ent_target", "ent_coef", "ent_coef_now"):
                    row[f"ent/{k.replace('ent_', '')}"] = float(v)
                elif k == "lr":
                    row["lr"] = float(v)
                else:
                    row[f"ppo/{k}"] = float(v)
            row["ent/entropy"] = row["loss/ent"]
            if self.sampler is not None:
                row.update(self.sampler.stats())
            last = self.env_steps >= c.total_env_steps
            ev = None
            if c.eval_every > 0 and (self.iteration % c.eval_every == 0 or last) \
                    and self.stop_signal is None:
                sync()
                te = time.time()
                ev = self.evaluate()
                row["time_eval_s"] = time.time() - te
                if ev[self.best_metric] > self.best_eval:
                    self.best_eval = ev[self.best_metric]
                    self.save("best.pt")
                ev["iteration"], ev["env_steps"] = self.iteration, self.env_steps
            row["wall_min"] = (time.time() - t_start) / 60
            log.train(row)
            if ev is not None:
                log.eval(ev)
                for es in self.eval_sets:
                    for pre in (es.prefix, es.prefix_samp):
                        if f"{pre}/all/return" not in ev:
                            continue
                        print(f"  {pre}: return {ev[f'{pre}/all/return']:+.4f} "
                              f"({ev[f'{pre}/all/pct_oracle']:.1f} % of oracle), P/Pref "
                              f"{ev[f'{pre}/all/path_frac']:.3f}, success "
                              f"{ev[f'{pre}/all/success']:.3f}, stop "
                              f"{ev[f'{pre}/all/stop_rate']:.3f}, len "
                              f"{ev[f'{pre}/all/length']:.1f}", flush=True)
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
                 use_wandb: bool, project: str, wandb_id: Optional[str],
                 init_info: Optional[dict] = None):
        self.run_dir = run_dir
        self.files: Dict[str, Any] = {}
        self.writers: Dict[str, csv.DictWriter] = {}
        self.wb = None
        self.wandb_id = wandb_id
        init_short = None if init_info is None else {
            k: init_info.get(k) for k in ("path", "name", "iteration", "env_steps", "skipped",
                                          "missing")}
        if use_wandb:
            import wandb

            self.wandb_id = wandb_id or uuid.uuid4().hex[:12]  # Kept in the checkpoint
            self.wb = wandb.init(project=project, name=name, id=self.wandb_id, resume="allow",
                                 config={**config, "encoder_info": encoder_info,
                                         "init_info": init_short},
                                 dir=os.environ.get("WANDB_DIR", run_dir))
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"name": name, "config": config, "encoder_info": encoder_info,
                       "init_info": init_short}, f, indent=1)

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
              f"ent {row['loss/ent']:.3f} (tgt {row['ent/target']:.2f}, a {row['ent/coef_now']:.4f}) "
              f"path {row['loss/path']:.4f} tilt {row['loss/tilt']:+.3f} scen {row['loss/scen']:+.3f} "
              f"kl {row['ppo/kl']:.4f} ev {row['ppo/explained_var']:+.3f} ep {int(row['ppo/epochs'])} "
              f"lr {row['lr']:.2e} | {row['fps']:.0f} fps (roll {row['time_rollout_s']:.1f} s, "
              f"upd {row['time_update_s']:.1f} s)", flush=True)
        if "plr/seen" in row:
            print(f"      plr: seen {row['plr/seen']} p_fresh {row['plr/p_fresh']:.2f} imp "
                  f"{row['plr/impossible_share']:.3f} shortfall fresh {row['plr/fresh_shortfall']:.3f} "
                  f"replay {row['plr/replay_shortfall']:.3f} score q50 {row.get('plr/score_q50', math.nan):.3f} "
                  f"top ({row.get('plr/top_sep', math.nan):g} mm, {row.get('plr/top_laser_pitch', math.nan):g}, "
                  f"{row.get('plr/top_laser_yaw', math.nan):g}) share {row.get('plr/top_draw_share', math.nan):.4f}",
                  flush=True)

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


def init_run_id(path: str) -> str:
    """Run id of an --init-from checkpoint: its directory name (the job id on the
    cluster)."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) or "x"


def run_name(c: PPOConfig, encoder_info: Optional[dict]) -> str:
    """Display / directory name from the config, built at start:
    ppo_cnn[-pre<S>]_k<K>_<scen>_b<B>_s<seed>[_ft<run>][_uf<it>|_e2e][_<tag>]_<jobid>;
    <scen>: full-plr / full-unif (atlas, prioritized / uniform good draws) or set<S>;
    -pre<S>: the encoder was initialized from a checkpoint pretrained on S setups;
    _ft<run>: initialized from that PPO run (--init-from)."""
    enc = "cnn"
    if encoder_info is not None:
        m = re.search(r"_grid(\d+)", encoder_info.get("name") or "")
        enc += f"-pregrid{m.group(1)}" if m else f"-pre{encoder_info.get('n_setups', 1)}"
    if c.setups == "full":
        scen = "full-plr" if c.plr else "full-unif"
    else:
        scen = f"set{len(SETUPS) if c.setups == 'all' else 1}"
    name = f"ppo_{enc}_k{c.history_len}_{scen}_b{c.num_envs}_s{c.seed}"
    if c.init_from:
        name += f"_ft{init_run_id(c.init_from)}"
    if not c.encoder_freeze:
        name += "_e2e"
    elif c.unfreeze_at >= 0:
        name += f"_uf{c.unfreeze_at}"
    if c.name_tag:
        name += f"_{c.name_tag}"
    return name + "_" + os.environ.get("SLURM_JOB_ID", "local")


# ── CLI ─────────────────────────────────────────────────────────────────────


RESUME_OVERRIDES = ("total_env_steps", "eval_every", "ckpt_minutes", "wandb", "wandb_project",
                    "unfreeze_at", "encoder_lr_scale", "compile", "target_kl", "eval_sampled",
                    "enc_checkpoint")
PATH_FIELDS = ("encoder_init", "init_from", "atlas")


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
        if isinstance(v, bool):
            ap.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=dflt(v))
        elif f.name == "init_center":
            ap.add_argument(flag, type=float, nargs=2, default=dflt(list(v)))
        elif f.name == "setups":
            ap.add_argument(flag, choices=SETUP_CHOICES, default=dflt(v))
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
        known = {f.name for f in fields(PPOConfig)}
        saved = {k: v for k, v in saved.items() if k in known}
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
        for k in PATH_FIELDS:
            if getattr(cfg, k):
                setattr(cfg, k, os.path.abspath(getattr(cfg, k)))
        trainer = Trainer(cfg, args.run_dir)
    trainer.train()


if __name__ == "__main__":
    main()
