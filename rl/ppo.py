"""
ppo.py - PPO trainer for the alignment agent (PLAN phase 6: P1-P6, T1-T6; stage 2)

Run from rl/:
    uv run python ppo.py --encoder-init runs/encoder/<run>/model.pt          # defaults below
    uv run python ppo.py --setups all --num-envs 64 --total-env-steps 100000 --no-wandb  # smoke
    uv run python ppo.py --run-dir <dir> --resume                            # continue latest.pt
Cluster: cluster/ppo_l40s.slurm.

    env        AlignEnv, auto-reset, history_len 0 (the agent keeps the windows), max_steps
               64, step_cost 0.01, 64 px; envs persist across iterations (episodes span
               rollouts). --setups full (default, stage 2) with --scenarios pool
               (default, D7): continuous scenarios (separation --sep-range, laser offset
               uniform over the M1 hole disk of --laser-offset-radius, laser pitch / yaw
               from a uniform M2 landing point within --laser-range, --laser-mode;
               beams clipped by the hole or missing M2 discarded) from the ScenarioPool
               (scenarios.py: --pool-size
               slots, --pool-refresh of them rescanned per iteration, PLR within the
               pool); --scenarios atlas: the grid atlas (--atlas, built there if
               missing) and the ScenarioSampler. Either way P_ref per scenario =
               max(P_max, P_floor), drawn per episode by the regret-prioritized sampler
               (--plr; --no-plr: uniform over the good set; 10 % impossible either way).
               --setups all: the four geometry.SETUPS, P_ref = per-setup P_max;
               measured: the measured cell
    init       --encoder-init: encoder weights of a pretrain_encoder.py checkpoint.
               --init-from: a previous ppo.py checkpoint, every tensor whose name and
               shape match (encoder, token, core, policy / value / path / tilt heads; the
               skipped and missing ones are printed and kept in config.json; the scenario
               head is re-initialized when its normalization range differs, e.g. atlas
               run -> pool run), its value-norm statistics; fresh optimizer, counters,
               entropy coefficient
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
               ring is re-encoded at each rollout start (and once more at the first rollout
               after the encoder stopped). --encoder-epochs N (0 = all): the encoder gets
               gradients in the first N epochs only; before epoch N + 1 the z of every
               stored frame (context included) is recomputed once without grad by the
               updated encoder and reused by the remaining epochs (no encoder gradients
               there: AdamW neither steps nor decays it). --encoder-train-until F (1 =
               never): the encoder trains only in iterations whose env steps before the
               rollout are below round(F x total_env_steps), then the update reuses the
               rollout's z as with the frozen encoder (Trainer.encoder_trains: one function
               of the iteration and the config for the ring re-encoding, the update and
               the logs; the iteration of the switch is printed)
    eval       every --eval-every iterations and at the end, separate envs without
               auto-reset, deterministic (argmax) and sampled (one seeded sample per
               episode and step, --eval-sampled) policies; oracle reference
               (baselines.plan_oracle) computed once per set and cached in the run dir.
               Sets (eval_set_names; built by build_eval_sets, shared with
               eval_policy.py): "full" (--setups full): --eval-n (initial tilt, scenario)
               pairs, scenarios 90 % uniform over the good set and 10 % over the
               impossible set (scenarios.eval_scenarios; with a hold-out: over the
               training scenarios only), splits all / blind / informative / good /
               impossible / separation bins (good scenarios); "cont" (--scenarios pool):
               eval_n continuous scenarios of the training ranges, the same filter and
               90 / 10 mix (scenarios.ContConfig, table cached in --eval-tables, seeded
               by --eval-seed, one episode per scenario), separation bins 20-50,
               50-100, ..., 350-400; with the pool the grid sets below stay as fixed
               benchmarks (measured laser offsets); "holdout" (--holdout-frac
               > 0): the same over the held-out grid scenarios only; "offgrid"
               (--eval-offgrid, default on): scenarios between the grid points in all
               three parameters, one episode per scenario (scenarios.py off-grid
               tables, cached in --eval-tables, built if missing); "extrap_short" /
               "extrap_long" (--eval-extrap, default on): separation 20-49 / 201-350 mm,
               outside the trained range, own separation bins; "set4": the four
               setups' set of run 4398136 (with --setups full only with --eval-set4);
               "measured" with --setups measured. Every set uses the same initial tilts.
               Prefixes: eval_cont/, eval_full/, eval_holdout/, eval_offgrid/,
               eval_extrap_short/, eval_extrap_long/ (+ _samp for the sampled policy)
               and, for the four-setup set, eval/ (the keys of run 4398136) and
               eval_samp/
    holdout    --holdout-frac F (default 0): a seeded (--holdout-seed) random share F of
               the grid scenarios, stratified good / impossible, never drawn in
               training (scenarios.holdout_mask, removed from the sampler's segments)
    logging    train.csv / eval.csv in the run dir; W&B (project herriott-rl) unless
               --no-wandb, display name built at start from the config
    checkpoint latest.pt (every --ckpt-minutes and at exit), best.pt (best deterministic
               % of the oracle: on the cont set with the pool, the full set with the
               atlas, else on the set of the run): agent, optimizer, value norm, entropy
               coefficient, sampler (with the pool: its slots), counters, RNG, env and
               ring state, config. SIGTERM / SIGUSR1: abort the current rollout, write
               latest.pt, exit 0
Host syncs: none inside a rollout step; one per epoch (KL early stop), the pool refresh
once per iteration (scans) and a few per iteration for logging (sampler statistics);
evaluation syncs per step (evaluation.evaluate).
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
SCENARIO_CHOICES = ("pool", "atlas")  # --setups full: continuous pool or grid atlas


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    """Trainer settings (CLI arguments with the same names)."""

    # ── Env ──
    setups: str = "full"  # full = scenario range (--scenarios), all = geometry.SETUPS, measured
    scenarios: str = "pool"  # --setups full: pool (continuous, D7) or atlas (grid)
    atlas: str = ""  # Grid atlas ("" = scenarios.DEFAULT_ATLAS), built if missing: training
    #                  with --scenarios atlas, the fixed benchmark sets (eval_full, off-grid)
    num_envs: int = 1024
    rollout_len: int = 128
    max_steps: int = 64
    step_cost: float = 0.01
    img_res: int = 64
    path_ref: str = "max"  # --setups all / measured; full: the atlas P_ref
    init_center: Tuple[float, float] = (0.0, 0.0)
    init_radius: float = 2.0
    seed: int = 0

    # ── Continuous scenarios (--scenarios pool) ──
    sep_range: Tuple[float, float] = (20.0, 400.0)  # mm
    laser_range: Tuple[float, float] = (-8.0, 8.0)  # deg, laser pitch and yaw (bounds)
    laser_offset_radius: float = 1.5  # mm, offset disk around the hole centre; 0 = measured
    laser_mode: str = "landing"  # landing: angles from a uniform M2 landing point; angle
    pool_size: int = 16384
    pool_refresh: float = 0.0625  # Share of the pool rescanned per iteration (was 0.125)

    # ── Scenario sampler (--setups full) ──
    plr: bool = True  # Regret-prioritized replay; False: good scenarios uniform
    plr_impossible: float = 0.1  # Share of draws from the impossible set
    plr_alpha: float = 0.3  # Score EMA
    plr_p_fresh: float = 0.3  # Fresh uniform share of the good draws (1 until min_seen)
    plr_min_seen: int = 2048
    plr_beta: float = 1.0  # Rank temperature (0.3 put ~half of all episodes on one scenario)
    plr_rho: float = 0.1  # Staleness share
    holdout_frac: float = 0.0  # Share of the grid scenarios never trained on (eval_holdout/)
    holdout_seed: int = 0

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
    encoder_epochs: int = 0  # Unfrozen encoder: gradients in the first N PPO epochs only; 0 = all
    encoder_train_until: float = 1.0  # Encoder frozen from env_steps >= this x total_env_steps; 1 = never
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
    eval_offgrid: bool = True  # --setups full: also the off-grid (interpolation) set
    eval_extrap: bool = True  # --setups full: also the two extrapolation sets
    eval_tables: str = ""  # Off-grid tables dir ("" = scenarios.EVAL_TABLES_DIR), built if missing
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


def pool_mode(c: PPOConfig) -> bool:
    return c.setups == "full" and c.scenarios == "pool"


def cont_ranges_of(c: PPOConfig):
    """scenarios.ContRanges of the run (--sep-range, --laser-range, --laser-offset-radius,
    --laser-mode)."""
    from scenarios import cont_ranges

    return cont_ranges(c.sep_range, c.laser_range, c.laser_offset_radius, c.laser_mode)


def agent_config(c: PPOConfig, n_bins: int) -> AgentConfig:
    """AgentConfig of the run; with the pool the scenario head covers its ranges (five
    parameters, three with --laser-offset-radius 0), else the atlas grid (default)."""
    kw = {}
    if pool_mode(c):
        kw["scen_lo"], kw["scen_hi"] = cont_ranges_of(c).head_range()
    return AgentConfig(window=c.history_len, d_model=c.d_model, n_layers=c.n_layers,
                       n_heads=c.n_heads, mlp_ratio=c.mlp_ratio, n_bins=n_bins, amp=c.amp,
                       enc_chunk=c.enc_chunk, enc_checkpoint=c.enc_checkpoint, **kw)


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
                 device: torch.device, n_scen: int = 3):
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
        self.scen = z(B, T, n_scen)  # Scenario parameters in scenario-head units
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
    Accumulates head diagnostics over active steps and over the first frames (step 0:
    the same initial tilts in every eval set, independent of the episode lengths); with
    n_episodes > 0 also per episode (evaluation.evaluate's info["episode"]), for
    diagnostics of any split."""

    KEYS = ("n", "path_mae", "tilt_mae_deg", "n_seen", "tilt_mae_deg_ge3", "tilt_std_deg",
            "sep_mae_mm", "laser_pitch_mae_deg", "laser_yaw_mae_deg", "laser_dx_mae_mm",
            "laser_dy_mae_mm")
    SCEN_KEYS = KEYS[6:]  # Per scenario parameter (env.SCEN_NAMES order)

    def __init__(self, agent: Agent, gen: Optional[torch.Generator] = None, n_episodes: int = 0):
        self.agent = agent
        self.gen = gen
        self.sums: Dict[str, torch.Tensor] = {}
        self.ep_sums: Optional[torch.Tensor] = None  # (n_episodes, 2 x len(KEYS)) float64
        self.n_episodes = n_episodes

    def _acc(self, k: str, v: torch.Tensor):
        self.sums[k] = self.sums.get(k, 0.0) + v.double().sum()

    @torch.no_grad()
    def __call__(self, obs, info, state):
        ag = self.agent
        if state is None:
            state = ag.init_ring(obs["image"].shape[0], obs["image"].device)
        out, state = ag.act(obs["image"], obs["action"], info["reset"], state)
        pn = info["path_mm"] / info["path_ref"]
        tilt_err = (out["tilt_mu"] * ag.cfg.tilt_scale - info["tilt"]).abs().mean(-1)
        n_sc = ag.n_scen  # Parameters the head predicts (3 or 5); the rest count 0
        scen_err = (ag.scen_denorm(out["scen_mu"]) - info["scenario_params"][:, :n_sc]).abs()
        scen_err = torch.cat((scen_err, scen_err.new_zeros(scen_err.shape[0], 5 - n_sc)), 1)
        ge3 = (info["bounces"] >= 3).float()
        tilt_std = (0.5 * out["tilt_logvar"]).exp().mean(-1) * ag.cfg.tilt_scale
        path_err = (out["path"] - pn).abs()
        m = info["active"].float()
        vals = []
        for w in (m, m * (info["step"] == 0).float()):  # All active steps, first frames
            vals += [w, path_err * w, tilt_err * w, ge3 * w, tilt_err * ge3 * w, tilt_std * w]
            vals += [scen_err[:, j] * w for j in range(5)]
        for j, v in enumerate(vals):
            self._acc(self.KEYS[j % len(self.KEYS)] + ("_s0" if j >= len(self.KEYS) else ""), v)
        if self.n_episodes > 0:
            if self.ep_sums is None:
                self.ep_sums = torch.zeros(self.n_episodes, len(vals), dtype=torch.float64,
                                           device=m.device)
            self.ep_sums.index_add_(0, info["episode"].clamp(min=0),
                                    torch.stack(vals, 1).double())
        act = greedy(out["logits"]) if self.gen is None else sample(out["logits"], self.gen)
        return act, state

    def diagnostics(self, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Head errors over the active steps (head_*) and over the first frames
        (head_*_step0): all episodes, or the episodes of mask (n,) (needs n_episodes)."""
        if mask is None:
            s = {k: float(v) for k, v in self.sums.items()}
        else:
            assert self.ep_sums is not None, "per-episode diagnostics need n_episodes"
            t = self.ep_sums[mask.to(self.ep_sums.device)].sum(0).tolist()
            keys = list(self.KEYS) + [k + "_s0" for k in self.KEYS]
            s = dict(zip(keys, t))
        out = {}
        for sfx, tag in (("", ""), ("_s0", "_step0")):
            g = lambda k: s.get(k + sfx, 0.0)
            n, ns = max(g("n"), 1.0), max(g("n_seen"), 1.0)
            out.update({f"head_path_mae{tag}": g("path_mae") / n,
                        f"head_tilt_mae_deg{tag}": g("tilt_mae_deg") / n,
                        f"head_tilt_mae_deg_ge3{tag}": g("tilt_mae_deg_ge3") / ns,
                        f"head_tilt_std_deg{tag}": g("tilt_std_deg") / n})
            for j, k in enumerate(self.SCEN_KEYS):  # NaN for parameters the head lacks
                out[f"head_{k}{tag}"] = g(k) / n if j < self.agent.n_scen else math.nan
        return out


@dataclass
class EvalSet:
    """A fixed evaluation set: env, (initial tilt, setup / scenario) pairs, oracle."""

    name: str  # full / holdout / offgrid / extrap_short / extrap_long / set4 / measured
    env: AlignEnv
    tilts: torch.Tensor
    setups: torch.Tensor
    oracle: Dict[str, Any]
    prefix: str  # Deterministic policy
    prefix_samp: str  # Sampled policy
    splits: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    table: Any = None  # Scenario table of the env (scenarios.Atlas), None for setups
    bins: Any = None  # Separation bins of the splits (None: scenarios.SEP_BINS / none)


def _digest(*ts: torch.Tensor) -> str:
    h = hashlib.sha1()
    for t in ts:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ── Evaluation sets (shared by the Trainer and eval_policy.py) ──────────────


TABLE_SETS = ("offgrid", "extrap_short", "extrap_long")  # scenarios.TABLE_KINDS
EVAL_SET_NAMES = ("cont", "full", "holdout") + TABLE_SETS + ("set4", "measured")
ATLAS_SETS = ("full", "holdout") + TABLE_SETS  # Need the grid atlas
FLAT_METRICS = ("return", "oracle_return", "pct_oracle", "regret", "path_frac", "success",
                "stop_rate", "stop_first", "length", "end_cap", "n")


def eval_set_names(c: PPOConfig) -> List[str]:
    """Sets a run evaluates: --setups full: cont (--scenarios pool), full, holdout
    (holdout_frac > 0), offgrid (eval_offgrid), extrap_short / extrap_long (eval_extrap),
    set4 (eval_set4); else the run's own set (set4 / measured)."""
    if c.setups != "full":
        return ["set4" if c.setups == "all" else c.setups]
    names = (["cont"] if pool_mode(c) else []) + ["full"]
    if c.holdout_frac > 0:
        names.append("holdout")
    if c.eval_offgrid:
        names.append("offgrid")
    if c.eval_extrap:
        names += ["extrap_short", "extrap_long"]
    if c.eval_set4:
        names.append("set4")
    return names


def eval_init_tilts(c: PPOConfig, n: int, limit: float, device: torch.device) -> torch.Tensor:
    """Seeded initial tilts mapped into the init box (the default box is the full +-limit
    box, identical to eval_initial_tilts); the same for every set."""
    u = eval_initial_tilts(n, c.eval_seed, limit)
    center = torch.tensor(c.init_center, dtype=torch.float32)
    return (center + (c.init_radius / limit) * u).clamp(-limit, limit).to(device)


def cached_oracle(env: AlignEnv, tilts: torch.Tensor, setups: torch.Tensor, path: str,
                  key: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Oracle plan of a set (baselines.plan_oracle), cached at path by key: ret,
    target_path_mm, target_bounces, moves (a cached file without the last two is
    recomputed and rewritten)."""
    from baselines import plan_oracle

    dev = env.device
    if os.path.exists(path):
        d = torch.load(path, map_location=dev)
        if d.get("key") == key and "target_bounces" in d:
            return d
    t0 = time.time()
    plan = plan_oracle(env, tilts, chunk=8, setups=setups)
    oracle = {"key": key, "ret": plan.ret.float(), "target_path_mm": plan.target_path_mm,
              "target_bounces": plan.target_bounces, "moves": plan.moves,
              "plan_s": time.time() - t0}
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(oracle, tmp)
    os.replace(tmp, path)
    print(f"oracle plan ({label}, {tilts.shape[0]} episodes): {oracle['plan_s']:.1f} s, "
          f"mean return {float(plan.ret.mean()):.4f}", flush=True)
    return oracle


def scenario_splits(table, bins=None) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Splits of a scenario-table set: all / blind / informative / good / impossible and
    separation bins (good scenarios; bins default scenarios.SEP_BINS)."""
    from scenarios import SEP_BINS, sep_bin_masks

    good_t, params = table.good, table.params
    bins = bins or SEP_BINS

    def splits(ep):
        blind = ep["init_bounces"] <= BLIND_MAX_BOUNCES
        good = good_t[ep["setup"]]
        m = {"all": torch.ones_like(blind), "blind": blind, "informative": ~blind,
             "good": good, "impossible": ~good}
        for nm, sm in sep_bin_masks(params[ep["setup"], 0], bins).items():
            m[nm] = sm & good
        return m

    return splits


def setup_eval_set(c: PPOConfig, kind: str, cache_dir: str, device: torch.device) -> EvalSet:
    """The four-setup (kind "all") or measured-cell set, as in run 4398136."""
    env = AlignEnv(env_config(c, c.eval_envs, False, c.eval_seed + 12345, setups=kind))
    tilts = eval_init_tilts(c, c.eval_n, env.limit, device)
    setups = (eval_setups(c.eval_n, env.n_setups, c.eval_seed) if env.n_setups > 1
              else torch.zeros(c.eval_n, dtype=torch.long)).to(device)
    key = {"n": c.eval_n, "seed": c.eval_seed, "setups": kind,
           "center": list(c.init_center), "radius": c.init_radius,
           "step_cost": c.step_cost, "path_ref": c.path_ref}
    oracle = cached_oracle(env, tilts, setups, os.path.join(cache_dir, "oracle_eval.pt"), key, kind)
    names = [g.name for g in env.setups]

    def splits(ep):
        blind = ep["init_bounces"] <= BLIND_MAX_BOUNCES
        m = {"all": torch.ones_like(blind), "blind": blind, "informative": ~blind}
        for i, nm in enumerate(names if len(names) > 1 else []):
            m[nm] = ep["setup"] == i
        return m

    name = "set4" if kind == "all" else kind
    return EvalSet(name, env, tilts, setups, oracle, "eval", "eval_samp", splits)


def _table_set(c: PPOConfig, name: str, table, scen: torch.Tensor, key: Dict[str, Any],
               file: str, prefix: str, bins, cache_dir: str, device: torch.device) -> EvalSet:
    env = AlignEnv(env_config(c, c.eval_envs, False, c.eval_seed + 12345),
                   scenarios=table.table())
    tilts = eval_init_tilts(c, c.eval_n, env.limit, device)
    scen = scen.to(device)
    key = {**key, "center": list(c.init_center), "radius": c.init_radius,
           "step_cost": c.step_cost}
    oracle = cached_oracle(env, tilts, scen, os.path.join(cache_dir, file), key, name)
    return EvalSet(name, env, tilts, scen, oracle, prefix, f"{prefix}_samp",
                   scenario_splits(table, bins), table, bins)


def full_eval_set(c: PPOConfig, atlas, cache_dir: str, device: torch.device) -> EvalSet:
    """The full-range set: (initial tilt, scenario) pairs over the atlas (with a hold-out:
    over the training scenarios only)."""
    from scenarios import eval_scenarios, holdout_mask

    key = {"n": c.eval_n, "seed": c.eval_seed, "impossible": c.eval_impossible,
           "atlas": _digest(atlas.params, atlas.path_ref)}
    g, im = atlas.good_ids, atlas.impossible_ids
    if c.holdout_frac > 0:
        ho = holdout_mask(atlas, c.holdout_frac, c.holdout_seed)
        g, im = g[~ho[g]], im[~ho[im]]
        key["holdout"] = [c.holdout_frac, c.holdout_seed]
    scen = eval_scenarios(atlas, c.eval_n, c.eval_seed, c.eval_impossible, g, im)
    return _table_set(c, "full", atlas, scen, key, "oracle_full.pt", "eval_full", None,
                      cache_dir, device)


def holdout_eval_set(c: PPOConfig, atlas, cache_dir: str, device: torch.device) -> EvalSet:
    """The grid hold-out set: (initial tilt, scenario) pairs over the held-out scenarios
    only (scenarios.holdout_mask; 90 / 10 good / impossible as the full set)."""
    from scenarios import eval_scenarios, holdout_mask

    assert c.holdout_frac > 0, "the hold-out set needs --holdout-frac > 0"
    ho = holdout_mask(atlas, c.holdout_frac, c.holdout_seed)
    g, im = atlas.good_ids, atlas.impossible_ids
    g, im = g[ho[g]], im[ho[im]]
    scen = eval_scenarios(atlas, c.eval_n, c.eval_seed, c.eval_impossible, g, im)
    key = {"n": c.eval_n, "seed": c.eval_seed, "impossible": c.eval_impossible,
           "atlas": _digest(atlas.params, atlas.path_ref),
           "holdout": [c.holdout_frac, c.holdout_seed]}
    return _table_set(c, "holdout", atlas, scen, key, "oracle_holdout.pt", "eval_holdout",
                      None, cache_dir, device)


def load_eval_table(c: PPOConfig, kind: str, atlas, device: torch.device, log=print):
    """Off-grid table of a kind (scenarios.TABLE_KINDS) for the run's grid: loaded from
    <eval_tables>/<kind>_v1.pt, built there if missing."""
    from scenarios import OffgridConfig, atlas_config, load_or_build_offgrid, table_path

    return load_or_build_offgrid(table_path(kind, c.eval_tables), atlas_config(atlas),
                                 OffgridConfig(kind=kind), device, log)


def table_eval_set(c: PPOConfig, kind: str, table, cache_dir: str,
                   device: torch.device) -> EvalSet:
    """An off-grid set (offgrid / extrap_short / extrap_long): eval_n episodes over the
    table, eval_impossible of them impossible, one per scenario
    (scenarios.offgrid_eval_scenarios); separation bins of the kind."""
    from scenarios import TABLE_SEP_BINS, offgrid_eval_scenarios

    scen = offgrid_eval_scenarios(table, c.eval_n, c.eval_seed, c.eval_impossible)
    key = {"n": c.eval_n, "seed": c.eval_seed, "impossible": c.eval_impossible,
           kind: _digest(table.params, table.path_ref)}
    return _table_set(c, kind, table, scen, key, f"oracle_{kind}.pt", f"eval_{kind}",
                      TABLE_SEP_BINS[kind], cache_dir, device)


def cont_eval_set(c: PPOConfig, cache_dir: str, device: torch.device, log=print) -> EvalSet:
    """The continuous set (eval_cont/): eval_n episodes, one per scenario, over the
    seeded continuous table of the run's ranges (scenarios.ContConfig: the same 5D
    distribution and beam filter as the pool, eval_impossible of them impossible; cached
    in eval_tables, built if missing); separation bins cont_sep_bins of the range."""
    from scenarios import (ContConfig, cont_sep_bins, cont_table_path, load_or_build_cont,
                           offgrid_eval_scenarios)

    ccfg = ContConfig(ranges=cont_ranges_of(c), n=c.eval_n, impossible_share=c.eval_impossible,
                      seed=c.eval_seed)
    table = load_or_build_cont(cont_table_path(ccfg, c.eval_tables), ccfg, device=device, log=log)
    scen = offgrid_eval_scenarios(table, c.eval_n, c.eval_seed, c.eval_impossible)
    key = {"n": c.eval_n, "seed": c.eval_seed, "impossible": c.eval_impossible,
           "cont": _digest(table.params, table.path_ref)}
    return _table_set(c, "cont", table, scen, key, "oracle_cont.pt", "eval_cont",
                      cont_sep_bins(*ccfg.ranges.sep), cache_dir, device)


def build_eval_sets(c: PPOConfig, names: List[str], atlas, cache_dir: str,
                    device: torch.device, log=print) -> List[EvalSet]:
    """Eval sets by name (EVAL_SET_NAMES); oracles cached in cache_dir."""
    sets = []
    for name in names:
        assert name in EVAL_SET_NAMES, f"eval set {name!r} not in {EVAL_SET_NAMES}"
        if name in ("set4", "measured"):
            sets.append(setup_eval_set(c, "all" if name == "set4" else name, cache_dir, device))
            continue
        if name == "cont":
            sets.append(cont_eval_set(c, cache_dir, device, log))
            continue
        assert atlas is not None, f"eval set {name} needs the atlas"
        if name == "full":
            sets.append(full_eval_set(c, atlas, cache_dir, device))
        elif name == "holdout":
            sets.append(holdout_eval_set(c, atlas, cache_dir, device))
        else:
            table = load_eval_table(c, name, atlas, device, log)
            sets.append(table_eval_set(c, name, table, cache_dir, device))
    return sets


@torch.no_grad()
def run_eval(agent: Agent, es: EvalSet, eval_seed: int, sampled: bool,
             per_episode: bool = False) -> Tuple[Dict[str, float], Dict[str, torch.Tensor],
                                                 EvalPolicy]:
    """One evaluation of a set: flat metrics <prefix>/<split>/<metric> (FLAT_METRICS) and
    <prefix>/head_*, the per-episode tensors (evaluation.evaluate) and the policy adapter
    (per_episode: with per-episode head diagnostics)."""
    gen = None
    if sampled:
        gen = torch.Generator(device=es.env.device)
        gen.manual_seed(eval_seed * 1_000_003 + 4099)  # Same noise at every eval
    pol = EvalPolicy(agent, gen, es.tilts.shape[0] if per_episode else 0)
    ep, met = evaluate(es.env, pol, es.tilts, es.oracle["target_path_mm"], es.oracle["ret"],
                       init_setups=es.setups)
    pre = es.prefix_samp if sampled else es.prefix
    ref = es.oracle["ret"].double().to(ep["ep_return"].device)
    cap = es.env.cfg.max_bounces
    flat: Dict[str, float] = {}
    for split, m in es.splits(ep).items():
        if not bool(m.any()):  # Keys always present (fixed CSV header)
            flat.update({f"{pre}/{split}/{k}": float("nan") for k in FLAT_METRICS})
            continue
        s = split_metrics(ep, m, cap)
        o = float(ref[m].mean())
        r = s["return_mean"]
        pct = 100.0 * r / o if r is not None and abs(o) > 1e-3 else None
        vals = (r, o, pct, s.get("regret_mean"), s.get("path_frac_pref_mean"),
                s.get("success"), s["terminated"], s["stop_first"], s["length_mean"],
                s["end_cap"], s["n"])
        for name, val in zip(FLAT_METRICS, vals):
            flat[f"{pre}/{split}/{name}"] = float("nan") if val is None else float(val)
    flat.update({f"{pre}/{k}": v for k, v in pol.diagnostics().items()})
    flat[f"{pre}/wall_s"] = met["wall_s"]
    return flat, ep, pol


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
        assert c.scenarios in SCENARIO_CHOICES, f"scenarios {c.scenarios!r} not in {SCENARIO_CHOICES}"
        assert c.encoder_epochs >= 0 and c.encoder_train_until >= 0, "encoder schedule"
        self.device = dev = torch.device(device or "cuda")
        torch.manual_seed(c.seed)

        # ── Scenarios and sampler (--setups full): pool or atlas ──
        self.atlas = self.sampler = self.pool = None
        scen_kw: Dict[str, Any] = {}
        log = lambda s: print(s, flush=True)  # noqa: E731
        ec = EnvConfig()
        if c.setups == "full":
            from scenarios import SamplerConfig, ScenarioSampler

            scfg = SamplerConfig(
                impossible_share=c.plr_impossible, alpha=c.plr_alpha, p_fresh=c.plr_p_fresh,
                min_seen=c.plr_min_seen, beta=c.plr_beta, rho=c.plr_rho, replay=c.plr,
                holdout_frac=c.holdout_frac, holdout_seed=c.holdout_seed)
            # The grid atlas: training scenarios (atlas mode) or the fixed benchmark sets
            if not pool_mode(c) or (build_eval and any(n in ATLAS_SETS for n in eval_set_names(c))):
                self._load_atlas()
        if pool_mode(c):
            from scenarios import PoolConfig, ScenarioPool

            assert c.holdout_frac == 0, "--holdout-frac needs --scenarios atlas"
            self.pool = self.sampler = ScenarioPool(
                PoolConfig(size=c.pool_size, ranges=cont_ranges_of(c), refresh_frac=c.pool_refresh,
                           seed=c.seed), scfg, device=dev, log=log)
            self.pool.atlas.check(ec.max_bounces, ec.geometry.m2_tilt_limit, ec.step_unit)
            scen_kw = {"pool": self.pool}
        elif c.setups == "full":
            self.sampler = ScenarioSampler(self.atlas, scfg)
            if c.holdout_frac > 0:
                ho = self.sampler.holdout
                print(f"hold-out: {int(ho.sum())} grid scenarios never drawn in training "
                      f"({int((ho & self.atlas.good).sum())} good, "
                      f"{int((ho & ~self.atlas.good).sum())} impossible; frac "
                      f"{c.holdout_frac}, seed {c.holdout_seed})", flush=True)
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
            # Expected: the old setup head; the scenario head when its range or width changed
            scen_ok = lambda k: k.startswith("scenario.") and (  # noqa: E731
                info.get("scen_reinit") or info["skipped"][k].startswith("shape"))
            other = sorted(k for k in info["skipped"] if not k.startswith("setup.") and not scen_ok(k))
            print(f"init from {info['path']} ({info['name']}, iteration {info['iteration']}): "
                  f"{len(info['loaded'])} tensors loaded; skipped {sorted(info['skipped'])}; "
                  f"fresh {info['missing']}; value norm from the checkpoint", flush=True)
            if any(scen_ok(k) for k in info["skipped"]):
                why = info.get("scen_reinit") or "head width changed"
                print(f"init from: scenario head re-initialized ({why})", flush=True)
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
        self.ring_stale = False  # The encoder changed since the ring's z: re-encode it
        self.next_first = torch.ones(B, dtype=torch.bool, device=dev)
        self.n_slots = math.ceil(c.rollout_len / c.max_steps)
        self.buf = Rollout(B, c.rollout_len, K, self.agent.cfg.encoder.z_dim, c.img_res,
                           self.n_slots, dev, self.agent.n_scen)
        self.iteration = 0
        self.env_steps = 0
        self.best_eval = -math.inf
        self.best_metric = ("eval_cont/all/pct_oracle" if pool_mode(c) else "eval_full/all/pct_oracle"
                            if c.setups == "full" else "eval/all/pct_oracle")
        self.wandb_id: Optional[str] = None
        self.name = run_name(c, self.encoder_info)
        self.run_dir = run_dir or os.path.join(RUNS_DIR, self.name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.stop_signal: Optional[int] = None
        self.eval_sets: Optional[List[EvalSet]] = None
        self._build_eval = build_eval

    # ── Helpers ──

    def _load_atlas(self):
        """The grid atlas (--atlas, built there if missing), checked against the env."""
        from scenarios import DEFAULT_ATLAS, load_or_build

        c = self.cfg
        c.atlas = os.path.abspath(c.atlas or DEFAULT_ATLAS)
        t0 = time.time()
        self.atlas = load_or_build(c.atlas, self.device, log=lambda s: print(s, flush=True))
        ec = EnvConfig()
        self.atlas.check(ec.max_bounces, ec.geometry.m2_tilt_limit, ec.step_unit)
        print(f"{self.atlas.summary()}\natlas ready in {time.time() - t0:.1f} s", flush=True)

    def encoder_stop_iteration(self) -> Optional[int]:
        """First iteration without encoder training under --encoder-train-until (the env
        steps before its rollout, iteration x num_envs x rollout_len, reach
        round(encoder_train_until x total_env_steps)); None = never (>= 1)."""
        c = self.cfg
        if c.encoder_train_until >= 1.0:
            return None
        steps = round(c.encoder_train_until * c.total_env_steps)
        return -(-steps // (c.num_envs * c.rollout_len))

    def encoder_trains(self, iteration: Optional[int] = None) -> bool:
        """Whether the encoder trains in this iteration (default the current one):
        unfrozen (--no-encoder-freeze, or from --unfreeze-at) and before
        encoder_stop_iteration. A function of the iteration and the config only, used by
        every code path: the ring re-encoding (rollout), the z of the update, the logs
        (the encoder's AdamW groups get no gradients otherwise: not stepped, no decay)."""
        c = self.cfg
        it = self.iteration if iteration is None else iteration
        if c.encoder_freeze and not (c.unfreeze_at >= 0 and it >= c.unfreeze_at):
            return False
        stop = self.encoder_stop_iteration()
        return stop is None or it < stop

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
        if self.encoder_trains() or self.ring_stale:  # Ring z from the current encoder
            self.ring = ag.reencode_ring(self.ring)
        self.ring_stale = False
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
                                     info["scenario_src"], info["scenario_gen"])
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

    @torch.no_grad()
    def reencode_buffer(self):
        """z of every stored frame (the K - 1 context positions and the T steps)
        recomputed by the current encoder without grad, in blocks of the update's
        minibatch (num_envs / minibatches envs)."""
        b, ag = self.buf, self.agent
        Bm = b.B // self.cfg.minibatches
        for i in range(0, b.B, Bm):
            b.z[i:i + Bm] = ag.encode(b.frames[i:i + Bm].flatten(0, 1)).view(-1, *b.z.shape[1:])

    def update(self) -> Dict[str, torch.Tensor]:
        """GAE, value-norm update and the PPO epochs on self.buf; returns stats. Sets
        the learning rate of this iteration (lr_at) and, with ent_adaptive, steps the
        entropy coefficient after every minibatch. When the encoder trains: z from the
        stored frames with grad in the first encoder_epochs epochs (0 = all), then
        self.buf.z recomputed once by the updated encoder (reencode_buffer) and used
        without grad; the ring is marked stale for the next rollout."""
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
        enc_ep = c.encoder_epochs if 0 < c.encoder_epochs < c.epochs else c.epochs
        B = b.B
        Bm = B // c.minibatches
        assert Bm * c.minibatches == B, "num_envs must be divisible by minibatches"
        params = [p for p in ag.parameters() if p.requires_grad]
        keys = ("loss", "pg", "vf", "ent", "path", "tilt", "scen", "kl", "clipfrac", "gn",
                "ent_coef", "ratio_dev")
        acc = {k: torch.zeros((), device=self.device) for k in keys}
        n_mb = 0
        epochs = enc_epochs = 0
        for epoch in range(c.epochs):
            if trains_enc and epoch == enc_ep:  # The encoder's epochs done (no KL stop)
                self.reencode_buffer()
            enc_grad = trains_enc and epoch < enc_ep
            perm = torch.randperm(B, generator=self.gen_mb, device=self.device)
            ekl = torch.zeros((), device=self.device)
            for i in range(c.minibatches):
                if self.stop_signal is not None:
                    break
                idx = perm[i * Bm:(i + 1) * Bm]
                if enc_grad:
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
            enc_epochs += int(enc_grad)
            if self.stop_signal is not None:
                break
            if float(ekl) / c.minibatches > c.target_kl:  # Host sync, once per epoch
                break
        if trains_enc and n_mb > 0:
            self.ring_stale = True  # The ring's z predate this update
        out = {k: v / max(n_mb, 1) for k, v in acc.items() if k != "ratio_dev"}
        out["ratio_dev_first_mb"] = acc["ratio_dev"]
        out["epochs"] = torch.tensor(float(epochs))
        out["enc_epochs"] = torch.tensor(float(enc_epochs))
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
        return eval_init_tilts(self.cfg, n, limit, self.device)

    def _oracle(self, env: AlignEnv, tilts: torch.Tensor, setups: torch.Tensor, file: str,
                key: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Oracle plan of the set (cached_oracle), cached in the run dir by key."""
        return cached_oracle(env, tilts, setups, os.path.join(self.run_dir, file), key, label)

    def _setup_set(self, kind: str) -> EvalSet:
        return setup_eval_set(self.cfg, kind, self.run_dir, self.device)

    def _full_set(self) -> EvalSet:
        return full_eval_set(self.cfg, self.atlas, self.run_dir, self.device)

    def setup_eval(self):
        """Eval envs, sets and oracle references (cached in the run dir), eval_set_names."""
        names = eval_set_names(self.cfg)
        if self.atlas is None and any(n in ATLAS_SETS for n in names):
            self._load_atlas()
        self.eval_sets = build_eval_sets(self.cfg, names, self.atlas,
                                         self.run_dir, self.device,
                                         log=lambda s: print(s, flush=True))

    @torch.no_grad()
    def _run_eval(self, es: EvalSet, sampled: bool) -> Dict[str, float]:
        """One evaluation of a set; flat metrics <prefix>/<split>/<metric>."""
        return run_eval(self.agent, es, self.cfg.eval_seed, sampled)[0]

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
            "ring_stale": self.ring_stale,
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
        # Older checkpoints: stale if the encoder trained in the last iteration
        self.ring_stale = bool(d.get("ring_stale", self.encoder_trains(self.iteration - 1)))
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
        stop = self.encoder_stop_iteration()
        if (not c.encoder_freeze or c.unfreeze_at >= 0) and (stop is not None or c.encoder_epochs > 0):
            ne = min(c.encoder_epochs or c.epochs, c.epochs)
            print(f"encoder schedule: gradients in the first {ne} of {c.epochs} PPO epochs; "
                  + ("trains until the end" if stop is None else
                     f"frozen from iteration {stop} ({stop * steps_per_iter} env steps >= "
                     f"{c.encoder_train_until:g} x {c.total_env_steps})"), flush=True)
        while self.env_steps < c.total_env_steps and self.stop_signal is None:
            it = self.iteration
            if it > 0 and self.encoder_trains() != self.encoder_trains(it - 1):
                if self.encoder_trains():
                    print(f"iteration {it}: encoder unfrozen (lr x {c.encoder_lr_scale})", flush=True)
                else:
                    print(f"iteration {it}: encoder frozen for the rest of the run (env steps "
                          f"{self.env_steps} >= {c.encoder_train_until:g} x {c.total_env_steps}); "
                          f"the update reuses the rollout's z", flush=True)
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
        if "pool/candidates" in row:
            hist = " ".join(f"{k[len('pool/hist_sep'):]} {100 * v:.0f}" for k, v in row.items()
                            if k.startswith("pool/hist_"))
            print(f"      pool: refresh {row['pool/refresh_s']:.1f} s, {int(row['pool/replaced'])} slots, "
                  f"{int(row['pool/candidates'])} candidates ({int(row['pool/prefix'])} used): angle "
                  f"{100 * row['pool/angle_rejected_share']:.1f} %, clipped "
                  f"{100 * row['pool/clipped_share']:.1f} %, missed {100 * row['pool/missed_share']:.1f} %, "
                  f"valid {100 * row['pool/valid_share']:.1f} %, {int(row['pool/scans'])} box scans, good "
                  f"{100 * row['pool/good_share']:.1f} % of valid | good slots % by sep: {hist}",
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


def encoder_origin(encoder_info: Optional[dict]) -> str:
    """Encoder part of the run name: cnn, cnn-pre<S> (pretrained on S setups),
    cnn-pregrid<N> (N grid scenarios), cnn-precont / cnn-ftcont (continuous scenarios,
    from scratch / fine-tuned from another pretrain_encoder.py checkpoint)."""
    enc = "cnn"
    if encoder_info is not None:
        nm = encoder_info.get("name") or ""
        m = re.search(r"_grid(\d+)", nm)
        if "_cont" in nm:
            enc += "-ftcont" if "_ft-" in nm else "-precont"
        elif m:
            enc += f"-pregrid{m.group(1)}"
        else:
            enc += f"-pre{encoder_info.get('n_setups', 1)}"
    return enc


def run_name(c: PPOConfig, encoder_info: Optional[dict]) -> str:
    """Display / directory name from the config, built at start:
    ppo_<enc>_k<K>_<scen>_b<B>_s<seed>[_ft<run>][_uf<it>|_e2e][_ee<N>][_eu<F>][_pr<F>][_<tag>]_<jobid>;
    <enc>: encoder_origin; <scen>: <ranges>-plr / -unif with the pool (ContRanges.tag,
    e.g. cont20-400-plr), full-plr / full-unif with the atlas (prioritized / uniform
    good draws), or set<S>, with -ho<pct>[s<seed>] when grid scenarios are held out
    (--holdout-frac); _ft<run>: initialized from that PPO run (--init-from); _ee<N> /
    _eu<F>: --encoder-epochs / --encoder-train-until when set; _pr<F>: --pool-refresh
    when not the default (pool runs)."""
    enc = encoder_origin(encoder_info)
    if pool_mode(c):
        scen = cont_ranges_of(c).tag() + ("-plr" if c.plr else "-unif")
    elif c.setups == "full":
        scen = "full-plr" if c.plr else "full-unif"
        if c.holdout_frac > 0:
            scen += f"-ho{100 * c.holdout_frac:g}" + (f"s{c.holdout_seed}" if c.holdout_seed else "")
    else:
        scen = f"set{len(SETUPS) if c.setups == 'all' else 1}"
    name = f"ppo_{enc}_k{c.history_len}_{scen}_b{c.num_envs}_s{c.seed}"
    if c.init_from:
        name += f"_ft{init_run_id(c.init_from)}"
    if not c.encoder_freeze:
        name += "_e2e"
    elif c.unfreeze_at >= 0:
        name += f"_uf{c.unfreeze_at}"
    if c.encoder_epochs > 0:
        name += f"_ee{c.encoder_epochs}"
    if c.encoder_train_until < 1.0:
        name += f"_eu{c.encoder_train_until:g}"
    if pool_mode(c) and c.pool_refresh != PPOConfig.pool_refresh:
        name += f"_pr{c.pool_refresh:g}"
    if c.name_tag:
        name += f"_{c.name_tag}"
    return name + "_" + os.environ.get("SLURM_JOB_ID", "local")


# ── CLI ─────────────────────────────────────────────────────────────────────


RESUME_OVERRIDES = ("total_env_steps", "eval_every", "ckpt_minutes", "wandb", "wandb_project",
                    "unfreeze_at", "encoder_lr_scale", "compile", "target_kl", "eval_sampled",
                    "enc_checkpoint", "eval_offgrid", "eval_extrap", "eval_tables",
                    "encoder_epochs", "encoder_train_until")
PATH_FIELDS = ("encoder_init", "init_from", "atlas", "eval_tables")
TUPLE_FIELDS = tuple(f.name for f in fields(PPOConfig) if isinstance(getattr(PPOConfig(), f.name), tuple))
LEGACY_DEFAULTS = {"scenarios": "atlas"}  # Checkpoints from before the field: the grid atlas


def config_from_dict(d: Dict[str, Any]) -> PPOConfig:
    """PPOConfig of a saved config dict (checkpoint cfg): unknown keys dropped, lists to
    tuples, fields missing in older checkpoints from LEGACY_DEFAULTS (else the default)."""
    known = {f.name for f in fields(PPOConfig)}
    d = {**LEGACY_DEFAULTS, **{k: v for k, v in d.items() if k in known}}
    for k in TUPLE_FIELDS:
        if k in d:
            d[k] = tuple(d[k])
    return PPOConfig(**d)


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
        elif isinstance(v, tuple):
            ap.add_argument(flag, type=float, nargs=len(v), default=dflt(list(v)),
                            metavar=("MIN", "MAX") if f.name.endswith("_range") else None)
        elif f.name == "setups":
            ap.add_argument(flag, choices=SETUP_CHOICES, default=dflt(v))
        elif f.name == "scenarios":
            ap.add_argument(flag, choices=SCENARIO_CHOICES, default=dflt(v))
        elif f.name == "laser_mode":
            ap.add_argument(flag, choices=("landing", "angle"), default=dflt(v))
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
    for k in TUPLE_FIELDS:
        cli[k] = tuple(cli[k])
    if args.resume:
        assert args.run_dir, "--resume needs --run-dir"
        path = os.path.join(args.run_dir, "latest.pt")
        saved = asdict(config_from_dict(torch.load(path, map_location="cpu", weights_only=False)["cfg"]))
        given = vars(build_parser(suppress=True).parse_args(argv))
        for k in RESUME_OVERRIDES:  # Run control from the flags given; the rest as saved
            if k in given:
                saved[k] = os.path.abspath(cli[k]) if k in PATH_FIELDS and cli[k] else cli[k]
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
