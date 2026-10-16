"""
evaluation.py - Deterministic evaluation of a policy on a fixed set of initial tilts

Shared by the baselines (baselines.py) and the PPO trainer.

    init_tilts   eval_initial_tilts(n, seed): uniform in the +-limit box, seeded on the
                 CPU, the same set on every device
    init_setups  eval_setups(n, n_setups, seed): setup index per episode, balanced
                 (n / S each) in a seeded random order; (tilt, setup) pairs
    episodes     exactly one per (initial tilt, setup); waves of env.num_envs envs reset
                 with explicit tilts and setups, every env runs a single episode and is
                 then ignored (with auto-reset the env restarts it at a random tilt,
                 without it idles; the policy still sees it with info["active"] False)
    policy       policy(obs, info, state) -> (actions (B, 2) int64, state); state is
                 None at the start of each wave, the policy builds its own
    info keys    reset (B,) bool        env is at step 0 of an episode (incl. auto-resets)
                 step (B,) int64        step index within the episode
                 active (B,) bool       env still runs its evaluated episode
                 episode (B,) int64     index into init_tilts (-1: inactive or padding)
                 privileged (diagnostics, oracle and setup-aware homing only):
                 tilt (B, 2), path_mm (B,), bounces (B,) of the current obs frame,
                 setup (B,), scenario_params (B, 5) and path_ref (B,) of the episode

Metrics (summarize): the aggregate over all episodes and split by the start region:
blind start = 1-2 bounces at the initial tilt (one M2 spot at a fixed place), informative
start = >= 3 bounces; with several setups also one split per setup. Per episode the final
path is also compared with the env's P_ref of its setup (path_frac_pref, success = final
path >= SUCCESS_FRAC * P_ref; with path_ref "max" P_ref is the setup's P_max).
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from env import AlignEnv

BLIND_MAX_BOUNCES = 2  # Initial bounces <= this: blind start
LEVELS = (31, 57, 89)  # Bounce levels of the hole-exit patterns; the cap is added
BOUNCE_BINS = ((0, 2), (3, 30), (31, 31), (32, 56), (57, 57), (58, 88), (89, 89))  # Inclusive
SUCCESS_FRAC = 0.9  # Success: final path >= this fraction of P_ref (PLAN stage 2)

Policy = Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Any], Tuple[torch.Tensor, Any]]


# ── Initial tilts ───────────────────────────────────────────────────────────


def eval_initial_tilts(
    n: int = 4096, seed: int = 0, limit: float = 2.0, device: Optional[torch.device] = None
) -> torch.Tensor:
    """(n, 2) float32 initial tilts (deg), uniform in [-limit, limit]^2, deterministic."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    u = torch.rand(n, 2, generator=gen, dtype=torch.float64)
    t = (limit * (2.0 * u - 1.0)).float()
    return t.to(device) if device is not None else t


def eval_setups(n: int, n_setups: int, seed: int = 0,
                device: Optional[torch.device] = None) -> torch.Tensor:
    """(n,) int64 setup per episode: i % n_setups in a seeded random order (balanced)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1)
    s = (torch.arange(n) % n_setups)[torch.randperm(n, generator=gen)]
    return s.to(device) if device is not None else s


# ── Rollout ─────────────────────────────────────────────────────────────────


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _policy_info(
    env: AlignEnv, active: torch.Tensor, episode: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Info dict for the policy, describing the current obs frame."""
    si = env.state_info()  # Scenario parameters and P_ref of the episodes (also pool mode)
    return {
        "reset": env.step_count == 0,
        "step": env.step_count,
        "active": active,
        "episode": torch.where(active, episode, -1),
        "tilt": env.tilt,
        "path_mm": env.path,
        "bounces": env.bounces,
        "setup": env.setup,
        "scenario_params": si["scenario_params"],
        "path_ref": si["path_ref"],
    }


@torch.no_grad()
def evaluate(
    env: AlignEnv,
    policy: Policy,
    init_tilts: torch.Tensor,
    ref_path_mm: Optional[torch.Tensor] = None,
    ref_return: Optional[torch.Tensor] = None,
    init_setups: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Run one episode per (initial tilt, setup); returns (per-episode tensors, metrics).

    init_setups (n,): setup indices; default 0 with one setup, eval_setups(n, S) with S.
    Per-episode tensors (n leading), on env.device:
        setup, path_ref (P_ref of the setup), init_tilt (n, 2), init_path_mm,
        init_bounces, final_tilt (n, 2), final_path_mm,
        final_bounces, ep_return, ep_length, ep_moves, terminated (bool), truncated
        (bool), first_reach (n, len(LEVELS) + 1) step of the first frame with bounces
        >= 31 / 57 / 89 / cap (0 = initial frame, -1 = never), ref_path_mm and
        ref_return if given.
    ref_path_mm (n,): reference path for the final-path fraction (the oracle's target);
    ref_return (n,): reference return for the regret (the oracle's planned return).
    metrics: summarize(...) plus wall time.
    """
    dev = env.device
    B = env.num_envs
    n = init_tilts.shape[0]
    cap = env.cfg.max_bounces
    levels = torch.tensor(LEVELS + (cap,), device=dev)
    init_tilts = init_tilts.to(device=dev, dtype=torch.float32)
    if init_setups is None:
        init_setups = (torch.zeros(n, dtype=torch.long) if env.n_setups == 1
                       else eval_setups(n, env.n_setups))
    init_setups = init_setups.to(device=dev, dtype=torch.long)

    ep = {
        "setup": init_setups.clone(),
        "path_ref": env.ref_of(init_setups),
        "init_tilt": torch.zeros(n, 2, device=dev),
        "init_path_mm": torch.zeros(n, device=dev),
        "init_bounces": torch.zeros(n, dtype=torch.long, device=dev),
        "final_tilt": torch.zeros(n, 2, device=dev),
        "final_path_mm": torch.zeros(n, device=dev),
        "final_bounces": torch.zeros(n, dtype=torch.long, device=dev),
        "ep_return": torch.zeros(n, device=dev),
        "ep_length": torch.zeros(n, dtype=torch.long, device=dev),
        "ep_moves": torch.zeros(n, dtype=torch.long, device=dev),
        "terminated": torch.zeros(n, dtype=torch.bool, device=dev),
        "truncated": torch.zeros(n, dtype=torch.bool, device=dev),
        "first_reach": torch.full((n, len(levels)), -1, dtype=torch.long, device=dev),
    }

    _sync(dev)
    t0 = time.perf_counter()
    for w0 in range(0, n, B):
        m = min(B, n - w0)
        idx = torch.arange(B, device=dev)
        valid = idx < m
        episode = torch.where(valid, w0 + idx, 0)
        tilts = init_tilts[episode]  # Padding rows repeat episode w0, inactive
        obs = env.reset(tilts=tilts, setups=init_setups[episode])

        e = episode[valid]
        ep["init_tilt"][e] = env.tilt[valid]
        ep["init_path_mm"][e] = env.path[valid]
        ep["init_bounces"][e] = env.bounces[valid]
        first = torch.where(env.bounces[:, None] >= levels, 0, -1)  # (B, L)

        active = valid.clone()
        state = None
        for t in range(env.cfg.max_steps):
            info_p = _policy_info(env, active, episode)
            action, state = policy(obs, info_p, state)
            obs, _, term, trunc, info = env.step(action)

            hit = (info["bounces"][:, None] >= levels) & (first < 0) & active[:, None]
            first = torch.where(hit, t + 1, first)
            done = info["done"] & active
            if bool(done.any()):
                d = done.nonzero().squeeze(1)
                e = episode[d]
                ep["final_tilt"][e] = info["tilt"][d]
                ep["final_path_mm"][e] = info["ep_final_path_mm"][d]
                ep["final_bounces"][e] = info["ep_final_bounces"][d]
                ep["ep_return"][e] = info["ep_return"][d]
                ep["ep_length"][e] = info["ep_length"][d]
                ep["ep_moves"][e] = info["ep_moves"][d]
                ep["terminated"][e] = term[d]
                ep["truncated"][e] = trunc[d]
                ep["first_reach"][e] = first[d]
                active = active & ~done
            if not bool(active.any()):
                break
        assert not bool(active.any()), "episodes left after max_steps"
    _sync(dev)
    wall = time.perf_counter() - t0

    if ref_path_mm is not None:
        ep["ref_path_mm"] = ref_path_mm.to(dev)
    if ref_return is not None:
        ep["ref_return"] = ref_return.to(dev)
    names = [g.name for g in env.setups] if env.n_setups > 1 else None
    metrics = summarize(ep, cap, names)
    metrics["wall_s"] = wall
    return ep, metrics


# ── Metrics ─────────────────────────────────────────────────────────────────


def bounce_bins(cap: int):
    """[(label, lo, hi)] inclusive bounce bins of the final-bounce distribution."""
    bins = [(f"{lo}" if lo == hi else f"{max(lo, 1)}-{hi}", lo, hi) for lo, hi in BOUNCE_BINS]
    bins.append((f"90-{cap - 1}", 90, cap - 1))
    bins.append(("cap", cap, 10**9))
    return bins


def _q(x: torch.Tensor, q: float) -> Optional[float]:
    return float(torch.quantile(x.double(), q)) if x.numel() else None


def _mean(x: torch.Tensor) -> Optional[float]:
    return float(x.double().mean()) if x.numel() else None


def _frac(m: torch.Tensor) -> Optional[float]:
    return float(m.double().mean()) if m.numel() else None


def split_metrics(ep: Dict[str, torch.Tensor], m: torch.Tensor, cap: int) -> Dict[str, Any]:
    """Metrics over the episodes selected by mask m (any split, e.g. ppo.py's)."""
    fb = ep["final_bounces"][m]
    fp = ep["final_path_mm"][m] / 1000.0
    length = ep["ep_length"][m]
    out: Dict[str, Any] = {
        "n": int(m.sum()),
        "share": _frac(m),
        "return_mean": _mean(ep["ep_return"][m]),
        "init_path_m_mean": _mean(ep["init_path_mm"][m] / 1000.0),
        "final_path_m_mean": _mean(fp),
        "final_path_m_median": _q(fp, 0.5),
        "end_ge31": _frac(fb >= 31),
        "end_ge57": _frac(fb >= 57),
        "end_ge89": _frac(fb >= 89),
        "end_cap": _frac(fb >= cap),
        "terminated": _frac(ep["terminated"][m]),
        "stop_first": _frac(ep["terminated"][m] & (length == 1)),  # Not move at step 0
        "truncated": _frac(ep["truncated"][m]),
        "length_mean": _mean(length),
        "length_median": _q(length, 0.5),
        "length_p90": _q(length, 0.9),
        "moves_mean": _mean(ep["ep_moves"][m]),
    }
    reach = {}
    for k, name in enumerate([str(v) for v in LEVELS] + ["cap"]):
        f = ep["first_reach"][m, k]
        r = f[f >= 0]
        reach[name] = {"frac": _frac(f >= 0), "steps_mean": _mean(r), "steps_median": _q(r, 0.5)}
    out["first_reach"] = reach

    out["final_bounce_bins"] = {
        label: _frac((fb >= lo) & (fb <= hi)) for label, lo, hi in bounce_bins(cap)
    }
    vals, cnt = torch.unique(fb, return_counts=True)
    out["final_bounce_hist"] = {str(int(v)): int(c) for v, c in zip(vals.tolist(), cnt.tolist())}

    if "path_ref" in ep:
        r = ep["final_path_mm"][m].double() / ep["path_ref"][m].double()
        out["path_frac_pref_mean"] = _mean(r)
        out["success"] = _frac(r >= SUCCESS_FRAC)
    if "ref_path_mm" in ep:
        r = ep["final_path_mm"][m].double() / ep["ref_path_mm"][m].double()
        out["path_frac_ref_mean"] = _mean(r)
        out["path_frac_ref_median"] = _q(r, 0.5)
    if "ref_return" in ep:
        out["regret_mean"] = _mean(ep["ref_return"][m].double() - ep["ep_return"][m].double())
    return out


def summarize(
    ep: Dict[str, torch.Tensor], cap: int, setup_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Metrics for all episodes, blind / informative starts and (with setup_names) per
    setup ("setup:<name>")."""
    blind = ep["init_bounces"] <= BLIND_MAX_BOUNCES
    splits = {"all": torch.ones_like(blind), "blind": blind, "informative": ~blind}
    if setup_names is not None:
        for i, name in enumerate(setup_names):
            splits[f"setup:{name}"] = ep["setup"] == i
    return {name: split_metrics(ep, m, cap) for name, m in splits.items()}
