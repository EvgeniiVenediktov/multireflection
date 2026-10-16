"""
scenarios.py - Scenario atlas and the regret-prioritized scenario sampler (PLAN stage 2)

Run from rl/:
    uv run python scenarios.py build [--atlas PATH] [--sep 50 200 2] [--pitch 0 8 0.25]
                                     [--yaw 0 8 0.25] [--no-named]
    uv run python scenarios.py summary [--atlas PATH]
Default atlas path runs/atlas/atlas_v1.pt (gitignored); ppo.py --setups full builds it
there (or at --atlas) when the file is missing and loads it otherwise.

Scenario = (separation, laser pitch, laser yaw); laser offset, mirror offsets, mirrors and
M1 as in the measured cell (geometry.CellGeometry). Grid: separation 50-200 mm step 2 x
laser pitch / yaw 0-8 deg step 0.25 (82,764 points, find_setups.py's coarse grid), plus
the named geometry.SETUPS added explicitly (a named setup on a grid point maps to it).

Atlas (one torch file, CPU tensors, N scenarios): per scenario a scan of the M2 +-2 deg
box on the 0.01 deg grid (the grid of AlignEnv.path_max, 401^2 cells) at cap 128 with the
Triton kernel without per-hit outputs, in chunks of scenarios:
    pmax, pmax_tilt (first cell reaching it), pmax_bounces
    plateau_order / plateau_cells / plateau_c / plateau_path: the hole-exit order >= 20
              bounces with the most cells (its area over the box, not one connected
              region), centroid and mean path (float64 sums: the atlas does not
              depend on the chunking)
    best_bounces, best_cells: highest bounce count (incl. the cap) and its cells
    near_cells: cells with path >= 0.9 P_max (the success region, evaluation.SUCCESS_FRAC)
    blind_cells: cells with <= 2 bounces
    first_hit: M2-local (x, y) of the first hit at zero M2 tilt (NaN if it misses M2)
    p_floor = floor_bounces x separation (path of a 20-bounce pattern); good = pmax >=
    p_floor, the rest "impossible"; path_ref = max(pmax, p_floor) is the env's reward
    scale (impossible scenarios give small rewards, the step cost dominates: the right
    behavior there is to stop at once)
Cost: 82,768 x 160,801 = 13.3 G rays; the coarse search (543 M rays incl. per-hit outputs)
took 24 s on the RTX A2000, so about 10 min there and 1-2 min on an L40S (estimate).

ScenarioSampler (all on the device; draw() and observe() have no host syncs):
    draws      one categorical over [impossible | fresh good | replay good] segments:
               impossible_share (10 %) uniform over the impossible set; the rest a
               good scenario, with probability p_fresh (0.3; 1.0 until min_seen = 2048
               good scenarios have a score) uniform over the good set, otherwise from
               the replay distribution over the seen good scenarios. The env passes one
               counter-based uniform per (env, episode) (env.AlignEnv._draw); the
               segment of a draw is its source (SRC_IMPOSSIBLE / SRC_FRESH / SRC_REPLAY)
    scores     per scenario: EMA (alpha 0.3) of the shortfall 1 - P_final / P_max over
               finished episodes (terminated or truncated); episodes of one step that
               share a scenario enter as their mean (scatter ops)
    replay     PLR (Jiang et al. 2021): P = (1 - rho) P_S + rho P_C, rank priority
               P_S ~ (1 / rank(score))^(1 / beta) (beta 0.3, rank 1 = largest shortfall),
               staleness P_C ~ iterations since the last score update (rho 0.1)
    refresh    once per iteration (refresh(iteration)): the CDF of the draws
    logging    stats(): seen count, score quantiles, impossible share, mean shortfall of
               fresh and replayed episodes, the top-priority scenario and its draw share
"""

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from env import Scenarios
from find_setups import SearchConfig, box_grid, candidate_states, coarse_candidates
from geometry import SETUPS, CellGeometry
from herriott_sim import DEVICE, STOP_HOLE

RL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ATLAS = os.path.join(RL_DIR, "runs", "atlas", "atlas_v1.pt")
ATLAS_VERSION = 1
PARAM_NAMES = ("separation", "laser_pitch", "laser_yaw")
SRC_EXPLICIT, SRC_IMPOSSIBLE, SRC_FRESH, SRC_REPLAY = -1, 0, 1, 2
SEP_BINS = ((50.0, 80.0), (80.0, 110.0), (110.0, 140.0), (140.0, 170.0), (170.0, 200.0))


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class AtlasConfig:
    """Atlas grid and scan settings."""

    sep: Tuple[float, float, float] = (50.0, 200.0, 2.0)  # mm (lo, hi, step), inclusive
    pitch: Tuple[float, float, float] = (0.0, 8.0, 0.25)  # deg, laser
    yaw: Tuple[float, float, float] = (0.0, 8.0, 0.25)  # deg, laser
    named: bool = True  # Add geometry.SETUPS
    limit: float = 2.0  # M2 box half-width (deg)
    step: float = 0.01  # M2 grid (deg)
    cap: int = 128  # Bounce cap (D1)
    min_order: int = 20  # Hole-exit orders that count as a plateau
    floor_bounces: int = 20  # P_floor = floor_bounces x separation
    near_frac: float = 0.9  # near_cells: path >= near_frac x P_max
    chunk_states: int = 1 << 23  # States per sim call (~1.3 GB peak with the statistics)

    def key(self) -> Dict[str, Any]:
        """What the atlas values depend on (chunking excluded) and the cell."""
        d = {k: v for k, v in asdict(self).items() if k != "chunk_states"}
        d["geometry"] = asdict(CellGeometry())
        return d


def grid_params(cfg: AtlasConfig) -> np.ndarray:
    """(n, 3) float64 grid points [sep, laser pitch, laser yaw], separation-major."""
    return coarse_candidates(SearchConfig(sep=cfg.sep, pitch=cfg.pitch, yaw=cfg.yaw))


def scenario_params(cfg: AtlasConfig) -> Tuple[np.ndarray, Dict[str, int], int]:
    """All scenario parameters (grid, then the named setups not on the grid), the index
    of each named setup and the number of grid points."""
    p = grid_params(cfg)
    n_grid = p.shape[0]
    named: Dict[str, int] = {}
    if cfg.named:
        tmpl = CellGeometry()
        extra = []
        for g in SETUPS:
            same = replace(g, name="", separation=tmpl.separation, laser_pitch=tmpl.laser_pitch,
                           laser_yaw=tmpl.laser_yaw, m2_pitch=tmpl.m2_pitch, m2_yaw=tmpl.m2_yaw)
            assert same == replace(tmpl, name=""), f"{g.name}: differs beyond the scenario"
            v = np.array([g.separation, g.laser_pitch, g.laser_yaw])
            hit = np.nonzero(np.abs(p - v).max(1) < 1e-9)[0]
            if hit.size:
                named[g.name] = int(hit[0])
            else:
                named[g.name] = n_grid + len(extra)
                extra.append(v)
        if extra:
            p = np.concatenate((p, np.stack(extra)))
    return p, named, n_grid


def sample_grid(n: int, seed: int, cfg: Optional[AtlasConfig] = None) -> np.ndarray:
    """(n, 3) float64 scenarios drawn uniformly from the grid points (with replacement),
    without the atlas (eval_encoder.py --scenarios)."""
    cfg = cfg or AtlasConfig()
    axes = []
    for lo, hi, st in (cfg.sep, cfg.pitch, cfg.yaw):
        axes.append(lo + st * np.arange(int(round((hi - lo) / st)) + 1, dtype=np.float64))
    rng = np.random.default_rng(seed)
    return np.stack([a[rng.integers(0, a.size, n)] for a in axes], 1)


# ── Scan ────────────────────────────────────────────────────────────────────


@torch.no_grad()
def scan_scenarios(sim, geom: CellGeometry, params: torch.Tensor, grid: torch.Tensor,
                   cfg: AtlasConfig) -> Dict[str, torch.Tensor]:
    """Atlas statistics of scenarios params (n, 3) float32 over the M2 grid (G, 2)."""
    n, G = params.shape[0], grid.shape[0]
    dev = grid.device
    cap, mo = cfg.cap, cfg.min_order
    out = sim.simulate(candidate_states(geom, params, grid), store_world=False, store_hits=False)
    b = out["hit_counts"].view(n, G)
    hole = (out["stop_reason"] == STOP_HOLE).view(n, G)
    p = out["total_path_length"].view(n, G)
    del out
    rows = torch.arange(n, device=dev)
    pmax, imax = p.max(1)
    idx = (rows[:, None] * (cap + 1) + b)[hole]
    hist_hole = torch.bincount(idx, minlength=n * (cap + 1)).view(n, cap + 1)
    pc, po = hist_hole[:, mo:].max(1)
    plateau_order = torch.where(pc > 0, po + mo, 0)
    pmask = hole & (b == plateau_order[:, None]) & (pc[:, None] > 0)
    best_b = b.amax(1)
    first = sim.simulate(candidate_states(geom, params, torch.zeros(1, 2, device=dev)),
                         store_world=False)
    on_m2 = first["hit_mirror"][:, 0] == 2
    return {
        "pmax": pmax,
        "pmax_tilt": grid[imax],
        "pmax_bounces": b[rows, imax].to(torch.int16),
        "plateau_order": plateau_order.to(torch.int16),
        "plateau_cells": pc.to(torch.int32),
        "plateau_c": ((pmask.double() @ grid.double()) / pc.double()[:, None]).float(),
        "plateau_path": (torch.where(pmask, p.double(), 0.0).sum(1) / pc.clamp(min=1)).float(),
        "best_bounces": best_b.to(torch.int16),
        "best_cells": (b == best_b[:, None]).sum(1).to(torch.int32),
        "near_cells": (p >= cfg.near_frac * pmax[:, None]).sum(1).to(torch.int32),
        "blind_cells": (b <= 2).sum(1).to(torch.int32),
        "first_hit": torch.where(on_m2[:, None], first["hit_local_xy"][:, 0], math.nan),
    }


@torch.no_grad()
def build_atlas(cfg: Optional[AtlasConfig] = None, device: Optional[torch.device] = None,
                log=print) -> Dict[str, Any]:
    """Scan every scenario (module docstring); returns the atlas dict (CPU tensors)."""
    cfg = cfg or AtlasConfig()
    dev = torch.device(device) if device is not None else DEVICE
    geom = CellGeometry()
    sim = geom.make_sim(dev, max_bounces=cfg.cap)
    params, named, n_grid = scenario_params(cfg)
    grid, W = box_grid(cfg.limit, cfg.step, dev)
    G = grid.shape[0]
    per = max(1, cfg.chunk_states // G)
    pt = torch.tensor(params, dtype=torch.float32, device=dev)
    N = pt.shape[0]
    log(f"atlas: {N} scenarios ({n_grid} grid + {N - n_grid} named) x {W}^2 M2 cells "
        f"({N * G / 1e9:.2f} G rays), {per} scenarios per sim call")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    parts: Dict[str, list] = {}
    for i in range(0, N, per):
        for k, v in scan_scenarios(sim, geom, pt[i:i + per], grid, cfg).items():
            parts.setdefault(k, []).append(v)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    build_s = time.perf_counter() - t0
    d: Dict[str, Any] = {k: torch.cat(v).cpu() for k, v in parts.items()}
    pf = cfg.floor_bounces * pt[:, 0].cpu()
    d.update(params=pt.cpu(), p_floor=pf, good=d["pmax"] >= pf,
             path_ref=torch.maximum(d["pmax"], pf))
    d.update(version=ATLAS_VERSION, key=cfg.key(), named=named, n_grid=n_grid,
             build_s=build_s, device_name=torch.cuda.get_device_name(dev) if dev.type == "cuda"
             else str(dev), date=time.strftime("%Y-%m-%d"))
    log(f"atlas: built in {build_s:.1f} s on {d['device_name']}; {int(d['good'].sum())} good, "
        f"{int((~d['good']).sum())} impossible (P_max < {cfg.floor_bounces} x separation)")
    return d


def save_atlas(d: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(d, tmp)
    os.replace(tmp, path)


# ── Atlas ───────────────────────────────────────────────────────────────────


class Atlas:
    """Loaded atlas on a device: tensors as attributes (module docstring), plus
    good_ids / impossible_ids (int64), named (name -> index), key, path."""

    TENSORS = ("params", "pmax", "pmax_tilt", "pmax_bounces", "plateau_order", "plateau_cells",
               "plateau_c", "plateau_path", "best_bounces", "best_cells", "near_cells",
               "blind_cells", "first_hit", "p_floor", "good", "path_ref")

    def __init__(self, d: Dict[str, Any], device: Optional[torch.device] = None, path: str = ""):
        dev = torch.device(device) if device is not None else DEVICE
        assert d.get("version") == ATLAS_VERSION, f"atlas version {d.get('version')}"
        for k in self.TENSORS:
            setattr(self, k, d[k].to(dev))
        self.device = dev
        self.key: Dict[str, Any] = d["key"]
        self.named: Dict[str, int] = dict(d["named"])
        self.n_grid: int = int(d["n_grid"])
        self.build_s: float = float(d.get("build_s", float("nan")))
        self.path = path
        self.good_ids = torch.nonzero(self.good).squeeze(1)
        self.impossible_ids = torch.nonzero(~self.good).squeeze(1)

    @property
    def n(self) -> int:
        return self.params.shape[0]

    def table(self) -> Scenarios:
        """Scenario table for AlignEnv: parameters and P_ref = max(P_max, P_floor)."""
        return Scenarios(params=self.params, path_ref=self.path_ref)

    def check(self, cap: int, limit: float, step: float):
        """The atlas must describe the env's cell, cap, box and tilt grid."""
        k = self.key
        want = {"cap": cap, "limit": limit, "step": step, "geometry": asdict(CellGeometry())}
        bad = {n: (k.get(n), v) for n, v in want.items() if k.get(n) != v}
        assert not bad, f"atlas {self.path} does not match the env (atlas, env): {bad}"

    def summary(self) -> str:
        k = self.key
        n_good = int(self.good.sum())
        pm = self.pmax[self.good] / 1000
        q = torch.quantile(pm.double(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64,
                                                     device=pm.device)).tolist() if n_good else []
        lines = [f"atlas {self.path}: {self.n} scenarios ({self.n_grid} grid: sep "
                 f"{k['sep']}, laser pitch {k['pitch']}, yaw {k['yaw']}; named "
                 f"{self.named}), M2 box +-{k['limit']} deg at {k['step']} deg, cap {k['cap']}; "
                 f"built in {self.build_s:.1f} s",
                 f"good {n_good} ({100 * n_good / self.n:.1f} %), impossible "
                 f"{self.n - n_good} (P_max < {k['floor_bounces']} x separation)"]
        if q:
            lines.append("good P_max (m): q10 / q50 / q90 " + " / ".join(f"{v:.2f}" for v in q))
        for lo, hi in SEP_BINS:
            m = (self.params[:, 0] >= lo) & (self.params[:, 0] < hi + (hi == SEP_BINS[-1][1]))
            if bool(m.any()):
                lines.append(f"  sep {lo:g}-{hi:g} mm: {int(m.sum())} scenarios, good "
                             f"{100 * float(self.good[m].float().mean()):.1f} %")
        return "\n".join(lines)


def load_atlas(path: str, device: Optional[torch.device] = None) -> Atlas:
    return Atlas(torch.load(path, map_location="cpu", weights_only=False), device, path)


def load_or_build(path: str, device: Optional[torch.device] = None,
                  cfg: Optional[AtlasConfig] = None, log=print) -> Atlas:
    """Load the atlas at path; if the file is missing, build it (default grid) and save it
    there first (the build time is logged)."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        log(f"atlas: {path} missing, building")
        save_atlas(build_atlas(cfg, device, log), path)
        log(f"atlas: saved {path}")
    return load_atlas(path, device)


def sep_bin_masks(sep: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Separation bins of SEP_BINS ([lo, hi), the last one closed): name -> mask."""
    out = {}
    for lo, hi in SEP_BINS:
        top = sep <= hi if hi == SEP_BINS[-1][1] else sep < hi
        out[f"sep{lo:g}-{hi:g}"] = (sep >= lo) & top
    return out


def eval_scenarios(atlas: Atlas, n: int, seed: int = 0,
                   impossible_share: float = 0.1) -> torch.Tensor:
    """(n,) int64 scenario per episode of the "full" eval set (CPU, seeded): round(n x
    impossible_share) uniform over the impossible set, the rest uniform over the good
    set (with replacement), in a seeded random order."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 7)
    good, imp = atlas.good_ids.cpu(), atlas.impossible_ids.cpu()
    n_imp = int(round(n * impossible_share)) if imp.numel() else 0
    if good.numel() == 0:
        n_imp = n
    parts = []
    if n - n_imp:
        parts.append(good[torch.randint(good.numel(), (n - n_imp,), generator=gen)])
    if n_imp:
        parts.append(imp[torch.randint(imp.numel(), (n_imp,), generator=gen)])
    s = torch.cat(parts)
    return s[torch.randperm(n, generator=gen)]


# ── Sampler ─────────────────────────────────────────────────────────────────


@dataclass
class SamplerConfig:
    """ScenarioSampler settings (module docstring)."""

    impossible_share: float = 0.1
    alpha: float = 0.3  # Score EMA
    p_fresh: float = 0.3  # Fresh uniform share of the good draws once min_seen is reached
    min_seen: int = 2048  # Good scenarios with a score before replay starts
    beta: float = 0.3  # Rank temperature: P_S ~ rank^(-1 / beta)
    rho: float = 0.1  # Staleness share of the replay distribution
    replay: bool = True  # False: good draws always uniform (no prioritization)


class ScenarioSampler:
    """Regret-prioritized scenario draws over an atlas, see the module docstring."""

    def __init__(self, atlas: Atlas, cfg: Optional[SamplerConfig] = None):
        self.cfg = cfg or SamplerConfig()
        self.atlas = atlas
        dev = self.device = atlas.device
        N = atlas.n
        g, im = atlas.good_ids, atlas.impossible_ids
        self.n_good, self.n_imp = g.numel(), im.numel()
        assert self.n_good + self.n_imp > 0
        self.good = atlas.good
        self.pmax = atlas.pmax.clamp(min=1e-6)
        self.score = torch.zeros(N, device=dev)
        self.seen = torch.zeros(N, dtype=torch.bool, device=dev)
        self.last = torch.zeros(N, dtype=torch.long, device=dev)  # Iteration of the last update
        self.visits = torch.zeros(N, dtype=torch.long, device=dev)  # Finished episodes
        self.iteration = 0
        # Draw lookup: segments [impossible | fresh good | replay good]
        self.ids = torch.cat((im, g, g))
        self.src = torch.cat((torch.full_like(im, SRC_IMPOSSIBLE), torch.full_like(g, SRC_FRESH),
                              torch.full_like(g, SRC_REPLAY))).to(torch.int8)
        self._acc = torch.zeros(4, 3, dtype=torch.float64, device=dev)
        self.refresh(0)

    # ── Distribution ──

    def replay_probs(self) -> torch.Tensor:
        """(n_good,) float64 replay distribution over the good scenarios (0 if unseen)."""
        c = self.cfg
        gs = self.atlas.good_ids
        seen = self.seen[gs]
        n_seen = seen.sum()
        key = torch.where(seen, self.score[gs], -math.inf)
        order = torch.argsort(key, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(1, order.numel() + 1, device=order.device)
        h = torch.where(seen, rank.double() ** (-1.0 / c.beta), 0.0)
        p_s = h / h.sum().clamp(min=1e-300)
        stale = torch.where(seen, (self.iteration - self.last[gs]).double(), 0.0)
        tot = stale.sum()
        p_c = torch.where(tot > 0, stale / tot.clamp(min=1e-300),
                          seen.double() / n_seen.clamp(min=1).double())
        return (1.0 - c.rho) * p_s + c.rho * p_c

    @torch.no_grad()
    def refresh(self, iteration: int):
        """Recompute the draw distribution (once per iteration; no host sync) and reset
        the per-iteration statistics."""
        c = self.cfg
        self.iteration = int(iteration)
        dev = self.device
        n_seen = self.seen[self.atlas.good_ids].sum()
        if c.replay and self.n_good:
            p_fresh = torch.where(n_seen >= c.min_seen, c.p_fresh, 1.0).double()
            p_r = self.replay_probs()
        else:
            p_fresh = torch.ones((), dtype=torch.float64, device=dev)
            p_r = torch.zeros(self.n_good, dtype=torch.float64, device=dev)
        q_imp = c.impossible_share if self.n_imp and self.n_good else float(self.n_good == 0)
        w = torch.cat((
            torch.full((self.n_imp,), q_imp / max(self.n_imp, 1), dtype=torch.float64, device=dev),
            (1.0 - q_imp) * p_fresh / max(self.n_good, 1)
            * torch.ones(self.n_good, dtype=torch.float64, device=dev),
            (1.0 - q_imp) * (1.0 - p_fresh) * p_r,
        ))
        cdf = torch.cumsum(w, 0)
        self.cdf = cdf / cdf[-1]
        self.p_fresh_eff = p_fresh
        self.p_replay = p_r
        self.q_imp = q_imp
        self._acc.zero_()

    # ── Draws and updates (no host syncs) ──

    def draw(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scenario indices (B,) int64 and their sources (B,) int8 for uniforms u (B,)."""
        j = torch.searchsorted(self.cdf, u.double().contiguous(), right=True)
        j = j.clamp(max=self.cdf.numel() - 1)
        return self.ids[j], self.src[j]

    @torch.no_grad()
    def observe(self, scenario: torch.Tensor, final_path: torch.Tensor, done: torch.Tensor,
                src: Optional[torch.Tensor] = None):
        """Episode ends: scenario (B,) index, final_path (B,) mm (P_final), done (B,) bool,
        src (B,) the draw source (statistics only). Updates the scores of the good
        scenarios by scatter ops."""
        a = self.cfg.alpha
        N = self.score.numel()
        good = self.good[scenario] & done
        short = (1.0 - final_path / self.pmax[scenario]).clamp(0.0, 1.0)
        w = good.float()
        cnt = torch.zeros(N, device=self.device).index_add_(0, scenario, w)
        tot = torch.zeros(N, device=self.device).index_add_(0, scenario, short * w)
        upd = cnt > 0
        mean = tot / cnt.clamp(min=1.0)
        self.score = torch.where(upd, torch.where(self.seen, (1 - a) * self.score + a * mean, mean),
                                 self.score)
        self.seen = self.seen | upd
        self.last = torch.where(upd, self.iteration, self.last)
        self.visits = self.visits + cnt.long()
        # Per-iteration statistics: rows (episodes, good episodes, shortfall sum, -) per
        # source column (impossible, fresh, replay); explicit sources are not counted
        if src is not None:
            s = src.long()
            ok = done & (s >= 0)
            col = s.clamp(min=0)
            d = ok.double()
            self._acc[0].index_add_(0, col, d)
            self._acc[1].index_add_(0, col, d * good.double())
            self._acc[2].index_add_(0, col, (short * w).double() * d)

    # ── Logging (host syncs) ──

    def stats(self) -> Dict[str, float]:
        """Flat statistics (plr/...) of the current state and the iteration so far."""
        gs = self.atlas.good_ids
        seen = self.seen[gs]
        n_seen = int(seen.sum())
        acc = self._acc.cpu()
        n_ep = float(acc[0].sum())
        out = {"plr/seen": n_seen, "plr/seen_frac": n_seen / max(self.n_good, 1),
               "plr/p_fresh": float(self.p_fresh_eff),
               "plr/episodes": n_ep,
               "plr/impossible_share": float(acc[0, 0]) / n_ep if n_ep else math.nan}
        for j, name in ((1, "fresh"), (2, "replay")):
            ng = float(acc[1, j])
            out[f"plr/{name}_episodes"] = float(acc[0, j])
            out[f"plr/{name}_shortfall"] = float(acc[2, j]) / ng if ng else math.nan
        nan = math.nan  # Every key always present (fixed CSV header)
        out.update({k: nan for k in ("plr/score_q10", "plr/score_q50", "plr/score_q90",
                                     "plr/score_mean", "plr/score_max", "plr/top_sep",
                                     "plr/top_laser_pitch", "plr/top_laser_yaw", "plr/top_score",
                                     "plr/top_pmax_m", "plr/top_draw_share",
                                     "plr/top10_draw_share")})
        if n_seen:
            sc = self.score[gs][seen].double()
            q = torch.quantile(sc, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64,
                                                device=sc.device)).tolist()
            out.update({"plr/score_q10": q[0], "plr/score_q50": q[1], "plr/score_q90": q[2],
                        "plr/score_mean": float(sc.mean()), "plr/score_max": float(sc.max())})
        p = self.p_replay
        if p.numel() and float(p.sum()) > 0:
            k = int(p.argmax())
            i = int(gs[k])
            share = (1.0 - self.q_imp) * (1.0 - float(self.p_fresh_eff))
            top = torch.topk(p, min(10, p.numel())).values
            prm = self.atlas.params[i].tolist()
            out.update({"plr/top_sep": prm[0], "plr/top_laser_pitch": prm[1],
                        "plr/top_laser_yaw": prm[2], "plr/top_score": float(self.score[i]),
                        "plr/top_pmax_m": float(self.atlas.pmax[i]) / 1000,
                        "plr/top_draw_share": share * float(p[k]),
                        "plr/top10_draw_share": share * float(top.sum())})
        return out

    # ── Checkpoint ──

    def state_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "seen": self.seen, "last": self.last,
                "visits": self.visits, "iteration": self.iteration}

    def load_state_dict(self, d: Dict[str, Any]):
        dev = self.device
        self.score, self.seen, self.last, self.visits = (
            d[k].to(dev) for k in ("score", "seen", "last", "visits"))
        self.refresh(d["iteration"])


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Scenario atlas (build / summary)")
    ap.add_argument("cmd", choices=("build", "summary"))
    ap.add_argument("--atlas", default=DEFAULT_ATLAS, help="Atlas file")
    d = AtlasConfig()
    ap.add_argument("--sep", type=float, nargs=3, default=list(d.sep), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--pitch", type=float, nargs=3, default=list(d.pitch), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--yaw", type=float, nargs=3, default=list(d.yaw), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--no-named", action="store_true", help="Do not add geometry.SETUPS")
    ap.add_argument("--chunk-states", type=int, default=d.chunk_states)
    ap.add_argument("--force", action="store_true", help="Rebuild an existing file")
    args = ap.parse_args()
    if args.cmd == "build":
        if os.path.exists(args.atlas) and not args.force:
            raise SystemExit(f"{args.atlas} exists (--force to rebuild)")
        cfg = AtlasConfig(sep=tuple(args.sep), pitch=tuple(args.pitch), yaw=tuple(args.yaw),
                          named=not args.no_named, chunk_states=args.chunk_states)
        save_atlas(build_atlas(cfg), args.atlas)
        print(f"saved {args.atlas}")
    print(load_atlas(args.atlas).summary())


if __name__ == "__main__":
    main()
