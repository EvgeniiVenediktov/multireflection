"""
scenarios.py - Scenario atlas, continuous scenario pool and the regret-prioritized
scenario sampler (PLAN stage 2, D7)

Run from rl/:
    uv run python scenarios.py build [--atlas PATH] [--sep 50 200 2] [--pitch 0 8 0.25]
                                     [--yaw 0 8 0.25] [--no-named]
    uv run python scenarios.py summary [--atlas PATH]
    uv run python scenarios.py offgrid [--atlas PATH] [--tables-dir DIR] [--kinds K,..]
                                       [--n 4096] [--impossible 0.1] [--seed 0] [--force]
    uv run python scenarios.py cont [--tables-dir DIR] [--sep-range 20 400]
                                    [--laser-range -8 8] [--laser-offset-radius 1.5]
                                    [--laser-mode landing] [--n 4096] [--impossible 0.1]
                                    [--seed 0] [--force]
    uv run python scenarios.py region [--tables-dir DIR] [ranges as cont] [--n 4096]
                                      [--seed 0] [--force]
Default atlas path runs/atlas/atlas_v1.pt (gitignored); ppo.py --setups full builds it
there (or at --atlas) when the file is missing and loads it otherwise. The off-grid eval
tables go to runs/eval_sets/<kind>_v1.pt (offgrid, extrap_short, extrap_long), the
continuous eval table to runs/eval_sets/<ranges tag>_n<n>_seed<seed>_v1.pt, the region
table to runs/eval_sets/<ranges tag>_region<edges>_n<n>_seed<seed>_v1.pt, likewise built
by ppo.py / eval_policy.py if missing.

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

Off-grid eval tables (scenarios never seen in training), one kind each (TABLE_KINDS):
    offgrid       interpolation: the midpoints of each grid axis (separation 51-199 mm
                  odd, laser pitch / yaw k x 0.25 + 0.125 = 0.125-7.875 deg; 75 x 32 x 32
                  = 76,800 points)
    extrap_short  extrapolation: separation 20-49 mm (integer), laser midpoints as above
    extrap_long   extrapolation: separation 201-350 mm (integer), laser midpoints
The lattice is visited in a seeded random order and scanned in rounds (same statistics,
cap and good / impossible rule as the atlas) until both groups have enough scenarios for
an eval set of OffgridConfig.n episodes, one per scenario (n x impossible_share impossible,
the rest good); the table is the shortest prefix of that order that fills both groups (it
does not depend on the round size). One torch file per kind with the atlas fields (loads
as an Atlas) plus lattice_index and offgrid (sizes, seed); its key = the grid's
AtlasConfig key plus the OffgridConfig. offgrid_eval_scenarios: the first scenarios of
each group in table order (a uniform random subset), cycled if more episodes are asked
for. Separation bins: TABLE_SEP_BINS (the extrapolation sets have their own).

Grid hold-out (holdout_mask, SamplerConfig.holdout_frac / holdout_seed): a seeded random
share of the grid scenarios, stratified (that share of the good and of the impossible grid
points, each a uniform subset; the named setups are never held out), removed from the
sampler's draw segments, so no training draw can return one; eval_scenarios(...,
good_ids, impossible_ids) draws the hold-out eval set from them only.

Continuous scenarios (ContRanges, D7): scenario = (separation, laser pitch, laser yaw,
laser dx, laser dy). Separation uniform over its range (default 20-400 mm), the laser
offset uniform over the M1 hole disk (|offset| < 1.5 mm around the hole centre; radius 0:
the measured offsets 0.75, 1.04 mm); laser angles constructive by default (the M2 landing
point at zero M2 tilt uniform over the aperture minus 0.5 mm, pitch / yaw derived from the
laser origin and that point in the sim's geometry, rejected outside the bounds, -8..8 deg),
or uniform in angle (mode "angle"). Candidates come from seeded streams
(ContRanges.candidates: blocks of CAND_BLOCK candidates per (stream, seed, block), so a
candidate does not depend on how the stream is read; separate streams for the pool, the
eval table and the encoder data). One generator for all of them: candidate_batch (angle
bounds, then the beam check Scanner.check, a 1-bounce sim at the M2 box centre: the beam
must pass the M1 hole and hit M2; clipped = stopped by the back of M1 at the hole edge,
missed = no hit) and sample_valid_scenarios (the first n valid ones; encoder data, viewer).
Invalid candidates are discarded, they are not "impossible" (= valid but P_max < P_floor).
Only valid candidates get the atlas box scan (Scanner.scan; zero_cells = box cells without
any hit: where the box scan disagrees with the centre check).

ScenarioPool (PoolConfig; a ScenarioSampler over its slots): size slots (16,384),
round(size x impossible_share) of them impossible, the rest good. Filled from the stream
(STREAM_POOL) in rounds sized from the running good / impossible shares: beam check, box
scan of the valid ones; the next good scenarios fill the good slots and the next
impossible ones the impossible slots in stream order (the good share varies strongly with
separation, so this is rejection sampling: the good slots follow the valid distribution
restricted to the good set). refresh(iteration) at a new iteration replaces the oldest
refresh_frac (1/8) of the good and of the impossible slots (ring order); their PLR scores
reset and their generation (slot_gen) increments: the env copies parameters, P_ref and
generation at the draw, and an episode updates its slot's score only if the generation
still matches. Draws and the PLR replay as ScenarioSampler, within the pool. stats() adds
pool/... (fill time, candidates, clipped / missed shares, good share of the valid ones per
separation bin, slot histogram, target-region bin shares of the good slots). Host syncs
only in refresh() and stats().
Region quotas (PoolConfig.region_mix, one share per target-region bin of REGION_EDGES;
empty = the natural distribution): the good slots are filled by rejection with per-bin
quotas (region_quotas: largest remainder of share x good slots), the next good scenarios
of each bin in stream order until its quota is met (good ones of a full bin inside the
used prefix are discarded), placed in stream order; a refresh replaces the same oldest
slots and refills each with a scenario of the bin it held, so the realized shares stay
exactly the initial quotas. The scan cost per fill grows with the rarest bin's quota /
natural share (tiny: 20 % / 6.3 % = 3.2x the natural cost).
ACCEL hard levels (LevelConfig, size K > 0; Parker-Holder et al. 2022): K level slots after
the pool slots, [size, size + K), each a level = scenario (atlas row) + start tilt + camera
look (the episode's camera parameters except the per-pixel textures, which are redrawn).
Candidates: finished episodes of good slots (fresh, replay or level draws) with shortfall
>= thresh, recorded by observe() (slot, start tilt, look; at most `candidates` per
iteration, the rest dropped). Once per iteration, before the slot refill, the existing
levels (score: the EMA of their replays, as the pool scores), the candidates (score:
their shortfall) and `edits` scenario edits (parents drawn from the level distribution,
parameters + Gaussian noise of edit_sigma, kept if within the ranges, beam-valid and good
after a box scan; score: the parent's, start tilt and look: the parent's) are merged:
entries with score >= thresh, at most family_cap per root pool scenario (the family, kept
through candidates of levels and edits), the top K by score. Kept levels stay in their
slot; new ones fill the freed slots, generation incremented. Draws: a share of the good
draws, share x the filled fraction of K, comes from the levels (SRC_LEVEL; PLR rank /
staleness distribution over the filled levels, beta and rho of the sampler); the env
starts such an episode at the level's tilt and look, and with probability jitter_p adds a
Gaussian start-tilt edit (sigma jitter per axis, source SRC_EDIT: it updates no level
score but can become a candidate). The level slots are excluded from the pool's own
draw segments and statistics; stats() adds accel/... and the level / edit episode shares.

Continuous eval table (ContConfig, build_cont_table; ppo.py eval_cont/): the valid
candidates of the shortest prefix of STREAM_EVAL (seeded) that holds n x 0.9 good and n x
0.1 impossible scenarios; loads as an Atlas (params (n, 5)); offgrid_eval_scenarios picks
the episodes; separation bins cont_sep_bins (20-50, 50-100, ..., 350-400).

Target-region size (region_area): area of the box cells with path >= 0.9 P_max (near_cells
x 0.01^2 deg^2; where an episode counts as success), from the same scan as P_max (every
table, the pool slots; AlignEnv.box_stats for the named setups); best_area: the cells at
the highest bounce count (in practice a subset of the target region). Bins REGION_EDGES
(tiny < 0.02, small 0.02-0.1, medium 0.1-0.5, large >= 0.5 deg^2; region_bin_masks).
Region eval table (RegionConfig, build_region_table; ppo.py eval_region/): n good valid
scenarios of the continuous distribution, n / 4 per bin, each the first good ones of its
bin in the seeded stream STREAM_REGION (sample_valid_scenarios, box-scanned); cached in
EVAL_TABLES_DIR (<ranges tag>_region<edges>_n<n>_seed<seed>_v1.pt) with the region sizes
of every good scenario of the used prefix (region_sample: the training distribution's
region sizes, region_stats in the build info).

ScenarioSampler (all on the device; draw() and observe() have no host syncs):
    draws      one categorical over [impossible | fresh good | replay good] segments
               (without the held-out scenarios; a pool with levels adds a level segment):
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
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from env import M2_ID, SCEN_STATE, Scenarios
from find_setups import SearchConfig, axis, box_grid, coarse_candidates
from geometry import SETUPS, CellGeometry
from herriott_sim import DEVICE, S_M2_PITCH, S_M2_YAW, STOP_BLOCKED, STOP_HOLE

RL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ATLAS = os.path.join(RL_DIR, "runs", "atlas", "atlas_v1.pt")
EVAL_TABLES_DIR = os.path.join(RL_DIR, "runs", "eval_sets")
DEFAULT_OFFGRID = os.path.join(EVAL_TABLES_DIR, "offgrid_v1.pt")
ATLAS_VERSION = 1
PARAM_NAMES = ("separation", "laser_pitch", "laser_yaw")
SRC_EXPLICIT, SRC_IMPOSSIBLE, SRC_FRESH, SRC_REPLAY = -1, 0, 1, 2
SRC_LEVEL, SRC_EDIT = 3, 4  # ACCEL level draws: as stored / with a start-tilt edit
N_SRC = 5
SEP_BINS = ((50.0, 80.0), (80.0, 110.0), (110.0, 140.0), (140.0, 170.0), (170.0, 200.0))
# Off-grid eval tables (module docstring): interpolation and the two extrapolation sets,
# their separation axes (None: midpoints of the grid axis) and separation bins
TABLE_KINDS = ("offgrid", "extrap_short", "extrap_long")
TABLE_SEP = {"offgrid": None, "extrap_short": (20.0, 49.0, 1.0),
             "extrap_long": (201.0, 350.0, 1.0)}
TABLE_SEP_BINS = {"offgrid": SEP_BINS,
                  "extrap_short": ((20.0, 30.0), (30.0, 40.0), (40.0, 50.0)),
                  "extrap_long": ((200.0, 250.0), (250.0, 300.0), (300.0, 350.0))}


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


def box_states(geom: CellGeometry, params: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """(n * G, 13) states: scenarios params (n, 3) [sep, laser pitch, laser yaw] or (n, 5)
    [..., laser dx, laser dy] x M2 tilts grid (G, 2) (find_setups.candidate_states for
    three parameters)."""
    n, G = params.shape[0], grid.shape[0]
    s = geom.base_state(n * G, grid.device).view(n, G, -1)
    p = params.to(torch.float32)
    for j, k in enumerate(SCEN_STATE[:p.shape[1]]):
        s[..., k] = p[:, j, None]
    s[..., S_M2_PITCH] = grid[:, 0]
    s[..., S_M2_YAW] = grid[:, 1]
    return s.view(n * G, -1)


@torch.no_grad()
def scan_scenarios(sim, geom: CellGeometry, params: torch.Tensor, grid: torch.Tensor,
                   cfg: AtlasConfig, zero_cells: bool = False) -> Dict[str, torch.Tensor]:
    """Atlas statistics of scenarios params (n, 3 | 5) float32 over the M2 grid (G, 2);
    zero_cells: also the number of cells without any hit (beam check vs box scan)."""
    n, G = params.shape[0], grid.shape[0]
    dev = grid.device
    cap, mo = cfg.cap, cfg.min_order
    out = sim.simulate(box_states(geom, params, grid), store_world=False, store_hits=False)
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
    first = sim.simulate(box_states(geom, params, torch.zeros(1, 2, device=dev)),
                         store_world=False)
    on_m2 = first["hit_mirror"][:, 0] == M2_ID
    extra = {"zero_cells": (b == 0).sum(1).to(torch.int32)} if zero_cells else {}
    return {**extra,
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


class Scanner:
    """Box scans and beam checks of scenario batches on one device (persistent sim and
    M2 grid of the AtlasConfig scan settings)."""

    def __init__(self, cfg: Optional[AtlasConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg = cfg or AtlasConfig()
        self.device = dev = torch.device(device) if device is not None else DEVICE
        self.geom = CellGeometry()
        self.sim = self.geom.make_sim(dev, max_bounces=cfg.cap)
        self.grid, _ = box_grid(cfg.limit, cfg.step, dev)
        self.per = max(1, cfg.chunk_states // self.grid.shape[0])
        self.sim1 = self.geom.make_sim(dev, max_bounces=1)  # Beam check: the first hit only

    @torch.no_grad()
    def scan(self, params: torch.Tensor, zero_cells: bool = False) -> Dict[str, torch.Tensor]:
        """Atlas fields (on the device) of scenarios params (n, 3 | 5): scan_scenarios in
        chunks of cfg.chunk_states states, then params (float32), p_floor, good and
        path_ref."""
        cfg = self.cfg
        pt = params.to(device=self.device, dtype=torch.float32)
        parts: Dict[str, list] = {}
        for i in range(0, pt.shape[0], self.per):
            for k, v in scan_scenarios(self.sim, self.geom, pt[i:i + self.per], self.grid, cfg,
                                       zero_cells).items():
                parts.setdefault(k, []).append(v)
        d = {k: torch.cat(v) for k, v in parts.items()}
        pf = cfg.floor_bounces * pt[:, 0]
        d.update(params=pt, p_floor=pf, good=d["pmax"] >= pf, path_ref=torch.maximum(d["pmax"], pf))
        return d

    @torch.no_grad()
    def check(self, params: torch.Tensor, chunk: int = 1 << 16) -> Dict[str, torch.Tensor]:
        """Beam check of scenarios params (n, 3 | 5) at the M2 box centre (zero tilt), on
        the device, with a 1-bounce sim (the first hit of the full sim): valid = the
        first hit is on M2; clipped = stopped by the back of M1 (the hole edge) before any
        hit (STOP_BLOCKED, 0 bounces); missed = the rest (no hit on M2). The first M2 hit
        hardly moves over the box (scan zero_cells checks it)."""
        pt = params.to(device=self.device, dtype=torch.float32)
        z = torch.zeros(1, 2, device=self.device)
        on, clip = [], []
        for i in range(0, pt.shape[0], chunk):
            out = self.sim1.simulate(box_states(self.geom, pt[i:i + chunk], z), store_world=False)
            b0 = out["hit_counts"] == 0
            on.append(~b0 & (out["hit_mirror"][:, 0] == M2_ID))
            clip.append(b0 & (out["stop_reason"] == STOP_BLOCKED))
        valid = torch.cat(on) if on else torch.zeros(0, dtype=torch.bool, device=self.device)
        clipped = torch.cat(clip) if clip else torch.zeros_like(valid)
        return {"valid": valid, "clipped": clipped, "missed": ~valid & ~clipped}


@torch.no_grad()
def scan_table(params: np.ndarray, cfg: AtlasConfig,
               device: torch.device) -> Dict[str, torch.Tensor]:
    """Atlas fields (CPU tensors) of the scenarios params (n, 3 | 5) float64
    (Scanner.scan)."""
    pt = torch.tensor(params, dtype=torch.float32, device=device)
    return {k: v.cpu() for k, v in Scanner(cfg, device).scan(pt).items()}


def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _device_name(dev: torch.device) -> str:
    return torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)


@torch.no_grad()
def build_atlas(cfg: Optional[AtlasConfig] = None, device: Optional[torch.device] = None,
                log=print) -> Dict[str, Any]:
    """Scan every scenario (module docstring); returns the atlas dict (CPU tensors)."""
    cfg = cfg or AtlasConfig()
    dev = torch.device(device) if device is not None else DEVICE
    params, named, n_grid = scenario_params(cfg)
    W = axis(-cfg.limit, cfg.limit, cfg.step).size
    G = W * W
    per = max(1, cfg.chunk_states // G)
    N = params.shape[0]
    log(f"atlas: {N} scenarios ({n_grid} grid + {N - n_grid} named) x {W}^2 M2 cells "
        f"({N * G / 1e9:.2f} G rays), {per} scenarios per sim call")
    _sync(dev)
    t0 = time.perf_counter()
    d: Dict[str, Any] = scan_table(params, cfg, dev)
    _sync(dev)
    build_s = time.perf_counter() - t0
    d.update(version=ATLAS_VERSION, key=cfg.key(), named=named, n_grid=n_grid,
             build_s=build_s, device_name=_device_name(dev), date=time.strftime("%Y-%m-%d"))
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
    good_ids / impossible_ids (int64), named (name -> index), key, path; an off-grid
    table also loads as an Atlas (offgrid: its build info, None for the atlas)."""

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
        self.offgrid: Optional[Dict[str, Any]] = d.get("offgrid")  # Off-grid tables only
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


def sep_bin_masks(sep: torch.Tensor, bins=SEP_BINS) -> Dict[str, torch.Tensor]:
    """Separation bins ([lo, hi), the last one closed): name -> mask."""
    out = {}
    for lo, hi in bins:
        top = sep <= hi if hi == bins[-1][1] else sep < hi
        out[f"sep{lo:g}-{hi:g}"] = (sep >= lo) & top
    return out


def eval_scenarios(atlas: Atlas, n: int, seed: int = 0, impossible_share: float = 0.1,
                   good_ids: Optional[torch.Tensor] = None,
                   impossible_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(n,) int64 scenario per episode of the "full" eval set (CPU, seeded): round(n x
    impossible_share) uniform over the impossible set, the rest uniform over the good
    set (with replacement), in a seeded random order. good_ids / impossible_ids: draw
    from these instead of the atlas's sets (the hold-out set)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 7)
    good = (atlas.good_ids if good_ids is None else good_ids).cpu()
    imp = (atlas.impossible_ids if impossible_ids is None else impossible_ids).cpu()
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


def atlas_config(atlas: Atlas) -> AtlasConfig:
    """AtlasConfig of a loaded atlas (from its key; default chunking)."""
    k = atlas.key
    names = [f.name for f in fields(AtlasConfig) if f.name != "chunk_states" and f.name in k]
    return AtlasConfig(**{n: tuple(k[n]) if isinstance(k[n], (list, tuple)) else k[n]
                          for n in names})


def table_stats(at: Atlas, rows: Optional[torch.Tensor] = None, bins=SEP_BINS) -> Dict[str, Any]:
    """Summary of the scenarios rows (default: the grid points of an atlas, every row of
    an off-grid table): count, good share, P_max of the good ones (m: mean, quantiles),
    share of good ones whose P_max is at the bounce cap, mean share of the box cells at
    the cap (all and good scenarios), median success-region area (cells >= 0.9 P_max,
    deg^2) and region_stats of the good ones (target-region bins), mean blind share of
    the box; per separation bin (bins) count, good share, median good P_max and cap
    shares."""
    if rows is None:
        rows = torch.arange(at.n_grid, device=at.device)
    rows = rows.to(at.device)
    k = at.key
    cap = int(k["cap"])
    cell = float(k["step"]) ** 2
    G = axis(-k["limit"], k["limit"], k["step"]).size ** 2
    good = at.good[rows]
    cap_cells = torch.where(at.best_bounces >= cap, at.best_cells, 0).double() / G
    nan = math.nan
    mean = lambda x: float(x.double().mean()) if x.numel() else nan

    def part(r: torch.Tensor, g: torch.Tensor) -> Dict[str, Any]:
        rg = r[g]
        pm = at.pmax[rg].double() / 1000
        qs = (0.1, 0.25, 0.5, 0.75, 0.9)
        q = torch.quantile(pm, torch.tensor(qs, dtype=torch.float64, device=pm.device)).tolist() \
            if rg.numel() else [nan] * len(qs)
        out = {"n": int(r.numel()), "good": int(rg.numel()), "good_share": mean(g),
               "pmax_m_mean": mean(pm)}
        out.update({f"pmax_m_q{int(100 * a)}": v for a, v in zip(qs, q)})
        out.update(
            pmax_cap_share=mean(at.pmax_bounces[rg] >= cap),
            cap_cells_share=mean(cap_cells[r]),
            cap_cells_share_good=mean(cap_cells[rg]),
            near_area_deg2_median=float(at.near_cells[rg].double().median()) * cell
            if rg.numel() else nan,
            blind_share_mean=mean(at.blind_cells[rg].double() / G),
            region=region_stats(at.near_cells[rg].double() * cell))
        return out

    out = part(rows, good)
    out["sep_bins"] = {}
    for name, m in sep_bin_masks(at.params[rows, 0], bins).items():
        b = part(rows[m], good[m])
        out["sep_bins"][name] = {x: b[x] for x in ("n", "good", "good_share", "pmax_m_q50",
                                                   "pmax_cap_share", "cap_cells_share")}
    return out


# ── Off-grid tables: interpolation and extrapolation ───────────────────────


OFFGRID_VERSION = 1


@dataclass
class OffgridConfig:
    """Off-grid eval table (module docstring). kind: offgrid (midpoints in all three
    parameters), extrap_short / extrap_long (separation TABLE_SEP outside the grid range,
    laser pitch / yaw midpoints)."""

    kind: str = "offgrid"
    n: int = 4096  # Eval episodes the table is sized for, one per scenario
    impossible_share: float = 0.1  # Of those episodes (as the "full" set)
    seed: int = 0  # Order in which the lattice is visited
    round_n: int = 2048  # Candidates per scan round (the table does not depend on it)

    def counts(self) -> Tuple[int, int]:
        """(good, impossible) scenarios the table must hold."""
        n_imp = int(round(self.n * self.impossible_share))
        return self.n - n_imp, n_imp

    def key(self, grid: AtlasConfig) -> Dict[str, Any]:
        """Key of the table: the grid's atlas key plus the off-grid settings."""
        assert self.kind in TABLE_KINDS, f"kind {self.kind!r} not in {TABLE_KINDS}"
        d = grid.key()
        d["offgrid"] = {"kind": self.kind, "version": OFFGRID_VERSION, "n": self.n,
                        "impossible_share": self.impossible_share, "seed": self.seed,
                        "sep": TABLE_SEP[self.kind]}
        return d


def table_path(kind: str, directory: str = "") -> str:
    """Cached table file of a kind: <directory or EVAL_TABLES_DIR>/<kind>_v1.pt."""
    return os.path.join(directory or EVAL_TABLES_DIR, f"{kind}_v{OFFGRID_VERSION}.pt")


def grid_axes(cfg: AtlasConfig) -> List[np.ndarray]:
    """Training grid axes (separation, laser pitch, laser yaw), float64."""
    return [axis(*a) for a in (cfg.sep, cfg.pitch, cfg.yaw)]


def offgrid_axes(cfg: AtlasConfig, kind: str = "offgrid") -> List[np.ndarray]:
    """Lattice axes of a table kind: midpoints between neighbouring grid values of each
    axis (exact in float32 for the default grid: odd mm, k x 0.25 + 0.125 deg); the
    extrapolation kinds replace the separation axis by TABLE_SEP[kind]."""
    ax = [0.5 * (a[1:] + a[:-1]) for a in grid_axes(cfg)]
    if TABLE_SEP[kind] is not None:
        ax[0] = axis(*TABLE_SEP[kind])
    return ax


def offgrid_lattice(cfg: AtlasConfig, kind: str = "offgrid") -> np.ndarray:
    """(n, 3) float64 lattice points, separation-major (offgrid: 75 x 32 x 32 = 76,800;
    extrap_short 30 x 32 x 32; extrap_long 150 x 32 x 32 by default)."""
    S, P, Y = np.meshgrid(*offgrid_axes(cfg, kind), indexing="ij")
    return np.stack((S.ravel(), P.ravel(), Y.ravel()), 1)


def on_grid_axes(params: np.ndarray, cfg: AtlasConfig, tol: float = 1e-6) -> np.ndarray:
    """(n, 3) bool: coordinate j of each row of params (n, 3) lies on grid axis j."""
    params = np.asarray(params, dtype=np.float64)
    return np.stack([np.abs(params[:, j, None] - a[None]).min(1) <= tol
                     for j, a in enumerate(grid_axes(cfg))], 1)


@torch.no_grad()
def build_offgrid(ocfg: Optional[OffgridConfig] = None, cfg: Optional[AtlasConfig] = None,
                  device: Optional[torch.device] = None, log=print) -> Dict[str, Any]:
    """Off-grid table of the grid cfg (module docstring); returns the dict (CPU tensors,
    the atlas fields plus lattice_index and offgrid). If the whole lattice holds fewer
    good or impossible scenarios than asked for, the table is the whole lattice
    (offgrid["short"] True, logged) and the eval set cycles over the group."""
    ocfg = ocfg or OffgridConfig()
    cfg = cfg or AtlasConfig()
    dev = torch.device(device) if device is not None else DEVICE
    lat = offgrid_lattice(cfg, ocfg.kind)
    order = np.random.default_rng(ocfg.seed).permutation(lat.shape[0])
    n_good, n_imp = ocfg.counts()
    tag = ocfg.kind
    log(f"{tag}: lattice of {lat.shape[0]} points (separation {offgrid_axes(cfg, tag)[0][0]:g}-"
        f"{offgrid_axes(cfg, tag)[0][-1]:g} mm), scanning in rounds of {ocfg.round_n} until "
        f"{n_good} good + {n_imp} impossible scenarios")
    _sync(dev)
    t0 = time.perf_counter()
    parts: Dict[str, list] = {}
    cnt_g = cnt_i = done = 0
    while (cnt_g < n_good or cnt_i < n_imp) and done < lat.shape[0]:
        idx = order[done:done + ocfg.round_n]
        for k, v in scan_table(lat[idx], cfg, dev).items():
            parts.setdefault(k, []).append(v)
        g = int(parts["good"][-1].sum())
        cnt_g, cnt_i, done = cnt_g + g, cnt_i + idx.size - g, done + idx.size
    _sync(dev)
    build_s = time.perf_counter() - t0
    short = cnt_g < n_good or cnt_i < n_imp
    if short:
        log(f"WARNING {tag}: lattice exhausted with {cnt_g} good / {cnt_i} impossible "
            f"scenarios; the eval set repeats them")
    d: Dict[str, Any] = {k: torch.cat(v) for k, v in parts.items()}
    good = d["good"]
    last = lambda ids, k: int(ids[min(k, ids.numel()) - 1]) if min(k, ids.numel()) else -1
    L = max(last(torch.nonzero(good).squeeze(1), n_good),
            last(torch.nonzero(~good).squeeze(1), n_imp)) + 1  # Shortest prefix
    d = {k: v[:L] for k, v in d.items()}
    d.update(version=ATLAS_VERSION, key=ocfg.key(cfg), named={}, n_grid=L, build_s=build_s,
             device_name=_device_name(dev), date=time.strftime("%Y-%m-%d"),
             lattice_index=torch.from_numpy(order[:L].copy()),
             offgrid={"kind": tag, "lattice": int(lat.shape[0]), "candidates": L,
                      "scanned": done, "n_good": n_good, "n_impossible": n_imp,
                      "seed": ocfg.seed, "short": short})
    ng = int(d["good"].sum())
    log(f"{tag}: {L} candidates ({done} scanned) in {build_s:.1f} s on {d['device_name']}: "
        f"{ng} good ({100 * ng / max(L, 1):.1f} %), {L - ng} impossible")
    return d


def load_or_build_offgrid(path: str, cfg: AtlasConfig, ocfg: Optional[OffgridConfig] = None,
                          device: Optional[torch.device] = None, log=print) -> Atlas:
    """Off-grid table at path for the grid cfg (build_offgrid; built and saved there if the
    file is missing). A file built for another grid or OffgridConfig is an error."""
    ocfg = ocfg or OffgridConfig()
    path = os.path.abspath(path)
    if not os.path.exists(path):
        log(f"{ocfg.kind}: {path} missing, building")
        save_atlas(build_offgrid(ocfg, cfg, device, log), path)
        log(f"{ocfg.kind}: saved {path}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    want, have = ocfg.key(cfg), d.get("key") or {}
    bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    assert not bad, f"table {path} does not match the grid / OffgridConfig: {bad}"
    return Atlas(d, device, path)


def offgrid_eval_scenarios(table: Atlas, n: int, seed: int = 0,
                           impossible_share: float = 0.1) -> torch.Tensor:
    """(n,) int64 scenario per episode of an off-grid eval set (CPU, seeded): round(n x
    impossible_share) impossible, the rest good, each the first scenarios of its group in
    table order (a uniform random subset; one episode per scenario, cycled when the group
    is smaller), in a seeded random order."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 7)
    good, imp = table.good_ids.cpu(), table.impossible_ids.cpu()
    n_imp = int(round(n * impossible_share)) if imp.numel() else 0
    if good.numel() == 0:
        n_imp = n
    parts = []
    if n - n_imp:
        parts.append(good[torch.arange(n - n_imp) % good.numel()])
    if n_imp:
        parts.append(imp[torch.arange(n_imp) % imp.numel()])
    s = torch.cat(parts)
    return s[torch.randperm(n, generator=gen)]


# ── Continuous scenarios ────────────────────────────────────────────────────


CAND_BLOCK = 4096  # Candidates per seeded block of a candidate stream
STREAM_POOL, STREAM_EVAL, STREAM_ENCODER = 1, 2, 3  # Independent streams per use
CONT_SEP_EDGES = (50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0)
CONT_VERSION = 1


def cont_sep_bins(lo: float, hi: float) -> Tuple[Tuple[float, float], ...]:
    """Separation bins of a continuous range: edges lo, the CONT_SEP_EDGES inside (lo,
    hi), hi (20-400 mm: 20-50, 50-100, ..., 350-400)."""
    e = [float(lo)] + [x for x in CONT_SEP_EDGES if lo < x < hi] + [float(hi)]
    return tuple(zip(e[:-1], e[1:]))


LASER_Z_MM = -5.0  # Laser origin behind the M1 surface, M1-local z (herriott_sim, sim_triton)
OFFSET_SHRINK = 1.0 - 1e-6  # Disk radii x this: float32 parameters stay strictly inside


@dataclass(frozen=True)
class ContRanges:
    """Continuous scenario distribution (module docstring). Scenario = (separation, laser
    pitch, laser yaw, laser dx, laser dy) (env.SCEN_NAMES). Separation uniform over sep,
    laser (dx, dy) uniform over the disk of offset_radius around the M1 hole centre (0:
    the measured offsets of geometry.CellGeometry, fixed). Laser angles:
        landing  (default) constructive: the M2 landing point of the beam at zero M2 tilt
                 uniform over the M2 aperture minus landing_margin; pitch / yaw follow
                 from laser origin and landing point (the sim's geometry: origin at
                 (dx, hole y + dy, -5 mm) behind the untilted M1, landing point on the M2
                 sphere); candidates whose angles fall outside pitch / yaw are rejected
                 (angle_ok), so the angle range shrinks with separation by construction
        angle    uniform over pitch x yaw (for comparison)"""

    sep: Tuple[float, float] = (20.0, 400.0)  # mm
    pitch: Tuple[float, float] = (-8.0, 8.0)  # deg, laser (landing: bounds)
    yaw: Tuple[float, float] = (-8.0, 8.0)  # deg, laser (landing: bounds)
    offset_radius: float = 1.5  # mm (the M1 hole radius)
    mode: str = "landing"  # landing / angle
    landing_margin: float = 0.5  # mm inside the M2 aperture edge

    def candidates(self, seed: int, start: int, count: int, stream: int) -> np.ndarray:
        """(count, 5) float64 candidates start .. start + count - 1 of the seeded stream
        (block b of CAND_BLOCK candidates from numpy's Generator seeded by (stream, seed,
        b), so a candidate does not depend on how the stream is read). Landing mode:
        the angles may lie outside the bounds (angle_ok)."""
        assert seed >= 0 and start >= 0 and self.mode in ("landing", "angle"), self.mode
        out = np.empty((count, 5))
        if count <= 0:
            return out
        b0, b1 = start // CAND_BLOCK, (start + count - 1) // CAND_BLOCK
        u = np.concatenate([np.random.default_rng([stream, seed, b]).random((CAND_BLOCK, 5))
                            for b in range(b0, b1 + 1)])
        u = u[start - b0 * CAND_BLOCK:start - b0 * CAND_BLOCK + count]
        g = CellGeometry()
        out[:, 0] = self.sep[0] + (self.sep[1] - self.sep[0]) * u[:, 0]
        if self.offset_radius > 0:  # Uniform over the disk, |offset| < radius
            r = self.offset_radius * OFFSET_SHRINK * np.sqrt(u[:, 3])
            t = 2.0 * np.pi * u[:, 4]
            out[:, 3], out[:, 4] = r * np.cos(t), r * np.sin(t)
        else:
            out[:, 3], out[:, 4] = g.laser_dx, g.laser_dy
        if self.mode == "angle":
            out[:, 1] = self.pitch[0] + (self.pitch[1] - self.pitch[0]) * u[:, 1]
            out[:, 2] = self.yaw[0] + (self.yaw[1] - self.yaw[0]) * u[:, 2]
            return out
        # Landing point (M2-local lx, ly) uniform over the aperture minus the margin
        rl = (g.diameter / 2 - self.landing_margin) * OFFSET_SHRINK * np.sqrt(u[:, 1])
        tl = 2.0 * np.pi * u[:, 2]
        lx, ly = rl * np.cos(tl), rl * np.sin(tl)
        pz = out[:, 0] - (g.roc - np.sqrt(g.roc ** 2 - rl ** 2))  # On the M2 sphere
        d = np.stack((g.m2_tx + lx - out[:, 3], g.m2_ty + ly - (g.m1_hole_offset_y + out[:, 4]),
                      pz - LASER_Z_MM), 1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # herriott_sim: direction = Ry(yaw) Rx(pitch) (0, 0, 1) = (cos p sin y, -sin p, cos p cos y)
        out[:, 1] = np.degrees(np.arcsin(np.clip(-d[:, 1], -1.0, 1.0)))
        out[:, 2] = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
        return out

    def angle_ok(self, params: np.ndarray) -> np.ndarray:
        """(n,) bool: laser pitch and yaw of params (n, 5) within the bounds."""
        return ((params[:, 1] >= self.pitch[0]) & (params[:, 1] <= self.pitch[1])
                & (params[:, 2] >= self.yaw[0]) & (params[:, 2] <= self.yaw[1]))

    def head_range(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Scenario-head normalization (lo, hi): separation, laser pitch, yaw and, with
        offset_radius > 0, laser dx, dy (+- offset_radius); 3 entries otherwise."""
        lo = (self.sep[0], self.pitch[0], self.yaw[0])
        hi = (self.sep[1], self.pitch[1], self.yaw[1])
        r = self.offset_radius
        if r > 0:
            lo, hi = lo + (-r, -r), hi + (r, r)
        return tuple(float(v) for v in lo), tuple(float(v) for v in hi)

    def tag(self) -> str:
        """Name marker: cont<sep lo>-<sep hi>, plus -lp<a> (laser bounds +-a) or
        -l<lo>to<hi>x<lo>to<hi> if not the default +-8 deg, -off<r> if the offset radius
        is not 1.5 mm, -ang in angle mode."""
        t = f"cont{self.sep[0]:g}-{self.sep[1]:g}"
        if self.pitch != (-8.0, 8.0) or self.yaw != (-8.0, 8.0):
            if self.pitch == self.yaw and self.pitch[0] == -self.pitch[1]:
                t += f"-lp{self.pitch[1]:g}"
            else:
                t += f"-l{self.pitch[0]:g}to{self.pitch[1]:g}x{self.yaw[0]:g}to{self.yaw[1]:g}"
        if self.offset_radius != 1.5:
            t += f"-off{self.offset_radius:g}"
        if self.mode != "landing":
            t += "-ang"
        return t


def cont_ranges(sep: Tuple[float, float], laser: Tuple[float, float], offset_radius: float,
                mode: str = "landing") -> ContRanges:
    """ContRanges from CLI values (the same range for laser pitch and yaw)."""
    return ContRanges(sep=(float(sep[0]), float(sep[1])), pitch=(float(laser[0]), float(laser[1])),
                      yaw=(float(laser[0]), float(laser[1])), offset_radius=float(offset_radius),
                      mode=mode)


def scenario_geometry(p, name: str = "") -> CellGeometry:
    """CellGeometry of a scenario (separation, laser pitch, laser yaw[, laser dx, laser
    dy]): the measured cell with these values (encoder data, FrameData setups)."""
    kw = dict(separation=float(p[0]), laser_pitch=float(p[1]), laser_yaw=float(p[2]))
    if len(p) > 3:
        kw.update(laser_dx=float(p[3]), laser_dy=float(p[4]))
    return replace(CellGeometry(), name=name, **kw)


def sep_band(sep: np.ndarray, bins) -> np.ndarray:
    """Index of the separation bin ([lo, hi), the last closed) of each value, -1 outside."""
    out = np.full(sep.shape, -1, dtype=np.int64)
    for j, (lo, hi) in enumerate(bins):
        top = sep <= hi if j == len(bins) - 1 else sep < hi
        out[(sep >= lo) & top] = j
    return out


STAGES = ("candidates", "angle_rejected", "clipped", "missed", "valid")  # Generator stages


@torch.no_grad()
def candidate_batch(ranges: ContRanges, seed: int, start: int, count: int, stream: int,
                    scanner: "Scanner") -> Dict[str, np.ndarray]:
    """Candidates start .. start + count - 1 of a seeded stream with their validity
    stages (host arrays): params (count, 5) float64; angle_ok (laser angles within the
    bounds; landing mode); clipped / missed (Scanner.check at the box centre, the
    angle_ok ones only); valid = angle_ok and neither. No box scan."""
    prm = ranges.candidates(seed, start, count, stream)
    ok = ranges.angle_ok(prm)
    clipped = np.zeros(count, dtype=bool)
    missed = np.zeros(count, dtype=bool)
    idx = np.nonzero(ok)[0]
    if idx.size:
        chk = scanner.check(torch.from_numpy(prm[idx]))
        clipped[idx] = chk["clipped"].cpu().numpy()
        missed[idx] = chk["missed"].cpu().numpy()
    return {"params": prm, "angle_ok": ok, "clipped": clipped, "missed": missed,
            "valid": ok & ~clipped & ~missed}


def stage_counts(b: Dict[str, np.ndarray], end: int) -> Dict[str, int]:
    """STAGES counts of the first end candidates of a candidate_batch."""
    return {"candidates": int(end), "angle_rejected": int((~b["angle_ok"][:end]).sum()),
            "clipped": int(b["clipped"][:end].sum()), "missed": int(b["missed"][:end].sum()),
            "valid": int(b["valid"][:end].sum())}


@torch.no_grad()
def sample_valid_scenarios(n: int, ranges: ContRanges, seed: int = 0, stream: int = STREAM_ENCODER,
                           device: Optional[torch.device] = None, scanner: Optional["Scanner"] = None,
                           start: int = 0, round_n: int = 8192) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The scenario generator (pool, eval table and encoder data use it through
    candidate_batch): the first n valid scenarios (angles within the bounds, beam through
    the M1 hole onto M2 at the box centre) of a seeded candidate stream from position
    start, (n, 5) float64, and info: STAGES counts up to the last one, positions (n,)
    stream positions, next (the position after the last one). No box scans."""
    sc = scanner or Scanner(device=device)
    keep, pos = [], []
    cnt = dict.fromkeys(STAGES, 0)
    got, at = 0, start
    while got < n:
        b = candidate_batch(ranges, seed, at, round_n, stream, sc)
        idx = np.nonzero(b["valid"])[0][:n - got]
        end = int(idx[-1]) + 1 if got + idx.size >= n else round_n
        for k, v in stage_counts(b, end).items():
            cnt[k] += v
        keep.append(b["params"][idx])
        pos.append(idx + at)
        got += idx.size
        at += round_n
        assert at - start < 1000 * max(n, 1) + 10 * round_n, "almost no valid scenarios in the ranges"
    nxt = int(pos[-1][-1]) + 1 if n else start
    return np.concatenate(keep), {**cnt, "positions": np.concatenate(pos), "next": nxt}


# ── Continuous eval table (eval_cont) ───────────────────────────────────────


@dataclass
class ContConfig:
    """Continuous eval table: the valid candidates (Scanner.check) of the shortest prefix
    of a seeded candidate stream (STREAM_EVAL) that holds n x impossible_share impossible
    and the rest good scenarios (as OffgridConfig, one episode per scenario)."""

    ranges: ContRanges = field(default_factory=ContRanges)
    n: int = 4096
    impossible_share: float = 0.1
    seed: int = 0
    round_n: int = 8192  # Max candidates per round, 2 x n for small tables (no effect on the table)

    def counts(self) -> Tuple[int, int]:
        n_imp = int(round(self.n * self.impossible_share))
        return self.n - n_imp, n_imp

    def key(self, scan: AtlasConfig) -> Dict[str, Any]:
        """Scan settings (AtlasConfig.key without the grid) plus the table settings."""
        d = {k: v for k, v in scan.key().items() if k not in ("sep", "pitch", "yaw", "named")}
        d["cont"] = {"version": CONT_VERSION, "ranges": asdict(self.ranges), "n": self.n,
                     "impossible_share": self.impossible_share, "seed": self.seed}
        return d


def cont_table_path(ccfg: ContConfig, directory: str = "") -> str:
    """<directory or EVAL_TABLES_DIR>/<ranges tag>_n<n>_seed<seed>_v1.pt."""
    return os.path.join(directory or EVAL_TABLES_DIR,
                        f"{ccfg.ranges.tag()}_n{ccfg.n}_seed{ccfg.seed}_v{CONT_VERSION}.pt")


@torch.no_grad()
def build_cont_table(ccfg: Optional[ContConfig] = None, scan: Optional[AtlasConfig] = None,
                     device: Optional[torch.device] = None, log=print) -> Dict[str, Any]:
    """Continuous eval table (ContConfig); the atlas fields (CPU tensors, params (n, 5))
    of the valid candidates (candidate_batch, box scan of the valid ones only),
    candidate_index (stream position) and the build info under "offgrid" (kind "cont":
    candidates of the prefix, generated, STAGES counts, valid candidates with box cells
    without any hit)."""
    ccfg = ccfg or ContConfig()
    scan = scan or AtlasConfig()
    dev = torch.device(device) if device is not None else DEVICE
    sc = Scanner(scan, dev)
    n_good, n_imp = ccfg.counts()
    tag = ccfg.ranges.tag()
    rn = min(ccfg.round_n, max(256, 2 * ccfg.n))  # Small tables: small rounds
    log(f"cont table {tag}: generating the seeded stream (seed {ccfg.seed}) in rounds of "
        f"{rn} until {n_good} good + {n_imp} impossible valid scenarios")
    _sync(dev)
    t0 = time.perf_counter()
    parts: Dict[str, list] = {}
    batches = []
    cnt_g = cnt_i = done = 0
    while cnt_g < n_good or cnt_i < n_imp:
        b = candidate_batch(ccfg.ranges, ccfg.seed, done, rn, STREAM_EVAL, sc)
        vid = np.nonzero(b["valid"])[0]
        d = sc.scan(torch.from_numpy(b["params"][vid]), zero_cells=True)
        d["candidate_index"] = torch.from_numpy(vid + done)
        for k, v in d.items():
            parts.setdefault(k, []).append(v.cpu())
        batches.append(b)
        g = int(d["good"].sum())
        cnt_g, cnt_i, done = cnt_g + g, cnt_i + vid.size - g, done + rn
        assert done < 2000 * max(ccfg.n, 1), "the ranges hold almost no good or impossible scenarios"
    _sync(dev)
    build_s = time.perf_counter() - t0
    d = {k: torch.cat(v) for k, v in parts.items()}
    good, pos = d["good"], d["candidate_index"]
    last = lambda ids, k: int(pos[ids[k - 1]]) if k else -1  # noqa: E731
    L = max(last(torch.nonzero(good).squeeze(1), n_good),
            last(torch.nonzero(~good).squeeze(1), n_imp)) + 1  # Shortest prefix
    keep = pos < L
    d = {k: v[keep] for k, v in d.items()}
    zc = d.pop("zero_cells")
    whole = {k: np.concatenate([b[k] for b in batches]) for k in ("angle_ok", "clipped", "missed", "valid")}
    st = stage_counts(whole, L)
    nv = int(keep.sum())
    assert nv == st["valid"]
    info = {"kind": "cont", "lattice": None, "candidates": L, "scanned": done, **st,
            "box_zero_cells": int((zc > 0).sum()), "n_good": n_good, "n_impossible": n_imp,
            "seed": ccfg.seed, "short": False, "ranges": asdict(ccfg.ranges)}
    d.update(version=ATLAS_VERSION, key=ccfg.key(scan), named={}, n_grid=nv, build_s=build_s,
             device_name=_device_name(dev), date=time.strftime("%Y-%m-%d"), offgrid=info)
    ng = int(d["good"].sum())
    log(f"cont table {tag}: {L} candidates ({done} generated) in {build_s:.1f} s on "
        f"{d['device_name']}: angles out of bounds {100 * st['angle_rejected'] / L:.1f} %, "
        f"clipped {100 * st['clipped'] / L:.1f} %, missed M2 {100 * st['missed'] / L:.1f} %, "
        f"{nv} valid: {ng} good ({100 * ng / max(nv, 1):.1f} %), {nv - ng} impossible; valid "
        f"with box cells without a hit: {info['box_zero_cells']}")
    return d


def load_or_build_cont(path: str, ccfg: Optional[ContConfig] = None,
                       scan: Optional[AtlasConfig] = None, device: Optional[torch.device] = None,
                       log=print) -> Atlas:
    """Continuous eval table at path (build_cont_table; built and saved there if the file
    is missing). A file built with other settings is an error."""
    ccfg = ccfg or ContConfig()
    scan = scan or AtlasConfig()
    path = os.path.abspath(path)
    if not os.path.exists(path):
        log(f"cont table: {path} missing, building")
        save_atlas(build_cont_table(ccfg, scan, device, log), path)
        log(f"cont table: saved {path}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    want, have = ccfg.key(scan), d.get("key") or {}
    bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    assert not bad, f"table {path} does not match the ContConfig / scan settings: {bad}"
    return Atlas(d, device, path)


# ── Target-region size, region eval table (eval_region) ─────────────────────


STREAM_REGION = 4  # Candidate stream of the region eval table
REGION_VERSION = 1
# Target-region bins (deg^2, [lo, hi)): round values. Region sizes of the good scenarios of
# the training distribution (cont20-400, landing mode, offsets over the 1.5 mm disk): the
# 16,241 good candidates of the region table (seed 0) q05 0.013, q10 0.046, q35 0.51, q50
# 1.18, q65 2.29 deg^2 (14,746 good pool slots of run 4410759: 0.045, 0.52, 1.18, 2.33);
# shares of the bins below 6.3 / 8.9 / 19.3 / 65.5 %. Success of run 4410759 (it 1600,
# argmax) on the 3686 good eval_cont episodes per bin: 26 / 61 / 79 / 96 %, each bin
# holding about a quarter of the failures; edges at the percentiles (round: 0.05 / 0.5 /
# 2.0) would put three quarters of them in the two lowest bins and leave the top bin at
# 98 %. The measured cell (0.016) and sep169 (0.011) fall in tiny, sep116 (0.23) and
# sep144 (0.20) in medium.
REGION_EDGES = (0.02, 0.1, 0.5)
REGION_BINS = ("tiny", "small", "medium", "large")
REGION_QS = (0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 0.95, 0.99)


def cell_area(t: Atlas) -> float:
    """Area of one box cell of a table's scan (deg^2)."""
    return float(t.key["step"]) ** 2


def region_area(t: Atlas) -> torch.Tensor:
    """(N,) float64 target-region size per scenario (deg^2): the box cells with path >=
    near_frac x P_max (near_cells: where an episode counts as success) x cell area."""
    return t.near_cells.double() * cell_area(t)


def best_area(t: Atlas) -> torch.Tensor:
    """(N,) float64 area of the box cells at the highest bounce count (deg^2)."""
    return t.best_cells.double() * cell_area(t)


def region_bin(area: torch.Tensor, edges=REGION_EDGES) -> torch.Tensor:
    """(n,) int64 target-region bin of sizes area (deg^2): j for edges[j - 1] <= area <
    edges[j] (0 below the first edge, len(edges) from the last one)."""
    e = torch.tensor(edges, dtype=area.dtype, device=area.device)
    return torch.bucketize(area, e, right=True)


def region_bin_masks(area: torch.Tensor, edges=REGION_EDGES,
                     names=REGION_BINS) -> Dict[str, torch.Tensor]:
    """Target-region bins: region_<name> -> mask (region_bin)."""
    assert len(names) == len(edges) + 1, "one name per bin"
    b = region_bin(area, edges)
    return {f"region_{nm}": b == j for j, nm in enumerate(names)}


def region_bin_labels(edges=REGION_EDGES, names=REGION_BINS) -> Dict[str, str]:
    """region_<name> -> range text in deg^2 (< 0.02, 0.02-0.1, ..., >= 0.5)."""
    e = list(edges)
    out = {}
    for j, nm in enumerate(names):
        out[f"region_{nm}"] = (f"< {e[0]:g}" if j == 0 else f">= {e[-1]:g}" if j == len(e)
                               else f"{e[j - 1]:g}-{e[j]:g}")
    return out


def region_stats(area: torch.Tensor, edges=REGION_EDGES, names=REGION_BINS) -> Dict[str, Any]:
    """Distribution of target-region sizes (deg^2): count, mean, quantiles REGION_QS and
    the share of each bin."""
    a = area.double().cpu()
    out: Dict[str, Any] = {"n": int(a.numel())}
    if not a.numel():
        return out
    q = torch.quantile(a, torch.tensor(REGION_QS, dtype=torch.float64)).tolist()
    out["mean"] = float(a.mean())
    out.update({f"q{round(100 * x):02d}": v for x, v in zip(REGION_QS, q)})
    out["bin_share"] = {k: float(m.double().mean()) for k, m in region_bin_masks(a, edges, names).items()}
    return out


@dataclass
class RegionConfig:
    """Region eval table: n good valid scenarios of the continuous distribution (ranges),
    stratified by target-region size (region_bin of edges): counts()[j] per bin, each the
    first good scenarios of its bin in the seeded stream STREAM_REGION (within a bin the
    training distribution restricted to it)."""

    ranges: ContRanges = field(default_factory=ContRanges)
    n: int = 4096
    seed: int = 0
    edges: Tuple[float, ...] = REGION_EDGES
    round_n: int = 8192  # Valid scenarios per round, 2 x n for small tables (no effect on the table)

    def counts(self) -> List[int]:
        """Scenarios per bin (n / bins, the remainder to the first bins)."""
        k = len(self.edges) + 1
        return [self.n // k + (j < self.n % k) for j in range(k)]

    def key(self, scan: AtlasConfig) -> Dict[str, Any]:
        """Scan settings (AtlasConfig.key without the grid) plus the table settings."""
        d = {k: v for k, v in scan.key().items() if k not in ("sep", "pitch", "yaw", "named")}
        d["region"] = {"version": REGION_VERSION, "ranges": asdict(self.ranges), "n": self.n,
                       "seed": self.seed, "edges": [float(e) for e in self.edges]}
        return d


def region_table_path(rcfg: RegionConfig, directory: str = "") -> str:
    """<directory or EVAL_TABLES_DIR>/<ranges tag>_region<edges>_n<n>_seed<seed>_v1.pt."""
    e = "-".join(f"{x:g}" for x in rcfg.edges)
    return os.path.join(directory or EVAL_TABLES_DIR,
                        f"{rcfg.ranges.tag()}_region{e}_n{rcfg.n}_seed{rcfg.seed}_v{REGION_VERSION}.pt")


@torch.no_grad()
def build_region_table(rcfg: Optional[RegionConfig] = None, scan: Optional[AtlasConfig] = None,
                       device: Optional[torch.device] = None, log=print) -> Dict[str, Any]:
    """Region eval table (RegionConfig): valid scenarios of the seeded stream STREAM_REGION
    in rounds (sample_valid_scenarios), box scan of each (Scanner.scan: P_max and the
    target-region size from the same scan), the good ones assigned to their bin; a bin
    keeps its first counts()[j] in stream order (the table does not depend on the round
    size), generation stops when every bin is full. Returns the atlas fields (CPU tensors,
    params (n, 5)) of the kept scenarios in stream order, candidate_index (stream
    position), region_bin (int8) and region_sample (float32: the target-region sizes of
    every good scenario of the used prefix, the distribution the bins cut), and the build
    info under "offgrid" (kind "region": candidates of the prefix, STAGES counts, good,
    region_stats of the sample, counts per bin)."""
    rcfg = rcfg or RegionConfig()
    scan = scan or AtlasConfig()
    dev = torch.device(device) if device is not None else DEVICE
    sc = Scanner(scan, dev)
    need = rcfg.counts()
    tag = rcfg.ranges.tag()
    cell = float(scan.step) ** 2
    rn = min(rcfg.round_n, max(256, 2 * rcfg.n))  # Small tables: small rounds
    log(f"region table {tag}: valid scenarios of the seeded stream (seed {rcfg.seed}) in rounds "
        f"of {rn}, box scans until {'/'.join(map(str, need))} good ones per target-region bin "
        f"(edges {', '.join(f'{e:g}' for e in rcfg.edges)} deg^2)")
    _sync(dev)
    t0 = time.perf_counter()
    parts: Dict[str, list] = {}
    at = 0
    while any(k > 0 for k in need):
        prm, info = sample_valid_scenarios(rn, rcfg.ranges, rcfg.seed, STREAM_REGION, scanner=sc,
                                           start=at)
        d = {k: v.cpu() for k, v in sc.scan(torch.from_numpy(prm)).items()}
        b = region_bin(d["near_cells"].double() * cell, rcfg.edges)
        keep = torch.zeros(prm.shape[0], dtype=torch.bool)
        for j in range(len(need)):
            rows = torch.nonzero(d["good"] & (b == j)).squeeze(1)[:need[j]]
            keep[rows] = True
            need[j] -= rows.numel()
        d.update(candidate_index=torch.from_numpy(info["positions"]), region_bin=b.to(torch.int8),
                 keep=keep)
        for k, v in d.items():
            parts.setdefault(k, []).append(v)
        at = info["next"]
        assert at < 20000 * max(rcfg.n, 1), "the ranges hold almost no scenarios of a region bin"
    _sync(dev)
    build_s = time.perf_counter() - t0
    d = {k: torch.cat(v) for k, v in parts.items()}
    keep, pos = d.pop("keep"), d["candidate_index"]
    L = int(pos[keep].max()) + 1  # Used prefix: up to the last kept scenario
    pre = pos < L
    sample = d["near_cells"][pre & d["good"]].double() * cell
    st = stage_counts(candidate_batch(rcfg.ranges, rcfg.seed, 0, L, STREAM_REGION, sc), L)
    assert st["valid"] == int(pre.sum()), "valid scenarios of the prefix != scanned"
    rs = region_stats(sample, rcfg.edges)
    info = {"kind": "region", "lattice": None, "candidates": L, "scanned": at, **st,
            "good": int((pre & d["good"]).sum()), "region": rs, "n_per_bin": rcfg.counts(),
            "edges": [float(e) for e in rcfg.edges], "seed": rcfg.seed, "short": False,
            "ranges": asdict(rcfg.ranges)}
    d = {k: v[keep] for k, v in d.items()}
    d.update(version=ATLAS_VERSION, key=rcfg.key(scan), named={}, n_grid=int(keep.sum()),
             build_s=build_s, device_name=_device_name(dev), date=time.strftime("%Y-%m-%d"),
             offgrid=info, region_sample=sample.float())
    log(f"region table {tag}: {L} candidates in {build_s:.1f} s on {d['device_name']}: angles out "
        f"of bounds {100 * st['angle_rejected'] / L:.1f} %, clipped {100 * st['clipped'] / L:.1f} %, "
        f"missed M2 {100 * st['missed'] / L:.1f} %, {st['valid']} valid (box-scanned), "
        f"{info['good']} good; their target-region size (deg^2) q10 / q50 / q90 "
        f"{rs['q10']:.3f} / {rs['q50']:.3f} / {rs['q90']:.3f}, bin shares "
        + " / ".join(f"{100 * v:.1f}" for v in rs["bin_share"].values()) + " %")
    return d


def load_or_build_region(path: str, rcfg: Optional[RegionConfig] = None,
                         scan: Optional[AtlasConfig] = None, device: Optional[torch.device] = None,
                         log=print) -> Atlas:
    """Region eval table at path (build_region_table; built and saved there if the file is
    missing). A file built with other settings is an error."""
    rcfg = rcfg or RegionConfig()
    scan = scan or AtlasConfig()
    path = os.path.abspath(path)
    if not os.path.exists(path):
        log(f"region table: {path} missing, building")
        save_atlas(build_region_table(rcfg, scan, device, log), path)
        log(f"region table: saved {path}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    want, have = rcfg.key(scan), d.get("key") or {}
    bad = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    assert not bad, f"table {path} does not match the RegionConfig / scan settings: {bad}"
    return Atlas(d, device, path)


# ── Grid hold-out ───────────────────────────────────────────────────────────


def holdout_mask(atlas: Atlas, frac: float, seed: int = 0) -> torch.Tensor:
    """(N,) bool on the atlas device: the held-out scenarios, round(frac x count) of the
    good and of the impossible grid points (each a uniform subset, CPU generator seeded
    by seed); the named setups are never held out. frac 0: none."""
    m = torch.zeros(atlas.n, dtype=torch.bool)
    if frac > 0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 11)
        free = torch.arange(atlas.n) < atlas.n_grid
        free[list(atlas.named.values())] = False
        good = atlas.good.cpu()
        for grp in (good, ~good):
            ids = torch.nonzero(grp & free).squeeze(1)
            k = int(round(frac * ids.numel()))
            m[ids[torch.randperm(ids.numel(), generator=gen)[:k]]] = True
    return m.to(atlas.device)


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
    holdout_frac: float = 0.0  # Share of the grid scenarios never drawn (holdout_mask)
    holdout_seed: int = 0


class ScenarioSampler:
    """Regret-prioritized scenario draws over an atlas, see the module docstring."""

    def __init__(self, atlas: Atlas, cfg: Optional[SamplerConfig] = None,
                 exclude: Optional[torch.Tensor] = None):
        """exclude (N,) bool: rows outside the draw segments as well (pool level slots)."""
        self.cfg = cfg or SamplerConfig()
        self.atlas = atlas
        dev = self.device = atlas.device
        N = atlas.n
        # Training scenarios: the atlas without the held-out ones (holdout_mask)
        self.holdout = holdout_mask(atlas, self.cfg.holdout_frac, self.cfg.holdout_seed)
        g, im = atlas.good_ids, atlas.impossible_ids
        drop = self.holdout if exclude is None else self.holdout | exclude
        if bool(drop.any()):
            g, im = g[~drop[g]], im[~drop[im]]
        self.good_ids, self.impossible_ids = g, im
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
        lv = self._level_slots()
        if lv is not None:  # Pool levels: a fourth segment
            self.ids = torch.cat((self.ids, lv))
            self.src = torch.cat((self.src, torch.full_like(lv, SRC_LEVEL).to(torch.int8)))
        self._acc = torch.zeros(4, N_SRC, dtype=torch.float64, device=dev)
        self.refresh(0)

    # ── Distribution ──

    def _rank_probs(self, ids: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
        """float64 PLR distribution over the rows ids where seen (0 elsewhere): (1 - rho)
        rank^(-1 / beta) of the scores + rho staleness."""
        c = self.cfg
        n_seen = seen.sum()
        key = torch.where(seen, self.score[ids], -math.inf)
        order = torch.argsort(key, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(1, order.numel() + 1, device=order.device)
        h = torch.where(seen, rank.double() ** (-1.0 / c.beta), 0.0)
        p_s = h / h.sum().clamp(min=1e-300)
        stale = torch.where(seen, (self.iteration - self.last[ids]).double(), 0.0)
        tot = stale.sum()
        p_c = torch.where(tot > 0, stale / tot.clamp(min=1e-300),
                          seen.double() / n_seen.clamp(min=1).double())
        return (1.0 - c.rho) * p_s + c.rho * p_c

    def replay_probs(self) -> torch.Tensor:
        """(n_good,) float64 replay distribution over the good training scenarios (0 if
        unseen)."""
        gs = self.good_ids
        return self._rank_probs(gs, self.seen[gs])

    def _level_slots(self) -> Optional[torch.Tensor]:
        """Slot indices of the level segment (ScenarioPool with levels); None: none."""
        return None

    def _level_draw(self) -> Optional[Tuple[float, torch.Tensor]]:
        """(share of the good draws, float64 distribution over _level_slots()) at a
        refresh; None without levels."""
        return None

    @torch.no_grad()
    def refresh(self, iteration: int):
        """Recompute the draw distribution (once per iteration; no host sync) and reset
        the per-iteration statistics."""
        c = self.cfg
        self.iteration = int(iteration)
        dev = self.device
        n_seen = self.seen[self.good_ids].sum()
        if c.replay and self.n_good:
            p_fresh = torch.where(n_seen >= c.min_seen, c.p_fresh, 1.0).double()
            p_r = self.replay_probs()
        else:
            p_fresh = torch.ones((), dtype=torch.float64, device=dev)
            p_r = torch.zeros(self.n_good, dtype=torch.float64, device=dev)
        q_imp = c.impossible_share if self.n_imp and self.n_good else float(self.n_good == 0)
        lv = self._level_draw()
        q_lv = 0.0 if lv is None else lv[0]
        q_g = (1.0 - q_imp) * (1.0 - q_lv)  # Fresh + replay share
        w = [
            torch.full((self.n_imp,), q_imp / max(self.n_imp, 1), dtype=torch.float64, device=dev),
            q_g * p_fresh / max(self.n_good, 1)
            * torch.ones(self.n_good, dtype=torch.float64, device=dev),
            q_g * (1.0 - p_fresh) * p_r,
        ]
        if lv is not None:
            w.append((1.0 - q_imp) * q_lv * lv[1])
        cdf = torch.cumsum(torch.cat(w), 0)
        self.cdf = cdf / cdf[-1]
        self.p_fresh_eff = p_fresh
        self.p_replay = p_r
        self.q_imp = q_imp
        self.q_level = (1.0 - q_imp) * q_lv  # Share of all draws from the levels
        self._acc.zero_()

    # ── Draws and updates (no host syncs) ──

    def draw(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scenario indices (B,) int64 and their sources (B,) int8 for uniforms u (B,)."""
        j = torch.searchsorted(self.cdf, u.double().contiguous(), right=True)
        j = j.clamp(max=self.cdf.numel() - 1)
        return self.ids[j], self.src[j]

    @torch.no_grad()
    def observe(self, scenario: torch.Tensor, final_path: torch.Tensor, done: torch.Tensor,
                src: Optional[torch.Tensor] = None, gen: Optional[torch.Tensor] = None,
                tilt0: Optional[torch.Tensor] = None, look: Optional[Dict[str, torch.Tensor]] = None):
        """Episode ends: scenario (B,) index, final_path (B,) mm (P_final), done (B,) bool,
        src (B,) the draw source (statistics; SRC_EDIT episodes update no score), gen (B,)
        the pool slot generation, tilt0 (B, 2) the start tilts and look the camera
        parameters (B rows) of the episodes (ScenarioPool; unused here). Updates the
        scores of the good scenarios by scatter ops."""
        a = self.cfg.alpha
        N = self.score.numel()
        good = self.good[scenario] & done
        short = (1.0 - final_path / self.pmax[scenario]).clamp(0.0, 1.0)
        gw = good.float()
        w = gw if src is None else (good & (src != SRC_EDIT)).float()
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
            self._acc[2].index_add_(0, col, (short * gw).double() * d)

    # ── Logging (host syncs) ──

    def stats(self) -> Dict[str, float]:
        """Flat statistics (plr/...) of the current state and the iteration so far."""
        gs = self.good_ids
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


# ── Continuous scenario pool ────────────────────────────────────────────────


@dataclass
class PoolConfig:
    """ScenarioPool settings (module docstring)."""

    size: int = 16384  # Slots; round(size x SamplerConfig.impossible_share) hold impossible ones
    ranges: ContRanges = field(default_factory=ContRanges)
    refresh_frac: float = 0.125  # Share of the good and of the impossible slots replaced per refresh
    seed: int = 0  # Candidate stream (STREAM_POOL)
    round_min: int = 1024  # Candidates per scan round: the expected need x 1.1, clamped
    round_max: int = 1 << 16
    max_candidates: int = 1 << 25  # Per fill; beyond: error (ranges without good scenarios)
    region_mix: Tuple[float, ...] = ()  # Good-slot share per target-region bin; () = natural


@dataclass
class LevelConfig:
    """ACCEL hard levels of the ScenarioPool (module docstring); size 0 = off."""

    size: int = 0  # Level slots K
    share: float = 0.2  # Share of the good draws from the levels, x their filled fraction
    thresh: float = 0.3  # Shortfall 1 - P_final / P_max to enter and to stay
    family_cap: int = 4  # Levels per root pool scenario
    jitter: float = 0.05  # deg, sigma per axis of the start-tilt edit
    jitter_p: float = 0.5  # Share of the level draws with a start-tilt edit (SRC_EDIT)
    edits: int = 256  # Scenario edits per iteration
    edit_sigma: Tuple[float, ...] = (5.0, 0.1, 0.1, 0.05, 0.05)  # sep mm, pitch / yaw deg, dx / dy mm
    candidates: int = 8192  # Candidate rows per iteration; the rest dropped (accel/dropped)


STREAM_EDIT = 5  # Generator stream of the level edits (seeded by pool seed and iteration)
LOOK_MAX_NUMEL = 256  # Camera parameters with more values per row (textures) are not stored
# Statistics of the last level update (accel/...)
LEVEL_UPDATE_KEYS = ("candidates", "dropped", "inserted", "inserted_edits", "evicted",
                     "edits_tried", "edits_valid", "edits_good", "update_s")


def region_quotas(mix, n: int) -> List[int]:
    """Per-bin counts of n slots for the shares mix (normalized): floor of share x n,
    the remainder one each to the largest fractional parts (ties: lower bin first)."""
    m = np.asarray(mix, dtype=np.float64)
    assert m.ndim == 1 and bool((m >= 0).all()) and float(m.sum()) > 0, f"region mix {mix}"
    x = m / m.sum() * n
    q = np.floor(x).astype(np.int64)
    for j in np.argsort(-(x - q), kind="stable")[:n - int(q.sum())]:
        q[j] += 1
    return [int(v) for v in q]


def parse_region_mix(spec: str) -> Tuple[float, ...]:
    """--pool-region-mix a,b,c,d: one share per target-region bin (REGION_BINS), >= 0,
    summing to 1 (within 1e-3); "" = () (the natural distribution)."""
    if not spec:
        return ()
    m = tuple(float(x) for x in spec.split(","))
    assert len(m) == len(REGION_BINS), f"region mix {spec!r}: one share per bin {REGION_BINS}"
    assert all(x >= 0 for x in m) and abs(sum(m) - 1.0) < 1e-3, f"region mix {spec!r}: shares >= 0, sum 1"
    return m


POOL_ROWS = ("candidates", "angle_rejected", "clipped", "missed", "valid", "good", "box_zero")
_POS = "_pos"  # Fill rows: stream position (not a slot field)
_RBIN = "_rbin"  # Fill rows: target-region bin


class ScenarioPool(ScenarioSampler):
    """Continuous scenario pool with PLR inside it, see the module docstring. Slots
    [0, n_good) hold good scenarios, [n_good, size) impossible ones; atlas is the slot
    table (an Atlas, params (size, 5)); slot_gen counts the replacements of each slot
    (an episode scores its slot only if the slot was not replaced meanwhile), born the
    iteration of the last fill. levels (LevelConfig, size K > 0): K ACCEL level slots
    [size, size + K) after them (atlas rows, scores and generations as the pool slots;
    lv_on, lv_tilt (K, 2), lv_look; uid: root pool scenario of every slot)."""

    def __init__(self, cfg: Optional[PoolConfig] = None, sampler: Optional[SamplerConfig] = None,
                 scan: Optional[AtlasConfig] = None, device: Optional[torch.device] = None,
                 log=print, levels: Optional[LevelConfig] = None):
        self.pcfg = pc = cfg or PoolConfig()
        self.lcfg = lc = levels or LevelConfig()
        sc = sampler or SamplerConfig()
        assert sc.holdout_frac == 0, "no hold-out in the pool"
        dev = torch.device(device) if device is not None else DEVICE
        self.scanner = Scanner(scan or AtlasConfig(), dev)
        self.log = log
        N = self.n_slots = pc.size
        K = self.n_levels = lc.size
        assert K >= 0 and len(lc.edit_sigma) == 5, "levels: size >= 0, five edit sigmas"
        n_imp = int(round(N * sc.impossible_share)) if sc.impossible_share > 0 else 0
        if sc.impossible_share > 0:
            n_imp = max(n_imp, 1)
        self.n_good_slots, self.n_imp_slots = N - n_imp, n_imp
        assert self.n_good_slots > 0, "pool too small"
        self.bins = cont_sep_bins(*pc.ranges.sep)
        self.cell = float(self.scanner.cfg.step) ** 2  # Box cell area (deg^2)
        self.quotas = region_quotas(pc.region_mix, self.n_good_slots) if pc.region_mix else None
        if pc.region_mix:
            assert len(pc.region_mix) == len(REGION_EDGES) + 1, "one region share per bin"
        # Lifetime good scenarios per target-region bin over the used stream prefix
        self.cum_rbin = np.zeros(len(REGION_EDGES) + 1, dtype=np.int64)
        self.cursor = 0  # Next candidate of the stream
        self.ptr = [0, 0]  # Oldest good / impossible slot (ring order)
        self.filled_iter = 0
        nb = len(self.bins)
        # Lifetime counts over the used stream prefix, per separation bin (POOL_ROWS):
        # candidates, angles out of bounds, clipped, missed, valid, good, valid with box
        # cells without a hit
        self.cum = np.zeros((len(POOL_ROWS), nb), dtype=np.int64)
        _sync(dev)
        t0 = time.perf_counter()
        g, i, st = self._fill(self.n_good_slots, self.n_imp_slots, self.quotas)
        d = {k: torch.cat((g[k], i[k])) for k in Atlas.TENSORS}
        _sync(dev)
        st["refresh_s"] = time.perf_counter() - t0
        st["replaced"] = N
        if K:  # Level slots: copies of slot 0, not good (never drawn, never scored) until filled
            d = {k: torch.cat((v, v[:1].expand(K, *v.shape[1:]))) for k, v in d.items()}
            d["good"][N:] = False
        d.update(version=ATLAS_VERSION, key=self.key(), named={}, n_grid=N,
                 build_s=st["refresh_s"], offgrid=None)
        atlas = Atlas(d, dev, "pool")
        assert bool(atlas.good[:self.n_good_slots].all()) and not bool(atlas.good[self.n_good_slots:].any())
        self.slot_gen = torch.zeros(N + K, dtype=torch.long, device=dev)
        self.born = torch.zeros(N + K, dtype=torch.long, device=dev)
        self.last_fill = st
        if K:
            self.lv_on = torch.zeros(K, dtype=torch.bool, device=dev)
            self.lv_tilt = torch.zeros(K, 2, device=dev)
            self.lv_look: Optional[Dict[str, torch.Tensor]] = None  # Allocated at the first look
            self.uid = torch.arange(N + K, device=dev)  # Root pool scenario (family) per slot
            self.next_uid = N + K
            self.levels_iter = 0
            C = lc.candidates  # Candidate rows; row C takes the discarded writes
            self.c_slot = torch.zeros(C + 1, dtype=torch.long, device=dev)
            self.c_gen = torch.zeros(C + 1, dtype=torch.long, device=dev)
            self.c_tilt = torch.zeros(C + 1, 2, device=dev)
            self.c_short = torch.zeros(C + 1, device=dev)
            self.c_look: Optional[Dict[str, torch.Tensor]] = None
            self.c_count = torch.zeros((), dtype=torch.long, device=dev)
            self.last_levels: Dict[str, float] = dict.fromkeys(LEVEL_UPDATE_KEYS, 0.0)
        super().__init__(atlas, sc, exclude=torch.arange(N + K, device=dev) >= N)
        log(f"pool: {N} slots ({self.n_good_slots} good, {self.n_imp_slots} impossible), "
            f"{pc.ranges.tag()} (laser pitch {pc.ranges.pitch}, yaw {pc.ranges.yaw} deg, offsets "
            f"over r = {pc.ranges.offset_radius} mm), filled in {st['refresh_s']:.1f} s: "
            + self._fill_text(st))
        log("pool: good slots per target-region bin (" + "/".join(REGION_BINS) + ") "
            + " / ".join(f"{100 * v:.1f}" for v in self.region_shares().values()) + " %"
            + (f" (quotas {'/'.join(map(str, self.quotas))} of mix {pc.region_mix}; "
               f"{st['rejected']} good candidates of the prefix discarded)" if self.quotas else
               " (natural)"))
        if K:
            log(f"pool: {K} ACCEL level slots: {lc.share:g} of the good draws x filled share, "
                f"shortfall >= {lc.thresh:g}, <= {lc.family_cap} per root scenario, start-tilt "
                f"edits {lc.jitter:g} deg on {lc.jitter_p:g} of the level draws, {lc.edits} "
                f"scenario edits per iteration (sigma {', '.join(f'{s:g}' for s in lc.edit_sigma)})")

    def key(self) -> Dict[str, Any]:
        """Scan settings of the slot table (Atlas.check) and the pool settings."""
        d = {k: v for k, v in self.scanner.cfg.key().items() if k not in ("sep", "pitch", "yaw", "named")}
        d["pool"] = asdict(self.pcfg)
        return d

    @staticmethod
    def _fill_text(st: Dict[str, float]) -> str:
        return (f"{st['candidates']} candidates generated ({st['prefix']} used): angles out of "
                f"bounds {100 * st['angle_rejected_share']:.1f} %, clipped "
                f"{100 * st['clipped_share']:.1f} %, missed M2 {100 * st['missed_share']:.1f} %, "
                f"valid {100 * st['valid_share']:.1f} %; {st['scans']} box scans; good "
                f"{100 * st['good_share']:.1f} % of the valid; box cells without a hit in "
                f"{100 * st['box_zero_share']:.2f} % of the valid")

    # ── Filling ──

    @torch.no_grad()
    def _fill(self, k_g: int, k_i: int, quotas: Optional[List[int]] = None
              ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
        """The next k_g good and k_i impossible valid scenarios of the candidate stream
        (atlas fields, on the device, plus _pos stream position and _rbin target-region
        bin per row) and the fill statistics (host syncs). Rounds of candidates from the
        cursor (candidate_batch: angle bounds, beam check), box scan of the valid ones
        only; the used prefix ends at the last scenario taken, the cursor moves to its end
        (candidates scanned past it are read again by the next fill). quotas (per
        target-region bin, sum k_g): the next quotas[j] good ones of each bin instead
        (rejection: good ones of a full bin inside the prefix are discarded), the good
        rows in stream order; round sizes from the running share of the scarcest bin."""
        pc, sc = self.pcfg, self.scanner
        nb, nr = len(self.bins), len(REGION_EDGES) + 1
        if quotas is not None:
            assert len(quotas) == nr and sum(quotas) == k_g, (quotas, k_g)
        need = [k_g, k_i] if quotas is None else list(quotas) + [k_i]  # Groups: good..., impossible
        ng = len(need) - 1
        taken: List[List[Dict[str, torch.Tensor]]] = [[] for _ in need]
        cnt = np.zeros((len(POOL_ROWS), nb), dtype=np.int64)
        cnt_r = np.zeros(nr, dtype=np.int64)  # Good per target-region bin, used prefix
        scanned = scans = 0
        while any(k > 0 for k in need):
            tot = (self.cum + cnt).sum(1)  # Running shares: lifetime and this fill
            share_i = (tot[4] - tot[5] + 1.0) / (tot[0] + 2.0)
            if quotas is None:
                share_g = (tot[5] + 1.0) / (tot[0] + 2.0)
                want = max(need[0] / share_g, need[1] / share_i)
            else:
                share_r = (self.cum_rbin + cnt_r + 1.0) / (tot[0] + 2.0)
                want = max(max(need[j] / share_r[j] for j in range(nr)), need[-1] / share_i)
            n = int(min(max(math.ceil(1.1 * want), pc.round_min), pc.round_max))
            b = candidate_batch(pc.ranges, pc.seed, self.cursor, n, STREAM_POOL, sc)
            vid = np.nonzero(b["valid"])[0]
            d = sc.scan(torch.from_numpy(b["params"][vid]), zero_cells=True)
            scans += vid.size
            good = d["good"]
            d[_POS] = torch.from_numpy(vid + self.cursor).to(good.device)
            d[_RBIN] = region_bin(d["near_cells"].double() * self.cell)
            end = n  # Used prefix of this round
            lasts = []
            grps = ((good,) if quotas is None else tuple(good & (d[_RBIN] == j) for j in range(nr)))
            for j, grp in enumerate(grps + (~good,)):
                rows = torch.nonzero(grp).squeeze(1)[:need[j]]
                if rows.numel():
                    taken[j].append({k: v[rows] for k, v in d.items()})
                    need[j] -= rows.numel()
                    lasts.append(int(vid[int(rows[-1])]))
            if all(k == 0 for k in need):
                end = max(lasts) + 1 if lasts else 0
            # Stage counts over the used prefix of the round, per separation bin
            band = sep_band(b["params"][:end, 0], self.bins)
            vin = vid < end
            bv = band[vid[vin]]
            gv = good.cpu().numpy()[vin]
            zv = (d["zero_cells"] > 0).cpu().numpy()[vin]
            for row, sel in enumerate((np.ones(end, bool), ~b["angle_ok"][:end], b["clipped"][:end],
                                       b["missed"][:end], b["valid"][:end])):
                np.add.at(cnt[row], band[sel], 1)
            np.add.at(cnt[5], bv[gv], 1)
            np.add.at(cnt[6], bv[zv], 1)
            np.add.at(cnt_r, d[_RBIN].cpu().numpy()[vin][gv], 1)
            self.cursor += end
            scanned += n
            if scanned > pc.max_candidates:
                raise RuntimeError(f"pool: {scanned} candidates without filling {k_g} good / {k_i} "
                                   f"impossible slots (region quotas {quotas}); check the ranges, "
                                   f"the impossible share or the region mix")
        self.cum += cnt
        self.cum_rbin += cnt_r
        tot = cnt.sum(1)
        cat = lambda lst: {k: torch.cat([t[k] for t in lst]) for k in lst[0]} \
            if lst else {k: v[:0] for k, v in d.items()}
        g = cat([t for j in range(ng) for t in taken[j]])
        if quotas is not None:  # Bins concatenated: back to stream order
            o = torch.argsort(g[_POS])
            g = {k: v[o] for k, v in g.items()}
        c0, cv = max(int(tot[0]), 1), max(int(tot[4]), 1)
        st = {"candidates": scanned, "prefix": int(tot[0]), "valid": int(tot[4]), "scans": scans,
              "angle_rejected_share": float(tot[1] / c0), "clipped_share": float(tot[2] / c0),
              "missed_share": float(tot[3] / c0), "valid_share": float(tot[4] / c0),
              "good_share": float(tot[5] / cv), "box_zero_share": float(tot[6] / cv),
              "rejected": int(tot[5]) - k_g}
        return g, cat(taken[ng]), st

    @torch.no_grad()
    def _replace(self, iteration: int):
        """Refill the oldest refresh_frac of the good and of the impossible slots; their
        scores reset, their generation increments (host syncs). Region quotas: the quotas
        of the fill are the bins of the replaced good slots, each refilled with a
        scenario of its bin (in slot order within a bin)."""
        pc = self.pcfg
        dev = self.device
        ng, ni = self.n_good_slots, self.n_imp_slots
        k_g = max(1, int(round(pc.refresh_frac * ng)))
        k_i = max(1, int(round(pc.refresh_frac * ni))) if ni else 0
        _sync(dev)
        t0 = time.perf_counter()
        sg = (self.ptr[0] + torch.arange(k_g, device=dev)) % ng
        si = ng + (self.ptr[1] + torch.arange(k_i, device=dev)) % max(ni, 1)
        if self.quotas is None:
            g, i, st = self._fill(k_g, k_i)
        else:
            rb = self.slot_rbin(sg)
            g, i, st = self._fill(k_g, k_i, torch.bincount(rb, minlength=len(self.quotas)).tolist())
            sg = sg[torch.argsort(rb, stable=True)]  # Slots and rows grouped by bin alike
            g = {k: v[torch.argsort(g[_RBIN], stable=True)] for k, v in g.items()}
        slots = torch.cat((sg, si))
        at = self.atlas
        for k in Atlas.TENSORS:
            getattr(at, k)[slots] = torch.cat((g[k], i[k])).to(getattr(at, k).dtype)
        self.pmax[slots] = at.pmax[slots].clamp(min=1e-6)
        self.score[slots] = 0.0
        self.seen[slots] = False
        self.last[slots] = iteration
        self.visits[slots] = 0
        self.slot_gen[slots] += 1
        self.born[slots] = iteration
        if self.n_levels:  # New root scenarios
            self.uid[slots] = self.next_uid + torch.arange(slots.numel(), device=dev)
            self.next_uid += slots.numel()
        self.ptr = [(self.ptr[0] + k_g) % ng, (self.ptr[1] + k_i) % max(ni, 1)]
        self.filled_iter = iteration
        _sync(dev)
        st["refresh_s"] = time.perf_counter() - t0
        st["replaced"] = k_g + k_i
        self.last_fill = st

    def slot_rbin(self, slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Target-region bin (int64) of slots (default all), from their near_cells."""
        nc = self.atlas.near_cells if slots is None else self.atlas.near_cells[slots]
        return region_bin(nc.double() * self.cell)

    def region_shares(self) -> Dict[str, float]:
        """Share of the good slots per target-region bin (region_<bin>; host sync)."""
        rb = self.slot_rbin(torch.arange(self.n_good_slots, device=self.device))
        c = torch.bincount(rb, minlength=len(REGION_BINS)).double() / max(self.n_good_slots, 1)
        return {f"region_{nm}": float(v) for nm, v in zip(REGION_BINS, c.tolist())}

    @torch.no_grad()
    def refresh(self, iteration: int):
        """Once per iteration: at a new iteration, replace the oldest refresh_frac of the
        slots with freshly scanned scenarios (host syncs), then recompute the draw
        distribution (ScenarioSampler.refresh)."""
        if self.n_levels and iteration > self.levels_iter:  # Before the refill: candidates' slots
            self._update_levels(iteration)
        if iteration > self.filled_iter and self.pcfg.refresh_frac > 0:
            self._replace(iteration)
        super().refresh(iteration)

    @torch.no_grad()
    def observe(self, scenario: torch.Tensor, final_path: torch.Tensor, done: torch.Tensor,
                src: Optional[torch.Tensor] = None, gen: Optional[torch.Tensor] = None,
                tilt0: Optional[torch.Tensor] = None, look: Optional[Dict[str, torch.Tensor]] = None):
        """ScenarioSampler.observe for episodes whose slot still holds their scenario
        (gen (B,) = the slot generation at the draw; None: all); with levels, the level
        candidates among them (needs src and tilt0; look: camera runs)."""
        if gen is not None:
            done = done & (gen == self.slot_gen[scenario])
        super().observe(scenario, final_path, done, src)
        if self.n_levels and src is not None:
            self._capture(scenario, final_path, done, src, tilt0, look)

    # ── ACCEL levels (module docstring) ──

    def _level_slots(self) -> Optional[torch.Tensor]:
        if not self.n_levels:
            return None
        return self.n_slots + torch.arange(self.n_levels, device=self.device)

    def level_probs(self) -> torch.Tensor:
        """(K,) float64 draw distribution over the filled levels (PLR rank and staleness)."""
        return self._rank_probs(self._level_slots(), self.lv_on)

    def _level_draw(self) -> Optional[Tuple[float, torch.Tensor]]:
        if not self.n_levels:
            return None
        n_on = int(self.lv_on.sum())
        p = self.level_probs() if n_on else torch.zeros(self.n_levels, dtype=torch.float64,
                                                        device=self.device)
        return self.lcfg.share * n_on / self.n_levels, p

    def level_start(self, t: torch.Tensor, slot: torch.Tensor, src: torch.Tensor,
                    v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Start tilts (B, 2) and sources (B,) of draws (slot, src) with the levels: a level
        draw (SRC_LEVEL) starts at its level's tilt, with probability jitter_p plus a
        Gaussian edit (sigma jitter per axis; source SRC_EDIT); the other rows keep t and
        src. v (B, 3) uniforms in [0, 1) (coin, Box-Muller pair). No sync; unclamped."""
        lc = self.lcfg
        lv = src == SRC_LEVEL
        j = (slot - self.n_slots).clamp(0, self.n_levels - 1)
        edit = lv & (v[:, 0] < lc.jitter_p)
        r = torch.sqrt(-2.0 * torch.log1p(-v[:, 1]))
        a = (2.0 * math.pi) * v[:, 2]
        z = r[:, None] * torch.stack((torch.cos(a), torch.sin(a)), 1)
        tl = self.lv_tilt[j] + torch.where(edit[:, None], lc.jitter * z, 0.0)
        return torch.where(lv[:, None], tl, t), torch.where(edit, SRC_EDIT, src).to(src.dtype)

    def level_look(self, slot: torch.Tensor, src: torch.Tensor,
                   look: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Camera parameters look (B rows) with the stored keys of each level draw's look
        (src SRC_LEVEL / SRC_EDIT); the textures stay as drawn. No sync."""
        if self.lv_look is None:
            return look
        m = src >= SRC_LEVEL
        j = (slot - self.n_slots).clamp(0, self.n_levels - 1)
        out = dict(look)
        for k, v in self.lv_look.items():
            out[k] = torch.where(m.view(-1, *([1] * (v.dim() - 1))), v[j], look[k])
        return out

    def _capture(self, scenario: torch.Tensor, final_path: torch.Tensor, done: torch.Tensor,
                 src: torch.Tensor, tilt0: Optional[torch.Tensor],
                 look: Optional[Dict[str, torch.Tensor]]):
        """Record the level candidates among the finished episodes: good slot, drawn
        (fresh, replay, level, edit), shortfall >= thresh. No sync: rows past the
        capacity go to the discard row."""
        assert tilt0 is not None, "levels need the episodes' start tilts"
        lc = self.lcfg
        C = lc.candidates
        short = (1.0 - final_path / self.pmax[scenario]).clamp(0.0, 1.0)
        m = done & self.good[scenario] & (src >= SRC_FRESH) & (short >= lc.thresh)
        pos = self.c_count + torch.cumsum(m.long(), 0) - 1
        idx = torch.where(m & (pos < C), pos, C)
        self.c_slot[idx] = scenario
        self.c_gen[idx] = self.slot_gen[scenario]
        self.c_tilt[idx] = tilt0
        self.c_short[idx] = short
        if look is not None:
            if self.c_look is None:
                keep = {k: v for k, v in look.items() if v[0].numel() <= LOOK_MAX_NUMEL}
                self.c_look = {k: torch.zeros(C + 1, *v.shape[1:], dtype=v.dtype, device=self.device)
                               for k, v in keep.items()}
                self.lv_look = {k: torch.zeros(self.n_levels, *v.shape[1:], dtype=v.dtype,
                                               device=self.device) for k, v in keep.items()}
            for k, v in self.c_look.items():
                v[idx] = look[k]
        self.c_count += m.sum()

    @torch.no_grad()
    def _edits(self, iteration: int, ex: torch.Tensor) -> Dict[str, Any]:
        """Scenario edits of the filled levels ex (level indices): edits parents drawn from
        the level distribution, parameters + N(0, edit_sigma) (laser offsets fixed with
        offset radius 0), kept within the ranges (separation, angle bounds, offset disk),
        beam-checked, box-scanned; returns parent (level index) and atlas rows of the good
        ones, and the counts (host syncs; numpy generator of (STREAM_EDIT, seed,
        iteration))."""
        lc, r = self.lcfg, self.pcfg.ranges
        out: Dict[str, Any] = {"parent": ex[:0], "rows": None, "tried": 0, "valid": 0}
        if lc.edits <= 0 or ex.numel() == 0:
            return out
        rng = np.random.default_rng([STREAM_EDIT, self.pcfg.seed, iteration])
        p = self.level_probs()[ex].cpu().numpy()
        exn = ex.cpu().numpy()
        par = exn[rng.choice(exn.size, size=lc.edits, p=p / p.sum())]
        sig = np.array(lc.edit_sigma, dtype=np.float64)
        if r.offset_radius <= 0:
            sig[3:] = 0.0
        prm = self.atlas.params[self.n_slots + torch.from_numpy(par).to(self.device)].double().cpu().numpy()
        prm = prm + rng.normal(size=prm.shape) * sig
        ok = (prm[:, 0] >= r.sep[0]) & (prm[:, 0] <= r.sep[1]) & r.angle_ok(prm)
        if r.offset_radius > 0:
            ok &= np.hypot(prm[:, 3], prm[:, 4]) < r.offset_radius * OFFSET_SHRINK
        idx = np.nonzero(ok)[0]
        out["tried"] = lc.edits
        if idx.size:
            chk = self.scanner.check(torch.from_numpy(prm[idx]))
            idx = idx[(~chk["clipped"] & ~chk["missed"]).cpu().numpy()]
        out["valid"] = int(idx.size)
        if idx.size:
            d = self.scanner.scan(torch.from_numpy(prm[idx]))
            g = d["good"]
            out["parent"] = torch.from_numpy(par[idx]).to(self.device)[g]
            out["rows"] = {k: d[k][g] for k in Atlas.TENSORS}
        return out

    @torch.no_grad()
    def _update_levels(self, iteration: int):
        """Once per new iteration, before the slot refill (host syncs): merge the filled
        levels (score: EMA of their replays), the candidates (score: their shortfall) and
        the edits (score: the parent's) into the level slots: score >= thresh, at most
        family_cap per root scenario (best first), the top K by score (ties: levels before
        candidates before edits); kept levels stay, new ones fill the free slots
        (generation, born, scores, start tilt and look set; the freed ones not refilled
        are emptied)."""
        lc, dev = self.lcfg, self.device
        N, K, C = self.n_slots, self.n_levels, lc.candidates
        at = self.atlas
        _sync(dev)
        t0 = time.perf_counter()
        n_cand = int(self.c_count)
        n = min(n_cand, C)
        ci = torch.arange(n, device=dev)
        ci = ci[self.c_gen[:n] == self.slot_gen[self.c_slot[:n]]]
        cs = self.c_slot[ci]
        ex = torch.nonzero(self.lv_on).squeeze(1)
        e = self._edits(iteration, ex)
        ep = e["parent"]
        n_ex, n_c, n_e = ex.numel(), ci.numel(), ep.numel()
        score = torch.cat((self.score[N + ex], self.c_short[ci], self.score[N + ep]))
        fam = torch.cat((self.uid[N + ex], self.uid[cs], self.uid[N + ep]))
        kind = torch.cat((torch.zeros(n_ex, dtype=torch.long, device=dev),
                          torch.ones(n_c, dtype=torch.long, device=dev),
                          torch.full((n_e,), 2, dtype=torch.long, device=dev)))
        sub = torch.cat((ex, ci, torch.arange(n_e, device=dev)))  # Row within its kind
        # Per family the best family_cap (score order, stable: table order on ties)
        order = torch.argsort(score, descending=True, stable=True)
        order = order[torch.argsort(fam[order], stable=True)]
        f = fam[order]
        pos = torch.arange(f.numel(), device=dev)
        head = torch.ones_like(f, dtype=torch.bool)
        head[1:] = f[1:] != f[:-1]
        start = torch.cummax(torch.where(head, pos, 0), 0).values
        in_cap = torch.zeros_like(head)
        in_cap[order] = (pos - start) < lc.family_cap
        rows = torch.nonzero((score >= lc.thresh) & in_cap).squeeze(1)
        top = rows[torch.argsort(score[rows], descending=True, stable=True)[:K]]
        sel = torch.zeros(score.numel(), dtype=torch.bool, device=dev)
        sel[top] = True
        # Sources of the new levels, gathered before any write
        new_c = sub[top[kind[top] == 1]]
        new_e = sub[top[kind[top] == 2]]
        src_slot = self.c_slot[new_c]
        vals = {k: torch.cat((getattr(at, k)[src_slot],
                              e["rows"][k][new_e].to(getattr(at, k).dtype) if n_e else
                              getattr(at, k)[:0])) for k in Atlas.TENSORS}
        par = ep[new_e]
        tilt = torch.cat((self.c_tilt[new_c], self.lv_tilt[par]))
        look = ({k: torch.cat((self.c_look[k][new_c], self.lv_look[k][par])) for k in self.lv_look}
                if self.lv_look is not None else None)
        sc = torch.cat((self.c_short[new_c], self.score[N + par]))
        fm = torch.cat((self.uid[src_slot], self.uid[N + par]))
        # Free the levels not kept, then fill the free slots
        out = ex[~sel[:n_ex]]
        self.lv_on[out] = False
        at.good[N + out] = False
        self.seen[N + out] = False
        self.slot_gen[N + out] += 1
        jn = torch.nonzero(~self.lv_on).squeeze(1)[:sc.numel()]
        s = N + jn
        for k in Atlas.TENSORS:
            getattr(at, k)[s] = vals[k]
        self.pmax[s] = at.pmax[s].clamp(min=1e-6)
        self.score[s] = sc
        self.seen[s] = True
        self.last[s] = iteration
        self.visits[s] = 0
        self.slot_gen[s] += 1
        self.born[s] = iteration
        self.uid[s] = fm
        self.lv_tilt[jn] = tilt
        if look is not None:
            for k, v in self.lv_look.items():
                v[jn] = look[k]
        self.lv_on[jn] = True
        self.c_count.zero_()
        self.levels_iter = iteration
        _sync(dev)
        self.last_levels = {"candidates": n_cand, "dropped": n_cand - n, "inserted": new_c.numel(),
                            "inserted_edits": new_e.numel(), "evicted": out.numel(),
                            "edits_tried": e["tried"], "edits_valid": e["valid"], "edits_good": n_e,
                            "update_s": time.perf_counter() - t0}

    # ── Logging (host syncs) ──

    def stats(self) -> Dict[str, float]:
        """ScenarioSampler.stats plus the top scenario's laser offsets and pool/...: the
        last fill (time, slots replaced, candidates generated / used, box scans, per stage: angles
        out of bounds, clipped, missed M2, valid (box-scanned) shares of the candidates,
        good share and box-check disagreement of the valid ones), lifetime good share of
        the valid candidates and valid share per separation bin, the separation
        histogram of the good slots, mean slot age, the target-region bin shares of the
        good slots (pool/region_<bin>) and the good candidates of the last fill's used
        prefix not taken (pool/rejected: of full region bins, or past the last good slot
        while impossible ones were still needed)."""
        out = super().stats()
        out["plr/top_laser_dx"] = out["plr/top_laser_dy"] = math.nan
        p = self.p_replay
        if p.numel() and float(p.sum()) > 0:
            prm = self.atlas.params[int(self.good_ids[int(p.argmax())])].tolist()
            out["plr/top_laser_dx"], out["plr/top_laser_dy"] = prm[3], prm[4]
        lf = self.last_fill
        out.update({f"pool/{k}": float(lf[k]) for k in (
            "refresh_s", "replaced", "candidates", "prefix", "valid", "scans", "angle_rejected_share",
            "clipped_share", "missed_share", "valid_share", "good_share", "box_zero_share")})
        c = self.cum
        sep = self.atlas.params[:self.n_good_slots, 0]
        for j, (lo, hi) in enumerate(self.bins):
            nm = f"sep{lo:g}-{hi:g}"
            out[f"pool/good_share_{nm}"] = float(c[5, j] / c[4, j]) if c[4, j] else math.nan
            out[f"pool/valid_share_{nm}"] = float(c[4, j] / c[0, j]) if c[0, j] else math.nan
        for nm, m in sep_bin_masks(sep, self.bins).items():
            out[f"pool/hist_{nm}"] = float(m.float().mean())
        out["pool/age_mean"] = float((self.iteration - self.born[:self.n_slots]).double().mean())
        out["pool/rejected"] = float(lf.get("rejected", math.nan))
        out.update({f"pool/{k}": v for k, v in self.region_shares().items()})
        if self.n_levels:
            out.update(self.level_stats())
        return out

    def level_stats(self) -> Dict[str, float]:
        """accel/...: filled levels and share, share of all draws, score quantiles, root
        scenarios (families), mean age (iterations) and replays of the filled levels, the
        last update (candidates recorded / dropped, inserted from candidates / edits,
        evicted, edits tried / valid / good, time); plr/level_... and plr/edit_...:
        episodes and mean shortfall of the level draws as stored / with a tilt edit."""
        N, K = self.n_slots, self.n_levels
        on = self.lv_on
        n_on = int(on.sum())
        acc = self._acc.cpu()
        nan = math.nan
        out = {"accel/levels": float(n_on), "accel/fill": n_on / K,
               "accel/draw_share": float(self.q_level)}
        for j, name in ((SRC_LEVEL, "level"), (SRC_EDIT, "edit")):
            ng = float(acc[1, j])
            out[f"plr/{name}_episodes"] = float(acc[0, j])
            out[f"plr/{name}_shortfall"] = float(acc[2, j]) / ng if ng else nan
        out.update({k: nan for k in ("accel/score_q10", "accel/score_q50", "accel/score_q90",
                                     "accel/families", "accel/age_mean", "accel/visits_mean")})
        if n_on:
            sc = self.score[N:][on].double()
            q = torch.quantile(sc, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64,
                                                device=sc.device)).tolist()
            out.update({"accel/score_q10": q[0], "accel/score_q50": q[1], "accel/score_q90": q[2],
                        "accel/families": float(torch.unique(self.uid[N:][on]).numel()),
                        "accel/age_mean": float((self.iteration - self.born[N:][on]).double().mean()),
                        "accel/visits_mean": float(self.visits[N:][on].double().mean())})
        out.update({f"accel/{k}": float(v) for k, v in self.last_levels.items()})
        return out

    # ── Checkpoint ──

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        d.update(pool={k: getattr(self.atlas, k) for k in Atlas.TENSORS}, slot_gen=self.slot_gen,
                 born=self.born, cursor=self.cursor, ptr=list(self.ptr),
                 filled_iter=self.filled_iter, cum=self.cum.copy(), last_fill=dict(self.last_fill),
                 pool_cfg=asdict(self.pcfg), cum_rbin=self.cum_rbin.copy())
        if self.n_levels:
            n = min(int(self.c_count), self.lcfg.candidates)
            d["levels"] = {
                "cfg": asdict(self.lcfg), "on": self.lv_on, "tilt": self.lv_tilt, "look": self.lv_look,
                "uid": self.uid, "next_uid": self.next_uid, "iter": self.levels_iter,
                "last": dict(self.last_levels), "c_count": self.c_count,
                "cand": {"slot": self.c_slot[:n], "gen": self.c_gen[:n], "tilt": self.c_tilt[:n],
                         "short": self.c_short[:n],
                         "look": {k: v[:n] for k, v in self.c_look.items()} if self.c_look else None}}
        return d

    def load_state_dict(self, d: Dict[str, Any]):
        """Also a checkpoint without levels into a pool with levels (levels empty)."""
        dev = self.device
        have = {"region_mix": (), **d["pool_cfg"]}  # Checkpoints from before region_mix
        assert have == asdict(self.pcfg), "pool settings differ from the checkpoint"
        n_ck = d["pool"]["params"].shape[0]
        assert n_ck in (self.n_slots, self.atlas.n), f"pool checkpoint with {n_ck} rows"
        for k in Atlas.TENSORS:
            getattr(self.atlas, k)[:n_ck].copy_(d["pool"][k].to(dev))
        self.pmax = self.atlas.pmax.clamp(min=1e-6)
        pad = lambda v: torch.cat((v.to(dev), torch.zeros(self.atlas.n - v.shape[0], dtype=v.dtype,  # noqa: E731
                                                          device=dev)))
        self.slot_gen, self.born = pad(d["slot_gen"]), pad(d["born"])
        self.cursor, self.ptr, self.filled_iter = int(d["cursor"]), list(d["ptr"]), int(d["filled_iter"])
        self.cum = np.array(d["cum"], dtype=np.int64)
        self.last_fill = dict(d["last_fill"])
        self.cum_rbin = np.array(d.get("cum_rbin", np.zeros_like(self.cum_rbin)), dtype=np.int64)
        lv = d.get("levels")
        assert lv is None or self.n_levels, "the checkpoint has levels, this pool none"
        if lv is not None:
            want = {**asdict(self.lcfg), "edit_sigma": list(self.lcfg.edit_sigma)}
            assert {**lv["cfg"], "edit_sigma": list(lv["cfg"]["edit_sigma"])} == want, \
                "level settings differ from the checkpoint"
            self.lv_on, self.lv_tilt = lv["on"].to(dev), lv["tilt"].to(dev)
            self.uid, self.next_uid, self.levels_iter = lv["uid"].to(dev), int(lv["next_uid"]), int(lv["iter"])
            self.last_levels = dict(lv["last"])
            self.c_count = lv["c_count"].to(dev)
            c = lv["cand"]
            n = c["slot"].shape[0]
            for name in ("slot", "gen", "tilt", "short"):
                getattr(self, f"c_{name}")[:n] = c[name].to(dev)
            if lv["look"] is not None:
                C = self.lcfg.candidates
                self.lv_look = {k: v.to(dev) for k, v in lv["look"].items()}
                self.c_look = {k: torch.zeros(C + 1, *v.shape[1:], dtype=v.dtype, device=dev)
                               for k, v in self.lv_look.items()}
                for k, v in (c["look"] or {}).items():
                    self.c_look[k][:n] = v.to(dev)
        elif self.n_levels:  # Levels start empty; the checkpoint's slots are the roots
            self.levels_iter = int(d["iteration"])
        d = {**d, **{k: pad(d[k]) for k in ("score", "seen", "last", "visits")}}
        super().load_state_dict(d)


# ── CLI ─────────────────────────────────────────────────────────────────────


def table_summary(st: Dict[str, Any], label: str) -> str:
    """One block of table_stats(...) as text."""
    lines = [f"{label}: {st['n']} scenarios, good {st['good']} ({100 * st['good_share']:.1f} %); "
             f"good P_max (m) mean {st['pmax_m_mean']:.2f}, q10 / q50 / q90 {st['pmax_m_q10']:.2f} / "
             f"{st['pmax_m_q50']:.2f} / {st['pmax_m_q90']:.2f}; P_max at the cap "
             f"{100 * st['pmax_cap_share']:.1f} % of good; box cells at the cap "
             f"{100 * st['cap_cells_share']:.2f} % (good {100 * st['cap_cells_share_good']:.2f} %)"]
    for name, b in st["sep_bins"].items():
        lines.append(f"  {name} mm: {b['n']} scenarios, good {100 * b['good_share']:.1f} %, "
                     f"good P_max q50 {b['pmax_m_q50']:.2f} m, at cap {100 * b['pmax_cap_share']:.1f} %")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Scenario atlas (build / summary) and the off-grid "
                                             "eval tables (offgrid)")
    ap.add_argument("cmd", choices=("build", "summary", "offgrid", "cont", "region"))
    ap.add_argument("--atlas", default=DEFAULT_ATLAS, help="Atlas file")
    d = AtlasConfig()
    ap.add_argument("--sep", type=float, nargs=3, default=list(d.sep), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--pitch", type=float, nargs=3, default=list(d.pitch), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--yaw", type=float, nargs=3, default=list(d.yaw), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--no-named", action="store_true", help="Do not add geometry.SETUPS")
    ap.add_argument("--chunk-states", type=int, default=d.chunk_states)
    ap.add_argument("--force", action="store_true", help="Rebuild an existing file")
    o = OffgridConfig()
    ap.add_argument("--kinds", default=",".join(TABLE_KINDS),
                    help="offgrid: tables to build / summarize (" + ", ".join(TABLE_KINDS) + ")")
    ap.add_argument("--tables-dir", default=EVAL_TABLES_DIR, help="offgrid: <dir>/<kind>_v1.pt")
    ap.add_argument("--n", type=int, default=o.n, help="offgrid: eval episodes (one per scenario)")
    ap.add_argument("--impossible", type=float, default=o.impossible_share)
    ap.add_argument("--seed", type=int, default=o.seed)
    r = ContRanges()
    ap.add_argument("--sep-range", type=float, nargs=2, default=list(r.sep), metavar=("MIN", "MAX"),
                    help="cont: separation range (mm)")
    ap.add_argument("--laser-range", type=float, nargs=2, default=list(r.pitch),
                    metavar=("MIN", "MAX"), help="cont: laser pitch and yaw range (deg)")
    ap.add_argument("--laser-offset-radius", type=float, default=r.offset_radius,
                    help="cont: laser offset disk radius (mm); 0 = the measured offsets")
    ap.add_argument("--laser-mode", choices=("landing", "angle"), default=r.mode,
                    help="cont: laser angles from a uniform M2 landing point, or uniform")
    args = ap.parse_args()
    if args.cmd == "region":
        rcfg = RegionConfig(ranges=cont_ranges(args.sep_range, args.laser_range,
                                               args.laser_offset_radius, args.laser_mode),
                            n=args.n, seed=args.seed)
        path = region_table_path(rcfg, args.tables_dir)
        if args.force and os.path.exists(path):
            os.remove(path)
        t = load_or_build_region(path, rcfg)
        b = t.offgrid
        rs = b["region"]
        print(f"region ({path}): {t.n} scenarios, per bin "
              + ", ".join(f"{k} {lab} {int(m.sum())}" for (k, m), lab in zip(
                  region_bin_masks(region_area(t)).items(), region_bin_labels().values())))
        print(f"  {b['candidates']} candidates, {b['valid']} valid (box-scanned), {b['good']} good; "
              f"target-region size of the good ones (deg^2): "
              + ", ".join(f"{k} {v:.4f}" for k, v in rs.items() if k.startswith("q"))
              + "; bin shares " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in rs["bin_share"].items()))
        return
    if args.cmd == "cont":
        ccfg = ContConfig(ranges=cont_ranges(args.sep_range, args.laser_range,
                                             args.laser_offset_radius, args.laser_mode),
                          n=args.n, impossible_share=args.impossible, seed=args.seed)
        path = cont_table_path(ccfg, args.tables_dir)
        if args.force and os.path.exists(path):
            os.remove(path)
        t = load_or_build_cont(path, ccfg)
        bins = cont_sep_bins(*ccfg.ranges.sep)
        print(table_summary(table_stats(t, bins=bins), f"cont ({path})"))
        b = t.offgrid
        print(f"  candidates {b['candidates']} (generated {b['scanned']}): angles out of bounds "
              f"{100 * b['angle_rejected'] / b['candidates']:.1f} %, clipped "
              f"{100 * b['clipped'] / b['candidates']:.1f} %, missed M2 "
              f"{100 * b['missed'] / b['candidates']:.1f} %, valid {b['valid']}; valid with box "
              f"cells without a hit {b['box_zero_cells']}")
        return
    if args.cmd == "offgrid":
        grid = load_atlas(args.atlas)
        cfg = atlas_config(grid)
        print(table_summary(table_stats(grid), "grid"))
        for kind in args.kinds.split(","):
            path = table_path(kind, args.tables_dir)
            if args.force and os.path.exists(path):
                os.remove(path)
            t = load_or_build_offgrid(path, cfg, OffgridConfig(
                kind=kind, n=args.n, impossible_share=args.impossible, seed=args.seed))
            print(table_summary(table_stats(t, bins=TABLE_SEP_BINS[kind]), f"{kind} ({path})"))
        return
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
