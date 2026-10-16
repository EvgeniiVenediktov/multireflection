"""
find_setups.py - Search for additional cell setups (separation x laser tilt) for multi-setup RL

Run from rl/:  uv run python find_setups.py [--force coarse refine final]

Why: with the single measured setup, knowing the absolute M2 tilt (from the mount limit)
solves the task open-loop (the image-free homing baseline gets 99 % of the oracle). With
several setups whose target regions lie at different M2 tilts, the policy has to tell the
setups apart from the images.

Search space: separation x laser pitch x laser yaw; laser offset, mirror offsets and the
mirrors stay at the measured values (geometry.CellGeometry), M1 untilted. Every candidate
gets a scan of the M2 +-2 deg box at cap 128 (decision D1). Stages, cached under
runs/setups/ (gitignored); a stage reruns with --force or when its inputs changed:

    coarse   separation 50-200 mm (2 mm) x laser pitch / yaw 0-8 deg (0.25 deg), M2 grid
             0.05 deg; per candidate: bounce and stop-reason histograms, the plateau
             (most common hole-exit order >= 20 bounces) and its area, the highest
             hole-exit order, the best region (highest bounce count incl. the cap: size,
             centroid), the longest path and its tilt, the first M2 hit at zero M2 tilt
    refine   1 mm / 0.05 deg neighbourhoods of the best eligible coarse candidates of
             each 10 mm separation band, same M2 grid
    final    M2 grid 0.01 deg for the best few per band (and the measured setup): the
             landscape (hole-exit orders, connected regions, sizes), cap cells re-traced
             at a high cap to split long hole-exit orders (> 128 bounces) from trapped
             beams (intensity cutoff), nominal tilt (deepest point of the largest
             plateau region), best region and target tilt (deepest point of its largest
             region), blind state (deepest point of the <= 2 bounce region)
    select   3 setups that, with the measured one, have separations >= min_sep_gap
             apart, a plateau region >= min_plateau_deg2, a compact best region inside
             the box (not touching the limits, <= max_best_deg2) and best regions
             >= min_best_dist apart; blind value V = max_T mean_s P_s(T) / P_max,s over
             the 0.01 deg grid (how well one fixed target tilt serves all four setups)
             <= max_blind_value; among those the highest score sum (fallback without
             any: the lowest V)

Coarse eligibility (for refine seeds): plateau >= min_plateau_deg2, best region
<= max_best_deg2 with its centroid >= 0.3 deg inside the limits, best bounce count
>= 1.5x the plateau order. Score (coarse and final): min(plateau area, 2 deg^2)
+ log2(best / plateau) (bounces coarse, path final) + 0.25 per hole-exit level above
the plateau (coarse: >= 2 cells, final: >= 0.005 deg^2).

Outputs: reports/setups.md, reports/setups.json, figures/setups.png. The chosen setups
are copied by hand into geometry.SETUPS / SETUP_INFO (nominal tilt, plateau order and
path, P_max, target); tests/test_env.py checks those values against the env.
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from geometry import CellGeometry
from herriott_sim import (
    DEVICE,
    S_LASER_PITCH,
    S_LASER_YAW,
    S_M2_PITCH,
    S_M2_YAW,
    S_SEP,
    STOP_CAP,
    STOP_HOLE,
    STOP_NAMES,
)

RL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(RL_DIR, "runs", "setups")
N_STOP = len(STOP_NAMES)


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class SearchConfig:
    """Search grids and selection thresholds."""

    # ── Coarse grid: (lo, hi, step) ──
    sep: Tuple[float, float, float] = (50.0, 200.0, 2.0)  # mm
    pitch: Tuple[float, float, float] = (0.0, 8.0, 0.25)  # deg, laser
    yaw: Tuple[float, float, float] = (0.0, 8.0, 0.25)  # deg, laser
    coarse_step: float = 0.05  # M2 grid (deg)

    # ── Refine: neighbourhoods of the best coarse candidates ──
    band: float = 10.0  # mm, separation bands
    per_band: int = 3  # Seeds per band
    refine_sep: Tuple[float, float] = (2.0, 1.0)  # +- half-width, step (mm)
    refine_tilt: Tuple[float, float] = (0.25, 0.05)  # +- half-width, step (deg)

    # ── Final: fine M2 scan ──
    final_per_band: int = 4
    fine_step: float = 0.01  # deg
    high_cap: int = 1024  # Re-trace of cap cells (the intensity cutoff stops trapped beams)

    # ── Landscape ──
    limit: float = 2.0  # M2 box half-width (deg)
    cap: int = 128
    min_order: int = 20  # Hole-exit orders that count as a pattern
    chunk_states: int = 420_000  # States per sim call (~0.5 GB peak at cap 128)

    # ── Selection ──
    n_new: int = 3
    min_sep_gap: float = 15.0  # mm, between setups and from the measured one
    min_plateau_deg2: float = 0.3
    max_best_deg2: float = 0.25
    min_best_dist: float = 0.5  # deg, between best regions
    best_margin: float = 0.3  # deg, coarse: best centroid inside the limits by this
    max_blind_value: float = 0.5  # A fixed target reaches <= half of P_max on average


MEASURED = (88.0, 4.08, 4.82)  # (separation, laser pitch, laser yaw) of CellGeometry()


def axis(lo: float, hi: float, step: float) -> np.ndarray:
    """Inclusive float64 grid lo..hi."""
    n = int(round((hi - lo) / step)) + 1
    return lo + step * np.arange(n, dtype=np.float64)


def box_grid(limit: float, step: float, device: torch.device) -> Tuple[torch.Tensor, int]:
    """(G, 2) M2 tilts on the box grid, pitch-major (row = pitch, col = yaw), and W."""
    a = torch.tensor(axis(-limit, limit, step), dtype=torch.float32, device=device)
    P, Y = torch.meshgrid(a, a, indexing="ij")
    return torch.stack((P.flatten(), Y.flatten()), 1), a.numel()


def setup_geometry(sep: float, pitch: float, yaw: float, nominal=None, name: str = "") -> CellGeometry:
    """CellGeometry of a candidate: measured cell with separation and laser tilt replaced."""
    g = replace(CellGeometry(), separation=float(sep), laser_pitch=float(pitch),
                laser_yaw=float(yaw))
    if nominal is not None:
        g = replace(g, m2_pitch=float(nominal[0]), m2_yaw=float(nominal[1]))
    if name and hasattr(g, "name"):
        g = replace(g, name=name)
    return g


# ── Scan ────────────────────────────────────────────────────────────────────


def candidate_states(geom: CellGeometry, cands: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """(n * G, 13) states: candidates (n, 3) [sep, laser pitch, laser yaw] x M2 grid."""
    n, G = cands.shape[0], grid.shape[0]
    s = geom.base_state(n * G, grid.device).view(n, G, -1)
    s[..., S_SEP] = cands[:, 0, None]
    s[..., S_LASER_PITCH] = cands[:, 1, None]
    s[..., S_LASER_YAW] = cands[:, 2, None]
    s[..., S_M2_PITCH] = grid[:, 0]
    s[..., S_M2_YAW] = grid[:, 1]
    return s.view(n * G, -1)


def _masked_centroid(mask: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """(n, 2) mean tilt of the masked cells (NaN where empty)."""
    m = mask.float()
    return (m @ grid) / m.sum(1, keepdim=True)


@torch.no_grad()
def scan_stats(
    sim, geom: CellGeometry, cands: torch.Tensor, grid: torch.Tensor, cfg: SearchConfig
) -> Dict[str, torch.Tensor]:
    """Per-candidate landscape statistics over the M2 grid (on the device).

    hist / hist_hole (n, cap + 1): bounce histogram over all cells / hole exits;
    stop_hist (n, N_STOP); plateau_order, plateau_cells, plateau_c (n, 2): most common
    hole-exit order >= min_order, its cells and centroid (0 / 0 / NaN if none);
    top_order, top_cells, top_path, top_c: highest hole-exit order with >= 2 cells;
    best_bounces, best_cells, best_c: highest bounce count incl. the cap; pmax,
    pmax_tilt (n, 2): longest path and its first cell; first_hit (n, 2): M2-local
    position of the first hit at the grid centre (NaN if the beam misses M2).
    """
    cap, mo = cfg.cap, cfg.min_order
    n, G = cands.shape[0], grid.shape[0]
    dev = grid.device
    out = sim.simulate(candidate_states(geom, cands, grid), store_world=False)
    b = out["hit_counts"].view(n, G)
    st = out["stop_reason"].view(n, G).long()
    p = out["total_path_length"].view(n, G)
    centre = G // 2
    fm = out["hit_mirror"].view(n, G, -1)[:, centre, 0]
    fxy = out["hit_local_xy"].view(n, G, -1, 2)[:, centre, 0]
    del out
    first_hit = torch.where((fm == 2)[:, None], fxy, math.nan)

    hole = st == STOP_HOLE
    row = torch.arange(n, device=dev)[:, None] * (cap + 1)
    idx = (row + b).flatten()
    hist = torch.bincount(idx, minlength=n * (cap + 1)).view(n, cap + 1)
    hist_hole = torch.bincount(idx, weights=hole.flatten().float(), minlength=n * (cap + 1))
    hist_hole = hist_hole.view(n, cap + 1).round().long()
    stop_hist = torch.bincount(
        (torch.arange(n, device=dev)[:, None] * N_STOP + st).flatten(), minlength=n * N_STOP
    ).view(n, N_STOP)

    pc, po = hist_hole[:, mo:].max(1)
    plateau_order = torch.where(pc > 0, po + mo, 0)
    plateau_c = _masked_centroid(hole & (b == plateau_order[:, None]) & (pc[:, None] > 0), grid)

    ords = torch.arange(cap + 1, device=dev)
    top_order = torch.where(hist_hole >= 2, ords, 0).amax(1)
    tmask = hole & (b == top_order[:, None]) & (top_order[:, None] > 0)
    top_cells = tmask.sum(1)

    best_b = b.amax(1)
    bmask = b == best_b[:, None]
    pmax, imax = p.max(1)
    return {
        "hist": hist.to(torch.int32),
        "hist_hole": hist_hole.to(torch.int32),
        "stop_hist": stop_hist.to(torch.int32),
        "plateau_order": plateau_order,
        "plateau_cells": pc,
        "plateau_c": plateau_c,
        "top_order": top_order,
        "top_cells": top_cells,
        "top_path": torch.where(tmask, p, 0.0).sum(1) / top_cells.clamp(min=1),
        "top_c": _masked_centroid(tmask, grid),
        "best_bounces": best_b,
        "best_cells": bmask.sum(1),
        "best_c": _masked_centroid(bmask, grid),
        "pmax": pmax,
        "pmax_tilt": grid[imax],
        "first_hit": first_hit,
    }


def scan_many(
    sim, geom: CellGeometry, cands: np.ndarray, step: float, cfg: SearchConfig, label: str
) -> Dict[str, np.ndarray]:
    """scan_stats over all candidates in chunks; numpy arrays incl. cands, step, scan_s."""
    dev = sim.device
    grid, _ = box_grid(cfg.limit, step, dev)
    per = max(1, cfg.chunk_states // grid.shape[0])
    ct = torch.tensor(cands, dtype=torch.float32, device=dev)
    parts: Dict[str, List[np.ndarray]] = {}
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for i in range(0, len(cands), per):
        for k, v in scan_stats(sim, geom, ct[i : i + per], grid, cfg).items():
            parts.setdefault(k, []).append(v.cpu().numpy())
    res = {k: np.concatenate(v) for k, v in parts.items()}
    res["cands"] = cands.astype(np.float64)
    res["scan_s"] = np.array(time.perf_counter() - t0)
    res["step"] = np.array(step)
    res["peak_gb"] = np.array(torch.cuda.max_memory_allocated(dev) / 2**30)
    print(f"  {label}: {len(cands)} candidates x {grid.shape[0]} M2 cells in "
          f"{float(res['scan_s']):.1f} s")
    return res


# ── Cache ───────────────────────────────────────────────────────────────────


def load_or_run(name: str, key: Dict[str, Any], force: bool, fn) -> Dict[str, np.ndarray]:
    """Cached stage result (runs/setups/<name>.npz): reruns when forced or the key changed."""
    path = os.path.join(OUT_DIR, f"{name}.npz")
    kstr = json.dumps(key, sort_keys=True, default=str)
    if not force and os.path.exists(path):
        d = dict(np.load(path, allow_pickle=False))
        if str(d.pop("_key")) == kstr:
            print(f"{name}: cached ({os.path.relpath(path, RL_DIR)})")
            return d
    print(f"{name}: running")
    d = fn()
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, _key=np.array(kstr), **d)
    return d


# ── Coarse and refine ───────────────────────────────────────────────────────


def _digest(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def coarse_candidates(cfg: SearchConfig) -> np.ndarray:
    S, P, Y = np.meshgrid(axis(*cfg.sep), axis(*cfg.pitch), axis(*cfg.yaw), indexing="ij")
    return np.stack((S.ravel(), P.ravel(), Y.ravel()), 1)


def coarse_score(d: Dict[str, np.ndarray], cfg: SearchConfig) -> np.ndarray:
    """Score of scan_many results; -inf where not eligible (module docstring)."""
    cell = float(d["step"]) ** 2
    area = d["plateau_cells"] * cell
    po = d["plateau_order"]
    bb = d["best_bounces"]
    inner = np.abs(np.nan_to_num(d["best_c"], nan=99.0)).max(1) <= cfg.limit - cfg.best_margin
    ok = ((po >= cfg.min_order) & (area >= cfg.min_plateau_deg2)
          & (d["best_cells"] * cell <= cfg.max_best_deg2) & inner & (bb >= 1.5 * po))
    ords = np.arange(d["hist_hole"].shape[1])
    levels = ((d["hist_hole"] >= 2) & (ords[None, :] > po[:, None])).sum(1)
    s = np.minimum(area, 2.0) + np.log2(np.maximum(bb / np.maximum(po, 1), 1.0))
    return np.where(ok, s + 0.25 * np.minimum(levels, 6), -np.inf)


def _band_top(c: np.ndarray, score: np.ndarray, cfg: SearchConfig, per: int) -> List[int]:
    """Indices of the per best-scoring candidates in each separation band, skipping
    near-duplicates (separation within 1 mm and laser tilt within 0.15 deg)."""
    out = []
    for lo in np.arange(cfg.sep[0], cfg.sep[1] + 1e-9, cfg.band):
        inb = np.nonzero((c[:, 0] >= lo - 1e-9) & (c[:, 0] < lo + cfg.band - 1e-9)
                         & np.isfinite(score))[0]
        chosen: List[int] = []
        for i in inb[np.argsort(-score[inb], kind="stable")]:
            if any(abs(c[i, 0] - c[j, 0]) <= 1 and np.abs(c[i, 1:] - c[j, 1:]).max() <= 0.15
                   for j in chosen):
                continue
            chosen.append(int(i))
            if len(chosen) == per:
                break
        out += chosen
    return out


def refine_candidates(coarse: Dict[str, np.ndarray], cfg: SearchConfig) -> np.ndarray:
    """Neighbourhoods of the best eligible coarse candidates per separation band."""
    c = coarse["cands"]
    seeds = _band_top(c, coarse_score(coarse, cfg), cfg, cfg.per_band)
    (hs, ss), (ht, ts) = cfg.refine_sep, cfg.refine_tilt
    out = [np.zeros((0, 3))]
    for i in seeds:
        s0, p0, y0 = c[i]
        S, P, Y = np.meshgrid(axis(s0 - hs, s0 + hs, ss), axis(p0 - ht, p0 + ht, ts),
                              axis(y0 - ht, y0 + ht, ts), indexing="ij")
        out.append(np.stack((S.ravel(), P.ravel(), Y.ravel()), 1))
    allc = np.round(np.concatenate(out), 4)
    allc = allc[(allc[:, 1] >= 0) & (allc[:, 2] >= 0)]
    return np.unique(allc, axis=0)


def finalist_candidates(coarse, refine, cfg: SearchConfig) -> np.ndarray:
    """Measured setup first, then the best few per band over coarse + refine."""
    keys = ("cands", "plateau_cells", "plateau_order", "best_bounces", "best_cells",
            "best_c", "hist_hole")
    allc = {k: np.concatenate((coarse[k], refine[k])) for k in keys}
    allc["step"] = coarse["step"]
    idx = _band_top(allc["cands"], coarse_score(allc, cfg), cfg, cfg.final_per_band)
    return np.concatenate((np.array([MEASURED]), allc["cands"][idx]))


# ── Final: fine landscape ───────────────────────────────────────────────────


def _components(mask: np.ndarray):
    """4-connected components: (labels, sizes), sizes[0] = 0 (background)."""
    from scipy import ndimage

    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    return lab, sizes


def _deepest(mask: np.ndarray) -> Tuple[int, int]:
    """Cell of mask farthest from its complement (the box edge counts as outside)."""
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return divmod(int(np.argmax(dist)), mask.shape[1])


def _bbox(mask: np.ndarray, ax: np.ndarray) -> List[float]:
    """[pitch_lo, pitch_hi, yaw_lo, yaw_hi] of the masked cells (deg)."""
    r = np.nonzero(mask.any(1))[0]
    c = np.nonzero(mask.any(0))[0]
    return [float(ax[r[0]]), float(ax[r[-1]]), float(ax[c[0]]), float(ax[c[-1]])]


@torch.no_grad()
def fine_maps(sim, sim_hi, geom: CellGeometry, cands: np.ndarray, cfg: SearchConfig):
    """(n, W, W) bounces / stop / path at the cap, bounces / stop of the cap cells re-traced
    at high_cap, and (n, 2) first M2 hit at zero tilt (NaN if it misses M2)."""
    dev = sim.device
    grid, W = box_grid(cfg.limit, cfg.fine_step, dev)
    maps: Dict[str, List[np.ndarray]] = {k: [] for k in
                                         ("bounces", "stop", "path", "bounces_hi", "stop_hi")}
    first = []
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for cand in cands:
        ct = torch.tensor(cand[None], dtype=torch.float32, device=dev)
        out = sim.simulate(candidate_states(geom, ct, grid), store_world=False)
        b, st, p = out["hit_counts"], out["stop_reason"].long(), out["total_path_length"]
        c = grid.shape[0] // 2
        ok = int(out["hit_mirror"][c, 0]) == 2
        first.append(out["hit_local_xy"][c, 0].cpu().numpy() if ok else np.full(2, np.nan))
        del out
        b_hi, st_hi = b.clone(), st.clone()
        idx = torch.nonzero(st == STOP_CAP).squeeze(1)
        per = max(1, cfg.chunk_states * cfg.cap // cfg.high_cap)
        for i in range(0, idx.numel(), per):
            j = idx[i : i + per]
            o = sim_hi.simulate(candidate_states(geom, ct, grid[j]), store_world=False)
            b_hi[j] = o["hit_counts"]
            st_hi[j] = o["stop_reason"].long()
            del o
        for k, v, t in (("bounces", b, np.int16), ("stop", st, np.int8), ("path", p, np.float32),
                        ("bounces_hi", b_hi, np.int16), ("stop_hi", st_hi, np.int8)):
            maps[k].append(v.view(W, W).cpu().numpy().astype(t))
    res = {k: np.stack(v) for k, v in maps.items()}
    res["first_hit"] = np.stack(first)
    res["cands"] = cands
    res["scan_s"] = np.array(time.perf_counter() - t0)
    res["peak_gb"] = np.array(torch.cuda.max_memory_allocated(dev) / 2**30)
    print(f"  final: {len(cands)} candidates x {W}^2 M2 cells in {float(res['scan_s']):.1f} s")
    return res


def landscape(F: Dict[str, np.ndarray], k: int, cfg: SearchConfig) -> Dict[str, Any]:
    """Landscape summary of finalist k from the fine maps."""
    ax = axis(-cfg.limit, cfg.limit, cfg.fine_step)
    cell = cfg.fine_step**2
    b, st, p = F["bounces"][k], F["stop"][k], F["path"][k]
    hole = st == STOP_HOLE
    cap_m = st == STOP_CAP
    long_m = cap_m & (F["stop_hi"][k] == STOP_HOLE)
    trap_m = cap_m & ~long_m

    # Hole-exit orders >= min_order: area, connected regions, largest region
    orders = []
    plateau, plateau_mask = None, None
    for o in np.unique(b[hole & (b >= cfg.min_order)]):
        m = hole & (b == o)
        lab, sizes = _components(m)
        j = int(np.argmax(sizes))
        big = lab == j
        orders.append({
            "order": int(o), "area_deg2": float(m.sum() * cell), "n_regions": int(len(sizes) - 1),
            "largest_deg2": float(sizes[j] * cell), "bbox": _bbox(big, ax),
            "path_mm": float(np.median(p[m])),
        })
        if plateau is None or orders[-1]["largest_deg2"] > plateau["largest_deg2"]:
            plateau, plateau_mask = orders[-1], big
    nominal = None
    if plateau is not None:
        i, j = _deepest(plateau_mask)
        nominal = [round(float(ax[i]), 2), round(float(ax[j]), 2)]

    # Best: highest bounce count incl. the cap; target = deepest point of its largest region
    bmax = int(b.max())
    bm = b == bmax
    lab, sizes = _components(bm)
    j = int(np.argmax(sizes))
    ti, tj = _deepest(lab == j)
    edge = bool(bm[0].any() or bm[-1].any() or bm[:, 0].any() or bm[:, -1].any())
    blind = b <= 2
    bi, bj = _deepest(blind) if blind.any() else (0, 0)

    pmax = float(p.max())
    p_plat = plateau["path_mm"] if plateau else 0.0
    levels = sum(1 for o in orders if plateau and o["order"] > plateau["order"]
                 and o["area_deg2"] >= 0.005)
    fh = F["first_hit"][k]
    L = {
        "sep": float(F["cands"][k, 0]), "laser_pitch": float(F["cands"][k, 1]),
        "laser_yaw": float(F["cands"][k, 2]),
        "plateau": plateau, "nominal": nominal, "levels": levels, "orders": orders,
        "pmax_mm": pmax, "gain": pmax / p_plat if p_plat > 0 else 0.0,
        "best_bounces": bmax, "best_deg2": float(bm.sum() * cell),
        "best_regions": int(len(sizes) - 1), "best_largest_bbox": _bbox(lab == j, ax),
        "target": [round(float(ax[ti]), 2), round(float(ax[tj]), 2)],
        "target_path_mm": float(p[ti, tj]), "best_on_edge": edge,
        "cap_cells": int(cap_m.sum()), "cap_long": int(long_m.sum()),
        "cap_trapped": int(trap_m.sum()),
        "long_orders": sorted(int(v) for v in np.unique(F["bounces_hi"][k][long_m])),
        "blind_frac": float(blind.mean()),
        "blind_state": [round(float(ax[bi]), 2), round(float(ax[bj]), 2)],
        "first_hit": None if np.isnan(fh).any() else [round(float(v), 3) for v in fh],
        "area20_deg2": float((hole & (b >= cfg.min_order)).sum() * cell),
        "stop_frac": {STOP_NAMES[s]: float((st == s).mean()) for s in range(N_STOP)},
    }
    if k == 0 and tuple(F["cands"][0]) == MEASURED:
        g = CellGeometry()
        nominal = L["nominal"] = [g.m2_pitch, g.m2_yaw]  # Measured nominal, unchanged
    if nominal is not None:
        L["nominal_bounces"] = int(b[i_of(ax, nominal[0]), i_of(ax, nominal[1])])
        L["nominal_path_mm"] = float(p[i_of(ax, nominal[0]), i_of(ax, nominal[1])])
    L["score"] = final_score(L)
    return L


def i_of(ax: np.ndarray, v: float) -> int:
    return int(np.argmin(np.abs(ax - v)))


def final_score(L: Dict[str, Any]) -> float:
    """min(plateau region, 2 deg^2) + log2(P_max / P_plateau) + 0.25 per level above."""
    if L["plateau"] is None:
        return -math.inf
    return (min(L["plateau"]["largest_deg2"], 2.0) + math.log2(max(L["gain"], 1.0))
            + 0.25 * min(L["levels"], 6))


# ── Selection ───────────────────────────────────────────────────────────────


def best_distance(F: Dict[str, np.ndarray], cfg: SearchConfig) -> np.ndarray:
    """(n, n) min distance (deg) between the best-region cells of the finalists."""
    from scipy.spatial import cKDTree

    ax = axis(-cfg.limit, cfg.limit, cfg.fine_step)
    pts = []
    for b in F["bounces"]:
        i, j = np.nonzero(b == b.max())
        pts.append(np.stack((ax[i], ax[j]), 1))
    trees = [cKDTree(q) for q in pts]
    n = len(pts)
    D = np.zeros((n, n))
    for a in range(n):
        for c in range(a + 1, n):
            small, big = (a, c) if len(pts[a]) < len(pts[c]) else (c, a)
            D[a, c] = D[c, a] = float(trees[big].query(pts[small])[0].min())
    return D


def eligible(L: Dict[str, Any], dist0: float, cfg: SearchConfig) -> str:
    """'' if finalist L can be a new setup, else the first failed criterion."""
    if L["plateau"] is None or L["plateau"]["largest_deg2"] < cfg.min_plateau_deg2:
        return "plateau"
    if L["best_on_edge"]:
        return "best on edge"
    if L["best_deg2"] > cfg.max_best_deg2:
        return "best too large"
    if abs(L["sep"] - MEASURED[0]) < cfg.min_sep_gap:
        return "sep near 88"
    if dist0 < cfg.min_best_dist:
        return "best near measured"
    return ""


@torch.no_grad()
def select(F, Ls, D, cfg: SearchConfig) -> Tuple[List[int], List[Dict[str, Any]], List[str]]:
    """Chosen finalist indices (measured = 0 first) and the ranked valid triples."""
    reasons = [eligible(L, D[0, k], cfg) if k else "measured" for k, L in enumerate(Ls)]
    E = [k for k, r in enumerate(reasons) if r == ""]
    dev = DEVICE
    f = torch.tensor(F["path"].reshape(len(Ls), -1), device=dev)
    f = f / f.amax(1, keepdim=True)  # P / P_max per finalist
    triples = []
    for a_i, a in enumerate(E):
        for b_i in range(a_i + 1, len(E)):
            b = E[b_i]
            if abs(Ls[a]["sep"] - Ls[b]["sep"]) < cfg.min_sep_gap or D[a, b] < cfg.min_best_dist:
                continue
            fab = f[0] + f[a] + f[b]
            cs = [c for c in E[b_i + 1:]
                  if abs(Ls[c]["sep"] - Ls[a]["sep"]) >= cfg.min_sep_gap
                  and abs(Ls[c]["sep"] - Ls[b]["sep"]) >= cfg.min_sep_gap
                  and D[a, c] >= cfg.min_best_dist and D[b, c] >= cfg.min_best_dist]
            if not cs:
                continue
            v = ((fab[None] + f[cs]) / 4).amax(1).tolist()
            for c, vc in zip(cs, v):
                triples.append({"idx": [a, b, c], "blind_value": vc,
                                "score": Ls[a]["score"] + Ls[b]["score"] + Ls[c]["score"]})
    assert triples, "no valid triple; relax the selection thresholds"
    if any(t["blind_value"] <= cfg.max_blind_value for t in triples):
        triples.sort(key=lambda t: (t["blind_value"] > cfg.max_blind_value, -t["score"],
                                    t["blind_value"]))
    else:
        triples.sort(key=lambda t: (t["blind_value"], -t["score"]))
    t = triples[0]
    chosen = [0] + sorted(t["idx"], key=lambda k: Ls[k]["sep"])
    return chosen, triples, reasons


def blind_value(F, idx: List[int], cfg: SearchConfig) -> Dict[str, Any]:
    """max_T mean_s P_s(T) / P_max,s over the setups idx, the argmax T and the per-setup
    fractions there."""
    ax = axis(-cfg.limit, cfg.limit, cfg.fine_step)
    f = np.stack([F["path"][k] / F["path"][k].max() for k in idx])
    m = f.mean(0)
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    return {"value": float(m[i, j]), "tilt": [round(float(ax[i]), 2), round(float(ax[j]), 2)],
            "per_setup": [float(v) for v in f[:, i, j]]}


# ── Report ──────────────────────────────────────────────────────────────────


def _bb(b: List[float]) -> str:
    return f"{b[1] - b[0] + 0.01:.2f} x {b[3] - b[2] + 0.01:.2f}"


def landscape_table(L: Dict[str, Any], F, k: int, cfg: SearchConfig) -> List[str]:
    """Markdown rows: bounces | path | leaves | area | regions | largest region."""
    b, st = F["bounces"][k], F["stop"][k]
    cell = cfg.fine_step**2
    n = b.size
    rows = ["| bounces | path (m) | how the beam leaves | area (deg^2, % of box) | regions "
            "| largest region (pitch x yaw deg) |", "|---|---|---|---|---|---|"]

    def pct(a: float) -> str:
        return f"{a:.3f} ({100 * a / (n * cell):.1f} %)"

    rows.append(f"| 1-2 | - | misses a mirror (blind) | {pct((b <= 2).sum() * cell)} | - | - |")
    low = (b >= 3) & (b < cfg.min_order)
    rows.append(f"| 3-{cfg.min_order - 1} | - | misses / blocked / short hole exits | "
                f"{pct(low.sum() * cell)} | - | - |")
    small = [o for o in L["orders"] if o["area_deg2"] < 0.001]
    for o in L["orders"]:
        if o["area_deg2"] < 0.001:
            continue
        tag = " (plateau)" if L["plateau"] and o["order"] == L["plateau"]["order"] else ""
        rows.append(f"| {o['order']}{tag} | {o['path_mm'] / 1000:.2f} | M1 hole | "
                    f"{pct(o['area_deg2'])} | {o['n_regions']} | {_bb(o['bbox'])} |")
    if small:
        rows.append(f"| {', '.join(str(o['order']) for o in small)} | - | M1 hole (edges) | "
                    f"{pct(sum(o['area_deg2'] for o in small))} | - | - |")
    other = (st != STOP_HOLE) & (st != STOP_CAP) & (b >= cfg.min_order)
    if other.any():
        rows.append(f"| {cfg.min_order}+ | - | misses / blocked after >= {cfg.min_order} | "
                    f"{pct(other.sum() * cell)} | - | - |")
    if L["cap_cells"]:
        lo = L["long_orders"]
        at = "-" if not lo else (f"{lo[0]}" if lo[0] == lo[-1] else f"{lo[0]}-{lo[-1]}")
        rows.append(f"| cap ({cfg.cap}) | {L['pmax_mm'] / 1000:.2f} | hole at {at} (long "
                    f"orders) | {pct(L['cap_long'] * cell)} | - | - |")
        rows.append(f"| cap ({cfg.cap}) | {L['pmax_mm'] / 1000:.2f} | never (trapped, "
                    f"intensity cutoff) | {pct(L['cap_trapped'] * cell)} | - | - |")
    return rows


def write_report(res: Dict[str, Any], F, cfg: SearchConfig, path_md: str):
    Ls, chosen = res["finalists"], res["chosen"]
    w = []
    w.append("# Multi-setup search\n")
    w.append(f"Generated by `uv run python find_setups.py` ({res['date']}); cache in "
             "`runs/setups/`, raw numbers in `setups.json`, figure `figures/setups.png`. "
             "Mirrors, laser offset (0.75, 1.04) mm and M2 offsets as measured, M1 untilted; "
             f"M2 box +-{cfg.limit:g} deg, cap {cfg.cap}. Search time: coarse "
             f"{res['time']['coarse_s']:.0f} s ({res['n']['coarse']} candidates x "
             f"{int(round(2 * cfg.limit / cfg.coarse_step)) + 1}^2 M2 cells), refine "
             f"{res['time']['refine_s']:.0f} s ({res['n']['refine']}), final "
             f"{res['time']['final_s']:.0f} s ({res['n']['final']} x "
             f"{int(round(2 * cfg.limit / cfg.fine_step)) + 1}^2), selection "
             f"{res['time']['select_s']:.0f} s; total {res['time']['total_s']:.0f} s "
             f"(peak GPU memory {res['peak_mem_gb']:.2f} GB).\n")
    w.append(f"Grids: separation {cfg.sep[0]:g}-{cfg.sep[1]:g} mm step {cfg.sep[2]:g}, laser "
             f"pitch / yaw {cfg.pitch[0]:g}-{cfg.pitch[1]:g} deg step {cfg.pitch[2]:g}; refine "
             f"+-{cfg.refine_sep[0]:g} mm step {cfg.refine_sep[1]:g} and +-{cfg.refine_tilt[0]:g} "
             f"deg step {cfg.refine_tilt[1]:g} around the best {cfg.per_band} eligible coarse "
             f"candidates per {cfg.band:g} mm band; final 0.01 deg scans of the best "
             f"{cfg.final_per_band} per band. Coarse: {res['n']['coarse_plateau']} candidates "
             f"have a hole-exit plateau >= {cfg.min_order} bounces of >= "
             f"{cfg.min_plateau_deg2:g} deg^2, {res['n']['coarse_eligible']} are eligible "
             "(compact best region inside the box, best >= 1.5x the plateau order).\n")
    w.append("Selection (all four setups): separations >= "
             f"{cfg.min_sep_gap:g} mm apart, plateau region >= {cfg.min_plateau_deg2:g} deg^2, "
             f"best region (highest bounce count, incl. the cap) <= {cfg.max_best_deg2:g} deg^2 "
             f"and not touching the limits, best regions >= {cfg.min_best_dist:g} deg apart "
             "(min distance between their cells), blind value V = max_T mean_s P_s(T) / "
             "P_max,s <= " f"{cfg.max_blind_value:g}" " (the best a single fixed target tilt "
             "can do over the four setups, path fraction only, 0.01 deg grid); among those the "
             "highest score sum.\n")

    w.append("## Chosen setups\n")
    w.append("| # | name | sep (mm) | laser pitch / yaw (deg) | nominal M2 (deg) | plateau | "
             "plateau region (deg^2) | best | P_max (m) | best region (deg^2) | target M2 "
             "(deg) | cap: long / trapped | first M2 hit (mm) | blind % |")
    w.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r, (k, name) in enumerate(zip(chosen, res["names"])):
        L = Ls[k]
        fh = L["first_hit"]
        w.append(f"| {r} | {name} | {L['sep']:g} | {L['laser_pitch']:.2f} / {L['laser_yaw']:.2f} | "
                 f"({L['nominal'][0]:+.2f}, {L['nominal'][1]:+.2f}) | {L['plateau']['order']} | "
                 f"{L['plateau']['largest_deg2']:.2f} | {L['best_bounces']} | "
                 f"{L['pmax_mm'] / 1000:.2f} | {L['best_deg2']:.3f} | "
                 f"({L['target'][0]:+.2f}, {L['target'][1]:+.2f}) | {L['cap_long']} / "
                 f"{L['cap_trapped']} | ({fh[0]:.2f}, {fh[1]:.2f}) | {100 * L['blind_frac']:.1f} |")
    w.append("")
    D = res["best_dist"]
    w.append("Min distance between best regions (deg):\n")
    w.append("| | " + " | ".join(res["names"]) + " |")
    w.append("|---|" + "---|" * len(chosen))
    for r, a in enumerate(chosen):
        w.append(f"| {res['names'][r]} | " + " | ".join(
            "-" if a == c else f"{D[a][c]:.2f}" for c in chosen) + " |")
    bv = res["blind"]
    w.append(f"\nBlind value V = {bv['value']:.3f} at M2 ({bv['tilt'][0]:+.2f}, "
             f"{bv['tilt'][1]:+.2f}) deg (P / P_max there: "
             + ", ".join(f"{v:.2f}" for v in bv["per_setup"]) + "). Cap cells: with cap "
             f"{cfg.cap} a beam that leaves through the hole after more than {cfg.cap} bounces "
             "(long order) and a trapped beam both score the cap path; the re-trace at cap "
             f"{cfg.high_cap} separates them (trapped beams stop at the intensity cutoff).\n")

    for r, k in enumerate(chosen):
        L = Ls[k]
        w.append(f"### {r}. {res['names'][r]}: {L['sep']:g} mm, laser "
                 f"({L['laser_pitch']:.2f}, {L['laser_yaw']:.2f}) deg\n")
        w.extend(landscape_table(L, F, k, cfg))
        w.append(f"\nNominal M2 ({L['nominal'][0]:+.2f}, {L['nominal'][1]:+.2f}): "
                 f"{L['nominal_bounces']} bounces, {L['nominal_path_mm'] / 1000:.2f} m. Best: "
                 f"{L['best_bounces']} bounces in {L['best_regions']} region(s), largest "
                 f"{_bb(L['best_largest_bbox'])} deg, target ({L['target'][0]:+.2f}, "
                 f"{L['target'][1]:+.2f}). Blind state ({L['blind_state'][0]:+.2f}, "
                 f"{L['blind_state'][1]:+.2f}).\n")

    w.append("## Ranked finalists\n")
    w.append("Score = min(plateau region, 2) + log2(P_max / P_plateau) + 0.25 per hole-exit "
             "level above the plateau (>= 0.005 deg^2). Status: chosen, eligible, or the first "
             "failed criterion. Dist: min distance of the best region to the measured setup's "
             "(deg).\n")
    w.append("| sep | laser p / y | plateau (deg^2) | levels | best | P_max (m) | best deg^2 "
             "| target | cap long / trapped | dist | score | status |")
    w.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = sorted(range(1, len(Ls)), key=lambda k: -Ls[k]["score"])
    for k in order:
        L = Ls[k]
        pl = L["plateau"]
        st = "chosen" if k in chosen else (res["reasons"][k] or "eligible")
        w.append(f"| {L['sep']:g} | {L['laser_pitch']:.2f} / {L['laser_yaw']:.2f} | "
                 f"{pl['order'] if pl else '-'} ({pl['largest_deg2'] if pl else 0:.2f}) | "
                 f"{L['levels']} | {L['best_bounces']} | {L['pmax_mm'] / 1000:.2f} | "
                 f"{L['best_deg2']:.3f} | ({L['target'][0]:+.2f}, {L['target'][1]:+.2f}) | "
                 f"{L['cap_long']} / {L['cap_trapped']} | {D[0][k]:.2f} | {L['score']:.2f} | "
                 f"{st} |")
    w.append("")
    w.append("Valid triples in selection order (V <= max, then score sum):\n")
    w.append("| separations (mm) | blind value V | score sum |")
    w.append("|---|---|---|")
    for t in res["triples"][:8]:
        seps = sorted(Ls[k]["sep"] for k in t["idx"])
        lt = " / ".join(f"{s:g} ({Ls[k]['laser_pitch']:.2f}, {Ls[k]['laser_yaw']:.2f})"
                        for s, k in zip(seps, sorted(t["idx"], key=lambda k: Ls[k]["sep"])))
        w.append(f"| {lt} | {t['blind_value']:.3f} | {t['score']:.2f} |")
    mt = res["min_blind_triple"]
    w.append(f"\n{res['n_triples']} valid triples, {res['n_triples_ok']} with V <= "
             f"{cfg.max_blind_value:g}. Lowest V: " + " / ".join(
                 f"{Ls[k]['sep']:g} ({Ls[k]['laser_pitch']:.2f}, {Ls[k]['laser_yaw']:.2f})"
                 for k in sorted(mt["idx"], key=lambda k: Ls[k]["sep"]))
             + f", V {mt['blind_value']:.3f}, score sum {mt['score']:.2f}.")
    with open(path_md, "w") as fh:
        fh.write("\n".join(w) + "\n")


def save_figure(res: Dict[str, Any], F, cfg: SearchConfig, path: str):
    """Per setup: bounce map with nominal / target / blind markers, 64 px renders."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle

    from env import SpotRenderer

    ramp = ["#f7f9fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("bounces", ramp)
    Ls, chosen = res["finalists"], res["chosen"]
    n = len(chosen)
    fig, axes = plt.subplots(n, 4, figsize=(13.5, 3.35 * n), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    lim = cfg.limit
    dev = DEVICE
    for r, k in enumerate(chosen):
        L = Ls[k]
        g = setup_geometry(L["sep"], L["laser_pitch"], L["laser_yaw"], L["nominal"])
        sim = g.make_sim(dev, max_bounces=cfg.cap)
        rend = SpotRenderer(64, g.diameter, 0.5, 1.6, g.reflectivity, cfg.cap, 0, dev)
        states = {"plateau (nominal)": L["nominal"], "best (target)": L["target"],
                  "blind": L["blind_state"]}
        tl = torch.tensor(list(states.values()), dtype=torch.float32, device=dev)
        s = g.base_state(3, dev)
        s[:, S_M2_PITCH], s[:, S_M2_YAW] = tl[:, 0], tl[:, 1]
        out = sim.simulate(s, store_world=False)
        img = rend(out["hit_mirror"], out["hit_local_xy"]).cpu().numpy()

        ax = axes[r, 0]
        im = ax.imshow(F["bounces"][k], cmap=cmap, vmin=0, vmax=cfg.cap, origin="lower",
                       extent=(-lim, lim, -lim, lim), interpolation="nearest")
        # Rows = pitch, cols = yaw: x axis yaw, y axis pitch
        for (name, t), mk, col in zip(states.items(), ("o", "*", "X"),
                                      ("#e8743b", "#d03b3b", "#6b6b6b")):
            ax.plot(t[1], t[0], mk, ms=9 if mk != "*" else 13, mfc=col, mec="white",
                    mew=1.2, label=name, ls="none")
        ax.set_xlabel("M2 yaw (deg)", fontsize=8)
        ax.set_ylabel("M2 pitch (deg)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{res['names'][r]}: {L['sep']:g} mm, laser ({L['laser_pitch']:.2f}, "
                     f"{L['laser_yaw']:.2f}) deg\nplateau {L['plateau']['order']}, best "
                     f"{L['best_bounces']} ({L['pmax_mm'] / 1000:.1f} m)", fontsize=8.5)
        if r == 0:
            ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("bounces (cap 128)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        R = rend.radius
        for c, (name, t) in enumerate(states.items()):
            a2 = axes[r, c + 1]
            a2.imshow(img[c, 0], cmap="gray", vmin=0, vmax=255, origin="upper",
                      extent=(-R, R, -R, R), interpolation="nearest")
            a2.add_patch(Circle((0, 0), R, fill=False, color="0.45", lw=0.6))
            a2.set_title(f"{name}: M2 ({t[0]:+.2f}, {t[1]:+.2f})\n"
                         f"{int(out['hit_counts'][c])} bounces, "
                         f"{float(out['total_path_length'][c]) / 1000:.2f} m, "
                         f"{STOP_NAMES[int(out['stop_reason'][c])]}", fontsize=8)
            a2.tick_params(labelsize=6.5)
            a2.set_xlabel("M2 local x (mm)", fontsize=7)
    fig.suptitle("Setups: bounce count over the M2 box and 64 px observations "
                 "(M2 face from behind, uint8)", fontsize=10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", nargs="*", default=[], choices=["coarse", "refine", "final"],
                    help="stages to rerun")
    args = ap.parse_args()
    cfg = SearchConfig()
    geom = CellGeometry()
    t_all = time.perf_counter()
    sim = geom.make_sim(DEVICE, max_bounces=cfg.cap)
    sim_hi = geom.make_sim(DEVICE, max_bounces=cfg.high_cap)
    torch.cuda.reset_peak_memory_stats()
    # Cache keys: only what a stage's output depends on (selection thresholds excluded)
    base_key = {k: getattr(cfg, k) for k in ("limit", "cap", "min_order", "coarse_step")}
    base_key.update(sep=cfg.sep, pitch=cfg.pitch, yaw=cfg.yaw)

    cc = coarse_candidates(cfg)
    coarse = load_or_run("coarse", {"cfg": base_key, "stage": "coarse"}, "coarse" in args.force,
                         lambda: scan_many(sim, geom, cc, cfg.coarse_step, cfg, "coarse"))
    rc = refine_candidates(coarse, cfg)
    refine = load_or_run("refine", {"cfg": base_key, "stage": "refine", "cands": _digest(rc)},
                         "refine" in args.force,
                         lambda: scan_many(sim, geom, rc, cfg.coarse_step, cfg, "refine"))
    fc = finalist_candidates(coarse, refine, cfg)
    fkey = {"limit": cfg.limit, "cap": cfg.cap, "fine_step": cfg.fine_step,
            "high_cap": cfg.high_cap}
    F = load_or_run("final", {"cfg": fkey, "stage": "final", "cands": _digest(fc)},
                    "final" in args.force, lambda: fine_maps(sim, sim_hi, geom, fc, cfg))

    t0 = time.perf_counter()
    Ls = [landscape(F, k, cfg) for k in range(len(fc))]
    D = best_distance(F, cfg)
    chosen, triples, reasons = select(F, Ls, D, cfg)
    t_sel = time.perf_counter() - t0
    names = ["measured"] + [f"sep{Ls[k]['sep']:g}" for k in chosen[1:]]
    cell = cfg.coarse_step**2
    res = {
        "date": time.strftime("%Y-%m-%d"),
        "cfg": asdict(cfg),
        "names": names,
        "chosen": chosen,
        "finalists": Ls,
        "reasons": reasons,
        "best_dist": D.round(4).tolist(),
        "triples": triples[:50],
        "n_triples": len(triples),
        "n_triples_ok": sum(t["blind_value"] <= cfg.max_blind_value for t in triples),
        "min_blind_triple": min(triples, key=lambda t: (t["blind_value"], -t["score"])),
        "blind": blind_value(F, chosen, cfg),
        "n": {"coarse": len(cc), "refine": len(rc), "final": len(fc),
              "coarse_plateau": int(((coarse["plateau_order"] >= cfg.min_order)
                                     & (coarse["plateau_cells"] * cell >= cfg.min_plateau_deg2)).sum()),
              "coarse_eligible": int(np.isfinite(coarse_score(coarse, cfg)).sum())},
        "time": {"coarse_s": float(coarse["scan_s"]), "refine_s": float(refine["scan_s"]),
                 "final_s": float(F["scan_s"]), "select_s": t_sel},
        "peak_mem_gb": max(torch.cuda.max_memory_allocated() / 2**30,
                           *(float(d.get("peak_gb", 0.0)) for d in (coarse, refine, F))),
    }
    res["time"]["total_s"] = sum(res["time"].values())
    res["setups"] = [
        {"name": nm, "separation": Ls[k]["sep"], "laser_pitch": Ls[k]["laser_pitch"],
         "laser_yaw": Ls[k]["laser_yaw"], "m2_pitch": Ls[k]["nominal"][0],
         "m2_yaw": Ls[k]["nominal"][1], "target": Ls[k]["target"], "pmax_mm": Ls[k]["pmax_mm"]}
        for nm, k in zip(names, chosen)
    ]
    os.makedirs(os.path.join(RL_DIR, "reports"), exist_ok=True)
    with open(os.path.join(RL_DIR, "reports", "setups.json"), "w") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    write_report(res, F, cfg, os.path.join(RL_DIR, "reports", "setups.md"))
    save_figure(res, F, cfg, os.path.join(RL_DIR, "figures", "setups.png"))
    print(f"chosen: {[(Ls[k]['sep'], Ls[k]['laser_pitch'], Ls[k]['laser_yaw']) for k in chosen]}")
    print(f"blind value {res['blind']['value']:.3f}; wall {time.perf_counter() - t_all:.0f} s; "
          f"peak {res['peak_mem_gb']:.2f} GB; wrote reports/setups.md, setups.json, "
          "figures/setups.png")


if __name__ == "__main__":
    main()
