"""
herriott_search.py — Herriott cell parameter search & stability
================================================================

STAGE 1 — Find peak configurations & laser angle tolerance
    For each separation in range:
        Simulate all (laser_pitch x laser_yaw) combos in one batch
        threshold = ceil(MIN_PATH_MM / separation)
        Record:
            - max bounces
            - all (pitch, yaw) pairs above threshold
            - contiguous region (flood-fill from center of winning zone)
            - pitch/yaw tolerance ranges from that contiguous region
    Save: results/stage1.json, results/stage1_maps.npz

STAGE 2 — Mechanical stability with M2 pitch/yaw compensation
    For each winning laser position from Stage 1:
        Build 5D disturbance grid (all offsets from nominal):
            sep_err x laser_dx x laser_dy x m2_tx x m2_ty
        At every disturbance point, tile all m2_pitch x m2_yaw
            compensator combos and keep best bounce count
        Analyze:
            - region where bounces >= threshold
            - flood-fill from center for contiguous stable region
            - marginal tolerances per axis
            - bounce histogram
    Save: results/stage2_summary.json, results/stage2_grids_sep{N}.npz
"""

import torch
import numpy as np
import json, time, os, itertools
from dataclasses import dataclass, asdict, field
from typing import List, Tuple
from tqdm import tqdm
from herriott_sim import create_sim, make_state, STATE_DIM, DEVICE, DTYPE


# ================================================================
# Configuration
# ================================================================

OUTPUT_DIR = "results"
MIN_PATH_MM = 4100.0
BATCH_SIZE = 180_000

# Stage 1: coarse search
SEP_RANGE = (80.0, 100.0, 1.0)  # (min, max, step) mm
LASER_ANGLE_RANGE = (-5.0, 5.0, 0.05)  # (min, max, step) deg

# Stage 2: m2 compensator (actively controlled)
M2_COMP_RANGE = (-3.0, 3.0, 0.2)  # (min, max, step) deg

# Stage 2: mechanical disturbance grids (offsets from nominal)
SEP_ERR_RANGE = (-5.0, 5.0, 0.5)  # mm
LDX_RANGE = (-0.5, 0.5, 0.25)  # mm
LDY_RANGE = (-0.5, 0.5, 0.25)  # mm
M2TX_RANGE = (-0.5, 0.5, 0.25)  # mm
M2TY_RANGE = (-0.5, 0.5, 0.25)  # mm

MAX_LASER_POS = 20  # max winning positions to test per separation


# ================================================================
# Utilities
# ================================================================


def grid1d(lo, hi, step):
    return torch.arange(lo, hi + step * 0.5, step, device=DEVICE, dtype=DTYPE)


def batched_sim(sim, states):
    N = states.shape[0]
    if N <= BATCH_SIZE:
        return sim.simulate(states)["hit_counts"]
    return torch.cat(
        [
            sim.simulate(states[i : i + BATCH_SIZE])["hit_counts"]
            for i in range(0, N, BATCH_SIZE)
        ]
    )


def flood_fill_nd(grid, center):
    """Flood-fill contiguous True region from center in N-D boolean array."""
    visited = np.zeros_like(grid, dtype=bool)
    if not grid[center]:
        return visited
    stack = [center]
    visited[center] = True
    while stack:
        pos = stack.pop()
        for dim in range(grid.ndim):
            for delta in (-1, 1):
                nb = list(pos)
                nb[dim] += delta
                nb = tuple(nb)
                if 0 <= nb[dim] < grid.shape[dim] and not visited[nb] and grid[nb]:
                    visited[nb] = True
                    stack.append(nb)
    return visited


def contiguous_range_1d(proj, vals, center_idx):
    if not proj[center_idx]:
        return [0.0, 0.0]
    lo, hi = center_idx, center_idx
    while lo > 0 and proj[lo - 1]:
        lo -= 1
    while hi < len(vals) - 1 and proj[hi + 1]:
        hi += 1
    return [float(vals[lo]), float(vals[hi])]


def subsample(lst, max_n):
    if len(lst) <= max_n:
        return lst
    step = len(lst) / max_n
    return [lst[int(i * step)] for i in range(max_n)]


def save_json(obj, path):
    def convert(o):
        if hasattr(o, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(o).items()}
        if isinstance(o, (list, tuple)):
            return [convert(i) for i in o]
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)


# ================================================================
# Stage 1: Peak finding + laser angle tolerance
# ================================================================


@dataclass
class Stage1Result:
    separation: float
    max_bounces: int
    bounce_threshold: int  # ceil(MIN_PATH_MM / sep)
    # All (pitch, yaw) above threshold
    winning_positions: List[Tuple[float, float]]
    pitch_range: Tuple[float, float]
    yaw_range: Tuple[float, float]
    # Contiguous region (flood-filled from centroid)
    contiguous_count: int
    contiguous_pitch_range: Tuple[float, float]
    contiguous_yaw_range: Tuple[float, float]
    centroid: Tuple[float, float]


def run_stage1(sim) -> List[Stage1Result]:
    seps = grid1d(*SEP_RANGE)
    pitches = grid1d(*LASER_ANGLE_RANGE)
    yaws = grid1d(*LASER_ANGLE_RANGE)
    pg, yg = torch.meshgrid(pitches, yaws, indexing="ij")
    p_flat, y_flat = pg.reshape(-1), yg.reshape(-1)
    n_p, n_y = len(pitches), len(yaws)
    N_angles = len(p_flat)

    # M2 compensator grid (same as Stage 2)
    cp_1d = grid1d(*M2_COMP_RANGE)
    cy_1d = grid1d(*M2_COMP_RANGE)
    cpg, cyg = torch.meshgrid(cp_1d, cy_1d, indexing="ij")
    comp_p, comp_y = cpg.reshape(-1), cyg.reshape(-1)
    N_comp = comp_p.shape[0]

    N_total = len(seps) * N_angles * N_comp
    print(
        f"  Grid: {len(seps)} seps x {n_p} pitches x {n_y} yaws x {N_comp} comp "
        f"= {N_total:,} total sims\n"
    )

    # -- Run all sims in one flat loop --
    all_counts = torch.zeros(N_total, dtype=torch.long, device=DEVICE)

    for start in tqdm(
        range(0, N_total, BATCH_SIZE),
        desc="  Simulating",
        total=(N_total + BATCH_SIZE - 1) // BATCH_SIZE,
    ):
        end = min(start + BATCH_SIZE, N_total)
        B = end - start
        flat_idx = torch.arange(start, end, device=DEVICE)

        sep_idx = flat_idx // (N_angles * N_comp)
        angle_idx = (flat_idx // N_comp) % N_angles
        comp_idx = flat_idx % N_comp

        states = torch.zeros(B, STATE_DIM, device=DEVICE, dtype=DTYPE)
        states[:, 4] = seps[sep_idx]
        states[:, 11] = p_flat[angle_idx]
        states[:, 12] = y_flat[angle_idx]
        states[:, 2] = comp_p[comp_idx]
        states[:, 3] = comp_y[comp_idx]

        all_counts[start:end] = batched_sim(sim, states)

    # Reshape: (N_seps, N_angles, N_comp) -> max over comp -> (N_seps, N_angles)
    best_per_angle = all_counts.reshape(len(seps), N_angles, N_comp).max(dim=2).values

    # Also compute uncompensated (comp at 0,0 = center of comp grid)
    comp_center = N_comp // 2
    uncomp = all_counts.reshape(len(seps), N_angles, N_comp)[:, :, comp_center]

    # Save maps and build results
    maps_to_save = {}
    maps_to_save["pitches"] = pitches.cpu().numpy()
    maps_to_save["yaws"] = yaws.cpu().numpy()

    results = []
    for si, sep in enumerate(seps):
        sf = sep.item()
        threshold = int(np.ceil(MIN_PATH_MM / sf))

        # Compensated bounce map
        bounce_map = best_per_angle[si].cpu().numpy().reshape(n_p, n_y)
        maps_to_save[f"bounces_comp_sep{sf:.0f}"] = bounce_map.astype(np.int16)

        # Uncompensated bounce map
        uncomp_map = uncomp[si].cpu().numpy().reshape(n_p, n_y)
        maps_to_save[f"bounces_uncomp_sep{sf:.0f}"] = uncomp_map.astype(np.int16)

        mb = int(bounce_map.max())
        above = bounce_map >= threshold
        above_uncomp = uncomp_map >= threshold
        if not above.any():
            tqdm.write(f"    sep={sf:.0f}mm: nothing above threshold ({threshold})")
            continue

        # Winning positions (from compensated map)
        mask = torch.from_numpy(above.reshape(-1))
        wp = list(zip(p_flat[mask].cpu().tolist(), y_flat[mask].cpu().tolist()))

        # Contiguous region
        ri, ci = np.where(above)
        centroid_ri = int(np.round(ri.mean()))
        centroid_ci = int(np.round(ci.mean()))
        if not above[centroid_ri, centroid_ci]:
            dists = (ri - centroid_ri) ** 2 + (ci - centroid_ci) ** 2
            nearest = dists.argmin()
            centroid_ri, centroid_ci = ri[nearest], ci[nearest]

        contiguous = flood_fill_nd(above, (centroid_ri, centroid_ci))
        maps_to_save[f"contiguous_sep{sf:.0f}"] = contiguous

        pitch_np = pitches.cpu().numpy()
        yaw_np = yaws.cpu().numpy()
        cont_rows = contiguous.any(axis=1)
        cont_cols = contiguous.any(axis=0)
        cp_range = (float(pitch_np[cont_rows].min()), float(pitch_np[cont_rows].max()))
        cy_range = (float(yaw_np[cont_cols].min()), float(yaw_np[cont_cols].max()))

        n_uncomp = int(above_uncomp.sum())
        n_comp_only = int(above.sum()) - int((above & above_uncomp).sum())

        res = Stage1Result(
            separation=sf,
            max_bounces=mb,
            bounce_threshold=threshold,
            winning_positions=wp,
            pitch_range=(min(p for p, _ in wp), max(p for p, _ in wp)),
            yaw_range=(min(y for _, y in wp), max(y for _, y in wp)),
            contiguous_count=int(contiguous.sum()),
            contiguous_pitch_range=cp_range,
            contiguous_yaw_range=cy_range,
            centroid=(float(pitch_np[centroid_ri]), float(yaw_np[centroid_ci])),
        )
        results.append(res)

        tqdm.write(
            f"    sep={sf:.0f}mm: max={mb}, thresh={threshold}, "
            f"{len(wp)} above (uncomp={n_uncomp}, comp_only={n_comp_only}), "
            f"{contiguous.sum()} contiguous\n"
            f"      pitch: [{cp_range[0]:+.2f}, {cp_range[1]:+.2f}]  "
            f"yaw: [{cy_range[0]:+.2f}, {cy_range[1]:+.2f}]  "
            f"center: ({res.centroid[0]:+.2f}, {res.centroid[1]:+.2f})"
        )

    np.savez_compressed(os.path.join(OUTPUT_DIR, "stage1_maps.npz"), **maps_to_save)
    return results


# ================================================================
# Stage 2: Mechanical stability mapping
# ================================================================


@dataclass
class LaserResult:
    laser_pitch: float
    laser_yaw: float
    peak_bounces: int
    bounce_threshold: int
    total_cells: int
    above_threshold_cells: int
    contiguous_cells: int
    contiguous_pct: float
    bounce_histogram: dict  # bounce_count -> num cells
    marginals: dict  # axis_name -> [lo, hi]


@dataclass
class Stage2Result:
    separation: float
    stage1_bounces: int
    bounce_threshold: int
    grid_shape: List[int]
    axes: dict
    laser_results: List[LaserResult] = field(default_factory=list)


def run_stage2(sim, s1_results: List[Stage1Result]) -> List[Stage2Result]:
    # Compensator grid
    cp_1d = grid1d(*M2_COMP_RANGE)
    cy_1d = grid1d(*M2_COMP_RANGE)
    cpg, cyg = torch.meshgrid(cp_1d, cy_1d, indexing="ij")
    comp_p, comp_y = cpg.reshape(-1), cyg.reshape(-1)
    N_comp = comp_p.shape[0]

    # Disturbance axes: (name, values, state_index)
    dist_defs = [
        ("sep_err", grid1d(*SEP_ERR_RANGE), 4),
        ("laser_dx", grid1d(*LDX_RANGE), 9),
        ("laser_dy", grid1d(*LDY_RANGE), 10),
        ("m2_tx", grid1d(*M2TX_RANGE), 7),
        ("m2_ty", grid1d(*M2TY_RANGE), 8),
    ]
    ax_names = [d[0] for d in dist_defs]
    ax_vals = [d[1] for d in dist_defs]
    ax_sidx = [d[2] for d in dist_defs]

    grid_shape = tuple(len(a) for a in ax_vals)
    center = tuple(len(a) // 2 for a in ax_vals)

    # Flat disturbance grid: (N_outer, 5)
    mg = torch.meshgrid(*ax_vals, indexing="ij")
    outer = torch.stack([g.reshape(-1) for g in mg], dim=-1)
    N_outer = outer.shape[0]

    # -- Flatten ALL work across all separations and laser positions --
    # Build job list: (sep_nominal, laser_pitch, laser_yaw) per job
    jobs = []  # (s1_index, lp, ly)
    for si, s1 in enumerate(s1_results):
        positions = subsample(s1.winning_positions, MAX_LASER_POS)
        for lp, ly in positions:
            jobs.append((si, lp, ly))

    N_jobs = len(jobs)
    N_total = N_jobs * N_outer * N_comp

    # Precompute per-job constants as tensors
    job_sep = torch.tensor(
        [s1_results[j[0]].separation for j in jobs], device=DEVICE, dtype=DTYPE
    )
    job_lp = torch.tensor([j[1] for j in jobs], device=DEVICE, dtype=DTYPE)
    job_ly = torch.tensor([j[2] for j in jobs], device=DEVICE, dtype=DTYPE)

    print(f"  Compensator: {len(cp_1d)}x{len(cy_1d)} = {N_comp} settings")
    print(
        f"  Disturbance: {' x '.join(f'{n}({s})' for n, s in zip(ax_names, grid_shape))}"
        f" = {N_outer:,} points"
    )
    print(f"  Jobs: {N_jobs} (laser positions across {len(s1_results)} separations)")
    print(f"  Total sims: {N_total:,}  (batched at {BATCH_SIZE:,})\n")

    # -- Run all sims --
    all_counts = torch.zeros(N_total, dtype=torch.long, device=DEVICE)

    for start in tqdm(
        range(0, N_total, BATCH_SIZE),
        desc="    Simulating",
        total=(N_total + BATCH_SIZE - 1) // BATCH_SIZE,
    ):
        end = min(start + BATCH_SIZE, N_total)
        B = end - start
        flat_idx = torch.arange(start, end, device=DEVICE)

        job_idx = flat_idx // (N_outer * N_comp)
        outer_idx = (flat_idx // N_comp) % N_outer
        comp_idx = flat_idx % N_comp

        states = torch.zeros(B, STATE_DIM, device=DEVICE, dtype=DTYPE)

        # Disturbance DOFs: offset (+ nominal sep for col 0)
        outer_batch = outer[outer_idx]  # (B, 5)
        for col in range(5):
            offset = outer_batch[:, col]
            if col == 0:  # sep_err: add nominal separation
                offset = offset + job_sep[job_idx]
            states[:, ax_sidx[col]] = offset

        # Laser angles
        states[:, 11] = job_lp[job_idx]
        states[:, 12] = job_ly[job_idx]

        # Compensator
        states[:, 2] = comp_p[comp_idx]
        states[:, 3] = comp_y[comp_idx]

        all_counts[start:end] = batched_sim(sim, states)

    # -- Reshape: (N_jobs, N_outer, N_comp) -> max over comp -> (N_jobs, N_outer) --
    best_per = all_counts.reshape(N_jobs, N_outer, N_comp).max(dim=2).values

    # -- Split results back per separation --
    job_cursor = 0
    all_results = []

    for si, s1 in enumerate(s1_results):
        positions = subsample(s1.winning_positions, MAX_LASER_POS)
        bounce_threshold = s1.bounce_threshold

        s2 = Stage2Result(
            separation=s1.separation,
            stage1_bounces=s1.max_bounces,
            bounce_threshold=bounce_threshold,
            grid_shape=list(grid_shape),
            axes={n: a.cpu().tolist() for n, a in zip(ax_names, ax_vals)},
        )
        grids_to_save = {}

        for lp_idx, (lp, ly) in enumerate(positions):
            grid = best_per[job_cursor].cpu().numpy().reshape(grid_shape)
            job_cursor += 1
            grids_to_save[f"bounces_{lp_idx}"] = grid.astype(np.int16)

            peak = int(grid.max())
            above = grid >= bounce_threshold
            cont = (
                flood_fill_nd(above, center) if above[center] else np.zeros_like(above)
            )
            cont_pct = 100 * cont.sum() / grid.size if grid.size > 0 else 0.0

            unique, cnt_arr = np.unique(grid, return_counts=True)
            histogram = {int(u): int(c) for u, c in zip(unique, cnt_arr)}

            marginals = {}
            for dim, name in enumerate(ax_names):
                reduce = tuple(d for d in range(5) if d != dim)
                proj = cont.any(axis=reduce)
                vals_np = ax_vals[dim].cpu().numpy()
                marginals[name] = contiguous_range_1d(proj, vals_np, len(vals_np) // 2)

            lr = LaserResult(
                laser_pitch=lp,
                laser_yaw=ly,
                peak_bounces=peak,
                bounce_threshold=bounce_threshold,
                total_cells=int(grid.size),
                above_threshold_cells=int(above.sum()),
                contiguous_cells=int(cont.sum()),
                contiguous_pct=round(cont_pct, 2),
                bounce_histogram=histogram,
                marginals=marginals,
            )
            s2.laser_results.append(lr)
            print(
                f"      ({lp:+.2f},{ly:+.2f}): peak={peak}, "
                f">={bounce_threshold}: {above.sum()}/{grid.size} "
                f"({100*above.sum()/grid.size:.1f}%), "
                f"contiguous={cont.sum()} ({cont_pct:.1f}%)"
            )

        npz_path = os.path.join(OUTPUT_DIR, f"stage2_grids_sep{s1.separation:.0f}.npz")
        meta = {n: a.cpu().numpy() for n, a in zip(ax_names, ax_vals)}
        meta["laser_positions"] = np.array(positions)
        np.savez_compressed(npz_path, **grids_to_save, **meta)
        print(f"    Saved {npz_path}")

        if s2.laser_results:
            best = max(s2.laser_results, key=lambda r: r.contiguous_cells)
            print(
                f"    Best: ({best.laser_pitch:+.2f},{best.laser_yaw:+.2f}), "
                f"peak={best.peak_bounces}, "
                f"above={best.above_threshold_cells}/{best.total_cells} "
                f"({100*best.above_threshold_cells/best.total_cells:.1f}%), "
                f"contiguous={best.contiguous_cells} ({best.contiguous_pct:.1f}%)"
            )
            print(f"    Bounce distribution:")
            for bc in sorted(best.bounce_histogram.keys(), key=int, reverse=True):
                n = best.bounce_histogram[bc]
                pct = 100 * n / best.total_cells
                tag = " <<" if int(bc) >= bounce_threshold else ""
                print(f"      {bc:>3} bounces: {n:6d} cells ({pct:5.1f}%){tag}")
            print(f"    Mechanical marginals (contiguous, m2 compensated):")
            for k, v in best.marginals.items():
                print(f"      {k:>12s}: [{v[0]:+.3f}, {v[1]:+.3f}]")

        all_results.append(s2)
        print()

    return all_results


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Herriott cell parameter search")
    parser.add_argument("--stage1", action="store_true", help="Run Stage 1 only")
    parser.add_argument(
        "--stage2", action="store_true", help="Run Stage 2 only (needs stage1.json)"
    )
    args = parser.parse_args()

    # Default: run both
    run_s1 = args.stage1 or not args.stage2
    run_s2 = args.stage2 or not args.stage1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sim = create_sim()
    print(f"Device: {DEVICE} | Min path: {MIN_PATH_MM / 1000:.1f}m\n")
    t0 = time.perf_counter()

    s1_path = os.path.join(OUTPUT_DIR, "stage1.json")

    if run_s1:
        print("=" * 60)
        print("STAGE 1: Peak finding + laser angle tolerance")
        print("=" * 60)
        s1 = run_stage1(sim)
        t1 = time.perf_counter()
        print(f"\n  Done in {t1 - t0:.1f}s, {len(s1)} viable separations\n")
        save_json(s1, s1_path)
    else:
        print(f"Loading Stage 1 results from {s1_path}")
        with open(s1_path) as f:
            raw = json.load(f)
        s1 = [
            Stage1Result(
                **{
                    k: (
                        tuple(v)
                        if k
                        in (
                            "pitch_range",
                            "yaw_range",
                            "contiguous_pitch_range",
                            "contiguous_yaw_range",
                            "centroid",
                        )
                        else v
                    )
                    for k, v in r.items()
                }
            )
            for r in raw
        ]
        s1 = [r for r in s1 if r.winning_positions]
        print(f"  Loaded {len(s1)} viable separations\n")

    if not s1:
        print("No configurations found.")
        exit()

    if run_s2:
        print("=" * 60)
        print("STAGE 2: Mechanical stability (m2 pitch/yaw compensated)")
        print("=" * 60)
        s2 = run_stage2(sim, s1)
        t2 = time.perf_counter()
        print(f"  Done in {t2 - t0:.1f}s\n")
        save_json(s2, os.path.join(OUTPUT_DIR, "stage2_summary.json"))

    print(f"Total: {time.perf_counter() - t0:.1f}s")
    print(f"Results in ./{OUTPUT_DIR}/")
