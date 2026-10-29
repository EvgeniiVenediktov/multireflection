"""
design_cpk.py - Cpk of the alignment on designed cells under manufacturing tolerances

Designs: properly designed cells of the training distribution (the sim's mirrors, ROC
200): laser through the M1 hole centre (dx = dy = 0), separation and laser pitch / yaw from
a search: --cands candidates of the continuous generator (scenarios.ContRanges, landing
mode, separation and angles kept one tolerance inside the training ranges, so the as-built
cells stay inside), the valid ones (Scanner.check) box-scanned (default AtlasConfig: M2
+-2 deg at 0.01 deg, cap 128). Per separation band (the cont bins 20-50, 50-100, ...,
350-400 mm): the --top good candidates with the highest P_max whose longest path ends
below the cap (no trapped-beam cap paths) at --min-bounces .. --max-bounces bounces (the
design order; default: any below the cap), each probed with --probe as-built cells (the
tolerance box below); the --keep with the highest median as-built P_max are probed again
with --reprobe fresh cells (against the winner's curse of the first probe) and the band's
design is the one with the highest median there (the expected delivered path; a beam that
fails the check counts 0).

As-built cells: per design the first --cells valid cells of a seeded stream with
separation +- tol_sep, laser pitch and yaw +- tol_laser, laser dx and dy +- tol_off
(independent uniform); the share failing the beam check (clipped by the hole edge or
missing M2) is reported; each cell box-scanned for its own P_max (the episode's P_ref).

Episodes: one per cell, the checkpoint's argmax policy from the run's seeded eval start
tilts in the run's eval env (ppo._table_set / run_eval: camera renders as in its evals).
Specs, one-sided, lower limit SUCCESS_FRAC (0.9): alignment, P_final / P_ref(as-built)
(the eval success) over the possible as-built cells (good: P_max >= P_floor; on an
impossible cell P_ref = P_floor is out of reach by definition, so those count as
manufacturing loss, reported as the possible share); delivered path, P_final / P_max(nominal
design), also over the possible cells. Cpk: the yield
equivalent Phi^-1(yield) / 3 (cpk.py) and the normal-theory (mean - LSL) / (3 sd) of the
per-episode ratio, per design and over all designs; manufacturing ceiling of the delivered
path: share of the possible cells with P_max(as-built) >= 0.9 P_max(nominal).

Outputs (--out): design_cpk.md, design_cpk.json, episodes.npz. The search scan is cached in
runs/design_cpk/ (by seed, candidates and tolerance inset).

    uv run python design_cpk.py --ckpt runs/ppo/<job>/<best>.pt [--tol-sep 1 --tol-laser 0.25 --tol-off 0.2]
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from cpk import entry
from eval_policy import load_checkpoint
from evaluation import SUCCESS_FRAC
from ppo import _digest, _table_set, run_eval
from scenarios import (ATLAS_VERSION, Atlas, AtlasConfig, ContRanges, Scanner, cont_sep_bins,
                       region_area)

RL_DIR = os.path.dirname(os.path.abspath(__file__))
STREAM_DESIGN, STREAM_TOL = 6, 7  # Generator streams (scenarios.py uses 1-5)
SEP_RANGE = (20.0, 400.0)  # Training ranges (r6 on)
LASER_RANGE = 8.0
SEARCH_KEYS = ("valid", "good", "pmax", "pmax_bounces", "plateau_order", "near_cells")


def tol_vec(a) -> np.ndarray:
    return np.array([a.tol_sep, a.tol_laser, a.tol_laser, a.tol_off, a.tol_off])


def as_built(base: np.ndarray, n: int, a, seed: int) -> np.ndarray:
    """(len(base) * n, 5) cells: each base row repeated n times plus uniform deviations
    within the tolerance box (seeded by (STREAM_TOL, seed))."""
    u = np.random.default_rng([STREAM_TOL, seed]).uniform(-1.0, 1.0, (len(base) * n, 5))
    return np.repeat(base, n, axis=0) + u * tol_vec(a)


def scan(sc: Scanner, prm: np.ndarray) -> Dict[str, np.ndarray]:
    """Beam check + box scan: valid (n,) and the scan fields of all rows (host arrays;
    rows failing the check have pmax 0)."""
    pt = torch.from_numpy(prm)
    valid = sc.check(pt)["valid"].cpu().numpy()
    d = {k: v.cpu().numpy() for k, v in sc.scan(pt[valid]).items()}
    out = {"valid": valid}
    for k, v in d.items():
        full = np.zeros((len(prm),) + v.shape[1:], dtype=v.dtype)
        full[valid] = v
        out[k] = full
    return out


def search_designs(sc: Scanner, a, log) -> List[Dict[str, Any]]:
    t = a.tol_laser
    rng = ContRanges(sep=(SEP_RANGE[0] + a.tol_sep, SEP_RANGE[1] - a.tol_sep),
                     pitch=(-LASER_RANGE + t, LASER_RANGE - t), yaw=(-LASER_RANGE + t, LASER_RANGE - t),
                     offset_radius=1e-6)  # Landing angles for a laser at the hole centre
    cache = os.path.join(RL_DIR, "runs", "design_cpk",
                         f"search_s{a.seed}_n{a.cands}_tol{a.tol_sep:g}-{a.tol_laser:g}.npz")
    t0 = time.time()
    if os.path.exists(cache):
        z = np.load(cache)
        prm, d = z["params"], {k: z[k] for k in z.files if k != "params"}
    else:
        prm = rng.candidates(a.seed, 0, a.cands, STREAM_DESIGN)
        prm = prm[rng.angle_ok(prm)]
        prm[:, 3:] = 0.0
        d = {k: v for k, v in scan(sc, prm).items() if k in SEARCH_KEYS}
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, params=prm, **d)
    pb = d["pmax_bounces"]
    elig = d["valid"] & d["good"] & (pb < sc.cfg.cap) & (pb >= a.min_bounces) & (pb <= a.max_bounces)
    log(f"search: {a.cands} candidates, {len(prm)} within the angle bounds, {int(d['valid'].sum())} "
        f"valid, {int(elig.sum())} good with P_max below the cap at {a.min_bounces}-{a.max_bounces} "
        f"bounces ({time.time() - t0:.1f} s)")
    area = d["near_cells"] * sc.cfg.step ** 2
    designs = []
    for j, (lo, hi) in enumerate(cont_sep_bins(*SEP_RANGE)):
        ids = np.nonzero(elig & (prm[:, 0] >= lo) & (prm[:, 0] < hi))[0]
        ids = ids[np.argsort(-d["pmax"][ids])][:a.top]
        if not ids.size:
            log(f"search: no eligible candidate in {lo:g}-{hi:g} mm")
            continue
        p = scan(sc, as_built(prm[ids], a.probe, a, a.seed * 1000 + j))
        ids = ids[np.argsort(-np.median(p["pmax"].reshape(len(ids), a.probe), axis=1))][:a.keep]
        p = scan(sc, as_built(prm[ids], a.reprobe, a, a.seed * 1000 + 50 + j))
        med = np.median(p["pmax"].reshape(len(ids), a.reprobe), axis=1)
        k = int(np.argmax(med))
        i = ids[k]
        good = p["good"].reshape(len(ids), a.reprobe)[k]
        designs.append({
            "band": f"{lo:g}-{hi:g}", "params": prm[i].tolist(), "pmax_m": float(d["pmax"][i]) / 1000,
            "pmax_bounces": int(d["pmax_bounces"][i]), "plateau_order": int(d["plateau_order"][i]),
            "region_deg2": float(area[i]), "probe_median_frac": float(med[k] / d["pmax"][i]),
            "probe_good": float(good.mean()),
            "candidates": int(ids.size)})
        log(f"design {lo:g}-{hi:g} mm: sep {prm[i, 0]:.1f}, laser ({prm[i, 1]:+.2f}, {prm[i, 2]:+.2f}) "
            f"deg, P_max {d['pmax'][i] / 1000:.2f} m ({d['pmax_bounces'][i]} bounces), region "
            f"{area[i]:.3f} deg^2, re-probe median {med[k] / d['pmax'][i]:.2f} x P_max, possible "
            f"{100 * good.mean():.0f} %")
    return designs


def build_cells(sc: Scanner, designs: List[Dict[str, Any]], a, dev, log):
    """Atlas of the as-built cells (design-major) and per-row design index, plus the
    beam-check rejection share per design."""
    rows, did, rej = [], [], []
    for j, dz in enumerate(designs):
        prm = as_built(np.array(dz["params"])[None], 4 * a.cells, a, a.seed * 1000 + 100 + j)
        ok = sc.check(torch.from_numpy(prm))["valid"].cpu().numpy()
        idx = np.nonzero(ok)[0][:a.cells]
        assert idx.size == a.cells, f"design {j}: only {idx.size} valid cells of {len(prm)}"
        rej.append(1.0 - float(ok[:idx[-1] + 1].mean()))
        rows.append(prm[idx])
        did.append(np.full(a.cells, j))
    prm = np.concatenate(rows)
    d = {k: v.cpu() for k, v in sc.scan(torch.from_numpy(prm)).items()}
    key = {**sc.cfg.key(), "design_cpk": {"designs": [dz["params"] for dz in designs],
                                          "tol": tol_vec(a).tolist(), "cells": a.cells, "seed": a.seed}}
    d.update(version=ATLAS_VERSION, key=key, named={}, n_grid=len(prm))
    log(f"cells: {len(prm)} as-built cells, beam-check rejects per design "
        + " / ".join(f"{100 * r:.1f}" for r in rej) + " %")
    return Atlas(d, dev), np.concatenate(did), rej


def stats(r1: np.ndarray, r2: np.ndarray, ceil2: np.ndarray, good: np.ndarray) -> Dict[str, Any]:
    out = {"possible": float(good.mean())}
    r1, r2, ceil2 = r1[good], r2[good], ceil2[good]
    for nm, r in (("align", r1), ("delivered", r2)):
        e = entry(float((r >= SUCCESS_FRAC).mean()), len(r))
        sd = float(r.std(ddof=1))
        e.update(mean=float(r.mean()), sd=sd, q05=float(np.quantile(r, 0.05)),
                 cpk_normal=(float(r.mean()) - SUCCESS_FRAC) / (3 * sd) if sd > 0 else float("inf"))
        out[nm] = e
    out["delivered"]["ceiling"] = float(ceil2.mean())
    return out


def fmt_cpk(e: Dict[str, float]) -> str:
    """Yield-equivalent Cpk [95 %]; without failures only the lower bound is known."""
    if e["yield"] >= 1.0:
        return f">= {e['cpk_lo']:+.2f} (no failures)"
    return f"{e['cpk']:+.2f} [{e['cpk_lo']:+.2f}, {e['cpk_hi']:+.2f}]"


def report(res: Dict[str, Any]) -> str:
    a = {"min_bounces": 0, "max_bounces": 1 << 30, **res["args"]}
    L = ["# Cpk of the alignment on designed cells under manufacturing tolerances", "",
         f"Generated by `design_cpk.py` ({res['date']}): checkpoint `{res['ckpt']}` ({res['name']}, "
         f"iteration {res['iteration']}), argmax policy, the run's eval env and start tilts.", "",
         f"Designs: laser at the M1 hole centre, separation and laser angles from a search over "
         f"{a['cands']} generator candidates (the longest P_max below the cap per separation "
         f"band at {a['min_bounces']}-{min(a['max_bounces'], AtlasConfig().cap - 1)} bounces, top {a['top']}, the best {a['keep']} by median as-built P_max over {a['probe']} "
         f"tolerance draws, then the highest median over {a['reprobe']} fresh draws). As-built "
         f"cells: separation +-{a['tol_sep']:g} mm, laser pitch / yaw "
         f"+-{a['tol_laser']:g} deg, laser dx / dy +-{a['tol_off']:g} mm, uniform; "
         f"{a['cells']} cells per design, one episode each, each cell's own P_max (box scan) as "
         "P_ref.", "",
         "Possible = share of as-built cells with P_max >= P_floor (20 bounces x separation); "
         "the others hold no long path at any M2 tilt (manufacturing loss). Specs (lower limit "
         "0.9), both over the possible cells: align = P_final / P_ref(as-built) (the eval success); "
         "delivered = P_final / P_max(nominal design). Cpk_y = Phi^-1(yield) / 3 (yield "
         "equivalent, [95 %]); Cpk_n = (mean - 0.9) / (3 sd) of the per-episode ratio. Ceiling = "
         "share of the possible cells with P_max(as-built) >= 0.9 P_max(nominal): the best yield any "
         "alignment can reach on the delivered spec.", "",
         "## Designs", "",
         "| band (mm) | sep (mm) | laser pitch / yaw (deg) | P_max (m) | bounces | target region (deg^2) | as-built P_max: median / q10 (x nominal) | beam-check rejects |",
         "|---|---|---|---|---|---|---|---|"]
    for dz in res["designs"]:
        p = dz["params"]
        L.append(f"| {dz['band']} | {p[0]:.1f} | {p[1]:+.2f} / {p[2]:+.2f} | {dz['pmax_m']:.2f} | "
                 f"{dz['pmax_bounces']} | {dz['region_deg2']:.3f} | {dz['built_median_frac']:.2f} / "
                 f"{dz['built_q10_frac']:.2f} | {100 * dz['reject']:.1f} % |")
    L += ["", "## Cpk", "",
          "| design | cells | possible | n (possible) | align: yield | Cpk_y | Cpk_n | delivered: ceiling | yield | Cpk_y | Cpk_n | mean length |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, s in res["cpk"].items():
        al, dl = s["align"], s["delivered"]
        L.append(f"| {nm} | {dl['n']} | {100 * s['possible']:.1f} % | {al['n']} | {100 * al['yield']:.1f} % | "
                 f"{fmt_cpk(al)} | {al['cpk_normal']:+.2f} | {100 * dl['ceiling']:.1f} % | "
                 f"{100 * dl['yield']:.1f} % | {fmt_cpk(dl)} | {dl['cpk_normal']:+.2f} | {s['length']:.1f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Cpk on designed cells under manufacturing tolerances")
    ap.add_argument("--ckpt", required=True, help="ppo.py checkpoint")
    ap.add_argument("--cands", type=int, default=131072, help="Search candidates")
    ap.add_argument("--top", type=int, default=64, help="Candidates per band probed for robustness")
    ap.add_argument("--probe", type=int, default=64, help="As-built draws per probed candidate")
    ap.add_argument("--keep", type=int, default=8, help="Candidates per band probed again")
    ap.add_argument("--min-bounces", type=int, default=0, help="Design order: lowest bounce count of P_max")
    ap.add_argument("--max-bounces", type=int, default=1 << 30, help="Design order: highest bounce count of P_max")
    ap.add_argument("--reprobe", type=int, default=256, help="Fresh as-built draws per kept candidate")
    ap.add_argument("--cells", type=int, default=512, help="As-built cells (episodes) per design")
    ap.add_argument("--tol-sep", type=float, default=1.0, help="Separation tolerance +- (mm)")
    ap.add_argument("--tol-laser", type=float, default=0.25, help="Laser pitch / yaw tolerance +- (deg)")
    ap.add_argument("--tol-off", type=float, default=0.2, help="Laser dx / dy tolerance +- (mm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RL_DIR, "reports", "design_cpk"))
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    sc = Scanner(AtlasConfig(), dev)
    designs = search_designs(sc, a, log)
    table, did, rej = build_cells(sc, designs, a, dev, log)

    agent, c, info = load_checkpoint(a.ckpt, dev)
    n = table.n
    c.eval_n, c.eval_envs = n, min(c.eval_envs, n)
    did_t = torch.from_numpy(did).to(dev)

    def splits(ep):
        m = {"all": torch.ones_like(ep["setup"], dtype=torch.bool)}
        m.update({f"d{j}": did_t[ep["setup"]] == j for j in range(len(designs))})
        return m

    key = {"design_cpk": _digest(table.params, table.path_ref), "n": n, "seed": c.eval_seed}
    es = _table_set(c, "design", table, torch.arange(n), key, "oracle_design.pt", "eval_design",
                    None, a.out, dev, splits)
    t0 = time.time()
    flat, ep, _ = run_eval(agent, es, c.eval_seed, False, per_episode=True)
    log(f"eval: {n} episodes in {time.time() - t0:.1f} s")

    setup = ep["setup"].cpu().numpy()
    fin = ep["final_path_mm"].double().cpu().numpy()
    ref = ep["path_ref"].double().cpu().numpy()
    pnom = np.array([dz["pmax_m"] * 1000 for dz in designs])[did[setup]]
    pbuilt = table.pmax.double().cpu().numpy()[setup]
    r1, r2, ceil2 = fin / ref, fin / pnom, pbuilt >= SUCCESS_FRAC * pnom
    good = table.good.cpu().numpy()[setup]
    area = region_area(table).cpu().numpy()
    res: Dict[str, Any] = {"ckpt": os.path.relpath(os.path.abspath(a.ckpt), RL_DIR), "name": info["name"],
                           "iteration": info["iteration"], "date": time.strftime("%Y-%m-%d"),
                           "args": vars(a), "designs": designs, "cpk": {}}
    groups = [("all designs", np.ones(n, dtype=bool))]
    for j, dz in enumerate(designs):
        m = did == j
        b = table.pmax.double().cpu().numpy()[m] / (dz["pmax_m"] * 1000)
        dz.update(reject=rej[j], built_median_frac=float(np.median(b)),
                  built_q10_frac=float(np.quantile(b, 0.1)), built_good=float(table.good.cpu().numpy()[m].mean()),
                  built_region_median_deg2=float(np.median(area[m])))
        groups.append((f"{dz['band']} mm", did[setup] == j))
    for (nm, m), sp in zip(groups, ["all"] + [f"d{j}" for j in range(len(designs))]):
        s = stats(r1[m], r2[m], ceil2[m], good[m])
        s["length"] = flat[f"eval_design/{sp}/length"]
        res["cpk"][nm] = s
    np.savez(os.path.join(a.out, "episodes.npz"), design=did[setup], params=table.params.cpu().numpy()[setup],
             final_path_mm=fin, path_ref_mm=ref, pmax_nominal_mm=pnom, pmax_built_mm=pbuilt, good=good,
             region_built_deg2=area[setup],
             init_bounces=ep["init_bounces"].cpu().numpy(), length=ep["ep_length"].cpu().numpy())
    with open(os.path.join(a.out, "design_cpk.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    md = report(res)
    with open(os.path.join(a.out, "design_cpk.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
