"""
cpk.py - Process capability (Cpk) of the alignment from the recorded evaluations

Characteristic and spec: simulation: final path P / P_ref, lower spec limit SUCCESS_FRAC
(0.9, evaluation.py), no upper limit; real raster: final spot count, lower spec limit 16
(the real cell's maximum; counts above it are counter artifacts). Both are bounded at the
target (P <= ~P_ref, count <= 16), so the per-episode distribution is a mass at the top plus
a tail of failures, and the normal-theory Cpk = (mean - LSL) / (3 sigma) does not describe
it (on the capped spot count it is <= 0 by construction); the reports also keep only means.
Cpk here is the yield equivalent: Cpk = Phi^-1(yield) / 3, yield = share of episodes at or
above the spec (the eval metric success / final_ge16), with the Wilson 95 % interval of
the yield mapped the same way. Cpk 1.00 = 1350 ppm below spec, 1.33 = 33 ppm.

Sources: eval_policy.py JSONs (--sim name=path), W&B runs (--wandb name=run id: the eval
row with the best eval_region/bin_mean/pct_oracle, i.e. the row of best.pt), eval_real.py
JSONs (--real name=path). Splits: cont all / good (the possible scenarios: impossible ones
have P_max below the floor, success unreachable by definition) / blind / informative /
good scenarios per target-region bin and per separation bin, the region bins of the region
set (all good), the set4 setups.

    uv run python cpk.py [--sim r6=reports/r6_it7630.json ...] [--wandb r12=<run id> ...]
        [--real r12=reports/real_eval/r12_best.json ...] [--out reports]
"""
import argparse
import json
import math
import os
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

RL_DIR = os.path.dirname(os.path.abspath(__file__))
WANDB_PROJECT = "e-venediktov-university-of-pittsburgh/herriott-rl"
BEST_KEY = "eval_region/bin_mean/pct_oracle"
REAL_LSL = 16

SIM = ["r6=reports/r6_it7630.json", "r7b=reports/r7b_it2950.json"]
WANDB = ["r8=be70292bb04d", "r9=8b4ec5020b88", "r10b=r10b-accel-42d9a35a",
         "r12=r12-tbonus1-095a83d7"]
REAL = ["r6=reports/real_eval/r6_it7630.json", "r7b=reports/real_eval/r7b_it2950.json",
        "r8=reports/real_eval/r8_best.json", "r9=reports/real_eval/r9_best.json",
        "r10b=reports/real_eval/r10b_best.json", "r12=reports/real_eval/r12_best.json"]

ROWS = ([("eval_cont", s) for s in ("all", "good", "blind", "informative")]
        + [("eval_cont", s) for s in ("region_tiny", "region_small", "region_medium",
                                      "region_large")]
        + [("eval_region", s) for s in ("region_tiny", "region_small", "region_medium",
                                        "region_large", "all")]
        + [("eval_cont", f"sep{a}-{b}") for a, b in ((20, 50), (50, 100), (100, 150),
           (150, 200), (200, 250), (250, 300), (300, 350), (350, 400))]
        + [("eval", s) for s in ("measured", "sep116", "sep144", "sep169")])
SET_NAME = {"eval_cont": "cont", "eval_region": "region", "eval": "set4"}

_N = NormalDist()


def cpk_yield(p: float) -> float:
    """Yield-equivalent Cpk: Phi^-1(p) / 3 (p clipped away from 0 and 1)."""
    return _N.inv_cdf(min(max(p, 1e-9), 1 - 1e-9)) / 3


def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def entry(p: float, n: int) -> Dict[str, float]:
    lo, hi = wilson(p, n)
    return {"yield": p, "n": n, "cpk": cpk_yield(p), "cpk_lo": cpk_yield(lo),
            "cpk_hi": cpk_yield(hi)}


def pairs(items: List[str]) -> List[Tuple[str, str]]:
    return [tuple(s.split("=", 1)) for s in items]


def sim_flat_json(path: str) -> Dict[str, float]:
    d = json.load(open(path if os.path.isabs(path) else os.path.join(RL_DIR, path)))
    flat: Dict[str, float] = {}
    for s in d["sets"].values():
        flat.update(s["policies"]["argmax"]["flat"])
    return flat


def sim_flat_wandb(run_id: str) -> Dict[str, float]:
    import wandb
    r = wandb.Api(timeout=60).run(f"{WANDB_PROJECT}/{run_id}")
    keys = [k for k in r.summary.keys() if k.startswith("eval") and "_samp" not in k
            and k.split("/")[-1] in ("success", "n")] + [BEST_KEY]
    sel = [x for x in r.scan_history(keys=[BEST_KEY, "_step"], page_size=5000)
           if x.get(BEST_KEY) is not None]
    step = int(max(sel, key=lambda x: x[BEST_KEY])["_step"])
    return next(iter(r.scan_history(keys=keys + ["_step"], min_step=step, max_step=step + 1)))


def sim_table(flat: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for st, sp in ROWS:
        p, n = flat.get(f"{st}/{sp}/success"), flat.get(f"{st}/{sp}/n")
        if p is not None and n:
            out[f"{SET_NAME[st]} {sp}"] = entry(float(p), int(n))
    bins = [flat.get(f"eval_region/region_{b}/pct_oracle") for b in ("tiny", "small", "medium", "large")]
    out["_best"] = flat.get(BEST_KEY, sum(bins) / 4 if None not in bins else None)
    return out


def real_table(path: str) -> Dict[str, Dict[str, float]]:
    d = json.load(open(path if os.path.isabs(path) else os.path.join(RL_DIR, path)))
    n = int(d["episodes"])
    out = {m: entry(float(v["final_ge16"]), n) for m, v in d["maps"].items()}
    pooled = sum(v["final_ge16"] for v in d["maps"].values()) / len(d["maps"])
    out["all maps"] = entry(pooled, n * len(d["maps"]))
    return out


def cell(e: Optional[Dict[str, float]], ci: bool = False) -> str:
    if not e:
        return "-"
    s = f"{e['cpk']:+.2f}"
    return s + f" [{e['cpk_lo']:+.2f}, {e['cpk_hi']:+.2f}]" if ci else s + f" ({100 * e['yield']:.1f} %)"


def report(sim: Dict[str, dict], real: Dict[str, dict]) -> str:
    L = ["# Process capability (Cpk) of the alignment", "",
         "Generated by `cpk.py` from the recorded evaluations (no new episodes). Cpk is the",
         "yield equivalent, Phi^-1(yield) / 3, one-sided: yield = share of episodes ending at or",
         "above the lower spec limit, P >= 0.9 P_ref in simulation (the `success` metric), >= 16",
         "spots on the real raster (`final_ge16`). The normal-theory (mean - LSL) / (3 sigma) does",
         "not apply: both characteristics are capped at the target (mass at the top + failure",
         "tail), and the reports keep only means. Cells: Cpk (yield).", "",
         "| Cpk | below spec | episodes to show it (0 failures, 95 %) |", "|---|---|---|"]
    for c in (0.33, 0.67, 1.0, 1.33, 1.67):
        q = 1 - _N.cdf(3 * c)
        ppm = f"{q * 1e6:,.0f}" if q * 1e6 >= 10 else f"{q * 1e6:.2g}"
        L.append(f"| {c:.2f} | {ppm} ppm | {math.ceil(3 / q):,} |")
    L += ["", "With 1024 episodes per bin and no failures the yield bound gives Cpk <= 0.92.", ""]
    tags = list(sim)
    L += ["## Simulation, best.pt of each run (argmax policy)", "",
          "r6, r7b: local eval_policy.py reports (r6 on clean renders, r7b on camera renders);",
          "r8 - r12: the W&B eval row of best.pt (camera renders). cont good = the possible",
          "scenarios (the 10 % impossible ones excluded); the cont region and separation bins are",
          "good scenarios of the training mix; the region set is good scenarios only, a quarter",
          "per target-region bin.", "",
          "| split | n | " + " | ".join(tags) + " |", "|---|---|" + "---|" * len(tags),
          "| region-bin mean pct_oracle | | " + " | ".join(
              f"{sim[t]['_best']:.2f}" if sim[t].get("_best") else "-" for t in tags) + " |"]
    for st, sp in ROWS:
        k = f"{SET_NAME[st]} {sp}"
        n = next((sim[t][k]["n"] for t in tags if k in sim[t]), None)
        L.append(f"| {k} | {n or ''} | " + " | ".join(cell(sim[t].get(k)) for t in tags) + " |")
    L.append("")
    tags = list(real)
    maps = list(next(iter(real.values())))
    L += ["## Real raster (eval_real.py, 1024 episodes per axis map)", "",
          "| axis map | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for m in maps:
        L.append(f"| {m} | " + " | ".join(cell(real[t].get(m)) for t in tags) + " |")
    L += ["", "95 % intervals (Wilson interval of the yield, mapped):", "",
          "| axis map | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for m in maps:
        L.append(f"| {m} | " + " | ".join(cell(real[t].get(m), ci=True) for t in tags) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Yield-equivalent Cpk from recorded evals")
    ap.add_argument("--sim", nargs="*", default=SIM, help="name=eval_policy.py JSON")
    ap.add_argument("--wandb", nargs="*", default=WANDB, help="name=W&B run id")
    ap.add_argument("--real", nargs="*", default=REAL, help="name=eval_real.py JSON")
    ap.add_argument("--out", default=os.path.join(RL_DIR, "reports"))
    a = ap.parse_args()
    sim = {t: sim_table(sim_flat_json(p)) for t, p in pairs(a.sim)}
    sim.update({t: sim_table(sim_flat_wandb(r)) for t, r in pairs(a.wandb)})
    real = {t: real_table(p) for t, p in pairs(a.real)}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"sim": sim, "real": real}, open(os.path.join(a.out, "cpk.json"), "w"), indent=1)
    md = report(sim, real)
    open(os.path.join(a.out, "cpk.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
