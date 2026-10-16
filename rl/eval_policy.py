"""
eval_policy.py - Evaluation of a ppo.py checkpoint on seen and unseen scenarios

Run from rl/:
    uv run python eval_policy.py --ckpt runs/ppo/4407578/best.pt --out reports
    uv run python eval_policy.py --ckpt <ckpt> [--sets cont,region,set4,full,offgrid,...]
                                 [--policy argmax,sampled] [--n 4096] [--out DIR] [--cache DIR]
                                 [--name STEM] [--compare OTHER.json]
Cluster: cluster/eval_l40s.slurm.

Sets (ppo.build_eval_sets, built exactly as the trainer builds them; the run's PPOConfig
from the checkpoint gives step cost, episode length, image size, initial-tilt box, eval
seed and impossible share; --n the episodes; every set has the same initial tilts).
Default DEFAULT_SETS cont, region, set4 (--setups all / measured runs: their own set):
    cont          continuous scenarios of the run's ranges (pool runs, D7: separation,
                  laser pitch / yaw, laser offset over the hole disk; valid beams only;
                  the default ranges for other runs), 90 / 10 good / impossible, one
                  episode per scenario
    region        good continuous scenarios of the same ranges, stratified by
                  target-region size (area of the M2 box cells with path >= 0.9 P_max):
                  n / 4 each tiny < 0.02, small 0.02-0.1, medium 0.1-0.5, large >= 0.5
                  deg^2 (scenarios.RegionConfig)
    set4          the four named setups of stage 1
and, with --sets, the grid benchmarks of the atlas runs (measured laser offsets, no beam
filter):
    full          scenarios of the training grid (the atlas: separation 50-200 mm step 2,
                  laser pitch / yaw 0-8 deg step 0.25), 90 / 10 good / impossible
    holdout       grid points held out of training (runs with --holdout-frac > 0)
    offgrid       midpoints between grid points in all three parameters, one episode per
                  scenario
    extrap_short  separation 20-49 mm, laser midpoints
    extrap_long   separation 201-350 mm, laser midpoints
The scenario tables (scenarios.py: atlas, off-grid, continuous and region tables in
--tables) and the oracle of each set (baselines.plan_oracle) are loaded from their caches
or built there. Oracles are cached in --cache, default the checkpoint's directory (where
ppo.py keeps them; oracle_cont.pt, oracle_region.pt, oracle_eval.pt, ... of the run are
reused) when --n is the run's eval_n, else <out>/oracles_n<N>.

Policies: argmax and sampled (ppo.EvalPolicy, the trainer's eval noise). Per set and
policy (ppo.run_eval + evaluation.split_metrics): % of the oracle's return, return,
regret, P / P_ref, success (P >= 0.9 P_ref), stop rate, stop at step 0, stop below 0.9
P_ref (and those episodes' mean P / P_ref and final bounces / the scenario's highest
bounce count: early stops on a lower-order pattern), step limit below 0.9 P_ref, length,
share of episodes ending at the bounce cap (and the oracle's), on the splits all / good /
impossible / blind / informative / separation bins / target-region bins (good
scenarios); head errors (path, tilt, scenario) per set and split from per-episode sums
over the active steps.

Writes <out>/<name>.json (every metric, the scenario-table summaries; default name
eval_<run id>_<ckpt stem>[_<tag>]) and .md (report): target-region size (distribution,
named setups, results per bin; with --compare those of another eval_policy.py JSON next
to them), scenario tables, overview, grid vs off-grid per separation bin (when both are
evaluated), head errors, extrapolation section, per-set details.
"""

import argparse
import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from agent import Agent, AgentConfig
from evaluation import SUCCESS_FRAC, split_metrics
from ppo import (ATLAS_SETS, EVAL_SET_NAMES, EvalSet, build_eval_sets, config_from_dict, init_run_id,
                 run_eval)

RL_DIR = os.path.dirname(os.path.abspath(__file__))
POLICIES = ("argmax", "sampled")
DEFAULT_SETS = ("cont", "region", "set4")  # --setups full runs (pool or atlas)
SET_INFO = {
    "cont": "continuous scenarios of the training ranges (pool runs)",
    "region": "good continuous scenarios, n / 4 per target-region bin",
    "full": "grid benchmark: training grid of the atlas runs",
    "holdout": "grid benchmark: grid points held out of training",
    "offgrid": "grid benchmark: midpoints between grid points",
    "extrap_short": "grid benchmark: separation 20-49 mm",
    "extrap_long": "grid benchmark: separation 201-350 mm",
    "set4": "the four named setups",
    "measured": "the measured cell",
}
SPLIT_METRICS = ("pct_oracle", "return", "oracle_return", "regret", "path_frac", "success",
                 "stop_rate", "stop_first", "stop_low", "stop_low_path_frac", "length",
                 "end_cap", "n")


# ── Checkpoint ──────────────────────────────────────────────────────────────


def load_checkpoint(path: str, device: torch.device, compile: bool = False):
    """Agent (eval mode) and PPOConfig of a ppo.py checkpoint, plus its info (name,
    iteration, env steps, loaded / skipped / missing tensors)."""
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = config_from_dict(ck.get("cfg") or {})
    acfg = AgentConfig.from_dict(ck["agent_cfg"])
    del ck
    agent = Agent(acfg)
    info = agent.init_from(path)
    info.pop("vnorm", None)
    agent.to(device).eval().requires_grad_(False)
    if compile:
        agent.compile()
    return agent, cfg, info


def logged_eval(ckpt: str, iteration: Optional[int]) -> Dict[str, float]:
    """The row of the run's eval.csv (next to the checkpoint) at the checkpoint's
    iteration, if any (consistency check of the sets the run logged)."""
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "eval.csv")
    if iteration is None or not os.path.exists(path):
        return {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("iteration") == str(iteration):
                out = {}
                for k, v in row.items():
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                return out
    return {}


def default_sets(cfg) -> List[str]:
    """DEFAULT_SETS for --setups full runs, else the run's own set."""
    if cfg.setups != "full":
        return ["set4" if cfg.setups == "all" else cfg.setups]
    return list(DEFAULT_SETS)


# ── Metrics ─────────────────────────────────────────────────────────────────


def _f(v) -> Optional[float]:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def set_metrics(es: EvalSet, ep: Dict[str, torch.Tensor], pol, flat: Dict[str, float],
                sampled: bool) -> Dict[str, Any]:
    """Per split: the flat metrics of ppo.run_eval, split_metrics extras, head errors,
    the oracle's cap share, the median target-region size of the split's scenarios and
    the final bounces / highest bounce count of the episodes that stop below 0.9 P_ref;
    plus the set's head errors."""
    pre = es.prefix_samp if sampled else es.prefix
    cap = es.env.cfg.max_bounces
    dev = ep["ep_return"].device
    o_b = es.oracle["target_bounces"].to(dev)
    o_m = es.oracle["moves"].to(dev)
    reg = None if es.region is None else es.region.to(dev)[ep["setup"]]
    top = None if es.best_bounces is None else es.best_bounces.to(dev)[ep["setup"]].double()
    frac = ep["final_path_mm"].double() / ep["path_ref"].double()
    low = ep["terminated"] & (frac < SUCCESS_FRAC)  # Stopped short of success
    out: Dict[str, Any] = {"splits": {}, "head": pol.diagnostics(), "wall_s": flat[f"{pre}/wall_s"]}
    for split, m in es.splits(ep).items():
        d = {k: _f(flat.get(f"{pre}/{split}/{k}")) for k in SPLIT_METRICS}
        if bool(m.any()):
            s = split_metrics(ep, m, cap)
            d.update({k: s[k] for k in ("final_path_m_mean", "init_path_m_mean", "end_ge31",
                                        "end_ge57", "end_ge89", "truncated", "length_p90",
                                        "moves_mean", "final_bounce_bins", "trunc_low")})
            d["oracle_end_cap"] = float((o_b[m] >= cap).double().mean())
            d["oracle_moves_mean"] = float(o_m[m].double().mean())
            d["head"] = pol.diagnostics(m)
            if reg is not None:
                d["region_median"] = float(reg[m].median())
            ml = m & low
            if top is not None and bool(ml.any()):
                d["stop_low_bounce_frac"] = float((ep["final_bounces"][ml].double() / top[ml]).mean())
        out["splits"][split] = d
    return out


def table_summary(es: EvalSet) -> Optional[Dict[str, Any]]:
    """Scenario-table statistics of the set: the whole table (all candidates; the grid
    points for the atlas sets; none for the region table, whose rows are the episodes)
    and the scenarios of the eval episodes."""
    from scenarios import SEP_BINS, TABLE_SEP_BINS, table_stats

    if es.table is None:
        return None
    bins = es.bins or TABLE_SEP_BINS.get(es.name, SEP_BINS)
    rows = torch.unique(es.setups)
    whole = None if es.name in ("full", "holdout", "region") else table_stats(es.table, bins=bins)
    off = getattr(es.table, "path", "")
    return {"table": whole, "eval": table_stats(es.table, rows, bins),
            "distinct_scenarios": int(rows.numel()), "file": off, "build": es.table.offgrid}


def setup_summary(es: EvalSet) -> Optional[List[Dict[str, Any]]]:
    """Per named setup of a setups set with box statistics (set4): parameters, P_max,
    highest bounce count, target-region size and area at the highest bounce count
    (deg^2), region bin."""
    from scenarios import REGION_BINS, region_bin

    if es.table is not None or es.region is None:
        return None
    pm = es.env.box_stats()["pmax"].tolist()
    b = region_bin(es.region.cpu()).tolist()
    out = []
    for i, g in enumerate(es.env.setups):
        out.append({"name": g.name, "separation": g.separation, "laser_pitch": g.laser_pitch,
                    "laser_yaw": g.laser_yaw, "laser_dx": g.laser_dx, "laser_dy": g.laser_dy,
                    "pmax_mm": pm[i], "best_bounces": int(es.best_bounces[i]),
                    "region_deg2": float(es.region[i]), "best_area_deg2": float(es.best_area[i]),
                    "region_bin": REGION_BINS[b[i]]})
    return out


# ── Report ──────────────────────────────────────────────────────────────────


def fmt(v, nd: int = 3, pct: bool = False, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    v = 100.0 * v if pct else v
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _table(head: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
        ["| " + " | ".join(r) + " |" for r in rows]


def _metric_row(d: Dict[str, Any]) -> List[str]:
    return [str(int(d["n"])) if d.get("n") is not None else "0", fmt(d.get("pct_oracle"), 1),
            fmt(d.get("return")), fmt(d.get("oracle_return")), fmt(d.get("regret")),
            fmt(d.get("path_frac")), fmt(d.get("success"), 1, True),
            fmt(d.get("stop_rate"), 1, True), fmt(d.get("stop_first"), 1, True),
            fmt(d.get("stop_low"), 1, True), fmt(d.get("length"), 1),
            f"{fmt(d.get('end_cap'), 1, True)} / {fmt(d.get('oracle_end_cap'), 1, True)}"]


METRIC_HEAD = ["n", "% oracle", "return", "oracle ret", "regret", "P / P_ref", "success %",
               "stop %", "stop first %", "stop < 0.9 %", "length", "at cap % (policy / oracle)"]


def _region_row(label: str, region: str, d: Dict[str, Any]) -> List[str]:
    return [label, region, str(int(d["n"])) if d.get("n") is not None else "0",
            fmt(d.get("pct_oracle"), 1), fmt(d.get("success"), 1, True), fmt(d.get("path_frac")),
            fmt(d.get("stop_rate"), 1, True), fmt(d.get("stop_first"), 1, True),
            fmt(d.get("length"), 1), fmt(d.get("stop_low"), 1, True),
            fmt(d.get("stop_low_path_frac")), fmt(d.get("stop_low_bounce_frac")),
            fmt(d.get("trunc_low"), 1, True)]


REGION_HEAD = ["split", "region (deg^2)", "n", "% oracle", "success %", "P / P_ref", "stop %",
               "stop first %", "length", "stop < 0.9 P_ref %", "their P / P_ref",
               "their bounces / max", "step limit < 0.9 %"]


def _stats_row(label: str, st: Dict[str, Any], extra: str = "") -> List[str]:
    sh = (st.get("region") or {}).get("bin_share") or {}
    return [label, str(st["n"]), str(st["good"]), fmt(st["good_share"], 1, True),
            f"{fmt(st['pmax_m_q10'], 2)} / {fmt(st['pmax_m_q50'], 2)} / {fmt(st['pmax_m_q90'], 2)}",
            fmt(st["pmax_m_mean"], 2), fmt(st["pmax_cap_share"], 1, True),
            fmt(st["cap_cells_share"], 2, True), fmt(st["near_area_deg2_median"], 3),
            " / ".join(fmt(v, 1, True) for v in sh.values()) or "-", extra]


STATS_HEAD = ["table", "scenarios", "good", "good %", "good P_max q10 / q50 / q90 (m)",
              "good P_max mean (m)", "P_max at cap % of good", "box cells at cap %",
              "success region median (deg^2)", "good per region bin % (tiny / small / medium / large)",
              "eval episodes / distinct scenarios"]


def _h2h(seen: Dict[str, Any], unseen: Dict[str, Any], splits: List[str]) -> List[str]:
    head = ["split", "n seen / unseen", "% oracle seen", "unseen", "delta", "success % seen",
            "unseen", "delta", "P / P_ref seen", "unseen", "regret seen", "unseen",
            "stop % seen", "unseen", "length seen", "unseen"]
    rows = []
    for sp in splits:
        a, b = seen["splits"].get(sp, {}), unseen["splits"].get(sp, {})
        d = lambda k, s=1.0: (None if a.get(k) is None or b.get(k) is None
                              else s * (b[k] - a[k]))
        rows.append([sp, f"{int(a.get('n') or 0)} / {int(b.get('n') or 0)}",
                     fmt(a.get("pct_oracle"), 1), fmt(b.get("pct_oracle"), 1),
                     fmt(d("pct_oracle"), 1, sign=True),
                     fmt(a.get("success"), 1, True), fmt(b.get("success"), 1, True),
                     fmt(d("success", 100.0), 1, sign=True),
                     fmt(a.get("path_frac")), fmt(b.get("path_frac")),
                     fmt(a.get("regret")), fmt(b.get("regret")),
                     fmt(a.get("stop_rate"), 1, True), fmt(b.get("stop_rate"), 1, True),
                     fmt(a.get("length"), 1), fmt(b.get("length"), 1)])
    return _table(head, rows)


HEAD_KEYS = (("head_path_mae", "path MAE (P / P_ref)", 3), ("head_tilt_mae_deg", "tilt MAE (deg)", 3),
             ("head_tilt_mae_deg_ge3", "tilt MAE >= 3 bounces", 3),
             ("head_tilt_std_deg", "tilt std (deg)", 3), ("head_sep_mae_mm", "separation MAE (mm)", 1),
             ("head_laser_pitch_mae_deg", "laser pitch MAE (deg)", 2),
             ("head_laser_yaw_mae_deg", "laser yaw MAE (deg)", 2),
             ("head_laser_dx_mae_mm", "laser dx MAE (mm)", 3),
             ("head_laser_dy_mae_mm", "laser dy MAE (mm)", 3))


def _head_rows(label: str, h: Dict[str, float], sfx: str = "") -> List[str]:
    return [label] + [fmt(h.get(k + sfx), nd) for k, _, nd in HEAD_KEYS]


def _region_splits(s: Dict[str, Any], pol: str) -> List[str]:
    """Target-region splits of a set (all first), in bin order."""
    sp = s["policies"][pol]["splits"]
    return ["all"] + [k for k in sp if k.startswith("region_")]


def region_section(res: Dict[str, Any]) -> List[str]:
    """Target-region size: definition and bins, the distribution, the named setups, the
    results per bin (and per named setup), with --compare the other run's next to them."""
    from scenarios import REGION_BINS, REGION_QS, region_bin_labels

    sets: Dict[str, Any] = res["sets"]
    pols: List[str] = res["policies"]
    labels = region_bin_labels()
    L = ["", "## Target-region size", ""]
    L.append("Target region of a scenario: the cells of the +-2 deg M2 box (0.01 deg grid, the "
             f"scan that gives P_max) whose path is >= {SUCCESS_FRAC:g} P_max, i.e. where an "
             "episode counts as success; its area in deg^2 (one cell 1e-4 deg^2). The cells "
             "at the highest bounce count lie inside it (equal or smaller area). Bins "
             "(scenarios.REGION_EDGES): " + ", ".join(
                 f"{nm} {labels['region_' + nm]}" for nm in REGION_BINS) + " deg^2.")
    # Distribution
    rows = []
    qk = [f"q{round(100 * x):02d}" for x in REGION_QS]
    for nm, s in sets.items():
        ts = s.get("tables") or {}
        b = ts.get("build") or {}
        if b.get("region"):
            rows.append([f"{nm}: candidate sample of the table ({b['good']} good of {b['valid']} "
                         f"valid, {b['candidates']} candidates)", b["region"]])
        st = ts.get("eval")
        if st and st.get("region") and nm != "region":
            rows.append([f"{nm}: good eval scenarios ({st['good']})", st["region"]])
    if rows:
        L += ["", "Distribution of the target-region size over good scenarios (deg^2) and the "
                  "share of each bin. The region table's candidate sample is the training "
                  "distribution (the good valid scenarios of a seeded stream of the ranges, up "
                  "to the last one the table keeps); the region set itself holds the same "
                  "number of episodes per bin.", ""]
        L += _table(["scenarios"] + qk + [f"{nm} %" for nm in REGION_BINS],
                    [[lab] + [fmt(st.get(k), 4) for k in qk]
                     + [fmt(st["bin_share"].get("region_" + nm), 1, True) for nm in REGION_BINS]
                     for lab, st in rows])
    # Named setups
    for nm, s in sets.items():
        if s.get("setups"):
            L += ["", f"Named setups ({nm}; P_max and both areas from the setup's 0.01 deg box "
                      "scan, AlignEnv.box_stats):", ""]
            L += _table(["setup", "separation (mm)", "laser pitch / yaw (deg)", "P_max (m)",
                         "bounces at P_max", "target region (deg^2)", "at max bounces (deg^2)", "bin"],
                        [[g["name"], f"{g['separation']:g}", f"{g['laser_pitch']:g} / {g['laser_yaw']:g}",
                          fmt(g["pmax_mm"] / 1000, 2), str(g["best_bounces"]), fmt(g["region_deg2"], 4),
                          fmt(g["best_area_deg2"], 4), g["region_bin"]] for g in s["setups"]])
    # Results per bin
    L += ["", "Results per target-region bin. Stop < 0.9 P_ref = episodes ended by not move "
              "below the success threshold (early stops), with their mean final P / P_ref and "
              "their final bounce count / the scenario's highest bounce count (< 1: stopped on "
              "a lower-order pattern); step limit < 0.9 = the 64 steps ran out below it. "
              "Success + stop < 0.9 + step limit < 0.9 = 100 %. cont: good scenarios only in "
              "the bins (all: every episode, 10 % impossible); region: good only; set4: the bins "
              "of the four setups (and per setup)."]
    for nm, s in sets.items():
        p0 = pols[0]
        sp = _region_splits(s, p0)
        if len(sp) == 1:
            continue
        L += ["", f"### {nm}", ""]
        for p in pols:
            spl = s["policies"][p]["splits"]
            rows = []
            for k in sp:
                d = spl.get(k, {})
                reg = labels.get(k, "all")
                rows.append(_region_row(k, reg, d))
            if nm == "set4":
                for g in s.get("setups") or []:
                    d = spl.get(g["name"], {})
                    rows.append(_region_row(g["name"], fmt(g["region_deg2"], 4), d))
            L += [f"**{p}**", ""] + _table(REGION_HEAD, rows) + [""]
    # Comparison
    cmp_ = res.get("compare")
    if cmp_:
        L += ["", f"### Comparison with run {cmp_['run_id']} ({cmp_['ckpt_stem']}.pt, iteration "
                  f"{cmp_['iteration']})", ""]
        L.append(f"`{cmp_['json']}`: {cmp_['name']}. Same sets, initial tilts and oracles; "
                 "this run first, then the other.")
        L.append("")
        head = ["set / policy / split", "n", "% oracle", "other", "success %", "other",
                "stop < 0.9 %", "other", "their P / P_ref", "other", "length", "other"]
        rows = []
        for nm, s in sets.items():
            if nm not in cmp_["sets"]:
                continue
            for p in pols:
                if p not in cmp_["sets"][nm]["policies"]:
                    continue
                a_sp = s["policies"][p]["splits"]
                b_sp = cmp_["sets"][nm]["policies"][p]["splits"]
                for k in _region_splits(s, p) + [g["name"] for g in s.get("setups") or []]:
                    a, b = a_sp.get(k, {}), b_sp.get(k, {})
                    rows.append([f"{nm} {p} {k}", str(int(a.get("n") or 0)),
                                 fmt(a.get("pct_oracle"), 1), fmt(b.get("pct_oracle"), 1),
                                 fmt(a.get("success"), 1, True), fmt(b.get("success"), 1, True),
                                 fmt(a.get("stop_low"), 1, True), fmt(b.get("stop_low"), 1, True),
                                 fmt(a.get("stop_low_path_frac")), fmt(b.get("stop_low_path_frac")),
                                 fmt(a.get("length"), 1), fmt(b.get("length"), 1)])
        L += _table(head, rows)
    return L


def report(res: Dict[str, Any]) -> str:
    """Markdown report of the result dict (main)."""
    sets: Dict[str, Any] = res["sets"]
    pols: List[str] = res["policies"]
    L = [f"# Evaluation of run {res['run_id']} ({res['ckpt_stem']}.pt)", ""]
    L.append(f"Generated by `uv run python eval_policy.py {res['cmd']}` ({res['date']}, "
             f"{res['device']}, {res['wall_min']:.1f} min). Checkpoint `{res['ckpt']}`: "
             f"{res['name']}, iteration {res['iteration']}, {res['env_steps'] / 1e6:.0f} M env "
             f"steps. Raw numbers: `{res['json_name']}`.")
    L.append("")
    L.append("Each set has the same initial tilts (uniform in the +-2 deg box, eval seed "
             f"{res['eval_seed']}) and {res['n']} episodes; `cont` and the grid sets 90 % good / "
             "10 % impossible scenarios (good: P_max >= 20 x separation), `region` good only, "
             "`set4` the four setups in equal shares. Return = (P_final - P_0) / P_ref - "
             f"{res['step_cost']} x moves; % oracle = mean return / mean oracle return on "
             "the same episodes; regret = oracle return - return; success = P_final >= 0.9 "
             "P_ref; stop % = episodes ended by not move (the rest hit the 64-step limit); "
             "stop first = not move at step 0; stop < 0.9 = not move below 0.9 P_ref; at cap = "
             "final frame at the 128-bounce cap (policy / oracle target). Separation and "
             "region bins count good scenarios only. Argmax = deterministic policy, sampled = "
             "one seeded sample per step (the trainer's eval noise).")
    L.append("")
    if res.get("train_dist"):
        L.append(res["train_dist"])
    else:
        L.append("Training distribution of this run: the atlas grid (separation 50-200 mm step 2, "
                 "laser pitch / yaw 0-8 deg step 0.25; 82,764 grid points + the named setups, "
                 "measured laser offsets), initial tilts uniform in the same box. `cont` and "
                 "`region` (separation 20-400 mm, laser offsets over the hole disk, laser angles "
                 "-8..8 deg) lie largely outside it; `full` measures in-distribution performance "
                 "only, `offgrid` interpolates, `extrap_short` / `extrap_long` extrapolate in "
                 "separation.")
    log = res.get("logged") or {}
    chk = []
    for nm, pre in (("cont", "eval_cont"), ("region", "eval_region"), ("full", "eval_full"),
                    ("set4", "eval")):
        if nm in sets and f"{pre}/all/pct_oracle" in log:
            for p, sfx in (("argmax", ""), ("sampled", "_samp")):
                k = f"{pre}{sfx}/all/pct_oracle"
                if p in sets[nm]["policies"] and k in log:
                    chk.append(f"{nm} {p} {sets[nm]['policies'][p]['splits']['all']['pct_oracle']:.2f}"
                               f" vs {log[k]:.2f}")
    if chk:
        L += ["", f"Consistency with the run's eval.csv at iteration {res['iteration']} "
                  f"(% oracle, here vs logged): " + "; ".join(chk) + "."]

    # ── Target-region size ──
    if any(any(k.startswith("region_") for k in s["policies"][pols[0]]["splits"]) for s in sets.values()):
        L += region_section(res)

    # ── Scenario tables ──
    L += ["", "## Scenario tables", ""]
    L.append("Whole table: every scanned candidate of the table (cont: the valid candidates of "
             "the shortest stream prefix holding the good and impossible counts; off-grid "
             "tables: the shortest prefix of a seeded random order of the lattice), the grid "
             "points for the atlas; none for the region table (its rows are the episodes). "
             "Eval: the distinct scenarios of the set's episodes. P_max from 0.01 deg scans of "
             "the +-2 deg M2 box at cap 128; box cells at cap = mean share of the 401^2 cells "
             "whose beam runs to the cap.")
    L.append("")
    rows = []
    if res.get("grid_stats"):
        rows.append(_stats_row("grid (atlas), all grid points", res["grid_stats"], "-"))
    for nm, s in sets.items():
        ts = s.get("tables")
        if not ts:
            continue
        if ts["table"] is not None:
            b = ts.get("build") or {}
            rows.append(_stats_row(f"{nm}, whole table", ts["table"],
                                   f"{b.get('candidates', '-')} candidates, "
                                   f"{b.get('scanned', '-')} scanned"))
        rows.append(_stats_row(f"{nm}, eval", ts["eval"],
                               f"{res['n']} / {ts['distinct_scenarios']}"))
    L += _table(STATS_HEAD, rows)

    # ── Overview ──
    L += ["", "## Overview (all episodes)", ""]
    rows = []
    for nm, s in sets.items():
        for p in pols:
            rows.append([nm, p] + _metric_row(s["policies"][p]["splits"]["all"]))
    L += _table(["set", "policy"] + METRIC_HEAD, rows)

    # ── Grid vs off-grid (interpolation) ──
    pool_run = bool(res.get("train_dist"))
    if "full" in sets and "offgrid" in sets:
        if pool_run:
            L += ["", "## Grid vs off-grid (fixed benchmarks, both inside the trained range)", ""]
            L.append("`full` (grid points of run 4407578's atlas) vs `offgrid` (midpoints between "
                     "them, separation 51-199 mm odd), same initial tilts; delta = offgrid - grid.")
        else:
            L += ["", "## Seen grid vs unseen off-grid (interpolation)", ""]
            L.append("`full` (training grid) vs `offgrid` (midpoints between grid points, separation "
                     "51-199 mm odd), same initial tilts; delta = unseen - seen.")
        sp = [k for k in sets["full"]["policies"][pols[0]]["splits"]]
        for p in pols:
            L += ["", f"**{p}**", ""]
            L += _h2h(sets["full"]["policies"][p], sets["offgrid"]["policies"][p], sp)
    if "full" in sets and "holdout" in sets:
        L += ["", "## Seen grid vs held-out grid points", ""]
        sp = [k for k in sets["full"]["policies"][pols[0]]["splits"]]
        for p in pols:
            L += ["", f"**{p}**", ""]
            L += _h2h(sets["full"]["policies"][p], sets["holdout"]["policies"][p], sp)

    # ── Head errors ──
    L += ["", "## Head errors (perception)", ""]
    L.append("Mean absolute error of the auxiliary heads (path: predicted P / P_ref of the "
             "current frame; tilt: M2 tilt; separation / laser pitch / yaw / offsets: scenario "
             "head, over the run's training ranges; '-' where the head lacks the parameter). "
             "All steps: over the active steps of the episodes, so sets with longer episodes "
             "weight later, better-informed frames more; first frame: step 0 only, the same "
             "initial tilts in every set (identical for both policies). A head error that "
             "grows with the return drop points at perception; similar head errors with a "
             "large return drop point at the policy.")
    L += ["", "All active steps:", ""]
    rows = []
    for nm, s in sets.items():
        for p in pols:
            rows.append(_head_rows(f"{nm} {p}", s["policies"][p]["head"]))
    L += _table(["set / policy"] + [h for _, h, _ in HEAD_KEYS], rows)
    L += ["", "First frame (step 0):", ""]
    rows = [_head_rows(nm, s["policies"][pols[0]]["head"], "_step0") for nm, s in sets.items()]
    L += _table(["set"] + [h for _, h, _ in HEAD_KEYS], rows)
    for a, b in (("full", "offgrid"),):
        if a in sets and b in sets:
            L += ["", f"Per split, {a} vs {b} (argmax; all steps / first frame):", ""]
            rows = []
            pa = sets[a]["policies"].get("argmax") or sets[a]["policies"][pols[0]]
            pb = sets[b]["policies"].get("argmax") or sets[b]["policies"][pols[0]]
            cols = (("head_path_mae", 3), ("head_path_mae_step0", 3), ("head_tilt_mae_deg_ge3", 3),
                    ("head_sep_mae_mm", 1), ("head_sep_mae_mm_step0", 1),
                    ("head_laser_pitch_mae_deg", 2), ("head_laser_yaw_mae_deg", 2))
            for sp in pa["splits"]:
                for nm, pp in ((a, pa), (b, pb)):
                    h = pp["splits"].get(sp, {}).get("head")
                    if h is not None:
                        rows.append([f"{sp} {nm}"] + [fmt(h.get(k), nd) for k, nd in cols])
            L += _table(["split / set", "path MAE", "path MAE step 0", "tilt MAE >= 3 bounces",
                         "separation MAE (mm)", "separation MAE step 0", "laser pitch MAE",
                         "laser yaw MAE"], rows)

    # ── Extrapolation ──
    ext = [nm for nm in ("extrap_short", "extrap_long") if nm in sets]
    if ext:
        if pool_run:
            L += ["", "## Separation 20-49 / 201-350 mm (fixed benchmarks of run 4407578)", ""]
            L.append("Inside this run's training range; for comparison with run 4407578, where "
                     "they were extrapolation. Laser pitch / yaw at the grid midpoints, measured "
                     "laser offsets, 0.01 deg box scans at cap 128, 90 / 10 good / impossible.")
        else:
            L += ["", "## Extrapolation (separation outside the trained 50-200 mm)", ""]
            L.append("Out-of-range generalization, not interpolation: separation 20-49 mm "
                     "(`extrap_short`) and 201-350 mm (`extrap_long`), laser pitch / yaw at the "
                     "midpoints k x 0.25 + 0.125 deg, 0.01 deg box scans at cap 128, the same "
                     "good / impossible rule and 90 / 10 mix. The frame encoder and the scenario "
                     "head were trained only on 50-200 mm. At short separations the 128-bounce "
                     "cap can flatten the landscape (e.g. 60 mm has patterns up to 403 bounces): "
                     "the at-cap columns show how much of it sits at the cap. Seen reference: the "
                     "`full` rows of the overview.")
        for nm in ext:
            s = sets[nm]
            ts = s.get("tables") or {}
            L += ["", f"### {nm}", ""]
            if ts.get("table"):
                t = ts["table"]
                L.append(f"Table: {t['n']} candidates, {t['good']} good ({100 * t['good_share']:.1f} %), "
                         f"good P_max at the cap {100 * t['pmax_cap_share']:.1f} %, box cells at the "
                         f"cap {100 * t['cap_cells_share']:.2f} % (good scenarios "
                         f"{100 * t['cap_cells_share_good']:.2f} %).")
                L.append("")
                rows = [[b, str(v["n"]), fmt(v["good_share"], 1, True), fmt(v["pmax_m_q50"], 2),
                         fmt(v["pmax_cap_share"], 1, True), fmt(v["cap_cells_share"], 2, True)]
                        for b, v in t["sep_bins"].items()]
                L += _table(["separation bin", "candidates", "good %", "good P_max q50 (m)",
                             "P_max at cap % of good", "box cells at cap %"], rows)
                L.append("")
            for p in pols:
                L += [f"**{p}**", ""]
                rows = [[sp] + _metric_row(d) for sp, d in s["policies"][p]["splits"].items()]
                L += _table(["split"] + METRIC_HEAD, rows)
                L.append("")

    # ── Details ──
    L += ["", "## Per-set details", ""]
    for nm, s in sets.items():
        if nm in ext:
            continue
        L += [f"### {nm} ({SET_INFO.get(nm, '')})", ""]
        for p in pols:
            L += [f"**{p}**", ""]
            rows = [[sp] + _metric_row(d) for sp, d in s["policies"][p]["splits"].items()]
            L += _table(["split"] + METRIC_HEAD, rows)
            L.append("")
    return "\n".join(L).rstrip() + "\n"


# ── Main ────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a ppo.py checkpoint on seen and unseen "
                                             "scenarios")
    ap.add_argument("--ckpt", required=True, help="ppo.py checkpoint (best.pt / latest.pt)")
    ap.add_argument("--sets", default="",
                    help="Comma list of " + ", ".join(EVAL_SET_NAMES) + " (default: "
                    + ",".join(DEFAULT_SETS) + "; --setups all / measured runs: their own set)")
    ap.add_argument("--policy", default="argmax,sampled", help="Comma list of argmax, sampled")
    ap.add_argument("--n", type=int, default=4096, help="Episodes per set")
    ap.add_argument("--envs", type=int, default=0, help="Eval envs (0 = the run's eval_envs)")
    ap.add_argument("--out", default=os.path.join(RL_DIR, "reports"), help="Report directory")
    ap.add_argument("--cache", default="", help="Oracle cache dir (default: the checkpoint's "
                    "dir when --n is the run's eval_n, else <out>/oracles_n<N>)")
    ap.add_argument("--atlas", default="", help="Atlas (default: the run's if it exists, else "
                    "scenarios.DEFAULT_ATLAS)")
    ap.add_argument("--tables", default="", help="Eval tables dir (default: the run's "
                    "eval_tables if it exists, else scenarios.EVAL_TABLES_DIR)")
    ap.add_argument("--tag", default="", help="Appended to the report name")
    ap.add_argument("--name", default="", help="Report name (default eval_<run id>_<ckpt stem>"
                    "[_<tag>])")
    ap.add_argument("--compare", default="", help="JSON of another eval_policy.py run: its "
                    "region-bin results next to these in the report")
    ap.add_argument("--compile", action="store_true", help="torch.compile the agent")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    t_start = time.time()
    pols = [p for p in args.policy.split(",") if p]
    assert all(p in POLICIES for p in pols), f"policies {pols} not in {POLICIES}"

    ckpt = os.path.abspath(args.ckpt)
    agent, cfg, info = load_checkpoint(ckpt, dev, args.compile)
    names = [s for s in args.sets.split(",") if s] or default_sets(cfg)
    assert all(s in EVAL_SET_NAMES for s in names), f"sets {names} not in {EVAL_SET_NAMES}"
    print(f"checkpoint {ckpt}: {info['name']}, iteration {info['iteration']}, "
          f"{len(info['loaded'])} tensors loaded; skipped {sorted(info['skipped'])}; "
          f"missing {info['missing']}", flush=True)
    run_n = cfg.eval_n
    cfg.eval_n = args.n
    if args.envs:
        cfg.eval_envs = args.envs
    cfg.eval_envs = min(cfg.eval_envs, args.n)
    if "holdout" in names:
        assert cfg.holdout_frac > 0, "the run has no hold-out (--holdout-frac 0)"
    from scenarios import DEFAULT_ATLAS, EVAL_TABLES_DIR, load_or_build, table_stats

    cfg.eval_tables = os.path.abspath(args.tables or (cfg.eval_tables if cfg.eval_tables
                                                      and os.path.isdir(cfg.eval_tables)
                                                      else EVAL_TABLES_DIR))
    atlas = None
    if any(s in ATLAS_SETS for s in names):
        path = args.atlas or (cfg.atlas if cfg.atlas and os.path.exists(cfg.atlas) else DEFAULT_ATLAS)
        atlas = load_or_build(path, dev, log=lambda s: print(s, flush=True))
        print(f"atlas {atlas.path}; off-grid tables {cfg.eval_tables}", flush=True)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.abspath(args.cache or (os.path.dirname(ckpt) if args.n == run_n
                                           else os.path.join(out_dir, f"oracles_n{args.n}")))
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    sets = build_eval_sets(cfg, names, atlas, cache, dev, log=lambda s: print(s, flush=True))
    print(f"{len(sets)} sets ready in {time.time() - t0:.1f} s (oracles in {cache})", flush=True)

    cmd = f"--ckpt {args.ckpt} --sets {','.join(names)} --policy {','.join(pols)} --n {args.n}"
    if args.name:
        cmd += f" --name {args.name}"
    if args.compare:
        cmd += f" --compare {args.compare}"
    res: Dict[str, Any] = {
        "ckpt": ckpt, "ckpt_arg": args.ckpt, "run_id": init_run_id(ckpt), "cmd": cmd,
        "ckpt_stem": os.path.splitext(os.path.basename(ckpt))[0], "name": info["name"],
        "iteration": info["iteration"], "env_steps": info["env_steps"] or 0, "n": args.n,
        "policies": pols, "eval_seed": cfg.eval_seed, "step_cost": cfg.step_cost,
        "date": time.strftime("%Y-%m-%d"), "device": torch.cuda.get_device_name(dev),
        "atlas": None if atlas is None else atlas.path, "tables_dir": cfg.eval_tables,
        "oracle_cache": cache, "tensors_skipped": sorted(info["skipped"]),
        "tensors_missing": info["missing"], "sets": {},
    }
    if cfg.setups == "full" and cfg.scenarios == "pool":
        grid = [nm for nm in names if nm in ATLAS_SETS]
        res["train_dist"] = (
            f"Training distribution of this run: continuous scenarios (D7), separation "
            f"{cfg.sep_range[0]:g}-{cfg.sep_range[1]:g} mm, laser offset uniform over the M1 "
            f"hole disk of radius {cfg.laser_offset_radius:g} mm (0: the measured offsets), "
            f"laser pitch / yaw within {cfg.laser_range[0]:g}..{cfg.laser_range[1]:g} deg "
            f"({'from a uniform M2 landing point' if cfg.laser_mode == 'landing' else 'uniform'}), "
            f"beams through the M1 hole onto M2 only, drawn from the scenario pool "
            f"({cfg.pool_size} slots, {cfg.pool_refresh:g} rescanned per iteration); `cont` "
            f"is a seeded set from the same distribution, `region` the same distribution "
            f"stratified by target-region size."
            + (f" The grid sets ({', '.join(f'`{g}`' for g in grid)}) keep the measured laser "
               f"offsets and serve as fixed benchmarks (comparable with run 4407578)." if grid else ""))
    if atlas is not None:
        res["grid_stats"] = table_stats(atlas)
    if args.compare:
        with open(args.compare) as f:
            other = json.load(f)
        res["compare"] = {"json": os.path.basename(args.compare), "sets": other["sets"],
                          **{k: other.get(k) for k in ("run_id", "ckpt_stem", "iteration", "name")}}
    for es in sets:
        s: Dict[str, Any] = {"info": SET_INFO.get(es.name, ""), "tables": table_summary(es),
                             "setups": setup_summary(es),
                             "oracle_return_mean": float(es.oracle["ret"].double().mean()),
                             "policies": {}}
        for p in pols:
            flat, ep, pol = run_eval(agent, es, cfg.eval_seed, p == "sampled", per_episode=True)
            s["policies"][p] = set_metrics(es, ep, pol, flat, p == "sampled")
            s["policies"][p]["flat"] = flat
            a = s["policies"][p]["splits"]["all"]
            print(f"  {es.name:12s} {p:7s}: {fmt(a['pct_oracle'], 1)} % of oracle, return "
                  f"{fmt(a['return'], 4, sign=True)}, success {fmt(a['success'])}, stop "
                  f"{fmt(a['stop_rate'])} (below 0.9 P_ref {fmt(a['stop_low'])}), len "
                  f"{fmt(a['length'], 1)} ({s['policies'][p]['wall_s']:.1f} s)", flush=True)
        res["sets"][es.name] = s
    res["logged"] = logged_eval(ckpt, info["iteration"])
    res["wall_min"] = (time.time() - t_start) / 60
    stem = args.name or (f"eval_{res['run_id']}_{res['ckpt_stem']}" + (f"_{args.tag}" if args.tag else ""))
    res["json_name"] = f"{stem}.json"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "compare"}, f, indent=1, default=float)
    with open(os.path.join(out_dir, f"{stem}.md"), "w") as f:
        f.write(report(res))
    print(f"wrote {os.path.join(out_dir, stem)}.md / .json ({res['wall_min']:.1f} min)", flush=True)
    return res


if __name__ == "__main__":
    main()
