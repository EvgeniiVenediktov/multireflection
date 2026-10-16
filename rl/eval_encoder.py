"""
eval_encoder.py - Evaluation of pretrained frame encoders (PLAN phase 5, Level 1)

Run from rl/:
    uv run python eval_encoder.py                      # all runs in runs/encoder/
    uv run python eval_encoder.py runs/encoder/<run>   # selected runs
    uv run python eval_encoder.py --figures-only       # figures from existing metrics

Per run, on fixed seeded sets (FrameData.eval_set(EVAL_N, EVAL_SEED): the same states
rendered clean, in-distribution and OOD; FrameData.probe_set_89_trapped):
    detection  local maxima of the predicted sqrt map above DET_THRESHOLD, log-parabola
               sub-pixel refinement, Hungarian matching to the true M2 centres:
               precision / recall at MATCH_MM, mean localization error of the matches;
               the same detector on the target map gives the ceiling
    count      exact and within +-1 accuracy
    ellipse    centre error, semi-axis MAE, major-axis angle error (true a / b > 1.2)
    probes     linear probes on frozen z, fit on the first half, reported on the second:
               ridge for the M2 tilt (samples with >= 3 bounces; R^2, RMSE deg) and the
               bounce count (R^2, MAE); logistic regression for 89 vs trapped
               (accuracy); the same probes on raw pixels as a reference
    speed      encoder forward time per batch of 256 (eager bf16 / fp32, compiled bf16)
Writes runs/encoder/<run>/metrics.json, reports/encoder_tables.md, and the figures
figures/encoder_examples.png and figures/encoder_training.png.
"""

import argparse
import csv
import glob
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from encoders import EncoderConfig, FrameModel, count_params, model_name
from frame_data import MODES, DataConfig, FrameData
from pretrain_encoder import ELL_SCALE_MM, RUNS_DIR
from render_random import OOD_FACTORS
from targets import REGIONS, vec_to_ellipse

RL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_N = 8192
EVAL_SEED = 2026
PROBE_SEED = 4242
PROBE_PER_CLASS = 1024
DET_THRESHOLD = 0.15  # sqrt-map units (raw 0.0225; the dimmest true peak is 0.078 raw)
MATCH_MM = 0.5
MAX_PEAKS = 96
RIDGE_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
COLORS = {"cnn": "#2a78d6", "vit-p8": "#eb6834", "vit-p4": "#1baf7a"}  # Categorical 1-3

# Named M2 tilts (deg), nominal geometry (as tests/test_env.py)
EXAMPLES = {
    "nominal (31)": (-0.38, 0.27),
    "57 island": (-0.55, 0.62),
    "89 island": (0.17, 0.97),
    "trapped": (0.10, 1.04),
    "blind (1 bounce)": (-1.8, -1.8),
}


# ── Model I/O ───────────────────────────────────────────────────────────────


def load_run(run_dir: str, device: torch.device) -> FrameModel:
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    cfg = EncoderConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items()})
    model = FrameModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def predict(model: FrameModel, images: torch.Tensor, bs: int = 512) -> Dict[str, torch.Tensor]:
    """z (float32), sqrt map (float32), count logits, ellipse vector (mm)."""
    outs: Dict[str, List[torch.Tensor]] = {"z": [], "map": [], "count": [], "ellipse": []}
    for i in range(0, images.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(images[i:i + bs])
        for k in outs:
            outs[k].append(o[k].float())
    d = {k: torch.cat(v) for k, v in outs.items()}
    d["ellipse"] = d["ellipse"] * ELL_SCALE_MM
    return d


# ── Detection ───────────────────────────────────────────────────────────────


def detect(sq_map: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor,
           thr: float = DET_THRESHOLD, k_max: int = MAX_PEAKS):
    """Peaks of (n, 1, H, W) sqrt maps: positions (n, k, 2) mm and valid (n, k)."""
    m = sq_map[:, 0].float()
    n, H, W = m.shape
    peaks = (m == F.max_pool2d(m[:, None], 3, 1, 1)[:, 0]) & (m > thr)
    score = torch.where(peaks, m, -1.0).flatten(1)
    val, idx = torch.topk(score, min(k_max, H * W), dim=1)
    valid = val > 0
    i, j = idx // W, idx % W
    L = torch.log(F.pad(m, (1, 1, 1, 1)).clamp_min(1e-6))  # Padded log map
    rows = torch.arange(n, device=m.device)[:, None]
    c = L[rows, i + 1, j + 1]

    def offset(lo, hi):
        den = lo - 2.0 * c + hi
        return torch.where(den < -1e-9, 0.5 * (lo - hi) / den, 0.0).clamp(-0.5, 0.5)

    dj = offset(L[rows, i + 1, j], L[rows, i + 1, j + 2])
    di = offset(L[rows, i, j + 1], L[rows, i + 2, j + 1])
    pitch = float(xs[1] - xs[0])
    pos = torch.stack((xs[j] + dj * pitch, ys[i] - di * pitch), -1)
    return pos, valid


def match(pred: torch.Tensor, pvalid: torch.Tensor, true: torch.Tensor, tvalid: torch.Tensor):
    """Per-sample Hungarian matching: TP within MATCH_MM, #pred, #true, sum of matched
    distances (n,) numpy arrays."""
    P, PV = pred.cpu().numpy(), pvalid.cpu().numpy()
    T, TV = true.cpu().numpy(), tvalid.cpu().numpy()
    n = P.shape[0]
    tp = np.zeros(n, np.int64)
    dsum = np.zeros(n)
    npred, ntrue = PV.sum(1), TV.sum(1)
    for s in range(n):
        p, t = P[s, PV[s]], T[s, TV[s]]
        if len(p) == 0 or len(t) == 0:
            continue
        d = np.sqrt(((p[:, None] - t[None]) ** 2).sum(-1))
        r, c = linear_sum_assignment(np.where(d <= MATCH_MM, d, 1e3))
        ok = d[r, c] <= MATCH_MM
        tp[s] = ok.sum()
        dsum[s] = d[r, c][ok].sum()
    return tp, npred, ntrue, dsum


def det_summary(tp, npred, ntrue, dsum, sel: Optional[np.ndarray] = None) -> Dict[str, float]:
    if sel is not None:
        tp, npred, ntrue, dsum = tp[sel], npred[sel], ntrue[sel], dsum[sel]
    T = tp.sum()
    return {"precision": float(T / max(1, npred.sum())), "recall": float(T / max(1, ntrue.sum())),
            "loc_err_mm": float(dsum.sum() / max(1, T)), "n": int(len(tp))}


# ── Probes ──────────────────────────────────────────────────────────────────


def _standardize(Xtr: torch.Tensor, Xte: torch.Tensor):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    Xb = torch.cat((X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)), 1)
    reg = lam * torch.eye(Xb.shape[1], dtype=X.dtype, device=X.device)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)


def _ridge_pred(X: torch.Tensor, Wt: torch.Tensor) -> torch.Tensor:
    return X @ Wt[:-1] + Wt[-1]


def ridge_probe(Xtr, Ytr, Xte, Yte) -> Dict[str, float]:
    """Ridge on standardized features, lambda from an inner 80 / 20 split; R^2 (mean
    over outputs), RMSE and MAE on the test split."""
    Xtr, Xte = _standardize(Xtr.double(), Xte.double())
    Ytr, Yte = Ytr.double(), Yte.double()
    k = int(0.8 * Xtr.shape[0])
    best = min(RIDGE_GRID, key=lambda lam: float(
        ((_ridge_pred(Xtr[k:], _ridge_fit(Xtr[:k], Ytr[:k], lam)) - Ytr[k:]) ** 2).mean()))
    P = _ridge_pred(Xte, _ridge_fit(Xtr, Ytr, best))
    sse = ((P - Yte) ** 2).sum(0)
    sst = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return {"r2": float((1 - sse / sst).mean()), "rmse": float(((P - Yte) ** 2).mean().sqrt()),
            "mae": float((P - Yte).abs().mean()), "lambda": best, "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0])}


def logistic_probe(Xtr, ytr, Xte, yte) -> Dict[str, float]:
    """L2 logistic regression (LBFGS) on standardized features, lambda from an inner
    80 / 20 split; test accuracy."""
    Xtr, Xte = _standardize(Xtr.double(), Xte.double())
    ytr, yte = ytr.double(), yte.double()

    def fit(X, y, lam):
        w = torch.zeros(X.shape[1] + 1, dtype=X.dtype, device=X.device, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X @ w[:-1] + w[-1], y) + lam * (w[:-1] ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach()

    acc = lambda X, y, w: float((((X @ w[:-1] + w[-1]) > 0).double() == y).double().mean())
    k = int(0.8 * Xtr.shape[0])
    grid = (1e-4, 1e-3, 1e-2, 1e-1)
    best = max(grid, key=lambda lam: acc(Xtr[k:], ytr[k:], fit(Xtr[:k], ytr[:k], lam)))
    return {"accuracy": acc(Xte, yte, fit(Xtr, ytr, best)), "lambda": best,
            "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}


def probes(feat: torch.Tensor, ev: Dict[str, torch.Tensor], feat89: torch.Tensor,
           pr: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Tilt, bounce and 89 vs trapped probes; features (n, D) of the eval set and of
    the probe set."""
    n = feat.shape[0]
    half = n // 2
    tr = torch.arange(n, device=feat.device) < half
    sel = ev["bounces"] >= 3
    out = {"tilt": ridge_probe(feat[tr & sel], ev["tilt"][tr & sel], feat[~tr & sel], ev["tilt"][~tr & sel])}
    b = ev["bounces"].float()[:, None]
    out["bounces"] = ridge_probe(feat[tr], b[tr], feat[~tr], b[~tr])
    y = (pr["region"] == REGIONS.index("trapped")).float()
    h = feat89.shape[0] // 2
    out["89_vs_trapped"] = logistic_probe(feat89[:h], y[:h], feat89[h:], y[h:])
    return out


# ── Speed ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def forward_time(model: FrameModel, images: torch.Tensor, reps: int = 50) -> Dict[str, float]:
    """Median encoder forward ms per batch of 256 (eager bf16, eager fp32, compiled bf16)."""
    x = images[:256]
    res = {}
    comp = torch.compile(model.encoder)
    for name, fn, amp in (("eager_bf16_ms", model.encoder, True), ("eager_fp32_ms", model.encoder, False),
                          ("compiled_bf16_ms", comp, True)):
        ts = []
        for r in range(reps + 5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                s.record()
                fn(x)
                e.record()
            torch.cuda.synchronize()
            if r >= 5:
                ts.append(s.elapsed_time(e))
        res[name] = float(np.median(ts))
    return res


# ── Evaluation ──────────────────────────────────────────────────────────────


def ellipse_metrics(pred_vec: torch.Tensor, ev: Dict[str, torch.Tensor]) -> Dict[str, float]:
    m = ev["ellipse_mask"]
    p = vec_to_ellipse(pred_vec[m])
    t = vec_to_ellipse(ev["ellipse"][m])
    centre = torch.linalg.norm(p[:, :2] - t[:, :2], dim=1)
    ecc = t[:, 2] / t[:, 3].clamp_min(1e-6) > 1.2
    dphi = torch.remainder(p[:, 4] - t[:, 4], torch.pi)
    dphi = torch.rad2deg(torch.minimum(dphi, torch.pi - dphi))[ecc]
    return {"centre_mm": float(centre.mean()), "a_mae_mm": float((p[:, 2] - t[:, 2]).abs().mean()),
            "b_mae_mm": float((p[:, 3] - t[:, 3]).abs().mean()), "angle_deg": float(dphi.mean()),
            "angle_median_deg": float(dphi.median()), "n": int(m.sum()), "n_angle": int(ecc.sum())}


def count_metrics(logits: torch.Tensor, count: torch.Tensor) -> Dict[str, float]:
    pc = logits.argmax(1)
    return {"exact": float((pc == count).float().mean()),
            "within1": float(((pc - count).abs() <= 1).float().mean())}


def evaluate_run(run_dir: str, data: FrameData, ev, pr, oracle, pix_probes) -> Dict:
    dev = data.device
    model = load_run(run_dir, dev)
    xs, ys = data.base.xs, data.base.ys
    region = ev["region"].cpu().numpy()
    res = {"run": os.path.basename(run_dir), "model": model_name(model.cfg),
           "params_encoder": count_params(model.encoder), "params_heads": count_params(model.heads),
           "sets": {}, "probes": {}}
    for mode in MODES:
        o = predict(model, ev[f"image_{mode}"])
        pos, pv = detect(o["map"], xs, ys)
        tp, npred, ntrue, dsum = match(pos, pv, ev["spots"], ev["spot_valid"])
        r = {"detection": det_summary(tp, npred, ntrue, dsum),
             "count": count_metrics(o["count"], ev["count"]),
             "ellipse": ellipse_metrics(o["ellipse"], ev),
             "map_mse_sqrt": float(((o["map"] - ev["spot_map"].sqrt()) ** 2).mean())}
        r["by_region"] = {
            name: {**det_summary(tp, npred, ntrue, dsum, region == i),
                   "count_exact": float((o["count"].argmax(1) == ev["count"])[ev["region"] == i].float().mean())}
            for i, name in enumerate(REGIONS) if (region == i).any()}
        if mode == "ood":
            fac = ev["ood_factor"].cpu().numpy()
            r["by_factor"] = {
                name: {**det_summary(tp, npred, ntrue, dsum, fac == i),
                       "count_exact": float((o["count"].argmax(1) == ev["count"])[ev["ood_factor"] == i].float().mean())}
                for i, name in enumerate(OOD_FACTORS)}
        o89 = predict(model, pr[f"image_{mode}"])
        res["probes"][mode] = probes(o["z"], ev, o89["z"], pr)
        pos, pv = detect(o89["map"], xs, ys)
        m89 = match(pos, pv, pr["spots"], pr["spot_valid"])
        reg = pr["region"].cpu().numpy()
        r["probe_set"] = {
            name: {**det_summary(*m89, reg == REGIONS.index(name)),
                   "count_exact": float((o89["count"].argmax(1) == pr["count"])[pr["region"] == REGIONS.index(name)].float().mean())}
            for name in ("89", "trapped")}
        res["sets"][mode] = r
    res["speed"] = forward_time(model, ev["image_id"])
    res["oracle_detection"] = oracle
    res["pixel_probes"] = pix_probes
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        res["train_time_s"] = c.get("train_time_s")
        res["skipped_steps"] = c.get("skipped_steps")
        res["peak_mem_gb"] = c.get("peak_mem_gb")
    return res


def build_sets(data: FrameData, n: int = EVAL_N):
    ev = data.eval_set(n, EVAL_SEED)
    pr = data.probe_set_89_trapped(PROBE_PER_CLASS, PROBE_SEED)
    # Detector ceiling: the target map itself
    pos, pv = detect(ev["spot_map"].sqrt(), data.base.xs, data.base.ys)
    oracle = det_summary(*match(pos, pv, ev["spots"], ev["spot_valid"]))
    pos, pv = detect(pr["spot_map"].sqrt(), data.base.xs, data.base.ys)
    m89 = match(pos, pv, pr["spots"], pr["spot_valid"])
    reg = pr["region"].cpu().numpy()
    oracle["probe_set"] = {name: det_summary(*m89, reg == REGIONS.index(name)) for name in ("89", "trapped")}
    pix = {}
    for mode in MODES:
        f = F.avg_pool2d(ev[f"image_{mode}"].float(), 2).flatten(1)  # 32 x 32 pixels
        f89 = F.avg_pool2d(pr[f"image_{mode}"].float(), 2).flatten(1)
        pix[mode] = probes(f, ev, f89, pr)
    return ev, pr, oracle, pix


# ── Tables ──────────────────────────────────────────────────────────────────


def _mean_std(vals: List[float], fmt: str) -> str:
    if len(vals) == 1:
        return format(vals[0], fmt)
    return f"{format(np.mean(vals), fmt)} +- {format(np.std(vals), fmt)}"


def tables(results: List[Dict]) -> str:
    """Markdown tables: per model (mean +- std over seeds), per set."""
    models = [m for m in ("cnn", "vit-p8", "vit-p4") if any(r["model"] == m for r in results)]
    by = {m: [r for r in results if r["model"] == m] for m in models}
    L = []
    L.append("| model | seeds | encoder params | train min | fwd ms (bf16 eager / fp32 / compiled) |")
    L.append("|---|---|---|---|---|")
    for m in models:
        rs = by[m]
        sp = rs[0]["speed"]
        tt = [r["train_time_s"] / 60 for r in rs if r.get("train_time_s")]
        tts = _mean_std(tt, '.1f') if tt else "-"
        L.append(f"| {m} | {len(rs)} | {rs[0]['params_encoder'] / 1e6:.2f} M | {tts} | "
                 f"{sp['eager_bf16_ms']:.1f} / {sp['eager_fp32_ms']:.1f} / {sp['compiled_bf16_ms']:.1f} |")
    for mode, title in (("id", "in-distribution"), ("clean", "clean"), ("ood", "OOD")):
        L.append(f"\n**{title}** (n = {EVAL_N})\n")
        L.append("| model | precision | recall | loc err (mm) | count exact | count +-1 | "
                 "ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for m in models:
            g = lambda f: [f(r["sets"][mode]) for r in by[m]]
            L.append(
                f"| {m} | {_mean_std(g(lambda s: s['detection']['precision']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['detection']['recall']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['detection']['loc_err_mm']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['count']['exact']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['count']['within1']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['ellipse']['centre_mm']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['ellipse']['a_mae_mm']), '.3f')} / "
                f"{_mean_std(g(lambda s: s['ellipse']['b_mae_mm']), '.3f')} | "
                f"{_mean_std(g(lambda s: s['ellipse']['angle_deg']), '.2f')} |")
    o = results[0]["oracle_detection"]
    L.append(f"\nDetector on the target map (ceiling): precision {o['precision']:.3f}, recall "
             f"{o['recall']:.3f}, loc err {o['loc_err_mm']:.3f} mm.")

    L.append("\n**OOD by held-out factor** (recall / count exact)\n")
    L.append("| model | " + " | ".join(OOD_FACTORS) + " |")
    L.append("|---|" + "---|" * len(OOD_FACTORS))
    for m in models:
        cells = []
        for f in OOD_FACTORS:
            rec = [r["sets"]["ood"]["by_factor"][f]["recall"] for r in by[m]]
            ce = [r["sets"]["ood"]["by_factor"][f]["count_exact"] for r in by[m]]
            cells.append(f"{np.mean(rec):.3f} / {np.mean(ce):.3f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    L.append("\n**In-distribution by region** (recall / count exact)\n")
    regs = [g for g in REGIONS if g in results[0]["sets"]["id"]["by_region"]]
    L.append("| model | " + " | ".join(f"{g} (n={results[0]['sets']['id']['by_region'][g]['n']})" for g in regs) + " |")
    L.append("|---|" + "---|" * len(regs))
    for m in models:
        cells = []
        for g in regs:
            rec = [r["sets"]["id"]["by_region"][g]["recall"] for r in by[m]]
            ce = [r["sets"]["id"]["by_region"][g]["count_exact"] for r in by[m]]
            cells.append(f"{np.mean(rec):.3f} / {np.mean(ce):.3f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")

    o89 = results[0]["oracle_detection"]["probe_set"]
    L.append(f"\n**89 vs trapped probe set** ({PROBE_PER_CLASS} each; precision / recall / count exact; "
             f"detector ceiling recall 89 {o89['89']['recall']:.3f}, trapped {o89['trapped']['recall']:.3f})\n")
    L.append("| model | set | 89 | trapped |")
    L.append("|---|---|---|---|")
    for m in models:
        for mode in MODES:
            cells = []
            for g in ("89", "trapped"):
                q = lambda k: np.mean([r["sets"][mode]["probe_set"][g][k] for r in by[m]])
                cells.append(f"{q('precision'):.3f} / {q('recall'):.3f} / {q('count_exact'):.3f}")
            L.append(f"| {m} | {mode} | " + " | ".join(cells) + " |")

    L.append("\n**Linear probes on z** (test half; tilt on >= 3 bounces)\n")
    L.append("| model | set | tilt R^2 | tilt RMSE (deg) | bounces R^2 | bounces MAE | 89 vs trapped acc |")
    L.append("|---|---|---|---|---|---|---|")
    rows = [(m, by[m], lambda r, mode: r["probes"][mode]) for m in models]
    rows.append(("pixels (32 x 32)", results[:1], lambda r, mode: r["pixel_probes"][mode]))
    for m, rs, get in rows:
        for mode in MODES:
            g = lambda f: [f(get(r, mode)) for r in rs]
            L.append(f"| {m} | {mode} | {_mean_std(g(lambda p: p['tilt']['r2']), '.3f')} | "
                     f"{_mean_std(g(lambda p: p['tilt']['rmse']), '.3f')} | "
                     f"{_mean_std(g(lambda p: p['bounces']['r2']), '.3f')} | "
                     f"{_mean_std(g(lambda p: p['bounces']['mae']), '.2f')} | "
                     f"{_mean_std(g(lambda p: p['89_vs_trapped']['accuracy']), '.3f')} |")
    return "\n".join(L) + "\n"


# ── Figures ─────────────────────────────────────────────────────────────────


def _read_csv(path: str) -> Dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def figure_training(run_dirs: List[str], path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for rd in sorted(run_dirs):
        with open(os.path.join(rd, "config.json")) as f:
            c = json.load(f)
        m = "cnn" if c["encoder"]["arch"] == "cnn" else f"vit-p{c['encoder']['patch']}"
        seed = c["args"]["seed"]
        t, v = _read_csv(os.path.join(rd, "train.csv")), _read_csv(os.path.join(rd, "val.csv"))
        kw = dict(color=COLORS[m], lw=1.5, ls="-" if seed == 0 else "--", label=f"{m} s{seed}")
        axs[0].plot(t["step"], t["loss"], **kw)
        axs[1].plot(v["step"], v["map"], marker="o", ms=3, **kw)
        axs[2].plot(t["step"], t["grad_norm_max"], **kw)
    axs[0].set(yscale="log", xlabel="step", title="train loss (mean of 25 steps)")
    axs[1].set(yscale="log", xlabel="step", title="val map loss (in-distribution, 1024)")
    axs[2].set(yscale="log", xlabel="step", title="max grad norm per 25 steps (pre-clip)")
    for a in axs:
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def figure_examples(run_dirs: Dict[str, str], data: FrameData, path: str):
    """Rows: EXAMPLES states (nominal geometry, one in-distribution nuisance draw);
    columns: input, target sqrt map, predicted sqrt map per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = data.device
    tilts = torch.tensor(list(EXAMPLES.values()), dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(5)
    d = data.simulate(data.states(tilts, gen, jitter=False))
    img = data.render(d, "id", gen)["image"]
    preds = {m: predict(load_run(rd, dev), img) for m, rd in run_dirs.items()}
    R = data.base.radius
    ext = (-R, R, -R, R)
    ncol = 2 + len(preds)
    fig, axs = plt.subplots(len(EXAMPLES), ncol, figsize=(2.15 * ncol, 2.2 * len(EXAMPLES)))
    tmap = d["spot_map"].sqrt()
    vmax = float(tmap.max())
    for r, name in enumerate(EXAMPLES):
        axs[r, 0].imshow(img[r, 0].cpu(), cmap="gray", vmin=0, vmax=255, extent=ext)
        axs[r, 0].set_ylabel(f"{name}\n{int(d['bounces'][r])} bounces, {int(d['count'][r])} spots",
                             fontsize=8)
        axs[r, 1].imshow(tmap[r, 0].cpu(), cmap="magma", vmin=0, vmax=vmax, extent=ext)
        for c, (m, o) in enumerate(preds.items()):
            ax = axs[r, 2 + c]
            ax.imshow(o["map"][r, 0].clamp_min(0).cpu(), cmap="magma", vmin=0, vmax=vmax, extent=ext)
            pos, pv = detect(o["map"][r:r + 1], data.base.xs, data.base.ys)
            tp, npred, ntrue, _ = match(pos, pv, d["spots"][r:r + 1], d["spot_valid"][r:r + 1])
            ax.set_xlabel(f"{int(npred[0])} peaks, {int(tp[0])}/{int(ntrue[0])} matched; "
                          f"count {int(o['count'][r].argmax())}", fontsize=7)
        if r == 0:
            for c, t in enumerate(["input (randomized)", "target sqrt map"] + [f"{m} prediction" for m in preds]):
                axs[0, c].set_title(t, fontsize=9)
    for a in axs.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("M2 face from behind (+x right, +y up); maps in sqrt intensity, same color scale",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Evaluate pretrained frame encoders")
    ap.add_argument("runs", nargs="*", help="Run directories (default: all in runs/encoder)")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--fig-seed", type=int, default=0, help="Seed of the runs shown in the example figure")
    ap.add_argument("--eval-n", type=int, default=EVAL_N, help="Eval set size (debug: smaller)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    runs = args.runs or sorted(d for d in glob.glob(os.path.join(RUNS_DIR, "*"))
                               if os.path.exists(os.path.join(d, "model.pt")))
    runs = [os.path.abspath(r) for r in runs]
    data = FrameData(DataConfig(), dev)
    os.makedirs(os.path.join(RL_DIR, "reports"), exist_ok=True)
    os.makedirs(os.path.join(RL_DIR, "figures"), exist_ok=True)

    if not args.figures_only:
        t0 = time.time()
        ev, pr, oracle, pix = build_sets(data, args.eval_n)
        print(f"sets built in {time.time() - t0:.0f} s: eval {args.eval_n}, probe {pr['region'].shape[0]}, "
              f"oracle recall {oracle['recall']:.3f}")
        for rd in runs:
            t0 = time.time()
            res = evaluate_run(rd, data, ev, pr, oracle, pix)
            with open(os.path.join(rd, "metrics.json"), "w") as f:
                json.dump(res, f, indent=1)
            s = res["sets"]
            print(f"{res['run']}: id P/R {s['id']['detection']['precision']:.3f}/"
                  f"{s['id']['detection']['recall']:.3f} count {s['id']['count']['exact']:.3f}; "
                  f"ood R {s['ood']['detection']['recall']:.3f}; "
                  f"89/trapped {res['probes']['id']['89_vs_trapped']['accuracy']:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)

    results = []
    for rd in runs:
        p = os.path.join(rd, "metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                results.append(json.load(f))
    if results:
        with open(os.path.join(RL_DIR, "reports", "encoder_tables.md"), "w") as f:
            f.write(tables(results))
    figure_training(runs, os.path.join(RL_DIR, "figures", "encoder_training.png"))
    sel = {}
    for rd in runs:
        with open(os.path.join(rd, "config.json")) as f:
            c = json.load(f)
        m = "cnn" if c["encoder"]["arch"] == "cnn" else f"vit-p{c['encoder']['patch']}"
        if c["args"]["seed"] == args.fig_seed:
            sel[m] = rd
    order = [m for m in ("cnn", "vit-p8", "vit-p4") if m in sel]
    figure_examples({m: sel[m] for m in order}, data, os.path.join(RL_DIR, "figures", "encoder_examples.png"))
    print("tables: reports/encoder_tables.md; figures: figures/encoder_examples.png, encoder_training.png")


if __name__ == "__main__":
    main()
