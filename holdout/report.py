"""Markdown report of spatial hold-out evaluations (holdout/eval_test.py outputs) for several models.

    python holdout/report.py r64=eval_results/<name>_holdout_test r128=... --out report.md [--quiet]

Each directory is one model's eval_test.py output (optionally with train_summary.json copied in).
Sections:
- TEST open-loop angular error per condition: mean over TEST positions, +- standard deviation of
  that mean over perturbation seeds, and (sd) the spread over positions.
- Clean error vs distance to TRAIN, with each model's nearest-neighbour baseline (mean +- sd over
  the positions in the bucket).
- Closed-loop replay from the origins: success % +- its standard error [95% Wilson interval],
  adjustments (converged starts) and final angular error as mean +- sd over origins.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in (None, "", "None", "nan") else None


def pm(mean, std, digits):
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if std is not None else "")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="LABEL=path/to/holdout_test dir")
    p.add_argument("--out", required=True)
    p.add_argument("--quiet", action="store_true", help="print only the output path")
    args = p.parse_args()

    models = []
    for spec in args.models:
        label, sep, path = spec.partition("=")
        if not sep:
            p.error(f"expected LABEL=path, got {spec}")
        models.append((label, Path(path)))
    labels = [m[0] for m in models]
    md = ["# Spatial hold-out evaluation", ""]

    # Settings
    md += ["## Models", ""]
    body = []
    for label, d in models:
        s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        t = json.loads((d / "train_summary.json").read_text()) if (d / "train_summary.json").exists() else {}
        nn = s.get("nn", {}) or {}
        m = rows(d / "metrics.csv")
        body.append([label, str(t.get("best_epoch", "n/a")), str(t.get("n_train", "n/a")), m[0]["n"] if m else "n/a",
                     f"{nn.get('resolution', 'n/a')} px, {nn.get('test_samples', 'n/a')} samples"])
    md += table(["model", "selected epoch", "train images", "TEST positions", "nearest neighbour"], body) + [""]

    # TEST open-loop
    metrics = {label: {r["condition"]: r for r in rows(d / "metrics.csv")} for label, d in models}
    conditions = []
    for label in labels:
        conditions += [c for c in metrics[label] if c not in conditions]
    md += ["## TEST open-loop angular error (deg)", "",
           "mean ± std over perturbation seeds (sd over TEST positions)", ""]
    body = []
    for c in conditions:
        cells = []
        for label in labels:
            r = metrics[label].get(c)
            if r is None:
                cells.append("-")
                continue
            cells.append(f"{pm(num(r['ang_err_mean']), num(r.get('ang_err_mean_std')), 4)} (sd {num(r['ang_err_std']):.3f})")
        body.append([c] + cells)
    md += table(["condition"] + labels, body) + [""]

    # Error vs distance, clean, with nearest neighbour
    md += ["## Clean error vs distance to TRAIN (deg)", "", "mean ± sd over the TEST positions in the bucket", ""]
    buckets, nn_buckets, dists = {}, {}, set()
    for label, d in models:
        buckets[label] = {r["dist_deg"]: r for r in rows(d / "buckets.csv") if r["condition"] == "clean"}
        nn_path = d / "nn_buckets.csv"
        nn_buckets[label] = {r["dist_deg"]: r for r in rows(nn_path) if r["method"] == "nn"} if nn_path.exists() else {}
        dists |= set(buckets[label])
    header = ["dist", "n"]
    for label in labels:
        header += [label, f"NN {label}"]
    body = []
    for dist in sorted(dists, key=float):
        n = next((buckets[l][dist]["n"] for l in labels if dist in buckets[l]), "")
        cells = [f"{float(dist):.2f}", n]
        for label in labels:
            b, q = buckets[label].get(dist), nn_buckets[label].get(dist)
            cells.append(pm(num(b["ang_err_mean"]), num(b["ang_err_std"]), 4) if b else "-")
            cells.append(pm(num(q["ang_err_mean"]), num(q["ang_err_std"]), 4) if q else "-")
        body.append(cells)
    md += table(header, body) + [""]

    # Closed loop
    cl_conditions = []
    for _, d in models:
        if (d / "closed_loop").is_dir():
            cl_conditions += [c.name for c in sorted((d / "closed_loop").iterdir()) if c.name not in cl_conditions]
    if cl_conditions:
        md += ["## Closed-loop replay from the origins", "",
               "success % ± standard error [95% Wilson]; adjustments (converged) and final error (deg) mean ± sd over origins", ""]
        body = []
        for c in cl_conditions:
            cells = []
            for _, d in models:
                f = d / "closed_loop" / c / "summary.json"
                if not f.exists():
                    cells.append("-")
                    continue
                s = json.loads(f.read_text())
                n, k = s["n_starts"], s["n_converged"]
                rate = k / n
                se = 100 * math.sqrt(rate * (1 - rate) / n)
                lo, hi = wilson(k, n)
                adj, err = s["adjustments_converged"], s["final_angular_error_deg"]
                cells.append(f"{100 * rate:.1f} ± {se:.1f} [{lo:.1f}, {hi:.1f}]; adj {pm(num(adj['mean']), num(adj['std']), 2)}; "
                             f"err {pm(num(err['mean']), num(err['std']), 3)}")
            body.append([c] + cells)
        md += table(["condition"] + labels, body) + [""]

    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"{len(conditions)} TEST conditions, {len(cl_conditions)} closed-loop conditions, {len(labels)} models -> {args.out}"
          if args.quiet else "\n".join(md))


if __name__ == "__main__":
    main()
