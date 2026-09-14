"""Markdown report of spatial hold-out evaluations for several models.

    python holdout/report.py r64=eval_results/<name>_holdout_test r128=... --out report.md [--quiet]

Each directory is one model's holdout/eval_test.py output (optionally with train_summary.json
copied in). Sections:
- Models: selected epoch, training images, TEST positions.
- Grid closed loop by start area: the clean replay from the 2,320 starts of the 0.1 deg grid,
  grouped by the split of each start's own position. Read from the sibling <run>_eval_grid
  directory written by holdout/heatmaps.py, when it exists.

Removed on purpose (see reports/holdout_resolution_study.md): the per-image TEST open-loop
tables, the nearest-neighbour baseline with its error-vs-distance buckets, and the closed-loop
replay from the 108 origins. The closed-loop sweep and the noise curve carry the robustness
results instead.
"""

import argparse
import csv
import json
from pathlib import Path

GROUP_ORDER = ["train", "val", "test"]
GROUP_LABELS = {"train": "TRAIN area", "val": "VALIDATION regions", "test": "TEST regions"}


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in (None, "", "None", "nan") else None


def table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def grid_dir_for(d):
    """Sibling <run>_eval_grid directory of a <run>_holdout_test[10|_s3_9] directory."""
    name = d.name
    for suffix in ("_holdout_test10", "_holdout_test_s3_9", "_holdout_test"):
        if name.endswith(suffix):
            return d.with_name(name[: -len(suffix)] + "_eval_grid")
    return d.with_name(name + "_eval_grid")


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
    md = ["# Spatial hold-out evaluation", ""]

    # Models
    md += ["## Models", ""]
    body = []
    for label, d in models:
        t = json.loads((d / "train_summary.json").read_text()) if (d / "train_summary.json").exists() else {}
        m = rows(d / "metrics.csv") if (d / "metrics.csv").exists() else []
        body.append([label, str(t.get("best_epoch", "n/a")), str(t.get("n_train", "n/a")),
                     m[0]["n"] if m else "n/a"])
    md += table(["model", "selected epoch", "train images", "TEST positions"], body) + [""]

    # Grid closed loop by start area
    grids = {}
    for label, d in models:
        f = grid_dir_for(d) / "corrections_by_split.csv"
        if f.exists():
            grids[label] = {r["group"]: r for r in rows(f)}
    if grids:
        starts = {g: next(iter(grids.values()))[g]["starts"] for g in GROUP_ORDER}
        md += ["## Grid closed loop by start area", "",
               "clean replay from the 0.1 deg grid; success %; mean corrections; final error (deg)", ""]
        header = ["model"] + [f"{GROUP_LABELS[g]} ({starts[g]})" for g in GROUP_ORDER]
        body = []
        for label, _ in models:
            if label not in grids:
                body.append([label] + ["-"] * len(GROUP_ORDER))
                continue
            cells = []
            for g in GROUP_ORDER:
                r = grids[label][g]
                cells.append(f"{100 * num(r['success_rate']):.1f}%; {num(r['adjustments_mean']):.2f}; "
                             f"{num(r['final_angular_error_mean']):.4f}")
            body.append([label] + cells)
        md += table(header, body) + [""]

    Path(args.out).write_text("\n".join(md) + "\n")
    print(f"{len(models)} models, {len(grids)} with grid replay -> {args.out}"
          if args.quiet else "\n".join(md))


if __name__ == "__main__":
    main()
