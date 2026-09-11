"""Merge the sweep.csv tables of several models (utils/eval_sweep.py) into one markdown table.

    python utils/eval_compare.py r512=runs/r512_.../sweep/sweep.csv r256=... [--out compare.md]

One row per condition, in the order of the first table (conditions only present in later
tables are appended), one column per model. Each cell is success rate / mean adjustments
over converged starts / mean final angular error in degrees over all starts; for a
multi-seed sweep, success and error carry +- the standard deviation over perturbation seeds.
"""

import argparse
import csv
from pathlib import Path


def load(path):
    with open(path, newline="") as f:
        return {row["condition"]: row for row in csv.DictReader(f)}


def cell(row):
    if row is None:
        return "-"
    success = float(row["success_rate"])
    err = float(row["final_angular_error_mean"])
    adj = row["adjustments_mean"]
    adj = f"{float(adj):.2f}" if adj not in ("", "None", "n/a") else "n/a"
    if int(row.get("seeds") or 1) > 1:
        # mean +- standard deviation over perturbation seeds (utils/eval_sweep.py --perturb-seeds)
        rate = f"{100 * success:.2f}±{100 * float(row['success_rate_std']):.2f}%"
        return f"{rate} / {adj} / {err:.3f}±{float(row['final_angular_error_mean_std']):.3f}"
    rate = "100%" if success == 1.0 else f"{100 * success:.2f}%"
    return f"{rate} / {adj} / {err:.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+", help="LABEL=path/to/sweep.csv")
    p.add_argument("--out", default=None, help="also write the table to this file")
    p.add_argument("--conditions", default=None, help="comma-separated rows to keep (default: all)")
    p.add_argument("--quiet", action="store_true", help="with --out: write the file, print only its path")
    args = p.parse_args()
    if args.quiet and not args.out:
        p.error("--quiet needs --out")

    labels, tables = [], []
    for spec in args.tables:
        label, sep, path = spec.partition("=")
        if not sep:
            p.error(f"expected LABEL=path, got {spec}")
        labels.append(label)
        tables.append(load(Path(path)))

    conditions = []
    for t in tables:
        conditions += [c for c in t if c not in conditions]
    if args.conditions:
        wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
        conditions = [c for c in conditions if c in wanted]

    lines = ["| condition | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for c in conditions:
        lines.append(f"| {c} | " + " | ".join(cell(t.get(c)) for t in tables) + " |")
    md = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(md)
    if args.quiet:
        print(f"{len(conditions)} rows x {len(labels)} models -> {args.out}")
    else:
        print(md, end="")


if __name__ == "__main__":
    main()
