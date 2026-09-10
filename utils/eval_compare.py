"""Merge the sweep.csv tables of several models (utils/eval_sweep.py) into one markdown table.

    python utils/eval_compare.py r512=runs/r512_.../sweep/sweep.csv r256=... [--out compare.md]

One row per condition, in the order of the first table (conditions only present in later
tables are appended), one column per model. Each cell is success rate / mean adjustments
over converged starts / mean final angular error in degrees over all starts.
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
    rate = "100%" if success == 1.0 else f"{100 * success:.2f}%"
    adj = row["adjustments_mean"]
    adj = f"{float(adj):.2f}" if adj not in ("", "None", "n/a") else "n/a"
    return f"{rate} / {adj} / {float(row['final_angular_error_mean']):.3f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+", help="LABEL=path/to/sweep.csv")
    p.add_argument("--out", default=None, help="also write the table to this file")
    args = p.parse_args()

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

    lines = ["| condition | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for c in conditions:
        lines.append(f"| {c} | " + " | ".join(cell(t.get(c)) for t in tables) + " |")
    md = "\n".join(lines) + "\n"
    print(md, end="")
    if args.out:
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
