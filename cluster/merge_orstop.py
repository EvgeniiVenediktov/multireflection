"""Merge the per-task outputs of cluster/eval_orstop.slurm into one table per model.

    uv run python cluster/merge_orstop.py eval_results/<stem>_sweep10_orstop [...]

Each directory holds task_<group>/<condition>/seed<k>/summary.json (as copied back from
<run dir>/sweep10_orstop/). Writes sweep_seeds.csv, sweep.csv and sweep.md into that directory,
one row per condition, mean over seeds, using utils/eval_sweep.py's columns and aggregation.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import eval_sweep  # noqa: E402

ORDER = [name for name, _ in eval_sweep.CONDITIONS]


def merge(root):
    root = Path(root)
    by_cond = {}
    for path in sorted(root.glob("task_*/*/seed*/summary.json")):
        cond, seed = path.parent.parent.name, int(path.parent.name[4:])
        summary = json.loads(path.read_text())
        by_cond.setdefault(cond, {})[seed] = {"seed": seed, **eval_sweep.row_from_summary(cond, summary)}
    rows, seed_rows = [], []
    for cond in sorted(by_cond, key=ORDER.index):
        runs = [by_cond[cond][s] for s in sorted(by_cond[cond])]
        seed_rows += runs
        rows.append(eval_sweep.aggregate(cond, runs))
    eval_sweep.write_table(rows, seed_rows, root)
    print(f"{root}: {len(rows)} conditions, " + ", ".join(f"{r['condition']}={r['seeds']}" for r in rows))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        merge(d)
