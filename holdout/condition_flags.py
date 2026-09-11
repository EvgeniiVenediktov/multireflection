"""Print the eval_batched.py flags of a condition from utils/eval_sweep.CONDITIONS (empty for clean).

    python holdout/condition_flags.py occlusion_2x30_rotbright
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.eval_sweep import CONDITIONS  # noqa: E402

if __name__ == "__main__":
    flags = dict(CONDITIONS)
    if len(sys.argv) != 2 or sys.argv[1] not in flags:
        sys.exit(f"usage: {sys.argv[0]} CONDITION, one of: {', '.join(flags)}")
    print(" ".join(flags[sys.argv[1]]))
