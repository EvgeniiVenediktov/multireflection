"""Render one example frame per closed-loop evaluation condition.

    python utils/perturbation_examples.py [--frame x0.00_y0.00.jpg] [--out-dir reports/perturbation_examples]

Every condition of utils/eval_sweep.CONDITIONS is applied to the same 512 px bank frame through
utils.eval_batched.Perturbation at perturbation seed 0, trajectory 0, step 0, which is the code
path the closed-loop sweep itself uses (brightness -> contrast -> noise -> occlusion -> 8 bit).
The frame is shown at the bank's 512 px, before the box average to a model's input size.

Outputs in --out-dir: <condition>.png per condition, contact_sheet.png with all of them, and
README.md listing what each condition draws.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from utils import eval_batched  # noqa: E402
from utils.eval_sweep import CONDITIONS  # noqa: E402


def render(frame, flags, device):
    """The condition's flags applied to one frame; returns (uint8 image, description)."""
    args = eval_batched.parse_args(list(flags))
    h, w = frame.shape
    x = torch.from_numpy(frame).to(device=device, dtype=torch.float32).div(255.0).view(1, 1, h, w)
    pert = eval_batched.Perturbation(args, 1, (h, w), device)
    if pert.enabled:
        x = pert.apply(x, [0], [0])
    img = (x.view(h, w).clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return img, pert.describe()


def contact_sheet(panels, frame_name, out_path, ncols=6):
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.05 * nrows))
    for ax, (name, img) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(name, fontsize=6, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Closed-loop evaluation conditions on {frame_name}, perturbation seed 0", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=eval_batched.DEFAULT_DATA_DIR, help="512 px bank")
    p.add_argument("--frame", default="x0.00_y0.00.jpg", help="bank file name to perturb")
    p.add_argument("--out-dir", default=os.path.join(REPO, "reports", "perturbation_examples"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    path = os.path.join(args.data_dir, args.frame)
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise SystemExit(f"cannot read {path}")
    os.makedirs(args.out_dir, exist_ok=True)

    panels, lines = [], []
    for name, flags in CONDITIONS:
        img, desc = render(frame, flags, args.device)
        cv2.imwrite(os.path.join(args.out_dir, f"{name}.png"), img)
        panels.append((name, img))
        lines.append(f"| `{name}` | {desc} |")

    contact_sheet(panels, args.frame, os.path.join(args.out_dir, "contact_sheet.png"))

    readme = [
        "# Perturbation examples",
        "",
        f"Every condition of the closed-loop sweep (`utils/eval_sweep.CONDITIONS`) applied to `{args.frame}`",
        "from the 512 px bank at perturbation seed 0, trajectory 0, step 0, through the same",
        "`utils.eval_batched.Perturbation` the evaluations use. Frames are shown at 512 px, before the",
        "box average to a model's input resolution.",
        "",
        "Regenerate with `python utils/perturbation_examples.py`.",
        "",
        "![all conditions](contact_sheet.png)",
        "",
        "| condition | what it draws |",
        "|---|---|",
        *lines,
        "",
    ]
    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("\n".join(readme))

    print(f"{len(panels)} conditions from {args.frame} -> {args.out_dir}")


if __name__ == "__main__":
    main()
