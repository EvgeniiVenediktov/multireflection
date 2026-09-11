"""make_split.py - spatial train/val/test split of a tilt-position image bank.

The image bank is a folder of JPEGs, one per 0.01 degree tilt position, named
x{X:.2f}_y{Y:.2f}.jpg. File names are identical across resolutions (dark512,
dark256, dark128, dark64), so a single split, computed from one folder's
listing, applies to all of them: this script only ever writes file names, it
never reads or copies image data.

The goal is a closed-loop evaluation grid (utils/eval_batched.py) with true
holdout: TEST is a set of square holes (Chebyshev radius 10, i.e. 21x21
positions) cut out around 108 evaluation origins spaced 0.5 degrees apart,
covering the whole bank. VAL is a second set of smaller holes (Chebyshev
radius 7) centered on the midpoints between origins, so validation loss is
measured on positions the model hasn't trained on but that are not part of
the closed-loop evaluation. TRAIN is everything left over. Distances are
Chebyshev (max of the two axis offsets) so a "hole" is a square, matching how
the closed loop is evaluated on a rectangular grid.

The exception is the origin at (0, 0): it is left as TRAIN because
utils/eval_batched.py --starts-file needs at least one origin whose
neighbourhood the model has actually seen, as a sanity check that the
closed-loop machinery itself works.

Outputs (in --out-dir):
  split.csv               ix, iy, x_deg, y_deg, split, dist_to_train_deg
  train_names.txt          real file names, sorted, one per line
  val_names.txt            (same)
  test_names.txt           (same)
  closed_loop_origins.txt  the 108 origins as file names, for
                           utils/eval_batched.py --starts-file
  counts.json              grid shape, per-split counts/percentages, hole
                           counts, clamped-origin count, distance histograms
  split_map.png            the grid colored by split, with origins marked

Run from the repository root:
    python holdout/make_split.py
    python holdout/make_split.py --data-dir /home/evv/data/dark64 --out-dir /home/evv/data/holdout_split
"""

import argparse
import json
import os
import re
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches

NAME_RE = re.compile(r"^x(-?\d+\.\d+)_y(-?\d+\.\d+)\.jpg$")

STEP = 50          # index units per 0.5 degree
TEST_RADIUS = 10
VAL_RADIUS = 7
VAL_OFFSET = 25     # index units between an origin and its midpoints

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def list_bank(data_dir):
    """Scan data_dir, return {(ix, iy): filename} and the index bounds."""
    positions = {}
    for name in os.listdir(data_dir):
        m = NAME_RE.match(name)
        if not m:
            continue
        ix = round(float(m.group(1)) * 100)
        iy = round(float(m.group(2)) * 100)
        positions[(ix, iy)] = name
    if not positions:
        raise RuntimeError(f"no x{{X}}_y{{Y}}.jpg files found in {data_dir}")
    ixs = [p[0] for p in positions]
    iys = [p[1] for p in positions]
    return positions, min(ixs), max(ixs), min(iys), max(iys)


def eval_origins():
    """108 (ox, oy) index pairs: X in [-2.0, 3.5] step 0.5, Y in [-2.0, 2.0] step 0.5."""
    ox_vals = range(-200, 351, STEP)
    oy_vals = range(-200, 201, STEP)
    return [(ox, oy) for ox in ox_vals for oy in oy_vals]


def val_midpoints():
    """88 (mx, my) index pairs, offset +0.25 deg from origins in [-2.0, 3.0] x [-2.0, 1.5]."""
    ox_vals = range(-200, 301, STEP)
    oy_vals = range(-200, 151, STEP)
    return [(ox + VAL_OFFSET, oy + VAL_OFFSET) for ox in ox_vals for oy in oy_vals]


def stamp_square(mask, cx, cy, radius, ix_min, iy_min):
    """Set mask True inside the Chebyshev ball of the given radius around (cx, cy),
    in array coordinates, clipped to the array bounds."""
    h, w = mask.shape
    lo_i = max(0, cx - ix_min - radius)
    hi_i = min(h - 1, cx - ix_min + radius)
    lo_j = max(0, cy - iy_min - radius)
    hi_j = min(w - 1, cy - iy_min + radius)
    if lo_i > hi_i or lo_j > hi_j:
        return 0
    before = mask[lo_i:hi_i + 1, lo_j:hi_j + 1].sum()
    mask[lo_i:hi_i + 1, lo_j:hi_j + 1] = True
    return int(mask[lo_i:hi_i + 1, lo_j:hi_j + 1].sum() - before)


def build_split(ix_min, ix_max, iy_min, iy_max):
    h, w = ix_max - ix_min + 1, iy_max - iy_min + 1
    test_mask = np.zeros((h, w), dtype=bool)
    val_mask = np.zeros((h, w), dtype=bool)

    n_test_holes = 0
    for ox, oy in eval_origins():
        if (ox, oy) == (0, 0):
            continue
        n_test_holes += 1
        stamp_square(test_mask, ox, oy, TEST_RADIUS, ix_min, iy_min)

    n_val_holes = 0
    for mx, my in val_midpoints():
        n_val_holes += 1
        stamp_square(val_mask, mx, my, VAL_RADIUS, ix_min, iy_min)

    assert not np.any(test_mask & val_mask), "TEST and VAL overlap"
    train_mask = ~test_mask & ~val_mask

    # Every position within Chebyshev 17 of the origin must be TRAIN (the
    # excluded (0, 0) origin, and no VAL midpoint reaches this close to it).
    ix_grid, iy_grid = np.meshgrid(
        np.arange(ix_min, ix_max + 1), np.arange(iy_min, iy_max + 1), indexing="ij")
    core = np.maximum(np.abs(ix_grid), np.abs(iy_grid)) <= 17
    assert np.all(train_mask[core]), "core region around (0, 0) is not all TRAIN"

    split = np.full((h, w), TRAIN, dtype=np.int8)
    split[val_mask] = VAL
    split[test_mask] = TEST

    return split, n_test_holes, n_val_holes


def compute_distance(split):
    non_train = split != TRAIN
    dist = distance_transform_edt(non_train) * 0.01
    return dist


def write_names(path, positions, split, code, ix_min, iy_min):
    names = []
    idx = np.argwhere(split == code)
    for i, j in idx:
        names.append(positions[(i + ix_min, j + iy_min)])
    names.sort()
    with open(path, "w") as f:
        f.write("\n".join(names) + "\n")
    return len(names)


def write_origins(path, positions, ix_min, ix_max, iy_min, iy_max):
    lines = []
    n_clamped = 0
    for ox, oy in eval_origins():
        cy = oy
        if cy not in range(iy_min, iy_max + 1):
            cy = max(iy_min, min(iy_max, cy))
            n_clamped += 1
        cx = max(ix_min, min(ix_max, ox))
        lines.append(positions[(cx, cy)])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines), n_clamped


def histogram(dist_values, bucket=0.01):
    if dist_values.size == 0:
        return {}
    n_buckets = int(np.ceil(dist_values.max() / bucket)) + 1
    edges = np.arange(n_buckets + 1) * bucket
    counts, _ = np.histogram(dist_values, bins=edges)
    return {f"{edges[i]:.2f}-{edges[i + 1]:.2f}": int(c) for i, c in enumerate(counts) if c}


def plot_split(split, dist, ix_min, ix_max, iy_min, iy_max, origins, out_path):
    x_min, x_max = ix_min / 100, ix_max / 100
    y_min, y_max = iy_min / 100, iy_max / 100
    # split is [ix, iy]; imshow wants [row=y, col=x].
    img = split.T
    cmap = matplotlib.colors.ListedColormap(["#4C72B0", "#DD8452", "#55A868"])  # train, val, test
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img, origin="lower", extent=[x_min, x_max, y_min, y_max],
               cmap=cmap, vmin=0, vmax=2, aspect="equal")
    ox = [o[0] / 100 for o in origins]
    oy = [o[1] / 100 for o in origins]
    origin_handle = ax.scatter(ox, oy, s=8, c="black", marker="x", linewidths=0.8, label="eval origins")
    zero_handle = ax.scatter([0], [0], s=40, facecolors="none", edgecolors="black", linewidths=1.2,
                              label="(0,0) origin (TRAIN)")
    handles = [
        matplotlib.patches.Patch(color="#4C72B0", label="train"),
        matplotlib.patches.Patch(color="#DD8452", label="val"),
        matplotlib.patches.Patch(color="#55A868", label="test"),
        origin_handle,
        zero_handle,
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8)
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    ax.set_title("Spatial train/val/test split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="/home/evv/data/dark64")
    parser.add_argument("--out-dir", default="/home/evv/data/holdout_split/")
    args = parser.parse_args()

    t0 = time.perf_counter()
    os.makedirs(args.out_dir, exist_ok=True)

    positions, ix_min, ix_max, iy_min, iy_max = list_bank(args.data_dir)
    h, w = ix_max - ix_min + 1, iy_max - iy_min + 1
    n_expected = h * w
    assert len(positions) == n_expected, (
        f"bank is not a dense rectangular grid: {len(positions)} files, expected {n_expected}")

    split, n_test_holes, n_val_holes = build_split(ix_min, ix_max, iy_min, iy_max)
    dist = compute_distance(split)

    # split.csv, sorted by ix, iy.
    csv_path = os.path.join(args.out_dir, "split.csv")
    with open(csv_path, "w") as f:
        f.write("ix,iy,x_deg,y_deg,split,dist_to_train_deg\n")
        for i in range(h):
            ix = i + ix_min
            for j in range(w):
                iy = j + iy_min
                f.write(f"{ix},{iy},{ix / 100:.2f},{iy / 100:.2f},"
                        f"{SPLIT_NAMES[split[i, j]]},{dist[i, j]:.4f}\n")

    n_train = write_names(os.path.join(args.out_dir, "train_names.txt"), positions, split, TRAIN, ix_min, iy_min)
    n_val = write_names(os.path.join(args.out_dir, "val_names.txt"), positions, split, VAL, ix_min, iy_min)
    n_test = write_names(os.path.join(args.out_dir, "test_names.txt"), positions, split, TEST, ix_min, iy_min)
    assert n_train + n_val + n_test == n_expected, "split does not cover every listed position exactly once"

    origins = eval_origins()
    n_origins, n_clamped = write_origins(
        os.path.join(args.out_dir, "closed_loop_origins.txt"), positions, ix_min, ix_max, iy_min, iy_max)

    test_dist = dist[split == TEST]
    val_dist = dist[split == VAL]

    counts = {
        "grid_shape": [h, w],
        "ix_range": [ix_min, ix_max],
        "iy_range": [iy_min, iy_max],
        "total_positions": n_expected,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "percentages": {
            "train": round(100 * n_train / n_expected, 2),
            "val": round(100 * n_val / n_expected, 2),
            "test": round(100 * n_test / n_expected, 2),
        },
        "n_test_holes": n_test_holes,
        "n_val_holes": n_val_holes,
        "n_origins": n_origins,
        "n_clamped_origins": n_clamped,
        "max_dist_test_deg": round(float(test_dist.max()) if test_dist.size else 0.0, 4),
        "max_dist_val_deg": round(float(val_dist.max()) if val_dist.size else 0.0, 4),
        "dist_histogram_test": histogram(test_dist),
        "dist_histogram_val": histogram(val_dist),
    }
    with open(os.path.join(args.out_dir, "counts.json"), "w") as f:
        json.dump(counts, f, indent=2)

    plot_split(split, dist, ix_min, ix_max, iy_min, iy_max, origins,
               os.path.join(args.out_dir, "split_map.png"))

    elapsed = time.perf_counter() - t0
    print(f"train: {n_train} ({100 * n_train / n_expected:.1f}%)")
    print(f"val:   {n_val} ({100 * n_val / n_expected:.1f}%)")
    print(f"test:  {n_test} ({100 * n_test / n_expected:.1f}%)")
    print(f"clamped origins: {n_clamped} (Y=2.00 row does not exist; (0,0) origin is TRAIN)")
    print(f"max dist_to_train_deg: test={counts['max_dist_test_deg']}, val={counts['max_dist_val_deg']}")
    print(f"elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
