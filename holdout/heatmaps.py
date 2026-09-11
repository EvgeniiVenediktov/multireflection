"""heatmaps.py - overlay the spatial hold-out split on a closed-loop eval trace.

Combines holdout/make_split.py's split.csv with a utils/eval_batched.py trace.csv
(a closed-loop replay over a start grid) to answer: how does the model do on
starts whose own position falls in TRAIN vs VAL vs TEST, and are the TEST/VAL
holes visible as worse regions in the corrections heatmap?

Outputs (in --out-dir):
  split_map.png            train/val/test over the position grid (like
                            make_split.py's split_map.png, but with a custom
                            title and reimplemented here to avoid depending on
                            the bank-scanning parts of make_split.py)
  corrections_heatmap.png  number of corrections per start, TEST/VAL hole
                            outlines overlaid, non-converged starts marked,
                            per-group stats in a text box
  angular_error_heatmap.png  (only with --also-error) same overlay, final
                            angular error instead of corrections
  corrections_by_split.csv   group, starts, success_rate, adjustments_mean,
                              final_angular_error_mean

A start is assigned to a group (train/val/test) by the split of its OWN
position (origin_x, origin_y), looked up in split.csv - not by whether its
trajectory ever crosses a hole.

Run from the repository root, e.g.:
    python holdout/heatmaps.py \\
        --split /home/evv/data/holdout_split/split.csv \\
        --trace eval_results/<run>_eval_grid/trace.csv \\
        --origins /home/evv/data/holdout_split/closed_loop_origins.txt \\
        --title "<run name>" \\
        --out-dir eval_results/<run>_eval_grid
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.lines
import matplotlib.ticker

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_CODE = {"train": TRAIN, "val": VAL, "test": TEST}
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
GROUP_ORDER = ["train", "val", "test"]

# distinct from SPLIT_COLORS so hole outlines read clearly on top of viridis
TEST_OUTLINE_COLOR = "#FF3B30"
VAL_OUTLINE_COLOR = "#FFD60A"
NOT_CONVERGED_COLOR = "#FF3B30"


def load_split_grid(path):
    """Read split.csv into a dense (ix, iy) -> split-code grid plus its bounds."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = list(reader)
    ix = np.array([int(r[0]) for r in rows], dtype=np.int64)
    iy = np.array([int(r[1]) for r in rows], dtype=np.int64)
    sp = np.array([r[4] for r in rows])

    ix_min, ix_max = int(ix.min()), int(ix.max())
    iy_min, iy_max = int(iy.min()), int(iy.max())
    h, w = ix_max - ix_min + 1, iy_max - iy_min + 1

    codes = np.full(sp.shape, TRAIN, dtype=np.int8)
    codes[sp == "val"] = VAL
    codes[sp == "test"] = TEST

    grid = np.zeros((h, w), dtype=np.int8)
    grid[ix - ix_min, iy - iy_min] = codes
    return grid, ix_min, ix_max, iy_min, iy_max


def split_of(grid, ix_min, ix_max, iy_min, iy_max, x_deg, y_deg):
    """Split code of the grid cell nearest (x_deg, y_deg); None if out of range."""
    ix = int(round(x_deg * 100))
    iy = int(round(y_deg * 100))
    if not (ix_min <= ix <= ix_max and iy_min <= iy <= iy_max):
        return None
    return int(grid[ix - ix_min, iy - iy_min])


def load_origins(path):
    """Origin (x, y) pairs from a closed_loop_origins.txt-style file of x{X}_y{Y}.jpg names."""
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(".jpg"):
                line = line[:-4]
            xpart, ypart = line.split("_y")
            xs.append(float(xpart[1:]))
            ys.append(float(ypart))
    return xs, ys


def load_trace_finals(path):
    """Final (max t) row per origin from trace.csv.

    Returns dict (origin_x, origin_y) -> dict(t, pos_x, pos_y, converged, err).
    """
    finals = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (float(row["origin_x"]), float(row["origin_y"]))
            t = int(row["t"])
            cur = finals.get(key)
            if cur is None or t > cur["t"]:
                x, y = float(row["pos_x"]), float(row["pos_y"])
                finals[key] = {
                    "t": t,
                    "pos_x": x,
                    "pos_y": y,
                    "converged": bool(int(row["converged"])),
                    "err": float(np.hypot(x, y)),
                }
    return finals


def to_grid(finals, xs, ys, field):
    ix = {v: i for i, v in enumerate(xs)}
    iy = {v: i for i, v in enumerate(ys)}
    z = np.full((len(ys), len(xs)), np.nan)
    for (x, y), d in finals.items():
        z[iy[y], ix[x]] = d[field]
    return z


def split_extent(ix_min, ix_max, iy_min, iy_max):
    """Cell-edge extent of the 0.01 deg position grid, so images and outlines line up exactly."""
    return ((ix_min - 0.5) / 100, (ix_max + 0.5) / 100, (iy_min - 0.5) / 100, (iy_max + 0.5) / 100)


def top_legend(ax, handles):
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
              fontsize=8, frameon=False, handlelength=1.5, columnspacing=1.2, borderaxespad=0.2)


def save_tight(fig, out_path):
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_split_map(grid, ix_min, ix_max, iy_min, iy_max, origins, title, out_path):
    extent = split_extent(ix_min, ix_max, iy_min, iy_max)
    cmap = matplotlib.colors.ListedColormap([SPLIT_COLORS["train"], SPLIT_COLORS["val"], SPLIT_COLORS["test"]])
    counts = {name: int(np.sum(grid == code)) for name, code in SPLIT_CODE.items()}

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.imshow(grid.T, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=2,
              aspect="equal", interpolation="nearest")
    handles = [matplotlib.patches.Patch(color=SPLIT_COLORS[name], label=f"{name} {100 * counts[name] / grid.size:.1f}%")
               for name in GROUP_ORDER]
    if origins is not None:
        ox, oy = origins
        handles.append(ax.scatter(ox, oy, s=8, c="black", marker="x", linewidths=0.8, label="origins"))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    top_legend(ax, handles)
    if title:
        ax.set_title(title, pad=18, fontsize=10)
    save_tight(fig, out_path)


def group_stats(finals, grid, ix_min, ix_max, iy_min, iy_max):
    """Per-group (train/val/test) success rate, mean corrections, mean final error."""
    buckets = {name: [] for name in GROUP_ORDER}
    for (x, y), d in finals.items():
        code = split_of(grid, ix_min, ix_max, iy_min, iy_max, x, y)
        if code is None:
            continue
        buckets[[k for k, v in SPLIT_CODE.items() if v == code][0]].append(d)

    stats = {}
    for name in GROUP_ORDER:
        items = buckets[name]
        n = len(items)
        if n == 0:
            stats[name] = dict(starts=0, success_rate=float("nan"),
                                adjustments_mean=float("nan"), final_angular_error_mean=float("nan"))
            continue
        conv = [d for d in items if d["converged"]]
        success_rate = len(conv) / n
        adjustments_mean = float(np.mean([d["t"] for d in conv])) if conv else float("nan")
        final_angular_error_mean = float(np.mean([d["err"] for d in items]))
        stats[name] = dict(starts=n, success_rate=success_rate,
                            adjustments_mean=adjustments_mean,
                            final_angular_error_mean=final_angular_error_mean)
    return stats


def stats_text(stats, sep="\n"):
    parts = []
    for name in GROUP_ORDER:
        s = stats[name]
        if s["starts"] == 0:
            parts.append(f"{name.upper()}: no starts")
            continue
        parts.append(f"{name.upper()}: n={s['starts']}  success={100 * s['success_rate']:.1f}%  "
                      f"adj={s['adjustments_mean']:.2f}  err={s['final_angular_error_mean']:.3f} deg")
    return sep.join(parts)


def add_hole_overlay(ax, grid, ix_min, ix_max, iy_min, iy_max):
    extent = split_extent(ix_min, ix_max, iy_min, iy_max)
    for code, color in ((TEST, TEST_OUTLINE_COLOR), (VAL, VAL_OUTLINE_COLOR)):
        ax.contour((grid == code).T.astype(float), levels=[0.5], colors=color, linewidths=0.8,
                   extent=extent, origin="lower")


def plot_metric_heatmap(finals, grid, ix_min, ix_max, iy_min, iy_max, field, label, integer_ticks,
                          title, out_path):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    xs = sorted({x for x, _ in finals})
    ys = sorted({y for _, y in finals})
    sx = xs[1] - xs[0] if len(xs) > 1 else 0.1
    sy = ys[1] - ys[0] if len(ys) > 1 else 0.1
    # Cell edges: every start owns a full cell, and the axes end where the start grid ends
    extent = (xs[0] - sx / 2, xs[-1] + sx / 2, ys[0] - sy / 2, ys[-1] + sy / 2)
    z = to_grid(finals, xs, ys, field)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(z, extent=extent, origin="lower", cmap="viridis", aspect="equal", interpolation="nearest")
    add_hole_overlay(ax, grid, ix_min, ix_max, iy_min, iy_max)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label, fontsize=9)
    if integer_ticks:
        vmin, vmax = int(np.nanmin(z)), int(np.nanmax(z))
        cbar.set_ticks(np.arange(vmin, vmax + 1, max(1, round((vmax - vmin) / 10))))
    else:
        cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))

    handles = [matplotlib.lines.Line2D([0], [0], color=TEST_OUTLINE_COLOR, lw=1.5, label="TEST hole"),
               matplotlib.lines.Line2D([0], [0], color=VAL_OUTLINE_COLOR, lw=1.5, label="VAL hole")]
    failed = [(x, y) for (x, y), d in finals.items() if not d["converged"]]
    if failed:
        handles.append(ax.scatter(*zip(*failed), marker="x", s=20, c=NOT_CONVERGED_COLOR, linewidths=1.1,
                                  label=f"not converged ({len(failed)})"))
    ax.set_xlabel("X origin (deg)")
    ax.set_ylabel("Y origin (deg)")
    top_legend(ax, handles)
    if title:
        ax.set_title(title, pad=18, fontsize=10)
    save_tight(fig, out_path)


def write_group_csv(stats, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "starts", "success_rate", "adjustments_mean", "final_angular_error_mean"])
        for name in GROUP_ORDER:
            s = stats[name]
            w.writerow([name, s["starts"], f"{s['success_rate']:.4f}" if s["starts"] else "",
                        f"{s['adjustments_mean']:.4f}" if s["starts"] else "",
                        f"{s['final_angular_error_mean']:.4f}" if s["starts"] else ""])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, help="split.csv from holdout/make_split.py")
    p.add_argument("--trace", required=True, help="trace.csv from utils/eval_batched.py")
    p.add_argument("--origins", default=None, help="closed_loop_origins.txt, marked on the split map")
    p.add_argument("--title", default=None, help="optional short title (e.g. '128 px, hold-out'); none by default")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--also-error", action="store_true",
                   help="also write angular_error_heatmap.png with the same overlay")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    grid, ix_min, ix_max, iy_min, iy_max = load_split_grid(args.split)
    origins = load_origins(args.origins) if args.origins else None
    finals = load_trace_finals(args.trace)
    stats = group_stats(finals, grid, ix_min, ix_max, iy_min, iy_max)

    plot_split_map(grid, ix_min, ix_max, iy_min, iy_max, origins, args.title,
                   os.path.join(args.out_dir, "split_map.png"))

    plot_metric_heatmap(finals, grid, ix_min, ix_max, iy_min, iy_max, "t", "Number of corrections", True,
                        args.title, os.path.join(args.out_dir, "corrections_heatmap.png"))

    if args.also_error:
        plot_metric_heatmap(finals, grid, ix_min, ix_max, iy_min, iy_max, "err", "Final angular error (deg)", False,
                            args.title, os.path.join(args.out_dir, "angular_error_heatmap.png"))

    write_group_csv(stats, os.path.join(args.out_dir, "corrections_by_split.csv"))

    print(f"{len(finals)} starts, split map: {sum(np.sum(grid == c) for c in (TRAIN, VAL, TEST))} positions")
    print(stats_text(stats).replace("\n", "  |  "))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
