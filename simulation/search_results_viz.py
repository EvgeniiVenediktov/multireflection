"""
herriott_viz_results.py — Visualize stability search results
=============================================================

Reads Stage 1 & 2 outputs from results/ and produces:

  1. Laser angle maps  — bounce count heatmap in (pitch, yaw) space
                          with contiguous above-threshold region outlined
                          (laser angle tolerance comes directly from here)
  2. Scorecard         — table comparing separations at a glance
  3. Marginals         — per-DOF mechanical tolerance bars
  4. 2D slices         — pairwise heatmaps (max-projected over other axes)
  5. Bounce histogram  — fraction of parameter space at each bounce count
  6. Tradeoff scatter  — path length vs contiguous %, Pareto frontier

Usage:
    python herriott_viz_results.py
"""

import numpy as np
import json, os, itertools
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

OUTPUT_DIR = "results"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


# ================================================================
# Load data
# ================================================================


def load_stage1():
    with open(os.path.join(OUTPUT_DIR, "stage1.json")) as f:
        return json.load(f)


def load_stage2():
    with open(os.path.join(OUTPUT_DIR, "stage2_summary.json")) as f:
        return json.load(f)


def load_stage1_maps():
    return np.load(os.path.join(OUTPUT_DIR, "stage1_maps.npz"))


def load_grids(separation):
    return np.load(os.path.join(OUTPUT_DIR, f"stage2_grids_sep{separation:.0f}.npz"))


def best_laser(s2_entry):
    return max(s2_entry["laser_results"], key=lambda r: r["contiguous_cells"])


# ================================================================
# 1. Laser angle maps (from Stage 1)
# ================================================================


def plot_laser_angle_maps(s1_list):
    maps = load_stage1_maps()
    pitches = maps["pitches"]
    yaws = maps["yaws"]
    ext = [yaws[0], yaws[-1], pitches[0], pitches[-1]]

    n = len(s1_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for i, s1 in enumerate(s1_list):
        ax = axes[i]
        sep = s1["separation"]
        threshold = s1["bounce_threshold"]

        bmap_comp = maps[f"bounces_comp_sep{sep:.0f}"]
        bmap_uncomp = maps[f"bounces_uncomp_sep{sep:.0f}"]
        cont = maps.get(f"contiguous_sep{sep:.0f}", None)

        vmin = max(0, threshold - 5)
        im = ax.imshow(
            bmap_comp,
            extent=ext,
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=bmap_comp.max(),
        )

        # Bottom layer: compensated threshold (solid white)
        ax.contour(
            yaws,
            pitches,
            bmap_comp,
            levels=[threshold - 0.5],
            colors="white",
            linewidths=1.5,
            linestyles="-",
        )

        # Middle layer: contiguous region (cyan)
        if cont is not None:
            ax.contour(
                yaws,
                pitches,
                cont.astype(float),
                levels=[0.5],
                colors="cyan",
                linewidths=2,
            )

        # Top layer: uncompensated threshold (dashed yellow, always visible)
        ax.contour(
            yaws,
            pitches,
            bmap_uncomp,
            levels=[threshold - 0.5],
            colors="yellow",
            linewidths=1.5,
            linestyles="--",
        )

        # Centroid
        cx, cy = s1["centroid"]
        ax.plot(cy, cx, "k+", markersize=12, markeredgewidth=2)

        ax.set_xlabel("Laser yaw (deg)")
        ax.set_ylabel("Laser pitch (deg)")
        cp = s1["contiguous_pitch_range"]
        cy_r = s1["contiguous_yaw_range"]
        ax.set_title(
            f"sep={sep:.0f}mm | max={s1['max_bounces']} | thresh={threshold}\n"
            f"pitch:[{cp[0]:+.1f},{cp[1]:+.1f}] "
            f"yaw:[{cy_r[0]:+.1f},{cy_r[1]:+.1f}]",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, label="Bounces (m2 compensated)", shrink=0.8)

        # Legend
        from matplotlib.lines import Line2D

        legend_items = [
            Line2D([0], [0], color="white", lw=1.5, label="Compensated threshold"),
            Line2D(
                [0],
                [0],
                color="yellow",
                lw=1.5,
                ls="--",
                label="Uncompensated threshold",
            ),
            Line2D([0], [0], color="cyan", lw=2, label="Contiguous region"),
        ]
        ax.legend(handles=legend_items, fontsize=7, loc="lower left")

    fig.suptitle(
        "Laser Angle Tolerance (heatmap=compensated best bounces)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "1_laser_angle_maps.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  1_laser_angle_maps.png")


# ================================================================
# 2. Scorecard
# ================================================================


def plot_scorecard(s1_list, s2_list):
    rows = []
    for s1, s2 in zip(s1_list, s2_list):
        lr = best_laser(s2)
        sep = s2["separation"]
        path_m = lr["peak_bounces"] * sep / 1000
        worst_mech = min(abs(v[1] - v[0]) for v in lr["marginals"].values())
        pitch_tol = s1["contiguous_pitch_range"][1] - s1["contiguous_pitch_range"][0]
        yaw_tol = s1["contiguous_yaw_range"][1] - s1["contiguous_yaw_range"][0]

        rows.append(
            {
                "sep": sep,
                "bounces": lr["peak_bounces"],
                "thresh": s1["bounce_threshold"],
                "path_m": path_m,
                "pitch_tol": pitch_tol,
                "yaw_tol": yaw_tol,
                "contig_pct": lr["contiguous_pct"],
                "worst_mech": worst_mech,
                "laser": f"({lr['laser_pitch']:+.1f},{lr['laser_yaw']:+.1f})",
            }
        )

    fig, ax = plt.subplots(figsize=(16, 1.5 + 0.5 * len(rows)))
    ax.axis("off")
    headers = [
        "Sep(mm)",
        "Peak",
        "Thresh",
        "Path(m)",
        "P tol(deg)",
        "Y tol(deg)",
        "Mech %",
        "Worst mech",
        "Laser",
    ]
    cell_text = [
        [
            f"{r['sep']:.0f}",
            f"{r['bounces']}",
            f"{r['thresh']}",
            f"{r['path_m']:.2f}",
            f"{r['pitch_tol']:.2f}",
            f"{r['yaw_tol']:.2f}",
            f"{r['contig_pct']:.1f}",
            f"{r['worst_mech']:.3f}",
            r["laser"],
        ]
        for r in rows
    ]
    table = ax.table(
        cellText=cell_text, colLabels=headers, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    if rows:
        best_i = max(range(len(rows)), key=lambda i: rows[i]["contig_pct"])
        for j in range(len(headers)):
            table[best_i + 1, j].set_facecolor("#d4edda")

    fig.suptitle("Separation Scorecard", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "2_scorecard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  2_scorecard.png")


# ================================================================
# 3. Marginal tolerance bars
# ================================================================


def plot_marginals(s1_list, s2_list):
    n = len(s2_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, (s1, s2) in enumerate(zip(s1_list, s2_list)):
        ax = axes[i]
        lr = best_laser(s2)

        # Mechanical marginals from Stage 2
        names = list(lr["marginals"].keys())
        lo = [lr["marginals"][k][0] for k in names]
        hi = [lr["marginals"][k][1] for k in names]

        # Add laser angle tolerances from Stage 1
        names += ["laser_pitch", "laser_yaw"]
        cp = s1["contiguous_pitch_range"]
        cy = s1["contiguous_yaw_range"]
        lo += [cp[0], cy[0]]
        hi += [cp[1], cy[1]]

        y = np.arange(len(names))
        ax.barh(y, hi, left=0, height=0.6, color="#4CAF50", label="+")
        ax.barh(y, lo, left=0, height=0.6, color="#F44336", label="-")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"sep={s2['separation']:.0f}mm | {lr['peak_bounces']} bounces")
        ax.set_xlabel("Tolerance")
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "All Tolerances (mechanical: m2 compensated; angular: Stage 1)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "3_marginals.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  3_marginals.png")


# ================================================================
# 4. 2D pairwise stability slices
# ================================================================


def plot_2d_slices(s2_list):
    for s2 in s2_list:
        sep = s2["separation"]
        lr = best_laser(s2)
        lp_idx = s2["laser_results"].index(lr)

        data = load_grids(sep)
        grid = data[f"bounces_{lp_idx}"]
        axes_vals = {k: np.array(v) for k, v in s2["axes"].items()}
        ax_names = list(axes_vals.keys())
        threshold = lr["bounce_threshold"]

        pairs = list(itertools.combinations(range(len(ax_names)), 2))
        n_pairs = len(pairs)
        cols = min(n_pairs, 5)
        rows_n = (n_pairs + cols - 1) // cols

        fig, axarr = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n))
        axarr = np.atleast_2d(np.array(axarr).reshape(rows_n, cols))

        vmin = max(0, threshold - 5)
        vmax = grid.max()
        norm = Normalize(vmin=vmin, vmax=vmax)

        for pi, (d1, d2) in enumerate(pairs):
            ax = axarr[pi // cols, pi % cols]
            reduce = tuple(d for d in range(grid.ndim) if d not in (d1, d2))
            proj = grid.max(axis=reduce)

            v1 = axes_vals[ax_names[d1]]
            v2 = axes_vals[ax_names[d2]]
            ext = [v2[0], v2[-1], v1[0], v1[-1]]

            im = ax.imshow(
                proj,
                extent=ext,
                origin="lower",
                aspect="auto",
                cmap="RdYlGn",
                norm=norm,
            )
            ax.contour(
                v2,
                v1,
                proj,
                levels=[threshold - 0.5],
                colors="white",
                linewidths=1.5,
                linestyles="--",
            )
            ax.set_xlabel(ax_names[d2])
            ax.set_ylabel(ax_names[d1])
            ax.axhline(0, color="w", lw=0.3, alpha=0.5)
            ax.axvline(0, color="w", lw=0.3, alpha=0.5)

        for pi in range(n_pairs, rows_n * cols):
            axarr[pi // cols, pi % cols].set_visible(False)

        fig.colorbar(
            ScalarMappable(norm=norm, cmap="RdYlGn"),
            ax=axarr.ravel().tolist(),
            label="Best bounces (m2 compensated)",
            shrink=0.6,
        )
        fig.suptitle(
            f"sep={sep:.0f}mm | laser=({lr['laser_pitch']:+.1f},{lr['laser_yaw']:+.1f}) | "
            f"thresh={threshold} | peak={lr['peak_bounces']}",
            fontsize=11,
        )
        fig.tight_layout()
        fname = f"4_slices_sep{sep:.0f}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}")


# ================================================================
# 5. Bounce histogram
# ================================================================


def plot_bounce_histograms(s2_list):
    n = len(s2_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for i, s2 in enumerate(s2_list):
        ax = axes[i]
        lr = best_laser(s2)
        hist = lr["bounce_histogram"]
        threshold = lr["bounce_threshold"]

        counts_k = sorted(hist.keys(), key=int)
        vals = [hist[c] for c in counts_k]
        total = sum(vals)
        pcts = [100 * v / total for v in vals]
        counts_int = [int(c) for c in counts_k]

        colors = ["#4CAF50" if c >= threshold else "#BDBDBD" for c in counts_int]
        ax.bar(counts_int, pcts, color=colors, edgecolor="none", width=0.8)
        ax.axvline(
            threshold - 0.5,
            color="red",
            ls="--",
            lw=1.5,
            label=f"threshold={threshold}",
        )
        ax.set_xlabel("Bounces")
        ax.set_ylabel("% of parameter space")
        ax.set_title(f"sep={s2['separation']:.0f}mm")
        ax.legend(fontsize=8)

    fig.suptitle("Bounce Distribution Across Mechanical Disturbance Space", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "5_bounce_hist.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  5_bounce_hist.png")


# ================================================================
# 6. Tradeoff scatter
# ================================================================


def plot_tradeoff(s1_list, s2_list):
    fig, ax = plt.subplots(figsize=(8, 6))

    seps, paths, contigs, pitch_tols, yaw_tols = [], [], [], [], []
    for s1, s2 in zip(s1_list, s2_list):
        lr = best_laser(s2)
        sep = s2["separation"]
        seps.append(sep)
        paths.append(lr["peak_bounces"] * sep / 1000)
        contigs.append(lr["contiguous_pct"])
        pitch_tols.append(
            s1["contiguous_pitch_range"][1] - s1["contiguous_pitch_range"][0]
        )
        yaw_tols.append(s1["contiguous_yaw_range"][1] - s1["contiguous_yaw_range"][0])

    # Size = min angular tolerance, color = separation
    min_ang = [min(p, y) for p, y in zip(pitch_tols, yaw_tols)]
    sc = ax.scatter(
        contigs,
        paths,
        s=[a * 30 + 50 for a in min_ang],
        c=seps,
        cmap="viridis",
        edgecolors="k",
        linewidth=0.5,
        zorder=3,
    )

    for i in range(len(seps)):
        ax.annotate(
            f"  {seps[i]:.0f}mm\n  ang:{min_ang[i]:.1f}deg",
            (contigs[i], paths[i]),
            fontsize=7,
            va="center",
        )

    # Pareto frontier
    pts = sorted(zip(contigs, paths, seps), reverse=True)
    pareto_x, pareto_y = [], []
    best_y = -1
    for cx, py, s in pts:
        if py > best_y:
            pareto_x.append(cx)
            pareto_y.append(py)
            best_y = py
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "r--", lw=1.5, alpha=0.7, label="Pareto frontier")
        ax.legend()

    ax.set_xlabel("Mechanical contiguous stable region (%)")
    ax.set_ylabel("Optical path length (m)")
    ax.set_title(
        "Tradeoff: Path Length vs Stability\n(dot size = min laser angle tolerance)"
    )
    fig.colorbar(sc, label="Separation (mm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "6_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  6_tradeoff.png")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Herriott search results")
    parser.add_argument(
        "--stage1", action="store_true", help="Plot Stage 1 only (laser angle maps)"
    )
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Plot Stage 2 only (needs stage1.json + stage2_summary.json)",
    )
    args = parser.parse_args()

    plot_s1 = args.stage1 or not args.stage2
    plot_s2 = args.stage2 or not args.stage1

    os.makedirs(FIG_DIR, exist_ok=True)

    s1 = load_stage1()
    s1_by_sep = {e["separation"]: e for e in s1}
    print(f"Loaded Stage 1: {len(s1)} separations")

    if plot_s1:
        print("\nStage 1 figures:")
        plot_laser_angle_maps(s1)

    if plot_s2:
        s2 = load_stage2()
        s2_by_sep = {e["separation"]: e for e in s2}
        common = sorted(set(s1_by_sep) & set(s2_by_sep))
        s1_m = [s1_by_sep[s] for s in common]
        s2_m = [s2_by_sep[s] for s in common]

        if not common:
            print("No matching Stage 1 + Stage 2 results found.")
            exit()

        print(f"\nStage 2 figures ({len(common)} separations):")
        plot_scorecard(s1_m, s2_m)
        plot_marginals(s1_m, s2_m)
        plot_2d_slices(s2_m)
        plot_bounce_histograms(s2_m)
        plot_tradeoff(s1_m, s2_m)

    print(f"\nAll figures saved to {FIG_DIR}/")
