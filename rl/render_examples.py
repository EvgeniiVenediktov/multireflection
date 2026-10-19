"""
render_examples.py - Example renders of render_camera.CameraRenderer (prototype)

Columns: the same 8 sim states (spot counts from a few to many, four setups; the
EDGE_COLS columns with the beam on the M2 rim where possible: a stored hit or the miss
point of a ray that ended past the rim). Rows: the
current clean render, then one factor group at a time (the rest neutral), then all
factors together (the training distribution; the last of these rows with the
camera-frame labels), then real frames at the encoder resolution for comparison.

    uv run python rl/render_examples.py [--out rl/figures/render_examples.png] [--seed 0]
"""

import argparse
import glob
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from frame_data import DataConfig, FrameData
from geometry import SETUPS
from render_camera import GROUPS, CameraRanges, CameraRenderer

RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(RL_DIR)
N_COL = 8
COUNTS = (1, 3, 6, 10, 16, 24, 34, 50)  # Wanted spot counts, nearest available
EDGE_COLS = (2, 4, 6)  # Prefer an edge-scatter source of weight >= EDGE_WEIGHT
EDGE_WEIGHT = 0.1

# (title, groups taken from the sampled parameters, range overrides, fixed values)
ROWS = (
    ("pose: rotation +-20 deg,\nshift, scale, perspective", ("pose",), {}, {"face": 0.08}),
    ("color camera: R / G / B laser,\ncrosstalk, tint, white bal.", ("color",),
     {"color_p": 1.0}, {"face": 0.1}),
    ("lit / dark face: rings, grain,\nedge glow, gradient", ("face", "light"), {}, {}),
    ("marks: engraving,\nscratches, dust", ("face", "marks"),
     {"face_p": 0.8, "engrave_p": 1.0}, {}),
    ("edge scatter: beam on the\nM2 rim (dark face)", ("scatter",), {}, {"face": 0.02}),
    ("spots: size, aspect,\nspeckle, laser power", ("spots",), {}, {}),
    ("sensor: exposure, gamma,\nnoise, PRNU, rows, hot px", ("sensor",),
     {"read_noise": (0.01, 0.04)}, {"face": 0.05}),
    ("all factors", tuple(GROUPS), {}, {}),
    ("all factors + labels\n(visible spots, count)", tuple(GROUPS), {}, {}),
)


def pick_states(fd: FrameData, cam: CameraRenderer, gen: torch.Generator):
    """N_COL frames with spot counts near COUNTS, edge scatter in EDGE_COLS, cycling the
    setups: hits, miss point, clean image."""
    d = fd.batch(gen, 2048, mode="clean")
    cnt = d["count"].cpu().numpy()
    setup = d["setup"].cpu().numpy()
    out = fd.sim.simulate(d["state"], store_world=False)  # FrameData keeps the hits only
    hm, lxy = out["hit_mirror"], out["hit_local_xy"]
    miss = (out["miss_mirror"], out["miss_local_xy"], out["intensity"])
    _, weight = cam.scatter_sources(hm, lxy, miss, cam.neutral_params(hm.shape[0]))
    on_edge = (weight[:, 0] >= EDGE_WEIGHT).cpu().numpy()
    used, idx = set(), []
    for j, want in enumerate(COUNTS):
        keys = (setup != j % fd.n_setups, np.abs(cnt - want))
        order = np.lexsort(keys + (~on_edge,) if j in EDGE_COLS else keys)
        i = next(int(i) for i in order if int(i) not in used)
        used.add(i)
        idx.append(i)
    sel = torch.tensor(idx, device=fd.device)
    return hm[sel], lxy[sel], tuple(t[sel] for t in miss), d["image"][sel]


def real_frames(n: int):
    paths = sorted(glob.glob(os.path.join(REPO, "graphs", "Input Examples", "*", "*.jpg")))
    paths = [p for p in paths if not os.path.basename(p).startswith("InputExample")]
    data = os.environ.get("MR_DATA_DIR", "/home/evv/data/dark128")
    paths += [os.path.join(data, f) for f in ("x0.00_y0.00.jpg", "x1.10_y-0.80.jpg")]
    out = []
    for p in paths:
        if os.path.exists(p):
            im = Image.open(p).convert("L").resize((64, 64), Image.BOX)
            out.append((np.repeat(np.asarray(im)[..., None], 3, -1), os.path.basename(p)))
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RL_DIR, "figures", "render_examples.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fd = FrameData(DataConfig(setups=SETUPS, cell_frac=0.7))
    gen = torch.Generator(device=fd.device)
    gen.manual_seed(args.seed)
    cam = CameraRenderer(fd.base, seed=args.seed)
    hm, lxy, miss, clean = pick_states(fd, cam, gen)

    rows = [("current env render\n(clean, mono)", clean.expand(-1, 3, -1, -1), None)]
    for title, groups, over, fixed in ROWS:
        p = cam.neutral_params(N_COL)
        s = cam.sample_params(N_COL, replace(CameraRanges(), **over), gen)
        for g in groups:
            p.update({k: s[k] for k in GROUPS[g]})
        for k, val in fixed.items():
            p[k] = torch.full_like(p[k], val)
        lab = cam.labels(hm, lxy, p) if "labels" in title else None
        rows.append((title, cam.render(hm, lxy, p, gen, miss), lab))

    real = real_frames(N_COL)
    n_rows = len(rows) + (1 if real else 0)
    fig, axes = plt.subplots(n_rows, N_COL, figsize=(1.25 * N_COL + 2.2, 1.3 * n_rows))
    fig.subplots_adjust(left=0.19, right=0.995, top=0.995, bottom=0.005, wspace=0.04, hspace=0.06)
    for i, (title, img, lab) in enumerate(rows):
        img = img.permute(0, 2, 3, 1).cpu().numpy()
        for j in range(N_COL):
            ax = axes[i, j]
            ax.imshow(img[j], interpolation="nearest")
            if lab is not None:
                vis = lab["visible"][j].cpu().numpy()
                xy = lab["pos_px"][j].cpu().numpy()[vis]
                ax.scatter(xy[:, 0], xy[:, 1], s=14, facecolors="none", edgecolors="#00e5ff",
                           linewidths=0.6)
                ax.text(1, 6, f"n={int(lab['count'][j])}", color="#00e5ff", fontsize=7)
                ax.set_xlim(-0.5, 63.5)
                ax.set_ylim(63.5, -0.5)
        axes[i, 0].text(-0.08, 0.5, title, transform=axes[i, 0].transAxes, ha="right",
                        va="center", fontsize=8)
    if real:
        for j in range(N_COL):
            if j < len(real):
                axes[-1, j].imshow(real[j][0], interpolation="nearest")
        axes[-1, 0].text(-0.08, 0.5, "real frames\n(64 x 64, other states)",
                         transform=axes[-1, 0].transAxes, ha="right", va="center", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(args.out)


if __name__ == "__main__":
    main()
