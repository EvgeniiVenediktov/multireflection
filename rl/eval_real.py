"""
eval_real.py - Policy evaluation on the real tilt raster

The real dataset (data_process/collect_real_data.py, /home/evv/data): 228,800 camera frames
of the real cell on the 572 x 400 tilt grid of the motorized mount, X -2.00 .. 3.71 deg
(vertical tilt), Y -2.00 .. 1.99 deg (horizontal tilt), 0.01 deg step; grayscale,
centre-cropped, circular-masked, stored at 64 / 128 / 256 px (dark64 / dark128 / dark256).

build   the raster cache in --out: raster64.npy (572, 400, 64, 64) uint8 (the dark64
        frames, the policy input), counts.npy (572, 400) int16 (spots per frame: the peak
        counter of tools/count_reflections.py on the dark128 frame), meta.json (grid)
eval    a ppo.py checkpoint (--encoder: with the encoder of a pretrain_encoder.py
        checkpoint instead, e.g. a 3-channel camera encoder) plays --episodes episodes on
        the raster: start at a seeded random grid tilt; after every action the frame of
        the new tilt is fetched. The ladder steps are multiples of 0.01 deg, so every move
        lands on the grid; moves are clamped to the grid edges and the clamped move is fed
        back as the applied action (as in the env); not move ends the episode, else
        max_steps of the run's env; argmax actions. No path or reward: the result is the
        pattern the policy ends on, its final frame and spot count. Frames of the bright
        elliptical band of the raster (mean grey level > BRIGHT_MEAN: beam on the mirror
        edge, scatter) make the counter count noise peaks, so their counts are left out
        of the count statistics and reported as the share ending there. Gray frames are
        copied to 3 channels for a 3-channel encoder.
        Axis map (--axis-map): the raster axis and sign each agent axis drives, agent axis
        0 = sim M2 pitch, 1 = yaw: "x,y" (default: +X, +Y), "-x,y", "y,x", ...; "all" =
        the eight maps.
Outputs (--out): real_eval.json (per map: share ending in the bright band, spot count at
start and end and the share ending on >= 16 / 29 / 45 spots (the 31 / 57 / 89 bounce
patterns) over the episodes outside it, final-count histogram, stop rate, mean length), episodes_<map>.npz (start / final grid index, final
count, length, stopped), final_frames.png (start and final frames of the first episodes
under the map with the most episodes ending on >= 16 spots).

    uv run python eval_real.py build --data /home/evv/data --out <cache dir>
    uv run python eval_real.py eval --ckpt <best.pt> [--encoder <model.pt>] --raster <cache dir>
        [--episodes 1024] [--seed 0] [--axis-map all] --out <dir>
"""

import argparse
import itertools
import json
import os
from dataclasses import replace
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import torch

NX, NY = 572, 400  # Grid size: X -2.00 .. 3.71, Y -2.00 .. 1.99 deg
X0 = Y0 = -2.0
STEP_DEG = 0.01
LEVELS = (16, 29, 45)  # M2 spots of the 31 / 57 / 89 bounce hole-exit patterns
BRIGHT_MEAN = 40.0  # Mean grey level above which a frame is in the bright band (counts unusable)

# Spot counter (tools/count_reflections.py, peak method; settings for 128 px)
THRESH = 70  # Grey levels
MIN_DIST = 3  # px, minimum separation of two spots
SMOOTH = 0.6  # px, Gaussian sigma before the peak search
MASK_FRAC = 0.93  # Tighter circular mask against the rim halo


# ── Raster cache ────────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    v = round(float(v), 2)
    return "%.2f" % (0.0 if v == 0 else v)  # No "-0.00" in the file names


def frame_name(ix: int, iy: int) -> str:
    return f"x{_fmt(X0 + STEP_DEG * ix)}_y{_fmt(Y0 + STEP_DEG * iy)}.jpg"


def count_spots(img: np.ndarray) -> int:
    """Spots in a preprocessed gray frame (tools/count_reflections.py, peak method)."""
    import cv2
    from scipy import ndimage

    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(w, h) // 2 * MASK_FRAC), 255, -1)
    img = np.where(mask == 255, img, 0).astype(np.uint8)
    scale = h / 128.0
    md = max(2, int(round(MIN_DIST * scale)))
    sm = cv2.GaussianBlur(img, (0, 0), SMOOTH * scale)
    mx = ndimage.maximum_filter(sm, size=2 * md + 1, mode="constant")
    ys, xs = np.nonzero((sm == mx) & (sm >= THRESH))
    keep: List[Tuple[float, float]] = []
    for i in np.argsort(-sm[ys, xs]):
        px, py = float(xs[i]), float(ys[i])
        if all((px - kx) ** 2 + (py - ky) ** 2 >= md**2 for kx, ky in keep):
            keep.append((px, py))
    return len(keep)


def _row(task: Tuple[str, int]) -> Tuple[int, np.ndarray, np.ndarray]:
    """Frames (NY, 64, 64) and spot counts (NY,) of one X row."""
    import cv2

    data, ix = task
    imgs = np.zeros((NY, 64, 64), np.uint8)
    cnt = np.zeros(NY, np.int16)
    for iy in range(NY):
        name = frame_name(ix, iy)
        im64 = cv2.imread(os.path.join(data, "dark64", name), cv2.IMREAD_GRAYSCALE)
        im128 = cv2.imread(os.path.join(data, "dark128", name), cv2.IMREAD_GRAYSCALE)
        assert im64 is not None and im128 is not None, f"missing {name}"
        imgs[iy] = im64
        cnt[iy] = count_spots(im128)
    return ix, imgs, cnt


def build(data: str, out: str, workers: int):
    os.makedirs(out, exist_ok=True)
    raster = np.lib.format.open_memmap(os.path.join(out, "raster64.npy"), mode="w+",
                                       dtype=np.uint8, shape=(NX, NY, 64, 64))
    counts = np.zeros((NX, NY), np.int16)
    with Pool(workers) as pool:
        for k, (ix, imgs, cnt) in enumerate(pool.imap_unordered(_row, [(data, i) for i in range(NX)])):
            raster[ix], counts[ix] = imgs, cnt
            if (k + 1) % 50 == 0:
                print(f"{k + 1} / {NX} rows", flush=True)
    raster.flush()
    np.save(os.path.join(out, "counts.npy"), counts)
    meta = {"nx": NX, "ny": NY, "x0": X0, "y0": Y0, "step_deg": STEP_DEG, "source": os.path.abspath(data),
            "frames": "dark64", "counts": "dark128, tools/count_reflections.py peak method",
            "count_max": int(counts.max()),
            "count_hist": np.bincount(counts.ravel().astype(np.int64)).tolist()}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"raster cache in {out}: max count {counts.max()}, frames with >= 16 / 29 / 45 spots "
          + " / ".join(str(int((counts >= v).sum())) for v in LEVELS), flush=True)


# ── Evaluation ──────────────────────────────────────────────────────────────


def parse_map(spec: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """'x,y' / '-y,x' / ... -> (raster axis per agent axis (0 = X, 1 = Y), sign per agent axis)."""
    toks = [t.strip().lower() for t in spec.split(",")]
    assert len(toks) == 2, f"axis map {spec!r}"
    axes = tuple("xy".index(t.lstrip("-")) for t in toks)
    assert sorted(axes) == [0, 1], f"axis map {spec!r}: each raster axis once"
    return axes, tuple(-1 if t.startswith("-") else 1 for t in toks)


def map_names(spec: str) -> List[str]:
    if spec != "all":
        return [spec]
    return [f"{s0}{a},{s1}{b}" for a, b in (("x", "y"), ("y", "x"))
            for s0, s1 in itertools.product(("", "-"), repeat=2)]


def load_agent(ckpt: str, encoder: str, dev: torch.device):
    """Agent of a ppo.py checkpoint (eval mode), its encoder replaced by --encoder's
    (the agent's input channels follow the encoder); returns (agent, ppo config dict)."""
    from agent import Agent, AgentConfig

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    acfg = AgentConfig.from_dict(ck["agent_cfg"])
    if encoder:
        in_ch = torch.load(encoder, map_location="cpu", weights_only=False)["cfg"].get("in_ch", 1)
        acfg = replace(acfg, encoder=replace(acfg.encoder, in_ch=in_ch))
    agent = Agent(acfg).to(dev)
    info = agent.init_from(ckpt)
    if encoder:
        agent.load_encoder(encoder)
    else:
        assert not info["skipped"], f"checkpoint tensors skipped: {sorted(info['skipped'])}"
    agent.eval()
    return agent, ck["cfg"]


@torch.no_grad()
def play(agent, raster: torch.Tensor, start: torch.Tensor, ladder: torch.Tensor, max_step: float,
         amap: Tuple[Tuple[int, int], Tuple[int, int]], max_steps: int) -> Dict[str, torch.Tensor]:
    """Episodes from the start grid indices (B, 2); see the module docstring."""
    from agent import greedy

    axes, signs = amap
    B, dev = start.shape[0], start.device
    lim = torch.tensor([NX - 1, NY - 1], device=dev)
    sg = torch.tensor(signs, device=dev)
    ij = start.clone()
    ring = agent.init_ring(B, dev)
    prev = torch.zeros(B, 2, device=dev)
    first = torch.ones(B, dtype=torch.bool, device=dev)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    stopped = torch.zeros(B, dtype=torch.bool, device=dev)
    length = torch.zeros(B, dtype=torch.long, device=dev)
    in_ch = agent.cfg.encoder.in_ch
    for _ in range(max_steps):
        img = raster[ij[:, 0], ij[:, 1]][:, None].expand(-1, in_ch, -1, -1)
        out, ring = agent.act(img, prev, first, ring)
        a = greedy(out["logits"])
        move = (a != 0).any(1) & active
        stopped |= active & ~move
        steps = ladder[a]  # (B, 2) grid cells per agent axis
        d = torch.zeros_like(ij)
        for k in range(2):
            d[:, axes[k]] += sg[k] * steps[:, k]
        new = torch.minimum((ij + d).clamp(min=0), lim)
        applied = torch.stack([sg[k] * (new - ij)[:, axes[k]] for k in range(2)], 1).float() / max_step
        ij = torch.where(move[:, None], new, ij)
        prev = torch.where(move[:, None], applied, prev)
        length += move.long()
        active = move
        first = torch.zeros_like(first)
        if not bool(active.any()):
            break
    return {"final": ij, "length": length, "stopped": stopped}


def summary(counts: torch.Tensor, bright: torch.Tensor, start: torch.Tensor,
            ep: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Pattern statistics; counts over the episodes whose frame is outside the bright band."""
    s0, s1 = (start[:, 0], start[:, 1]), (ep["final"][:, 0], ep["final"][:, 1])
    c0, c1 = counts[s0].float(), counts[s1].float()
    b0, b1 = bright[s0], bright[s1]
    m0, m1, both = ~b0, ~b1, ~b0 & ~b1
    mean = lambda x, m: (x[m].mean().item() if bool(m.any()) else float("nan"))  # noqa: E731
    out = {"start_bright": b0.float().mean().item(), "final_bright": b1.float().mean().item(),
           "start_count": mean(c0, m0), "final_count": mean(c1, m1),
           "improved": mean((c1 > c0).float(), both), "worse": mean((c1 < c0).float(), both),
           "stop_rate": ep["stopped"].float().mean().item(),
           "length": ep["length"].float().mean().item()}
    for v in LEVELS:  # Shares of all episodes (bright-band endings count as not reaching it)
        out[f"final_ge{v}"] = ((c1 >= v) & m1).float().mean().item()
        out[f"start_ge{v}"] = ((c0 >= v) & m0).float().mean().item()
    out["final_hist"] = torch.bincount(c1[m1].long(), minlength=int(counts.max()) + 1).tolist()
    return out


def figure(raster: np.ndarray, counts: np.ndarray, start: np.ndarray, final: np.ndarray,
           title: str, path: str, n: int = 24):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, start.shape[0])
    cols = 8
    rows = 2 * ((n + cols - 1) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.5 * rows))
    for k in range(rows * cols):
        axs.flat[k].axis("off")
    for e in range(n):
        r, c = 2 * (e // cols), e % cols
        for dr, ij, lab in ((0, start[e], "start"), (1, final[e], "end")):
            ax = axs[r + dr, c]
            ax.imshow(raster[ij[0], ij[1]], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{lab} {counts[ij[0], ij[1]]} sp\n({X0 + STEP_DEG * ij[0]:.2f}, "
                         f"{Y0 + STEP_DEG * ij[1]:.2f})", fontsize=6)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(args):
    from env import EnvConfig

    dev = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    raster_np = np.load(os.path.join(args.raster, "raster64.npy"), mmap_mode="r")
    counts_np = np.load(os.path.join(args.raster, "counts.npy"))
    raster = torch.from_numpy(np.array(raster_np)).to(dev)  # Copy: the memmap is read-only
    counts = torch.from_numpy(counts_np.astype(np.int64)).to(dev)
    bright = raster.flatten(2).float().mean(-1) > BRIGHT_MEAN  # (NX, NY)
    agent, pcfg = load_agent(args.ckpt, args.encoder, dev)
    ladder_units = EnvConfig().step_ladder
    assert len(ladder_units) == agent.cfg.n_bins, "ladder size != the agent's bins"
    ladder = torch.tensor(ladder_units, dtype=torch.long, device=dev)
    max_step = float(max(abs(v) for v in ladder_units))
    max_steps = args.max_steps or int(pcfg.get("max_steps", 64))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)
    start = torch.stack((torch.randint(0, NX, (args.episodes,), generator=gen),
                         torch.randint(0, NY, (args.episodes,), generator=gen)), 1).to(dev)
    res = {"ckpt": os.path.abspath(args.ckpt), "encoder": os.path.abspath(args.encoder) if args.encoder else "",
           "episodes": args.episodes, "seed": args.seed, "max_steps": max_steps,
           "raster_count_max": int(counts.max()), "bright_mean": BRIGHT_MEAN,
           "raster_bright": bright.float().mean().item(), "maps": {}}
    best = None
    for name in map_names(args.axis_map):
        ep = play(agent, raster, start, ladder, max_step, parse_map(name), max_steps)
        s = summary(counts, bright, start, ep)
        res["maps"][name] = s
        np.savez(os.path.join(args.out, f"episodes_{name.replace(',', '_')}.npz"),
                 start=start.cpu().numpy(), final=ep["final"].cpu().numpy(),
                 final_count=counts[ep["final"][:, 0], ep["final"][:, 1]].cpu().numpy(),
                 length=ep["length"].cpu().numpy(), stopped=ep["stopped"].cpu().numpy())
        print(f"map {name:>6}: bright band {100 * s['start_bright']:4.1f} -> "
              f"{100 * s['final_bright']:4.1f} %, count {s['start_count']:5.1f} -> {s['final_count']:5.1f}, "
              + ", ".join(f">= {v} sp {100 * s[f'final_ge{v}']:5.1f} %" for v in LEVELS)
              + f", improved {100 * s['improved']:5.1f} %, stop {100 * s['stop_rate']:5.1f} %, "
              f"len {s['length']:4.1f}", flush=True)
        if best is None or s["final_ge16"] > res["maps"][best[0]]["final_ge16"]:
            best = (name, ep["final"].cpu().numpy())
    with open(os.path.join(args.out, "real_eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(raster_np, counts_np, start.cpu().numpy(), best[1],
           f"{os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))}, map {best[0]} "
           f"(most episodes ending on >= 16 spots)", os.path.join(args.out, "final_frames.png"))
    print(f"wrote {args.out}/real_eval.json, final_frames.png", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", default="/home/evv/data")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True, help="ppo.py checkpoint")
    e.add_argument("--encoder", default="", help="pretrain_encoder.py checkpoint replacing the encoder")
    e.add_argument("--raster", required=True, help="Directory of the build output")
    e.add_argument("--episodes", type=int, default=1024)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--max-steps", type=int, default=0, help="0: the run's env max_steps")
    e.add_argument("--axis-map", default="x,y")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.data, args.out, args.workers)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
