"""
probe_render_res.py - Does a 64 px render lose what tells trapped states apart from
shorter-path states that look alike? Direct 64 px rendering (env.SpotRenderer: spots
sampled at the pixel centres, response per 64 px pixel) vs rendering at R and scaling
down to 64 px (env.DownscaleRenderer: response and 8-bit quantization per sensor pixel,
then the mean of each block, as a camera image resized for the encoder), for several
spot sizes.

Run from rl/:  uv run python probe_render_res.py [--step 0.01] [--res 128,256,512]
               [--sigma 0.5,0.25,0.125] [--short-frac 0.8] [--gain G] [--tag T]
               [--out reports/render_res]

States: the measured cell's +-limit M2 box on a --step grid (sim with the env's bounce
cap). Trapped: stop reason STOP_CAP (the beam stays in the cell to the cap). For every
trapped state the nearest image among the states whose path is below short_frac x its
path: d_short, the RMS difference over the 64 x 64 pixels in 8-bit levels. A large
d_short means the 64 px image tells the trapped state apart from every shorter-path
state. Per pipeline: d_short quantiles, the share of trapped states with d_short below
eps (1, 2, 5, 10 levels; the camera read noise of 0.02 full scale is 5.1 levels), the
bounce counts of the look-alikes, the trapped images' own RMS level and d_short relative
to it. --gain: the response gain of every pipeline (default the env's; lower = less
saturation). Figure: the trapped states with the smallest d_short under the current
renderer (64 px, sigma 0.5), their look-alikes and the difference x 8, for the direct and
the highest-resolution pipeline of the smallest and the largest sigma. Output:
<out>/render_res[_<tag>].md, .json, .jpg.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List

import torch

from env import AlignEnv, DownscaleRenderer, EnvConfig, SpotRenderer
from herriott_sim import STOP_CAP

RL_DIR = os.path.dirname(os.path.abspath(__file__))
EPS = (1.0, 2.0, 5.0, 10.0)  # 8-bit RMS levels
N_FIG = 6  # Trapped states in the figure


def scan(env: AlignEnv, step: float, chunk: int = 8192) -> Dict[str, torch.Tensor]:
    """Sim of the +-limit box on a step grid: tilts, path, bounces, stop reason, hits."""
    n = int(round(2 * env.limit / step)) + 1
    ax = torch.linspace(-env.limit, env.limit, n, device=env.device)
    P, Y = torch.meshgrid(ax, ax, indexing="ij")
    tilts = torch.stack((P.flatten(), Y.flatten()), 1)
    parts: Dict[str, List[torch.Tensor]] = {}
    for i in range(0, tilts.shape[0], chunk):
        out = env._simulate(tilts[i:i + chunk])
        for k in ("total_path_length", "hit_counts", "stop_reason", "hit_mirror", "hit_local_xy"):
            parts.setdefault(k, []).append(out[k])
    d = {k: torch.cat(v) for k, v in parts.items()}
    d["tilts"] = tilts
    return d


@torch.no_grad()
def render_all(renderer, st: Dict[str, torch.Tensor], chunk: int) -> torch.Tensor:
    """(N, res * res) uint8 images of all scanned states."""
    out = []
    for i in range(0, st["tilts"].shape[0], chunk):
        img = renderer(st["hit_mirror"][i:i + chunk], st["hit_local_xy"][i:i + chunk])
        out.append(img.reshape(img.shape[0], -1))
    return torch.cat(out)


@torch.no_grad()
def nearest_short(img: torch.Tensor, path: torch.Tensor, q: torch.Tensor, frac: float,
                  chunk: int = 8192):
    """For the query rows q: the RMS distance (8-bit levels) to the nearest image among
    the rows with path < frac x the query's path, and that row's index."""
    x = img.float()
    xq = x[q]
    nq2 = (xq * xq).sum(1)
    pq = path[q]
    best = torch.full((q.numel(),), math.inf, device=img.device, dtype=torch.float64)
    arg = torch.full((q.numel(),), -1, device=img.device, dtype=torch.long)
    npx = x.shape[1]
    for i in range(0, x.shape[0], chunk):
        xs = x[i:i + chunk]
        d2 = (nq2[:, None] + (xs * xs).sum(1)[None] - 2.0 * (xq @ xs.T)).double().clamp(min=0)
        ok = path[i:i + chunk][None] < frac * pq[:, None]
        d2 = torch.where(ok, d2, math.inf)
        v, j = d2.min(1)
        upd = v < best
        best = torch.where(upd, v, best)
        arg = torch.where(upd, j + i, arg)
    return (best / npx).sqrt(), arg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--step", type=float, default=0.01, help="deg, scan grid")
    ap.add_argument("--res", default="128,256,512", help="render resolutions scaled down to 64")
    ap.add_argument("--sigma", default="0.5,0.25,0.125", help="spot sigmas, mm")
    ap.add_argument("--short-frac", type=float, default=0.8,
                    help="look-alikes: path below this x the trapped path (0.8 keeps the 89-bounce exits)")
    ap.add_argument("--gain", type=float, default=0.0, help="response gain (0: the env's)")
    ap.add_argument("--tag", default="", help="output name suffix")
    ap.add_argument("--chunk", type=int, default=256, help="states per render call")
    ap.add_argument("--out", default=os.path.join(RL_DIR, "reports", "render_res"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    env = AlignEnv(EnvConfig(num_envs=1, history_len=0))
    c, g = env.cfg, env.cfg.geometry
    gain = a.gain or c.gain
    stem = os.path.join(a.out, "render_res" + (f"_{a.tag}" if a.tag else ""))
    st = scan(env, a.step)
    path = st["total_path_length"]
    trapped = st["stop_reason"] == STOP_CAP
    q = torch.nonzero(trapped).squeeze(1)
    n = path.numel()
    print(f"{n} states ({a.step} deg grid), {q.numel()} trapped ({100 * q.numel() / n:.2f} %), "
          f"path of the trapped {float(path[q].min()) / 1000:.2f}-{float(path[q].max()) / 1000:.2f} m; "
          f"shorter than {a.short_frac} x: {int((path < a.short_frac * path[q].min()).sum())} states "
          f"({time.time() - t0:.0f} s)", flush=True)
    assert q.numel() > 0, "no trapped states on this grid"

    res_list = [int(r) for r in a.res.split(",") if r]
    pipes = []
    for s in (float(x) for x in a.sigma.split(",")):
        pipes.append((f"64 direct, sigma {s:g}", s, 64))
        pipes += [(f"{r} -> 64, sigma {s:g}", s, r) for r in res_list]
    ref_name = f"64 direct, sigma {c.spot_sigma:g}"
    rows, fig_imgs = [], {}
    fig_q = None
    for name, s, r in pipes:
        t1 = time.time()
        full = SpotRenderer(r, g.diameter, s, gain, g.reflectivity, c.max_bounces, 0, env.device)
        rend = full if r == c.img_res else DownscaleRenderer(full, c.img_res)
        img = render_all(rend, st, a.chunk)
        d, arg = nearest_short(img, path, q, a.short_frac)
        lvl = (img[q].float().pow(2).mean(1)).sqrt()
        bn = st["hit_counts"][arg]
        row = {"pipeline": name, "sigma_mm": s, "render_res": r,
               "d_short_q10": float(d.quantile(0.1)), "d_short_median": float(d.median()),
               "d_short_mean": float(d.mean()),
               **{f"share_below_{e:g}": float((d < e).float().mean()) for e in EPS},
               "trapped_rms_median": float(lvl.median()),
               "d_short_rel_median": float((d / lvl.clamp(min=1e-6)).median()),
               "lookalike_bounces_median": float(bn.double().median()),
               "seconds": time.time() - t1}
        rows.append(row)
        print(f"{name:26s} d_short q10 {row['d_short_q10']:6.2f} median {row['d_short_median']:6.2f} | "
              + " ".join(f"<{e:g}: {100 * row[f'share_below_{e:g}']:5.1f} %" for e in EPS)
              + f" | trapped RMS {row['trapped_rms_median']:5.1f}, d / RMS {row['d_short_rel_median']:.3f}, look-alike bounces "
              f"{row['lookalike_bounces_median']:.0f} ({row['seconds']:.0f} s)", flush=True)
        if name == ref_name or fig_q is None:  # Figure: hardest under the current renderer
            fig_q = torch.argsort(d)[:N_FIG].cpu()
        fig_imgs[name] = (img[q].cpu(), img[arg].cpu(), arg.cpu())
        del img
    # Figure: per hard trapped state, its image and its look-alike in every pipeline
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = sorted({p[1] for p in pipes})
    names = [p[0] for p in pipes if p[1] in (sig[-1], sig[0]) and p[2] in (64, max(res_list + [64]))]
    names = list(dict.fromkeys(names))
    fig, axs = plt.subplots(N_FIG, 3 * len(names), figsize=(1.5 * 3 * len(names), 1.7 * N_FIG + 0.4),
                            squeeze=False)
    for i, k in enumerate(fig_q.tolist()):
        qi = int(q[k])
        for j, nm in enumerate(names):
            img_q, img_l, arg = fig_imgs[nm]
            li = int(arg[k])
            a_q = img_q[k].view(c.img_res, c.img_res).float()
            a_l = img_l[k].view(c.img_res, c.img_res).float()
            panels = ((a_q, f"trapped {int(st['hit_counts'][qi])} b", qi),
                      (a_l, f"look-alike {int(st['hit_counts'][li])} b", li),
                      (((a_q - a_l).abs() * 8).clamp(max=255), "|diff| x 8", None))
            for col, (im, lab, idx) in enumerate(panels):
                ax = axs[i, 3 * j + col]
                ax.imshow(im.numpy(), cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                if idx is not None:
                    t = st["tilts"][idx].tolist()
                    lab += f"\n({t[0]:.2f}, {t[1]:.2f})"
                ax.set_title(lab, fontsize=6)
            if i == 0:
                axs[0, 3 * j].text(0, 1.45, nm, transform=axs[0, 3 * j].transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(stem + ".jpg", dpi=110)
    info = {"step": a.step, "states": n, "trapped": int(q.numel()), "short_frac": a.short_frac,
            "cap": c.max_bounces, "img_res": c.img_res, "gain": gain, "pipelines": rows}
    with open(stem + ".json", "w") as f:
        json.dump(info, f, indent=1)
    lines = ["# Trapped states vs shorter-path look-alikes at 64 px", "",
             f"Measured cell, +-{env.limit:g} deg M2 box on a {a.step:g} deg grid: {n} states, "
             f"{q.numel()} trapped (beam to the {c.max_bounces}-bounce cap). d_short: per trapped "
             f"state, the RMS difference (8-bit levels, 64 x 64) to the nearest image among the "
             f"states with path < {a.short_frac:g} x its path. Camera read noise 0.02 full scale "
             "= 5.1 levels. `R -> 64`: rendered at R, response and 8-bit per sensor pixel, block "
             "mean to 64 px (env.DownscaleRenderer). Current renderer: 64 direct, sigma "
             f"{c.spot_sigma:g} mm, gain {c.gain:g}; this run: gain {gain:g}.", "",
             "| pipeline | d_short q10 | d_short median | " + " | ".join(f"< {e:g}" for e in EPS)
             + " | trapped RMS | d_short / RMS | look-alike bounces |",
             "|---" * (6 + len(EPS)) + "|"]
    for r in rows:
        lines.append(f"| {r['pipeline']} | {r['d_short_q10']:.2f} | {r['d_short_median']:.2f} | "
                     + " | ".join(f"{100 * r[f'share_below_{e:g}']:.1f} %" for e in EPS)
                     + f" | {r['trapped_rms_median']:.1f} | {r['d_short_rel_median']:.3f} | "
                     f"{r['lookalike_bounces_median']:.0f} |")
    lines += ["", f"Figure `{os.path.basename(stem)}.jpg`: the trapped states with the smallest "
              "d_short under the current renderer, their nearest shorter-path look-alike and the "
              "difference x 8 (title: bounces, tilt)."]
    with open(stem + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {stem}.md, .json, .jpg ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
