"""
bench_render.py - Timing of the frame data paths for encoder training (run on CRCD:
cluster/bench_render_l40s.slurm)

Per batch size B (states of geometry.SETUPS, FrameData tilt sampling):
    sim             HerriottSim.simulate, store_hits on (per-hit + miss outputs) / off
    current path    FrameData.simulate (sim + TargetBuilder targets) and the randomized
                    1-channel renderer (render_random.RandomRenderer)
    camera path     render_camera.CameraRenderer: parameters, 3-channel render in chunks
                    of C frames (eager and torch.compile'd layers), camera-frame labels,
                    a camera-frame spot-map target (spot layer at 1 px)
plus one CNN encoder + heads training step (3-channel stem, bf16 autocast,
channels_last, AdamW) for scale. Median of --reps timed calls after --warmup; peak
memory per entry. Writes bench_render.md / .json to --out.

    python bench_render.py --out DIR [--batches 256 1024 4096] [--chunks 128 512 2048]
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import N_COUNT, EncoderConfig, FrameModel
from frame_data import DataConfig, FrameData
from geometry import SETUPS
from render_camera import CameraRanges, CameraRenderer


def timed(fn, warmup: int, reps: int):
    """(median ms, peak MiB above the start, first-call ms) of fn()."""
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = 1000 * (time.perf_counter() - t)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return statistics.median(ts), (torch.cuda.max_memory_allocated() - base) / 2**20, first


def chunked(cam: CameraRenderer, hm, lxy, miss, p, gen, C: int):
    """Render in chunks of C frames (the (n, K, H, W) temporaries scale with n)."""
    n = hm.shape[0]
    out = []
    for i in range(0, n, C):
        s = slice(i, i + C)
        out.append(cam.render(hm[s], lxy[s], {k: v[s] for k, v in p.items()}, gen,
                              tuple(t[s] for t in miss)))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--chunks", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(dev), flush=True)

    fd = FrameData(DataConfig(setups=SETUPS), device=dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    cam = CameraRenderer(fd.base)
    cams = {"eager": cam}
    if not args.no_compile:
        cc = CameraRenderer(fd.base)
        for name in ("_spot_layer", "_scatter_layer", "_face_layer"):
            setattr(cc, name, torch.compile(getattr(cc, name), dynamic=False))
        cams["compiled"] = cc
    pix = fd.base.pitch  # 1 px in mm: the spot-map target width
    rows = []

    def add(name, B, fn, C=None):
        try:
            ms, mib, first = timed(fn, args.warmup, args.reps)
            rows.append({"entry": name, "B": B, "chunk": C, "ms": ms, "us_per_frame": 1000 * ms / B,
                         "peak_mib": mib, "first_ms": first})
            print(f"{name:34s} B={B:5d} C={str(C):5s} {ms:9.2f} ms  {1000 * ms / B:8.2f} us/frame  "
                  f"peak {mib:8.0f} MiB  first {first:9.0f} ms", flush=True)
        except Exception as e:  # OOM or compile failure: record and continue
            torch.cuda.empty_cache()
            rows.append({"entry": name, "B": B, "chunk": C, "error": f"{type(e).__name__}: {e}"[:300]})
            print(f"{name:34s} B={B:5d} C={str(C):5s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)

    for B in args.batches:
        setup = fd.sample_setups(B, gen)
        st = fd.states(fd.sample_tilts(B, gen, setup=setup), gen, setup=setup)
        out = fd.sim.simulate(st, store_world=False)
        hm, lxy = out["hit_mirror"], out["hit_local_xy"]
        miss = (out["miss_mirror"], out["miss_local_xy"], out["intensity"])

        add("sim store_hits=on", B, lambda: fd.sim.simulate(st, store_world=False))
        add("sim store_hits=off", B, lambda: fd.sim.simulate(st, store_world=False, store_hits=False))
        add("current: sim + targets", B, lambda: fd.simulate(st, setup))
        rp = fd.renderer.sample_params(B, fd.cfg.ranges, gen)
        add("current: RandomRenderer 1ch", B, lambda: fd.renderer(hm, lxy, rp, gen))

        add("camera: sample_params", B, lambda: cam.sample_params(B, CameraRanges(), gen))
        p = cam.sample_params(B, CameraRanges(), gen)
        add("camera: labels", B, lambda: cam.labels(hm, lxy, p))
        pt = {**cam.neutral_params(B), "warp": p["warp"], "scale": p["scale"],
              "sigma": torch.full((B,), pix, device=dev)}

        def map_target():
            uv, w, valid = cam.spots_camera(hm, lxy, pt)
            return cam._spot_layer(uv, w * valid, pt)

        add("camera: map target (1 px spots)", B, map_target)
        for C in args.chunks:
            if C > B:
                continue
            for tag, c in cams.items():
                add(f"camera: render 3ch ({tag})", B, lambda c=c, C=C: chunked(c, hm, lxy, miss, p, gen, C), C)
        torch.cuda.empty_cache()

    # One encoder + heads training step, 3-channel stem
    ecfg = EncoderConfig()
    model = FrameModel(ecfg).to(dev)
    model.encoder.stem = nn.Conv2d(3, ecfg.cnn_widths[0], 3, padding=1).to(dev)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for B in sorted({256, 1024} & set(args.batches) or {256}):
        img = torch.randint(0, 256, (B, 3, 64, 64), dtype=torch.uint8, device=dev)
        img = img.contiguous(memory_format=torch.channels_last)
        tgt = torch.rand(B, 1, 64, 64, device=dev)
        cnt = torch.randint(0, N_COUNT, (B,), device=dev)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(img)
            loss = F.mse_loss(o["map"].float(), tgt) + 0.1 * F.cross_entropy(o["count"].float(), cnt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        add("encoder train step (eager, bf16)", B, step)

    with open(os.path.join(args.out, "bench_render.json"), "w") as f:
        json.dump({"device": torch.cuda.get_device_name(dev), "args": vars(args), "rows": rows}, f, indent=1)
    lines = [f"# Frame data timing ({torch.cuda.get_device_name(dev)})", "",
             f"median of {args.reps} calls after {args.warmup} warm-up calls; peak = memory above the "
             "start of the timed calls; first = first call (compile included)", "",
             "| entry | B | chunk | ms | us / frame | peak MiB | first ms |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['entry']} | {r['B']} | {r['chunk'] or ''} | failed: {r['error'][:80]} | | | |")
        else:
            lines.append(f"| {r['entry']} | {r['B']} | {r['chunk'] or ''} | {r['ms']:.2f} | "
                         f"{r['us_per_frame']:.2f} | {r['peak_mib']:.0f} | {r['first_ms']:.0f} |")
    with open(os.path.join(args.out, "bench_render.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(args.out, "bench_render.md"))


if __name__ == "__main__":
    main()
