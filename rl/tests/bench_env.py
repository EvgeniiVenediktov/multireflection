"""
tests/bench_env.py - AlignEnv.step() timing with a sim / render / rest breakdown

Run from rl/:  uv run python tests/bench_env.py [--steps 100] [--warmup 10]

Cases: B = 256, 1024, 4096 x img_res 64, 128; random actions (uniform ladder index per
axis), default init box (full +-2 deg range), history stack K = 8. Each step simulates
and renders 2B states (post-step + reset candidates).

Wall time: mean over --steps steps, torch.cuda.synchronize around the timed loop.
Breakdown: CUDA events around AlignEnv._simulate and AlignEnv._render (GPU timeline
span of each phase, which includes launch gaps when the CPU is the bottleneck);
rest = wall - sim - render (action / reward / reset bookkeeping, history stack).
Target: < 5 ms per step at B = 256, img_res 64.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from env import AlignEnv, EnvConfig  # noqa: E402

DEV = torch.device("cuda")


def timed(fn, events: list):
    """Wrap fn to record a (start, end) CUDA event pair per call."""

    def wrapper(*args, **kw):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kw)
        end.record()
        events.append((start, end))
        return out

    return wrapper


def bench(B: int, res: int, steps: int, warmup: int) -> dict:
    env = AlignEnv(EnvConfig(num_envs=B, img_res=res, device="cuda"))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    actions = [
        torch.randint(0, env.n_bins, (B, 2), generator=gen, device=DEV)
        for _ in range(warmup + steps)
    ]
    for a in actions[:warmup]:
        env.step(a)

    sim_ev, ren_ev = [], []
    env._simulate = timed(env._simulate, sim_ev)
    env._render = timed(env._render, ren_ev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for a in actions[warmup:]:
        env.step(a)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / steps * 1e3
    sim = sum(s.elapsed_time(e) for s, e in sim_ev) / steps
    ren = sum(s.elapsed_time(e) for s, e in ren_ev) / steps
    return {
        "B": B,
        "res": res,
        "oversample": env.renderer.oversample,
        "wall": wall,
        "sim": sim,
        "render": ren,
        "rest": wall - sim - ren,
        "mem": torch.cuda.max_memory_allocated() / 2**20,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    print(f"{torch.cuda.get_device_name()}, steps {args.steps}, warmup {args.warmup}")
    print(f"{'B':>5} {'res':>4} {'S':>2} | {'step ms':>8} {'sim':>7} {'render':>7} "
          f"{'rest':>7} | {'peak MiB':>8}")
    for B in (256, 1024, 4096):
        for res in (64, 128):
            r = bench(B, res, args.steps, args.warmup)
            print(f"{r['B']:5d} {r['res']:4d} {r['oversample']:2d} | {r['wall']:8.2f} "
                  f"{r['sim']:7.2f} {r['render']:7.2f} {r['rest']:7.2f} | {r['mem']:8.0f}")
            del r
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
