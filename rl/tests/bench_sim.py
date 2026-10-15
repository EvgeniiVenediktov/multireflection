"""
tests/bench_sim.py - Simulator timing, torch reference vs fused Triton kernel

Run from rl/:  uv run python tests/bench_sim.py [--runs 20] [--no-torch]

Cases (CellGeometry, M2 tilt varied only):
    nominal   B = 256, all rays at the nominal state (31 bounces), cap 128
    trapped   B = 256, M2 tilt (0.10, 1.04) deg (never finds the hole), cap 128 and 456
    scan      B = 160801, M2 +-2 deg at 0.01 deg, cap 128 and 456

Mean wall time per call over --runs calls after warmup, torch.cuda.synchronize around
the timed loop. The torch backend computes the world hits either way, so it is timed
with store_world=True only. Target: < 2 ms per call at B = 256 for any bounce count.
"""

import argparse
import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from geometry import CellGeometry  # noqa: E402
from herriott_sim import S_M2_PITCH, S_M2_YAW  # noqa: E402

DEV = torch.device("cuda")
TRAPPED_TILT = (0.10, 1.04)  # deg, M2 pitch / yaw


def mean_ms(fn: Callable, runs: int, warmup: int = 3) -> float:
    """Mean wall time per call (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1e3


def scan_states(geom: CellGeometry) -> torch.Tensor:
    ax = (torch.arange(-200, 201, dtype=torch.float64) * 0.01).float().to(DEV)
    P, Y = torch.meshgrid(ax, ax, indexing="ij")
    s = geom.base_state(P.numel(), DEV)
    s[:, S_M2_PITCH] = P.reshape(-1)
    s[:, S_M2_YAW] = Y.reshape(-1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--no-torch", action="store_true", help="skip the torch reference")
    args = ap.parse_args()

    geom = CellGeometry()
    nominal = geom.base_state(256, DEV)
    trapped = geom.base_state(256, DEV)
    trapped[:, S_M2_PITCH], trapped[:, S_M2_YAW] = TRAPPED_TILT
    scan = scan_states(geom)
    cases = [
        ("nominal", nominal, 128),
        ("trapped", trapped, 128),
        ("trapped", trapped, 456),
        ("scan", scan, 128),
        ("scan", scan, 456),
    ]

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, runs {args.runs}")
    head = (
        f"{'case':8s} {'B':>7s} {'cap':>4s} {'bounces':>8s} | {'torch':>10s} | "
        f"{'triton':>9s} {'no world':>9s} | {'speedup':>8s}"
    )
    print(head)
    print("-" * len(head))
    for name, state, cap in cases:
        sim = geom.make_sim(DEV, cap)
        out = sim.simulate(state, backend="triton", store_world=False)
        n = out["hit_counts"]
        bounces = f"{int(n.max())}" if name != "scan" else f"<={int(n.max())}"
        t_tri = mean_ms(lambda: sim.simulate(state, backend="triton"), args.runs)
        t_lean = mean_ms(
            lambda: sim.simulate(state, backend="triton", store_world=False), args.runs
        )
        if args.no_torch:
            ref, speed = "-", "-"
        else:
            t_ref = mean_ms(lambda: sim.simulate(state, backend="torch"), args.runs, 1)
            ref, speed = f"{t_ref:8.2f}ms", f"{t_ref / t_tri:.0f}x"
        print(
            f"{name:8s} {state.shape[0]:7d} {cap:4d} {bounces:>8s} | {ref:>10s} | "
            f"{t_tri:7.3f}ms {t_lean:7.3f}ms | {speed:>8s}"
        )


if __name__ == "__main__":
    main()
