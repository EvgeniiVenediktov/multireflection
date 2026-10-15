"""
tests/test_sim.py - Simulator regression and backend agreement

Run from rl/:  uv run python tests/test_sim.py

    1. nominal cell (CellGeometry), both backends: 31 bounces, 2733.27 mm, hole exit,
       mirrors alternate starting with M2
    2. trapped beam: runs to the cap (128) or the intensity cutoff (456)
    3. full M2 scan, +-2 deg at 0.01 deg (401 x 401), cap 456: triton vs torch
    4. random 13-DOF states (all stop reasons), cap 456: triton vs torch
    5. output invariants (padding, alternation, local frame vs world hits, store_world)
    6. the old env self-test (herriott_env.py) still runs
"""

import os
import subprocess
import sys
import time

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

from geometry import NOMINAL_BOUNCES, NOMINAL_PATH_MM, CellGeometry  # noqa: E402
from herriott_sim import (  # noqa: E402
    S_LASER_DX,
    S_LASER_DY,
    S_LASER_PITCH,
    S_LASER_YAW,
    S_M1_PITCH,
    S_M1_TX,
    S_M1_TY,
    S_M1_YAW,
    S_M2_PITCH,
    S_M2_TX,
    S_M2_TY,
    S_M2_YAW,
    S_SEP,
    STOP_CAP,
    STOP_HOLE,
    STOP_INTENSITY,
    STOP_NAMES,
    HerriottSim,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKENDS = ("torch", "triton") if DEV.type == "cuda" else ("torch",)
SCAN_CAP = 456

# Agreement criteria (triton vs torch)
MIN_COUNT_AGREE = 0.999
MAX_PATH_REL = 1e-5
MAX_LOCAL_ABS = 1e-3  # mm


# ── Helpers ─────────────────────────────────────────────────────────────────


def run(sim: HerriottSim, state: torch.Tensor, backend: str, **kw) -> dict:
    out = sim.simulate(state, backend=backend, **kw)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return out


def stop_distribution(stop: torch.Tensor) -> str:
    bins = torch.bincount(stop.long(), minlength=len(STOP_NAMES)).tolist()
    n = stop.numel()
    return ", ".join(f"{name} {c} ({100 * c / n:.2f} %)" for name, c in zip(STOP_NAMES, bins))


def compare(name: str, ref: dict, out: dict) -> torch.Tensor:
    """Agreement of out (triton) with ref (torch); asserts the criteria.

    Returns the per-config mask where hit counts disagree.
    """
    eq = ref["hit_counts"] == out["hit_counts"]
    n = eq.numel()
    n_mis = int((~eq).sum())
    frac = eq.float().mean().item()

    path_r, path_o = ref["total_path_length"][eq], out["total_path_length"][eq]
    path_rel = ((path_r - path_o).abs() / path_r.abs().clamp_min(1e-6)).max().item()
    stop_same = bool((ref["stop_reason"] == out["stop_reason"])[eq].all())
    mirror_same = bool((ref["hit_mirror"] == out["hit_mirror"])[eq].all())
    local_abs = (ref["hit_local_xy"][eq] - out["hit_local_xy"][eq]).abs().max().item()
    world_abs = (ref["hit_sequence"][eq] - out["hit_sequence"][eq]).abs().max().item()
    inten_rel = (
        ((ref["intensity"] - out["intensity"]).abs() / ref["intensity"])[eq].max().item()
    )
    bit_exact = (
        (ref["total_path_length"] == out["total_path_length"]) & eq
    ).float().mean().item()

    print(f"[{name}] {n} configs")
    print(f"  hit_counts agree:      {100 * frac:.4f} % ({n_mis} mismatches)")
    print(f"  where counts agree:    path rel err max {path_rel:.2e}, "
          f"stop_reason identical {stop_same}, hit_mirror identical {mirror_same}")
    print(f"                         hit_local_xy max |diff| {local_abs:.2e} mm, "
          f"world xyz max |diff| {world_abs:.2e} mm, intensity rel err max {inten_rel:.2e}")
    print(f"  path bit-identical:    {100 * bit_exact:.4f} %")

    assert frac >= MIN_COUNT_AGREE, f"{name}: hit_counts agree on {frac:.5f}"
    assert path_rel < MAX_PATH_REL, f"{name}: path rel err {path_rel}"
    assert stop_same, f"{name}: stop_reason differs"
    assert mirror_same, f"{name}: hit_mirror differs"
    assert local_abs < MAX_LOCAL_ABS, f"{name}: hit_local_xy diff {local_abs}"
    return ~eq


def check_invariants(name: str, sim: HerriottSim, state: torch.Tensor, out: dict):
    """Output consistency of one backend's result."""
    B, N = out["hit_mirror"].shape
    counts = out["hit_counts"]
    hm = out["hit_mirror"].long()
    col = torch.arange(N, device=hm.device)
    stored = col[None, :] < counts[:, None]

    assert out["hit_counts"].dtype == torch.long
    assert out["hit_mirror"].dtype == torch.int8
    assert out["stop_reason"].dtype == torch.int8
    assert out["exited_hole"].dtype == torch.bool
    assert torch.equal(out["exited_hole"], out["stop_reason"] == STOP_HOLE)
    assert bool(((hm > 0) == stored).all()), "hit_mirror padding"
    assert bool((out["hit_local_xy"][~stored] == 0).all()), "hit_local_xy padding"
    assert bool((out["hit_sequence"][~stored] == 0).all()), "hit_sequence padding"
    same_twice = (hm[:, 1:] == hm[:, :-1]) & stored[:, 1:]
    assert not bool(same_twice.any()), "same mirror twice in a row"

    # Final position = last stored hit, intensity = reflectivity product
    last = (counts - 1).clamp_min(0)
    final = out["hit_sequence"][torch.arange(B, device=hm.device), last]
    assert torch.equal(out["final_positions"], final), "final_positions"
    refl = torch.where(
        hm == 1, sim.m1_cfg.reflectivity, torch.where(hm == 2, sim.m2_cfg.reflectivity, 1.0)
    ).double().prod(dim=1)
    inten_err = ((out["intensity"].double() - refl).abs() / refl).max().item()
    assert inten_err < 1e-4, f"intensity vs reflectivity product: {inten_err}"

    # Local (lx, ly) = projection of the world hit onto the hit mirror's local axes
    # (fp64, on a subset of rows to bound memory)
    rows = torch.arange(0, B, max(1, B // 20000), device=hm.device)
    st = state.to(hm.device)[rows]
    world = out["hit_sequence"][rows].double()
    local = out["hit_local_xy"][rows].double()
    zero = torch.zeros_like(st[:, 0])
    frames = {
        1: (S_M1_PITCH, S_M1_YAW, (st[:, S_M1_TX], st[:, S_M1_TY], zero)),
        2: (S_M2_PITCH, S_M2_YAW, (st[:, S_M2_TX], st[:, S_M2_TY], st[:, S_SEP])),
    }
    lxy_err = 0.0
    for m, (ip, iy, pos) in frames.items():
        R = HerriottSim.rotation_matrix(st[:, ip], st[:, iy]).double()
        rel = world - torch.stack(pos, dim=-1).double()[:, None, :]
        lx = (rel * R[:, None, :, 0]).sum(-1)
        ly = (rel * R[:, None, :, 1]).sum(-1)
        d = (torch.stack([lx, ly], -1) - local)[hm[rows] == m]
        if d.numel():
            lxy_err = max(lxy_err, d.abs().max().item())
    assert lxy_err < 1e-3, f"hit_local_xy vs world projection: {lxy_err}"
    print(f"  [{name}] invariants ok (local frame vs world projection max {lxy_err:.1e} mm)")


def random_states(n: int, geom: CellGeometry, seed: int = 0, near: bool = False) -> torch.Tensor:
    """Random 13-DOF states.

    wide: mirror tilts N(0, 2) deg, offsets N(0, 0.5) mm, separation U(60, 250) mm,
          laser nominal + N(0, 0.3) mm / N(0, 0.5) deg (mostly 0-5 bounces)
    near: every DOF nominal + noise: tilts 0.5 deg, separation 10 mm, offsets 0.2 mm,
          laser 0.1 mm / 0.2 deg (multi-bounce patterns, all stop reasons)
    """
    g = torch.Generator().manual_seed(seed)
    s = geom.base_state(n, "cpu")
    if near:
        s[:, S_M1_PITCH : S_M2_YAW + 1] += 0.5 * torch.randn(n, 4, generator=g)
        s[:, S_SEP] += 10.0 * torch.randn(n, generator=g)
        s[:, S_M1_TX : S_M2_TY + 1] += 0.2 * torch.randn(n, 4, generator=g)
        s[:, S_LASER_DX : S_LASER_DY + 1] += 0.1 * torch.randn(n, 2, generator=g)
        s[:, S_LASER_PITCH : S_LASER_YAW + 1] += 0.2 * torch.randn(n, 2, generator=g)
    else:
        s[:, S_M1_PITCH : S_M2_YAW + 1] = 2.0 * torch.randn(n, 4, generator=g)
        s[:, S_SEP] = 60.0 + 190.0 * torch.rand(n, generator=g)
        s[:, S_M1_TX : S_M2_TY + 1] = 0.5 * torch.randn(n, 4, generator=g)
        s[:, S_LASER_DX : S_LASER_DY + 1] += 0.3 * torch.randn(n, 2, generator=g)
        s[:, S_LASER_PITCH : S_LASER_YAW + 1] += 0.5 * torch.randn(n, 2, generator=g)
    return s.to(DEV)


def histogram(counts: torch.Tensor, stop: torch.Tensor, min_cells: int = 50) -> str:
    """Bounce-count histogram, bins with >= min_cells cells, with their stop reasons."""
    vals, cnt = torch.unique(counts, return_counts=True)
    lines, rest = [], 0
    for v, c in zip(vals.tolist(), cnt.tolist()):
        if c < min_cells:
            rest += c
            continue
        reasons = torch.bincount(stop[counts == v].long(), minlength=len(STOP_NAMES))
        why = ", ".join(f"{STOP_NAMES[i]} {k}" for i, k in enumerate(reasons.tolist()) if k)
        lines.append(f"    {v:4d} bounces: {c:6d} cells  ({why})")
    lines.append(f"    other counts (< {min_cells} cells each): {rest} cells")
    return "\n".join(lines)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_nominal(geom: CellGeometry):
    print("\n== nominal ==")
    sim = geom.make_sim(DEV)
    state = geom.base_state(1, DEV)
    for be in BACKENDS:
        out = run(sim, state, be)
        n = int(out["hit_counts"][0])
        path = float(out["total_path_length"][0])
        stop = int(out["stop_reason"][0])
        mirrors = out["hit_mirror"][0].tolist()
        print(f"  {be:6s}: {n} bounces, path {path:.3f} mm, stop {STOP_NAMES[stop]}")
        assert n == NOMINAL_BOUNCES, f"{be}: {n} bounces"
        assert abs(path - NOMINAL_PATH_MM) < 0.05, f"{be}: path {path}"
        assert stop == STOP_HOLE and bool(out["exited_hole"][0])
        assert mirrors[:n] == [2 if i % 2 == 0 else 1 for i in range(n)], mirrors[:n]
        assert all(m == 0 for m in mirrors[n:])
        check_invariants(be, sim, state, out)


def test_cap(geom: CellGeometry):
    print("\n== trapped beam, M2 tilt (0.10, 1.04) deg ==")
    state = geom.base_state(4, DEV)
    state[:, S_M2_PITCH], state[:, S_M2_YAW] = 0.10, 1.04
    for cap, want_n, want_stop in [(128, 128, STOP_CAP), (456, 456, STOP_INTENSITY)]:
        sim = geom.make_sim(DEV, cap)
        outs = {be: run(sim, state, be) for be in BACKENDS}
        for be, out in outs.items():
            n = int(out["hit_counts"][0])
            stop = int(out["stop_reason"][0])
            print(f"  cap {cap} {be:6s}: {n} bounces, path "
                  f"{float(out['total_path_length'][0]):.1f} mm, stop {STOP_NAMES[stop]}")
            assert n == want_n and stop == want_stop
        if "triton" in outs:
            for k, v in outs["torch"].items():
                assert torch.equal(v, outs["triton"][k]), f"cap {cap}: {k} differs"


def test_scan(geom: CellGeometry):
    print(f"\n== M2 scan +-2 deg at 0.01 deg, cap {SCAN_CAP} ==")
    sim = geom.make_sim(DEV, SCAN_CAP)
    ax = (torch.arange(-200, 201, dtype=torch.float64) * 0.01).float().to(DEV)
    P, Y = torch.meshgrid(ax, ax, indexing="ij")
    state = geom.base_state(P.numel(), DEV)
    state[:, S_M2_PITCH] = P.reshape(-1)
    state[:, S_M2_YAW] = Y.reshape(-1)

    t0 = time.perf_counter()
    ref = run(sim, state, "torch")
    t1 = time.perf_counter()
    print(f"  torch {t1 - t0:.2f} s")
    print("  bounce-count histogram (torch):")
    print(histogram(ref["hit_counts"], ref["stop_reason"]))
    print(f"  stop reasons: {stop_distribution(ref['stop_reason'])}")
    check_invariants("torch", sim, state, ref)
    if "triton" not in BACKENDS:
        return

    out = run(sim, state, "triton")
    mis = compare("scan", ref, out)
    check_invariants("triton", sim, state, out)

    # Where are the mismatches: at a region boundary (a 4-neighbour differs in count)?
    if mis.any():
        cg = ref["hit_counts"].view(401, 401)
        pad = torch.nn.functional.pad(cg[None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0]
        boundary = (
            (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
            | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
            | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
            | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
        ).reshape(-1)
        idx = mis.nonzero().flatten()
        print(f"  mismatches: {idx.numel()}, at a region boundary: {int(boundary[idx].sum())}")
        for i in idx[:20].tolist():
            print(
                f"    M2 ({state[i, S_M2_PITCH]:+.2f}, {state[i, S_M2_YAW]:+.2f}) deg: "
                f"torch {int(ref['hit_counts'][i])} {STOP_NAMES[int(ref['stop_reason'][i])]}, "
                f"triton {int(out['hit_counts'][i])} {STOP_NAMES[int(out['stop_reason'][i])]}, "
                f"boundary {bool(boundary[i])}"
            )
    else:
        print("  mismatches: none")


def test_random(geom: CellGeometry, n: int = 100_000, near: bool = False):
    kind = "near nominal" if near else "wide"
    print(f"\n== random 13-DOF states ({kind}), n = {n}, cap {SCAN_CAP} ==")
    sim = geom.make_sim(DEV, SCAN_CAP)
    state = random_states(n, geom, seed=int(near), near=near)
    ref = run(sim, state, "torch")
    print(f"  stop reasons (torch): {stop_distribution(ref['stop_reason'])}")
    zero = ref["hit_counts"] == 0
    at0 = torch.bincount(ref["stop_reason"][zero].long(), minlength=len(STOP_NAMES))
    print("  of which at 0 bounces: " + ", ".join(
        f"{STOP_NAMES[i]} {c}" for i, c in enumerate(at0.tolist()) if c))
    vals, cnt = torch.unique(ref["hit_counts"], return_counts=True)
    top = sorted(zip(cnt.tolist(), vals.tolist()), reverse=True)[:8]
    print("  most common bounce counts: " + ", ".join(f"{v}: {c}" for c, v in top))
    check_invariants("torch", sim, state, ref)
    if "triton" not in BACKENDS:
        return

    out = run(sim, state, "triton")
    print(f"  stop reasons (triton): {stop_distribution(out['stop_reason'])}")
    mis = compare(f"random {kind}", ref, out)
    check_invariants("triton", sim, state, out)
    for i in mis.nonzero().flatten()[:10].tolist():
        print(
            f"    #{i}: torch {int(ref['hit_counts'][i])} "
            f"{STOP_NAMES[int(ref['stop_reason'][i])]}, triton {int(out['hit_counts'][i])} "
            f"{STOP_NAMES[int(out['stop_reason'][i])]}"
        )

    # store_world=False gives the same outputs without the world hits
    lean = run(sim, state, "triton", store_world=False)
    assert lean["hit_sequence"] is None
    for k, v in out.items():
        if k != "hit_sequence":
            assert torch.equal(v, lean[k]), f"store_world=False changes {k}"
    print("  store_world=False: identical outputs, hit_sequence None")


def test_old_env():
    print("\n== old env self-test (herriott_env.py) ==")
    res = subprocess.run(
        [sys.executable, "herriott_env.py"], cwd=RL_DIR, capture_output=True, text=True
    )
    tail = "\n".join("    " + ln for ln in res.stdout.strip().splitlines()[-4:])
    print(tail)
    assert res.returncode == 0, res.stderr[-2000:]
    print("  ok")


if __name__ == "__main__":
    geom = CellGeometry()
    print(f"device {DEV}, backends {BACKENDS}")
    test_nominal(geom)
    test_cap(geom)
    test_scan(geom)
    test_random(geom)
    test_random(geom, near=True)
    test_old_env()
    print("\nall tests passed")
