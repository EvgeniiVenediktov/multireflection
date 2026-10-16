"""
sim_triton.py - Fused Triton kernel for the Herriott ray tracer (backend="triton")

Why: the torch reference (HerriottSim._simulate_torch) runs every bounce as ~100 small
elementwise kernels over the batch plus two active.any() host syncs. The arithmetic is
negligible; at RL batch sizes the call is launch/sync bound (3.5 ms per bounce at
B = 256, 110 ms for the 31-bounce pattern, 1.4 s for a trapped beam at a 400 cap).

Design:
    - one launch per call, grid = cdiv(B, BLOCK); a program traces BLOCK rays as
      vectors, one lane = one ray (rays are independent, no cross-lane traffic)
    - prologue in registers: the 13 DOFs are loaded once per ray; rotation matrices,
      mirror positions, normals, sphere centres, local axes and the laser ray are
      built in-kernel, mirror constants come in as scalar arguments
    - the bounce loop runs entirely in registers: both roots of both spheres, the four
      candidate hits with aperture / sag / hole / front-side tests, back-side blocking,
      hole exit, nearest valid hit on the other mirror, reflection, path, intensity,
      stop reason
    - block-level early exit: the loop runs while bounce < cap and max(active) over the
      block is 1, so a program ends as soon as all its rays are done
    - per-bounce outputs are masked stores (active lanes, column `bounce`) into
      zero-filled tensors allocated once by the wrapper; world xyz only with
      STORE_WORLD, mirror id and local xy only with STORE_HITS; per-ray outputs are
      stored once after the loop
    - no host syncs, no intermediate global tensors
    - same formulas as the reference with its rounding reproduced: IEEE sqrt / div /
      sin / cos, no FMA contraction (enable_fp_fusion=False), torch's einsum and size-3
      sum orders; the outputs are bit-identical to the torch backend on the test sets
      (a different torch build could change its reduction order, which would only
      shift region boundaries by rounding)
"""

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from herriott_sim import (
    STATE_DIM,
    STOP_BLOCKED,
    STOP_CAP,
    STOP_HOLE,
    STOP_INTENSITY,
    STOP_MISSED,
)

# Launch configuration, one ray per lane. 32 rays / 1 warp per program was fastest
# (RTX A2000, B = 256, trapped at cap 456: 0.69 ms vs 0.72 ms at 64 / 2 warps and
# 0.84 ms at 128 / 4 warps): the early-exit reduction is a warp shuffle and the
# programs spread over more SMs. Large batches are bound by the output zero-fill.
BLOCK = 32
NUM_WARPS = 1

_INF = tl.constexpr(float("inf"))
_DEG2RAD = tl.constexpr(0.017453292519943295)
_K_CAP = tl.constexpr(STOP_CAP)
_K_HOLE = tl.constexpr(STOP_HOLE)
_K_MISSED = tl.constexpr(STOP_MISSED)
_K_BLOCKED = tl.constexpr(STOP_BLOCKED)
_K_INTENSITY = tl.constexpr(STOP_INTENSITY)


# ── Device helpers ──────────────────────────────────────────────────────────


@triton.jit
def _rotation(pitch, yaw):
    """Ry @ Rx from pitch / yaw (deg), as in HerriottSim.rotation_matrix."""
    px = pitch * _DEG2RAD
    py = yaw * _DEG2RAD
    cx = libdevice.cos(px)
    sx = libdevice.sin(px)
    cy = libdevice.cos(py)
    sy = libdevice.sin(py)
    r00 = cy
    r01 = sx * sy
    r02 = cx * sy
    r11 = cx
    r12 = -sx
    r20 = -sy
    r21 = sx * cy
    r22 = cx * cy
    return r00, r01, r02, r11, r12, r20, r21, r22


@triton.jit
def _normalize(x, y, z):
    """F.normalize: v / max(|v|, 1e-12)."""
    n = tl.sqrt_rn((x * x + z * z) + y * y)
    n = tl.maximum(n, 1e-12)
    return tl.div_rn(x, n), tl.div_rn(y, n), tl.div_rn(z, n)


@triton.jit
def _sphere_roots(ox, oy, oz, dx, dy, dz, cx, cy, cz, rr, a):
    """Both ray-sphere roots, as in HerriottSim._ray_sphere_intersect."""
    ocx = ox - cx
    ocy = oy - cy
    ocz = oz - cz
    b = 2.0 * ((ocx * dx + ocz * dz) + ocy * dy)
    c = ((ocx * ocx + ocz * ocz) + ocy * ocy) - rr
    disc = b * b - 4.0 * a * c
    valid = disc > 0.0
    sq = tl.sqrt_rn(tl.maximum(disc, 1e-10))
    den = 2.0 * a + 1e-10
    t_n = tl.div_rn(-b - sq, den)
    t_f = tl.div_rn(-b + sq, den)
    v_n = (t_n > 1e-5) & valid
    v_f = (t_f > 1e-5) & valid
    t_n = tl.where(v_n, t_n, _INF)
    t_f = tl.where(v_f, t_f, _INF)
    return t_n, v_n, t_f, v_f


@triton.jit
def _candidate(
    t, ox, oy, oz, dx, dy, dz,
    px, py, pz, xax, xay, xaz, yax, yay, yaz, nx, ny, nz, cx, cy, cz,
    aperture, sag_p1, hole_r, hole_y,
    HAS_HOLE: tl.constexpr,
):
    """Hit point, local (lx, ly), aperture / hole / front-side tests of one root.

    As HerriottSim._check_aperture and _front_side.
    """
    hx = ox + t * dx
    hy = oy + t * dy
    hz = oz + t * dz
    rx = hx - px
    ry = hy - py
    rz = hz - pz
    lx = (rx * xax + rz * xaz) + ry * xay
    ly = (rx * yax + rz * yaz) + ry * yay
    r = tl.sqrt_rn(lx * lx + ly * ly)
    dn = (rx * nx + rz * nz) + ry * ny
    ap = (r < aperture) & (dn >= 0.0) & (dn < sag_p1)
    if HAS_HOLE:
        hly = ly - hole_y
        hd = tl.sqrt_rn(lx * lx + hly * hly)
        in_hole = ap & (hd < hole_r)
        ap = ap & (in_hole == 0)
    else:
        in_hole = ap & (ap == 0)
    front = ((dx * (cx - hx) + dz * (cz - hz)) + dy * (cy - hy)) < 0.0
    return hx, hy, hz, lx, ly, ap, in_hole, front


# ── Kernel ──────────────────────────────────────────────────────────────────


@triton.jit
def _trace_kernel(
    state_ptr,
    counts_ptr, path_ptr, inten_ptr, stop_ptr, exited_ptr, final_ptr,
    mirror_ptr, local_ptr, world_ptr,
    B, cap,
    m1_rr, m1_roc, m1_inv_roc, m1_ap, m1_sag_p1, m1_hole_r, m1_hole_y, m1_refl,
    m2_rr, m2_roc, m2_inv_roc, m2_ap, m2_sag_p1, m2_refl,
    threshold,
    STATE_STRIDE: tl.constexpr,
    STORE_WORLD: tl.constexpr,
    STORE_HITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    lane = offs < B
    row = offs.to(tl.int64)

    # ── Prologue: DOFs, mirror frames, laser ray ──
    sp = state_ptr + row * STATE_STRIDE
    m1_pitch = tl.load(sp + 0, mask=lane, other=0.0)
    m1_yaw = tl.load(sp + 1, mask=lane, other=0.0)
    m2_pitch = tl.load(sp + 2, mask=lane, other=0.0)
    m2_yaw = tl.load(sp + 3, mask=lane, other=0.0)
    sep = tl.load(sp + 4, mask=lane, other=0.0)
    m1_tx = tl.load(sp + 5, mask=lane, other=0.0)
    m1_ty = tl.load(sp + 6, mask=lane, other=0.0)
    m2_tx = tl.load(sp + 7, mask=lane, other=0.0)
    m2_ty = tl.load(sp + 8, mask=lane, other=0.0)
    l_dx = tl.load(sp + 9, mask=lane, other=0.0)
    l_dy = tl.load(sp + 10, mask=lane, other=0.0)
    l_pitch = tl.load(sp + 11, mask=lane, other=0.0)
    l_yaw = tl.load(sp + 12, mask=lane, other=0.0)

    # Mirror frames: local x axis = R[:, 0], y axis = R[:, 1], normal = facing * R[:, 2]
    # (row 1 of Ry @ Rx is (0, cos, -sin), so a10 = b10 = 0)
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    a00, a01, a02, a11, a12, a20, a21, a22 = _rotation(m1_pitch, m1_yaw)
    b00, b01, b02, b11, b12, b20, b21, b22 = _rotation(m2_pitch, m2_yaw)
    p1x = m1_tx
    p1y = m1_ty
    p1z = zero
    n1x = a02
    n1y = a12
    n1z = a22
    c1x = p1x + n1x * m1_roc
    c1y = p1y + n1y * m1_roc
    c1z = p1z + n1z * m1_roc
    p2x = m2_tx
    p2y = m2_ty
    p2z = sep
    n2x = -b02
    n2y = -b12
    n2z = -b22
    c2x = p2x + n2x * m2_roc
    c2y = p2y + n2y * m2_roc
    c2z = p2z + n2z * m2_roc

    # Laser origin: hole centre + (dx, dy), 5 mm behind M1, in the M1 frame.
    # Matrix-vector products as fma(r1, v1, r0 * v0) + r2 * v2, the rounding of the
    # reference's torch.einsum; 3-term dot products below as (x + z) + y, the order
    # of torch's .sum(-1) over a size-3 dim.
    hlx = l_dx
    hly = m1_hole_y + l_dy
    hlz = zero - 5.0
    ox = (tl.fma(a01, hly, a00 * hlx) + a02 * hlz) + p1x
    oy = (tl.fma(a11, hly, zero * hlx) + a12 * hlz) + p1y
    oz = (tl.fma(a21, hly, a20 * hlx) + a22 * hlz) + p1z

    # Laser direction: M1 rotation applied to (laser rotation applied to +z)
    e00, e01, e02, e11, e12, e20, e21, e22 = _rotation(l_pitch, l_yaw)
    ldx = tl.fma(a01, e12, a00 * e02) + a02 * e22
    ldy = tl.fma(a11, e12, zero * e02) + a12 * e22
    ldz = tl.fma(a21, e12, a20 * e02) + a22 * e22
    dx, dy, dz = _normalize(ldx, ldy, ldz)

    # ── Per-ray state ──
    active = lane
    last = tl.zeros([BLOCK], dtype=tl.int32)
    count = tl.zeros([BLOCK], dtype=tl.int32)
    stop = tl.zeros([BLOCK], dtype=tl.int32) + _K_CAP
    path = tl.zeros([BLOCK], dtype=tl.float32)
    inten = tl.zeros([BLOCK], dtype=tl.float32) + 1.0
    fx = tl.zeros([BLOCK], dtype=tl.float32)
    fy = tl.zeros([BLOCK], dtype=tl.float32)
    fz = tl.zeros([BLOCK], dtype=tl.float32)
    hit_base = row * cap

    bounce = 0
    n_active = tl.max(active.to(tl.int32), axis=0)
    while (bounce < cap) & (n_active > 0):
        # Both roots of both spheres
        a = (dx * dx + dz * dz) + dy * dy
        t1n, v1n, t1f, v1f = _sphere_roots(ox, oy, oz, dx, dy, dz, c1x, c1y, c1z, m1_rr, a)
        t2n, v2n, t2f, v2f = _sphere_roots(ox, oy, oz, dx, dy, dz, c2x, c2y, c2z, m2_rr, a)

        # Candidate hits, local coordinates, aperture / hole / front-side tests
        h1nx, h1ny, h1nz, l1nx, l1ny, ap1n, hole1n, f1n = _candidate(
            t1n, ox, oy, oz, dx, dy, dz,
            p1x, p1y, p1z, a00, zero, a20, a01, a11, a21, n1x, n1y, n1z, c1x, c1y, c1z,
            m1_ap, m1_sag_p1, m1_hole_r, m1_hole_y, True,
        )
        h1fx, h1fy, h1fz, l1fx, l1fy, ap1f, hole1f, f1f = _candidate(
            t1f, ox, oy, oz, dx, dy, dz,
            p1x, p1y, p1z, a00, zero, a20, a01, a11, a21, n1x, n1y, n1z, c1x, c1y, c1z,
            m1_ap, m1_sag_p1, m1_hole_r, m1_hole_y, True,
        )
        h2nx, h2ny, h2nz, l2nx, l2ny, ap2n, hole2n, f2n = _candidate(
            t2n, ox, oy, oz, dx, dy, dz,
            p2x, p2y, p2z, b00, zero, b20, b01, b11, b21, n2x, n2y, n2z, c2x, c2y, c2z,
            m2_ap, m2_sag_p1, 0.0, 0.0, False,
        )
        h2fx, h2fy, h2fz, l2fx, l2fy, ap2f, hole2f, f2f = _candidate(
            t2f, ox, oy, oz, dx, dy, dz,
            p2x, p2y, p2z, b00, zero, b20, b01, b11, b21, n2x, n2y, n2z, c2x, c2y, c2z,
            m2_ap, m2_sag_p1, 0.0, 0.0, False,
        )
        hole1n = v1n & hole1n
        hole1f = v1f & hole1f
        v1n = v1n & ap1n
        v1f = v1f & ap1f
        v2n = v2n & ap2n
        v2f = v2f & ap2f

        # Back-side hits block the ray
        t_block = tl.minimum(
            tl.minimum(
                tl.where(v1n & (f1n == 0), t1n, _INF),
                tl.where(v1f & (f1f == 0), t1f, _INF),
            ),
            tl.minimum(
                tl.where(v2n & (f2n == 0), t2n, _INF),
                tl.where(v2f & (f2f == 0), t2f, _INF),
            ),
        )
        v1n = v1n & f1n
        v1f = v1f & f1f
        v2n = v2n & f2n
        v2f = v2f & f2f

        # Nearest front-side M1 disk crossing in the hole
        hole1n = hole1n & f1n
        hole1f = hole1f & f1f
        td1n = tl.where(v1n | hole1n, t1n, _INF)
        td1f = tl.where(v1f | hole1f, t1f, _INF)
        td1 = tl.minimum(td1n, td1f)
        in_hole1 = tl.where(td1 == td1n, hole1n, hole1f) & (td1 < _INF)

        # Closer valid root per mirror
        t1n = tl.where(v1n, t1n, _INF)
        t1f = tl.where(v1f, t1f, _INF)
        t1 = tl.minimum(t1n, t1f)
        v1 = t1 < _INF
        s1 = t1 == t1n
        h1x = tl.where(s1, h1nx, h1fx)
        h1y = tl.where(s1, h1ny, h1fy)
        h1z = tl.where(s1, h1nz, h1fz)
        l1x = tl.where(s1, l1nx, l1fx)
        l1y = tl.where(s1, l1ny, l1fy)

        t2n = tl.where(v2n, t2n, _INF)
        t2f = tl.where(v2f, t2f, _INF)
        t2 = tl.minimum(t2n, t2f)
        v2 = t2 < _INF
        s2 = t2 == t2n
        h2x = tl.where(s2, h2nx, h2fx)
        h2y = tl.where(s2, h2ny, h2fy)
        h2z = tl.where(s2, h2nz, h2fz)
        l2x = tl.where(s2, l2nx, l2fx)
        l2y = tl.where(s2, l2ny, l2fy)

        # Not the same mirror twice; closer mirror wins; back side blocks
        v1 = v1 & (last != 1)
        v2 = v2 & (last != 2)
        t1 = tl.where(v1, t1, _INF)
        t2 = tl.where(v2, t2, _INF)
        hit_m1 = (t1 <= t2) & v1
        hit_m2 = (t2 < t1) & v2
        any_hit = (hit_m1 | hit_m2) & (tl.minimum(t1, t2) < t_block)
        hit_m1 = hit_m1 & any_hit

        # Rays ending here: hole exit, blocked or missed
        stopping = active & (any_hit == 0)
        exit_now = stopping & in_hole1 & (last != 1) & (td1 < t_block)
        reason = tl.where(
            exit_now, _K_HOLE, tl.where(t_block < _INF, _K_BLOCKED, _K_MISSED)
        )
        stop = tl.where(stopping, reason, stop)
        active = active & any_hit

        # Selected hit
        t = tl.where(hit_m1, t1, t2)
        hx = tl.where(hit_m1, h1x, h2x)
        hy = tl.where(hit_m1, h1y, h2y)
        hz = tl.where(hit_m1, h1z, h2z)
        lx = tl.where(hit_m1, l1x, l2x)
        ly = tl.where(hit_m1, l1y, l2y)
        nx = tl.where(hit_m1, -(h1x - c1x) * m1_inv_roc, (h2x - c2x) * m2_inv_roc)
        ny = tl.where(hit_m1, -(h1y - c1y) * m1_inv_roc, (h2y - c2y) * m2_inv_roc)
        nz = tl.where(hit_m1, -(h1z - c1z) * m1_inv_roc, (h2z - c2z) * m2_inv_roc)
        mirror = tl.where(hit_m1, 1, 2)

        # Store column `bounce` for active lanes
        hidx = hit_base + bounce
        if STORE_HITS:
            tl.store(mirror_ptr + hidx, mirror.to(tl.int8), mask=active)
            tl.store(local_ptr + hidx * 2, lx, mask=active)
            tl.store(local_ptr + hidx * 2 + 1, ly, mask=active)
        if STORE_WORLD:
            tl.store(world_ptr + hidx * 3, hx, mask=active)
            tl.store(world_ptr + hidx * 3 + 1, hy, mask=active)
            tl.store(world_ptr + hidx * 3 + 2, hz, mask=active)
        count = tl.where(active, count + 1, count)
        path = tl.where(active, path + t, path)
        last = tl.where(active, mirror, last)
        fx = tl.where(active, hx, fx)
        fy = tl.where(active, hy, fy)
        fz = tl.where(active, hz, fz)

        # Reflect, offset the origin
        dot = (dx * nx + dz * nz) + dy * ny
        dot2 = 2.0 * dot
        dx, dy, dz = _normalize(dx - dot2 * nx, dy - dot2 * ny, dz - dot2 * nz)
        ox = hx + 1e-4 * dx
        oy = hy + 1e-4 * dy
        oz = hz + 1e-4 * dz

        # Attenuate
        refl = tl.where(hit_m1, m1_refl, m2_refl)
        inten = tl.where(active, inten * refl, inten)
        dimmed = active & ((inten > threshold) == 0)
        stop = tl.where(dimmed, _K_INTENSITY, stop)
        active = active & (dimmed == 0)

        n_active = tl.max(active.to(tl.int32), axis=0)
        bounce += 1

    # ── Per-ray outputs ──
    tl.store(counts_ptr + row, count.to(tl.int64), mask=lane)
    tl.store(path_ptr + row, path, mask=lane)
    tl.store(inten_ptr + row, inten, mask=lane)
    tl.store(stop_ptr + row, stop.to(tl.int8), mask=lane)
    tl.store(exited_ptr + row, stop == _K_HOLE, mask=lane)
    tl.store(final_ptr + row * 3, fx, mask=lane)
    tl.store(final_ptr + row * 3 + 1, fy, mask=lane)
    tl.store(final_ptr + row * 3 + 2, fz, mask=lane)


# ── Python wrapper ──────────────────────────────────────────────────────────


def _f32(x: float) -> float:
    """Round a Python float to fp32 (what torch does with a scalar operand)."""
    return float(np.float32(x))


def _mirror_args(sim) -> tuple:
    """Scalar kernel arguments from the sim's mirror configs (fp32-rounded)."""
    m1, m2 = sim.m1_cfg, sim.m2_cfg
    return (
        _f32(m1.roc**2),
        _f32(m1.roc),
        float(np.float32(1.0) / np.float32(m1.roc)),  # torch divides by a scalar as * (1/x)
        _f32(sim.m1_aperture),
        _f32(sim.m1_sag + 1),
        _f32(m1.hole_radius),
        _f32(m1.hole_offset_y),
        _f32(m1.reflectivity),
        _f32(m2.roc**2),
        _f32(m2.roc),
        float(np.float32(1.0) / np.float32(m2.roc)),
        _f32(sim.m2_aperture),
        _f32(sim.m2_sag + 1),
        _f32(m2.reflectivity),
        _f32(sim.sim_cfg.intensity_threshold),
    )


def simulate_triton(
    sim,
    state: torch.Tensor,
    store_world: bool = True,
    block: Optional[int] = None,
    num_warps: Optional[int] = None,
    store_hits: bool = True,
) -> dict:
    """Trace (B, 13) states with the fused kernel; same dict as the torch backend.
    store_hits False: no per-hit outputs (hit_mirror / hit_local_xy None, the stores are
    compiled out and the (B, N) buffers are not allocated); the per-ray outputs are
    unchanged (large scans: the zero-fill of the hit buffers dominates otherwise)."""
    assert state.is_cuda, "triton backend needs a CUDA state"
    assert state.shape[1] == STATE_DIM, f"Expected state dim {STATE_DIM}"
    state = state.to(torch.float32).contiguous()
    B = state.shape[0]
    N = sim.sim_cfg.max_bounces
    dev = state.device
    blk = block or BLOCK
    nw = num_warps or (NUM_WARPS if block is None else max(1, blk // 32))

    counts = torch.empty(B, dtype=torch.long, device=dev)
    path = torch.empty(B, dtype=torch.float32, device=dev)
    intensity = torch.empty(B, dtype=torch.float32, device=dev)
    stop = torch.empty(B, dtype=torch.int8, device=dev)
    exited = torch.empty(B, dtype=torch.bool, device=dev)
    final = torch.empty(B, 3, dtype=torch.float32, device=dev)
    hn = (B, N) if store_hits else (1, 1)
    hit_mirror = torch.zeros(*hn, dtype=torch.int8, device=dev)
    hit_local = torch.zeros(*hn, 2, dtype=torch.float32, device=dev)
    hit_world = (
        torch.zeros(B, N, 3, dtype=torch.float32, device=dev) if store_world else None
    )

    if B > 0:
        _trace_kernel[(triton.cdiv(B, blk),)](
            state,
            counts, path, intensity, stop, exited, final,
            hit_mirror, hit_local, hit_world if store_world else hit_local,
            B, N,
            *_mirror_args(sim),
            STATE_STRIDE=STATE_DIM,
            STORE_WORLD=store_world,
            STORE_HITS=store_hits,
            BLOCK=blk,
            num_warps=nw,
            enable_fp_fusion=False,
        )

    return {
        "hit_counts": counts,
        "final_positions": final,
        "total_path_length": path,
        "hit_sequence": hit_world,
        "hit_mirror": hit_mirror if store_hits else None,
        "hit_local_xy": hit_local if store_hits else None,
        "intensity": intensity,
        "stop_reason": stop,
        "exited_hole": exited,
    }
