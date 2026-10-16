"""
env.py - Vectorized GPU environment for M2 alignment (PLAN phase 3: E1-E8, D1-D4)

B independent episodes, all state on the device, no host syncs in step() / reset().

    setups     one or more cells (geometry.CellGeometry, same mirrors), EnvConfig.setups;
               empty = the single EnvConfig.geometry (default: the measured cell). Each
               env samples a setup per episode (EnvConfig.setup_weights)
    state      13-DOF base state of the episode's setup; the controlled DOFs (default
               M2 pitch, M2 yaw) move, every other DOF stays at the setup value
    action     (B, 2) int64, per-axis index into the step ladder
               (0, +1, -1, +3, -3, +10, -10, +30, -30, +100, -100) x 0.01 deg;
               index 0 is the zero step, (0, 0) is "not move"
    transition tilt = clamp(tilt + step, -limit, +limit), limit 2 deg (mount range);
               the applied delta (after clamping) / largest ladder step is the action
               that led to the next frame
    reward     move:     (P_t - P_{t-1}) / P_ref - step_cost
               not move: 0, the episode terminates (decision D2)
               P_t = optical path after step t (bounce cap, D1); P_ref per setup
               (EnvConfig.path_ref): "nominal" = path at the setup's nominal tilt
               (2733.27 mm for the measured cell, the stage-1 P_nom), "max" = longest
               path in the +-limit box (0.01 deg scan at construction); the default
               "auto" is nominal for one setup and max for several. The return is
               (P_final - P_0) / P_ref - step_cost * moves
    truncation the step count reaches max_steps without termination
    auto-reset done envs restart inside step(): obs is the first frame of the new
               episode, info holds the pre-reset quantities (final frame and its
               stack, path, stop reason, true tilt, setup, episode stats).
               EnvConfig.auto_reset False: done envs idle (actions ignored, reward 0,
               state frozen at the final frame) until reset(mask)
    reset draw initial tilt (uniform in the init box) and setup of env i's episode j
               come from a counter-based hash of (seed, i, j): deterministic per
               episode, independent of the other envs and of step timing, no generator
               state and no host sync (a torch.Generator would have to draw for all B
               envs every step, or sync to learn how many reset)
    partial    rows that need no simulation (unmasked envs in reset(mask), not-move
               envs, envs that do not reset) are traced as a dead state that stops
               before the first bounce, so the kernel's 32-ray programs made only of
               such rows exit at once (packing live rows first was measured: no gain,
               the step is render bound); all rows are still rendered (fixed shapes)

Observation (dict, see AlignEnv.observe):
    image    (B, 1, H, W) uint8   camera image of the M2 face (SpotRenderer)
    action   (B, 2) float32       applied action that led to the frame (0 at reset)
    images   (B, K, 1, H, W) uint8, actions (B, K, 2), mask (B, K) bool
                                  history stack (HistoryStack), oldest first
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import torch

from geometry import NOMINAL_PATH_MM, CellGeometry
from herriott_sim import DEVICE, S_LASER_PITCH, S_M2_PITCH, S_M2_YAW

M2_ID = 2  # hit_mirror value of an M2 hit
AUTO_SIGMA_PX = 2.5  # Auto oversampling: spot sigma >= this many splat-grid pixels
PATH_REFS = ("auto", "nominal", "max")
DEAD_LASER_PITCH = 89.0  # deg: the laser misses M2, the ray stops before the first bounce


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class EnvConfig:
    """AlignEnv settings."""

    # ── Batch ──
    num_envs: int = 256
    device: Optional[str] = None  # None = herriott_sim.DEVICE
    seed: int = 0

    # ── Cell, controlled DOFs, simulation ──
    geometry: CellGeometry = field(default_factory=CellGeometry)
    controlled: Tuple[int, int] = (S_M2_PITCH, S_M2_YAW)  # State indices
    tilt_limit: Optional[float] = None  # Absolute, deg; None = geometry.m2_tilt_limit
    max_bounces: int = 128  # Bounce cap (D1)

    # ── Setups: sampled per env and episode; empty = (geometry,) ──
    setups: Tuple[CellGeometry, ...] = ()  # e.g. geometry.SETUPS; same mirrors
    setup_weights: Optional[Tuple[float, ...]] = None  # None = uniform
    path_ref: str = "auto"  # Reward scale P_ref: "nominal", "max", "auto" (see module)
    ref_scan_step: float = 0.01  # deg, box scan for P_max

    # ── Actions: per-axis ladder in units of step_unit, index 0 = zero step ──
    step_ladder: Tuple[float, ...] = (0, 1, -1, 3, -3, 10, -10, 30, -30, 100, -100)
    step_unit: float = 0.01  # deg

    # ── Episode and reward (D2) ──
    max_steps: int = 64
    step_cost: float = 0.01
    auto_reset: bool = True  # False: done envs idle until reset() (evaluation)

    # ── Reset: initial tilts uniform in center +- radius, clamped to the limits ──
    init_center: Tuple[float, float] = (0.0, 0.0)  # deg
    init_radius: float = 2.0  # deg

    # ── Observation (D3) ──
    img_res: int = 64
    spot_sigma: float = 0.5  # mm
    gain: float = 1.6  # 1 - exp(-gain * I): a unit spot (k = 0) peaks at 0.80
    oversample: int = 0  # Splat grid oversampling; 0 = auto (sigma >= 2.5 grid px)
    history_len: int = 8  # Frames K in the history stack; 0 = no stack


# ── Counter-based random numbers ────────────────────────────────────────────

_M32 = 0xFFFFFFFF
_STREAMS = (0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35)  # Tilt x, tilt y, setup


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """(x * c) mod 2^32 for int64 x in [0, 2^32), without int64 overflow."""
    lo = x & 0xFFFF
    hi = x >> 16
    return (lo * c + (((hi * c) & 0xFFFF) << 16)) & _M32


def hash32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer hash (C. Wellons) of int64 tensors holding uint32 values."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def uniforms_torch(key: torch.Tensor, episode: torch.Tensor) -> torch.Tensor:
    """(n, 3) float32 in [0, 1) per (key, episode), one column per stream (reference)."""
    h = hash32(key ^ hash32(episode & _M32))
    cols = [hash32(h ^ s) for s in _STREAMS]
    return (torch.stack(cols, 1) >> 8).to(torch.float32) * (1.0 / (1 << 24))


_uniform_kernel = None


def _build_uniform_kernel():
    """Triton kernel: uniforms_torch fused into one launch (the torch version is ~60
    tiny int64 kernels, launch bound)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _lowbias32(x):
        x = x ^ (x >> 16)
        x = x * tl.full(x.shape, 0x7FEB352D, tl.uint32)
        x = x ^ (x >> 15)
        x = x * tl.full(x.shape, 0x846CA68B, tl.uint32)
        return x ^ (x >> 16)

    @triton.jit
    def kernel(key_ptr, ep_ptr, out_ptr, n, s0, s1, s2, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        key = (tl.load(key_ptr + offs, mask=m, other=0) & 0xFFFFFFFF).to(tl.uint32)
        ep = (tl.load(ep_ptr + offs, mask=m, other=0) & 0xFFFFFFFF).to(tl.uint32)
        h = _lowbias32(key ^ _lowbias32(ep))
        scale = 1.0 / 16777216.0
        u0 = (_lowbias32(h ^ s0.to(tl.uint32)) >> 8).to(tl.float32) * scale
        u1 = (_lowbias32(h ^ s1.to(tl.uint32)) >> 8).to(tl.float32) * scale
        u2 = (_lowbias32(h ^ s2.to(tl.uint32)) >> 8).to(tl.float32) * scale
        tl.store(out_ptr + offs * 3, u0, mask=m)
        tl.store(out_ptr + offs * 3 + 1, u1, mask=m)
        tl.store(out_ptr + offs * 3 + 2, u2, mask=m)

    return kernel


def uniforms(key: torch.Tensor, episode: torch.Tensor) -> torch.Tensor:
    """uniforms_torch, one Triton launch on CUDA (bit-identical, tests/test_env.py)."""
    global _uniform_kernel
    if not key.is_cuda:
        return uniforms_torch(key, episode)
    if _uniform_kernel is None:
        _uniform_kernel = _build_uniform_kernel()
    n = key.shape[0]
    out = torch.empty(n, 3, dtype=torch.float32, device=key.device)
    block = 256
    _uniform_kernel[((n + block - 1) // block,)](
        key.contiguous(), episode.contiguous(), out, n, *_STREAMS, BLOCK=block
    )
    return out


# ── Renderer ────────────────────────────────────────────────────────────────


class SpotRenderer:
    """Camera image of the M2 face from the sim's per-hit outputs.

    Field of view: the mirror aperture (diameter D), res x res pixels of pitch
    h = D / res, in the M2-local (lx, ly) of herriott_sim (projections onto M2's
    rotated x / y axes; at zero tilt the world x / y).

    Orientation: column j <-> lx = -D/2 + (j + 0.5) h  (left to right, +lx right)
                 row i    <-> ly = +D/2 - (i + 0.5) h  (top to bottom, +ly up)
    M2 faces -z, so its local z (+lx x +ly) points away from the cavity, toward the
    viewer: the image shows M2 from behind, as a camera behind the mirror sees the
    transmitted spots. Seen from M1 (inside the cavity) it is mirrored left-right.
    imshow(img, origin="upper", extent=(-D/2, D/2, -D/2, D/2)) gives mm axes.

    Spot of the hit with 0-based bounce index k: incident intensity reflectivity^k,
    profile intensity * exp(-r^2 / (2 sigma^2)) (no normalization). Steps:
        1. bilinear splat of the M2 hits (M1 hits and padding weigh 0) into a grid
           oversampled by S and padded by the kernel half-width, index_add_ on flat
           indices
        2. separable Gaussian blur truncated at 3 sigma, sampled at the pixel centres
           only (stride S), as two matmuls with a banded (n_fine, res) matrix
        3. circular aperture mask
        4. fixed response 1 - exp(-gain * I) (no per-image normalization), 8 bit
    The bilinear splat widens a spot by up to 1/4 grid px^2 in variance; at sigma
    >= 2.5 grid px the error is < 4 % of the peak (auto S: 2 at 64 px, 1 at 128 px for
    sigma 0.5 mm). Without oversampling at 64 px (sigma 1.26 px) it is 12 %.
    """

    def __init__(
        self,
        res: int,
        diameter: float,
        sigma: float,
        gain: float,
        reflectivity: float,
        n_hits: int,
        oversample: int = 0,
        device: Optional[torch.device] = None,
    ):
        dev = torch.device(device) if device is not None else DEVICE
        self.res = res
        self.radius = diameter / 2
        self.pitch = diameter / res
        self.sigma = sigma
        self.gain = gain
        S = oversample or max(1, math.ceil(AUTO_SIGMA_PX * self.pitch / sigma - 1e-9))
        self.oversample = S
        self.fine_pitch = self.pitch / S
        self.half = P = math.ceil(3 * sigma / self.fine_pitch)  # Kernel half-width
        self.n_fine = (res - 1) * S + 2 * P + 1
        # Fine index m sits at x0 + (m - P) * fine_pitch (cols) / y0 - (...) (rows)
        self.x0 = -self.radius + self.pitch / 2
        self.y0 = self.radius - self.pitch / 2

        pix = torch.arange(res, device=dev, dtype=torch.float32)
        self.xs = -self.radius + (pix + 0.5) * self.pitch  # Column centres, lx (mm)
        self.ys = self.radius - (pix + 0.5) * self.pitch  # Row centres, ly (mm)
        self.aperture = (
            self.xs[None, :] ** 2 + self.ys[:, None] ** 2 <= self.radius**2
        ).float()  # (res, res)

        # Banded blur matrix: kern[m, j] = Gaussian weight of fine index m at pixel j
        d = torch.arange(self.n_fine, device=dev)[:, None] - (
            torch.arange(res, device=dev) * S + P
        )[None, :]
        dist = d.double() * self.fine_pitch
        kern = torch.exp(-(dist**2) / (2 * sigma**2)) * (d.abs() <= P)
        self.kern = kern.float().contiguous()  # (n_fine, res)
        self.kern_t = self.kern.T.contiguous()  # (res, n_fine)

        k = torch.arange(n_hits, device=dev, dtype=torch.float64)
        self.hit_intensity = (reflectivity**k).float()  # (N,), incident intensity

    def intensity(self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor) -> torch.Tensor:
        """(n, H, W) float32 image before the camera response."""
        n, N = hit_mirror.shape
        S, P, nf, hf = self.oversample, self.half, self.n_fine, self.fine_pitch
        w = torch.where(hit_mirror == M2_ID, self.hit_intensity[:N], 0.0)
        u = (hit_local_xy[..., 0] - self.x0) / hf + P
        v = (self.y0 - hit_local_xy[..., 1]) / hf + P
        u0 = u.floor()
        v0 = v.floor()
        fu = u - u0
        fv = v - v0
        inside = (u0 >= 0) & (u0 <= nf - 2) & (v0 >= 0) & (v0 <= nf - 2)
        w = torch.where(inside, w, 0.0)
        u0 = u0.clamp(0, nf - 2).long()
        v0 = v0.clamp(0, nf - 2).long()
        rows = torch.arange(n, device=w.device)[:, None] * (nf * nf)
        base = rows + v0 * nf + u0
        idx = torch.stack((base, base + 1, base + nf, base + nf + 1), dim=-1)
        gu = 1.0 - fu
        gv = 1.0 - fv
        wt = torch.stack((w * gu * gv, w * fu * gv, w * gu * fv, w * fu * fv), dim=-1)
        fine = torch.zeros(n * nf * nf, device=w.device, dtype=torch.float32)
        fine.index_add_(0, idx.reshape(-1), wt.reshape(-1))
        fine = fine.view(n, nf, nf)  # (rows v, cols u)
        img = torch.matmul(self.kern_t, fine @ self.kern)  # (n, res, res)
        return img * self.aperture

    def respond(self, intensity: torch.Tensor) -> torch.Tensor:
        """Saturating response 1 - exp(-gain * I), quantized: (n, 1, H, W) uint8."""
        resp = -torch.expm1(-self.gain * intensity)
        return (resp * 255.0).round().to(torch.uint8).unsqueeze(1)

    def __call__(self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor) -> torch.Tensor:
        return self.respond(self.intensity(hit_mirror, hit_local_xy))


# ── History stack ───────────────────────────────────────────────────────────


class HistoryStack:
    """Last K frames per env and the applied actions that led to them, on the device.

    Entries are oldest first, the newest at index K - 1. A reset clears an env's
    history and stores the first frame with a zero action; invalid entries are zero
    and False in the mask. Updates replace the stored tensors (no in-place writes), so
    tensors returned by get() / peek_with() stay unchanged after later pushes.
    """

    def __init__(
        self,
        num_envs: int,
        length: int,
        img_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
    ):
        dev = torch.device(device) if device is not None else DEVICE
        self.length = length
        self.images = torch.zeros(num_envs, length, *img_shape, dtype=torch.uint8, device=dev)
        self.actions = torch.zeros(num_envs, length, 2, dtype=torch.float32, device=dev)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=dev)  # Valid entries
        self._slot = torch.arange(length, device=dev)

    def _mask(self, count: torch.Tensor) -> torch.Tensor:
        return self._slot[None, :] >= (self.length - count)[:, None]

    def _appended(
        self, image: torch.Tensor, action: torch.Tensor, keep: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack shifted by one with (image, action) newest; rows not kept lose the rest."""
        old_i = self.images[:, 1:]
        old_a = self.actions[:, 1:]
        if keep is not None:
            old_i = torch.where(keep.view(-1, 1, 1, 1, 1), old_i, 0)
            old_a = torch.where(keep.view(-1, 1, 1), old_a, 0.0)
        images = torch.cat((old_i, image.unsqueeze(1)), dim=1)
        actions = torch.cat((old_a, action.to(torch.float32).unsqueeze(1)), dim=1)
        return images, actions

    def push(
        self,
        image: torch.Tensor,
        applied_action: torch.Tensor,
        reset_mask: torch.Tensor,
        active: Optional[torch.Tensor] = None,
    ):
        """Append a frame (B, 1, H, W) and its action (B, 2); reset_mask (B,) envs
        restart; envs with active (B,) False keep their stack unchanged."""
        keep = ~reset_mask
        action = torch.where(keep[:, None], applied_action, 0.0)
        images, actions = self._appended(image, action, keep)
        count = torch.where(keep, (self.count + 1).clamp(max=self.length), 1)
        if active is not None:
            images = torch.where(active.view(-1, 1, 1, 1, 1), images, self.images)
            actions = torch.where(active.view(-1, 1, 1), actions, self.actions)
            count = torch.where(active, count, self.count)
        self.images, self.actions, self.count = images, actions, count

    def reset(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Restart the masked envs (all if None) with image as their first frame."""
        if mask is None:
            mask = torch.ones_like(self.count, dtype=torch.bool)
        images, actions = self._appended(image, torch.zeros_like(self.actions[:, 0]), ~mask)
        self.images = torch.where(mask.view(-1, 1, 1, 1, 1), images, self.images)
        self.actions = torch.where(mask.view(-1, 1, 1), actions, self.actions)
        self.count = torch.where(mask, 1, self.count)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """images (B, K, 1, H, W) uint8, actions (B, K, 2) float32, mask (B, K) bool."""
        return self.images, self.actions, self._mask(self.count)

    def peek_with(
        self, image: torch.Tensor, applied_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stack after pushing (image, applied_action) without a reset; no mutation."""
        images, actions = self._appended(image, applied_action, None)
        return images, actions, self._mask((self.count + 1).clamp(max=self.length))


# ── Environment ─────────────────────────────────────────────────────────────


def _same_cell(a: CellGeometry, b: CellGeometry) -> bool:
    """Equal geometry apart from the name."""
    return replace(a, name="") == replace(b, name="")


class AlignEnv:
    """Vectorized alignment env, see the module docstring.

    step() simulates and renders one batch of 2B states (B without auto-reset): the B
    post-step states and B freshly drawn reset states; the reset half is used only
    where an episode ended (no data-dependent indexing, no host syncs).

    Setup attributes: setups (tuple of CellGeometry), n_setups, setup_probs (S,),
    path_refs (S,) float32 tensor (P_ref per setup), path_ref_mode ("nominal" / "max"),
    p_nom (float, P_ref of setup 0: 2733.27 mm by default). Per env: setup (B,) int64
    index of the episode's setup, episode (B,) int64 episode counter, idle (B,) bool
    (only without auto-reset).
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = c = cfg or EnvConfig()
        self.device = dev = torch.device(c.device) if c.device is not None else DEVICE
        self.num_envs = B = c.num_envs
        assert len(c.controlled) == 2, "two controlled DOFs"
        assert c.step_ladder[0] == 0, "ladder index 0 must be the zero step"
        assert c.path_ref in PATH_REFS, f"path_ref {c.path_ref!r} not in {PATH_REFS}"
        self.ctrl = tuple(int(i) for i in c.controlled)

        # ── Setups ──
        self.setups: Tuple[CellGeometry, ...] = tuple(c.setups) or (c.geometry,)
        S = self.n_setups = len(self.setups)
        g = self.setups[0]
        for s in self.setups[1:]:
            assert s.mirror_configs() == g.mirror_configs(), "setups must share the mirrors"
            assert s.m2_tilt_limit == g.m2_tilt_limit, "setups must share the tilt limit"
            assert s.intensity_threshold == g.intensity_threshold
        self.limit = float(g.m2_tilt_limit if c.tilt_limit is None else c.tilt_limit)
        self.setup_base = torch.cat([s.base_state(1, dev) for s in self.setups])  # (S, 13)
        self._dead = self.setup_base[0].clone()
        self._dead[S_LASER_PITCH] = DEAD_LASER_PITCH
        w = torch.tensor(
            c.setup_weights if c.setup_weights is not None else [1.0] * S, dtype=torch.float64
        )
        assert w.numel() == S and bool((w >= 0).all()) and float(w.sum()) > 0, "setup weights"
        self.setup_probs = (w / w.sum()).float().to(dev)
        self._setup_cdf = torch.cumsum(w / w.sum(), 0).float().to(dev)

        # ── Actions ──
        self.n_bins = len(c.step_ladder)
        self.ladder = (
            torch.tensor(c.step_ladder, dtype=torch.float64) * c.step_unit
        ).float().to(dev)  # (n_bins,) deg
        self.max_step = max(abs(s) for s in c.step_ladder) * c.step_unit  # deg

        # ── Simulator and renderer ──
        self.sim = g.make_sim(dev, max_bounces=c.max_bounces)
        self.renderer = SpotRenderer(
            c.img_res, g.diameter, c.spot_sigma, c.gain, g.reflectivity,
            c.max_bounces, c.oversample, dev,
        )
        self.img_shape = (1, c.img_res, c.img_res)

        # ── Reward scale per setup ──
        self.path_ref_mode = c.path_ref if c.path_ref != "auto" else (
            "nominal" if S == 1 else "max")
        self._path_max: Optional[torch.Tensor] = None
        if self.path_ref_mode == "max":
            refs = self.path_max().tolist()
        else:
            refs = self._nominal_paths()
        self.p_nom = float(refs[0])
        self.path_refs = torch.tensor(refs, dtype=torch.float32, device=dev)  # (S,)

        # ── Reset draws: per-env key of the counter-based hash ──
        self.gen = torch.Generator(device=dev)  # Kept for callers; resets use the hash
        self.gen.manual_seed(c.seed)
        sd = int(c.seed) & ((1 << 64) - 1)
        k = hash32(torch.full((B,), sd & _M32, dtype=torch.long, device=dev))
        k = hash32(k ^ hash32(torch.full((B,), (sd >> 32) & _M32, dtype=torch.long, device=dev)))
        self._env_key = hash32(k ^ torch.arange(B, device=dev))
        self._setup0 = torch.zeros(B, dtype=torch.long, device=dev)
        self.set_init_box(c.init_center, c.init_radius)

        # ── Per-env state ──
        self.tilt = torch.zeros(B, 2, device=dev)  # Controlled DOFs (deg)
        self.setup = torch.zeros(B, dtype=torch.long, device=dev)  # Index into setups
        self.episode = torch.full((B,), -1, dtype=torch.long, device=dev)  # Counter
        self.idle = torch.zeros(B, dtype=torch.bool, device=dev)  # Done, no auto-reset
        self.path = torch.zeros(B, device=dev)  # mm, current state
        self.bounces = torch.zeros(B, dtype=torch.long, device=dev)
        self.stop = torch.zeros(B, dtype=torch.int8, device=dev)
        self.image = torch.zeros(B, *self.img_shape, dtype=torch.uint8, device=dev)
        self.prev_action = torch.zeros(B, 2, device=dev)  # Led to the current frame
        self.step_count = torch.zeros(B, dtype=torch.long, device=dev)
        self.ep_return = torch.zeros(B, device=dev)
        self.ep_moves = torch.zeros(B, dtype=torch.long, device=dev)
        self.path0 = torch.zeros(B, device=dev)  # Path at episode start
        self.history = (
            HistoryStack(B, c.history_len, self.img_shape, dev) if c.history_len > 0 else None
        )
        self.reset()

    # ── Reward scale ──

    def _nominal_paths(self) -> List[float]:
        """Path at each setup's nominal tilt; the measured cell uses NOMINAL_PATH_MM."""
        out = []
        for s in self.setups:
            if _same_cell(s, CellGeometry()):
                out.append(NOMINAL_PATH_MM)
                continue
            st = s.base_state(1, self.device)
            out.append(float(self.sim.simulate(st, store_world=False)["total_path_length"][0]))
        return out

    @torch.no_grad()
    def path_max(self) -> torch.Tensor:
        """(S,) float32 longest path per setup over the +-limit box on the ref_scan_step
        grid (computed once, cached)."""
        if self._path_max is None:
            L, h = self.limit, self.cfg.ref_scan_step
            n = int(round(2 * L / h)) + 1
            ax = torch.tensor([-L + h * i for i in range(n)], dtype=torch.float32,
                              device=self.device)
            P, Y = torch.meshgrid(ax, ax, indexing="ij")
            grid = torch.stack((P.flatten(), Y.flatten()), 1)
            best = []
            chunk = 1 << 16
            for s in range(self.n_setups):
                m = torch.zeros((), device=self.device)
                for i in range(0, grid.shape[0], chunk):
                    g = grid[i : i + chunk]
                    st = self.states(g, torch.full((g.shape[0],), s, device=self.device))
                    m = torch.maximum(m, self.sim.simulate(st, store_world=False)
                                      ["total_path_length"].max())
                best.append(m)
            self._path_max = torch.stack(best)
        return self._path_max

    def ref_of(self, setup: torch.Tensor) -> torch.Tensor:
        """(n,) float32 P_ref of setup indices (n,)."""
        return self.path_refs[setup]

    def _norm(self, path: torch.Tensor, setup: torch.Tensor) -> torch.Tensor:
        """path / P_ref of the setup (one setup: division by the float p_nom, as stage 1)."""
        if self.n_setups == 1:
            return path / self.p_nom
        return path / self.path_refs[setup]

    # ── Reset draws ──

    def set_init_box(self, center=None, radius=None):
        """Initial-tilt box (curriculum): center (2,) or (B, 2), radius scalar, (2,) or
        (B, 1), deg. Takes effect at the next reset of each env."""
        if center is not None:
            self.init_center = torch.as_tensor(center, dtype=torch.float32).to(self.device)
        if radius is not None:
            self.init_radius = torch.as_tensor(radius, dtype=torch.float32).to(self.device)

    def _draw(self, episode: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initial tilts (B, 2) in the init box (clamped) and setups (B,) of the given
        per-env episode indices; a function of (seed, env, episode) only."""
        u = uniforms(self._env_key, episode)
        t = (self.init_center + self.init_radius * (2.0 * u[:, :2] - 1.0)).clamp(
            -self.limit, self.limit)
        if self.n_setups == 1:
            return t, self._setup0
        s = torch.searchsorted(self._setup_cdf, u[:, 2].contiguous(), right=True)
        return t, s.clamp(max=self.n_setups - 1)

    def _sample_tilts(self) -> torch.Tensor:
        """(B, 2) initial tilts of each env's next episode (as reset() would draw)."""
        return self._draw(self.episode + 1)[0]

    # ── Sim and render ──

    def states(self, tilts: torch.Tensor, setup: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(n, 13) states: base of each row's setup (default 0) with the controlled DOFs
        set to tilts (n, 2)."""
        n = tilts.shape[0]
        if setup is None:
            s = self.setup_base[0].expand(n, -1).clone()
        else:
            s = self.setup_base[setup]
        s[:, self.ctrl[0]] = tilts[:, 0]
        s[:, self.ctrl[1]] = tilts[:, 1]
        return s

    _states = states

    def _simulate(
        self,
        tilts: torch.Tensor,
        setup: Optional[torch.Tensor] = None,
        live: Optional[torch.Tensor] = None,
    ) -> dict:
        """Sim of (n, 2) tilts of setups (n,) (default 0). live (n,) bool: the other rows
        are traced as the dead state (0 bounces, blank image; no sync)."""
        s = self.states(tilts, setup)
        if live is not None:
            s = torch.where(live[:, None], s, self._dead)
        return self.sim.simulate(s, store_world=False)

    def _render(self, out: dict) -> torch.Tensor:
        return self.renderer(out["hit_mirror"], out["hit_local_xy"])

    def render_tilts(
        self, tilts: torch.Tensor, setup: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Diagnostics: sim + image of arbitrary (n, 2) tilts of setups (n,) (default 0);
        no env state change."""
        tilts = tilts.to(device=self.device, dtype=torch.float32)
        if setup is not None:
            setup = setup.to(device=self.device, dtype=torch.long)
        out = self._simulate(tilts, setup)
        inten = self.renderer.intensity(out["hit_mirror"], out["hit_local_xy"])
        return {
            "image": self.renderer.respond(inten),
            "intensity": inten,
            "path_mm": out["total_path_length"],
            "bounces": out["hit_counts"],
            "stop_reason": out["stop_reason"],
            "sim": out,
        }

    # ── State (checkpoint / resume) ──

    _STATE_KEYS = ("tilt", "setup", "episode", "idle", "path", "bounces", "stop", "image",
                   "prev_action", "step_count", "ep_return", "ep_moves", "path0",
                   "init_center", "init_radius")

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Per-env episode state and the init box (tensors, for a checkpoint); with the
        same config and seed, set_state() continues the same episode stream."""
        st = {k: getattr(self, k) for k in self._STATE_KEYS}
        if self.history is not None:
            st["hist_images"], st["hist_actions"], st["hist_count"] = (
                self.history.images, self.history.actions, self.history.count)
        return st

    def set_state(self, st: Dict[str, torch.Tensor]):
        """Restore get_state() output (same num_envs and config)."""
        for k in self._STATE_KEYS:
            setattr(self, k, st[k].to(self.device))
        if self.history is not None and "hist_images" in st:
            self.history.images = st["hist_images"].to(self.device)
            self.history.actions = st["hist_actions"].to(self.device)
            self.history.count = st["hist_count"].to(self.device)

    # ── Gym-style API ──

    def observe(self) -> Dict[str, torch.Tensor]:
        """Current observation (no state change)."""
        obs = {"image": self.image, "action": self.prev_action}
        if self.history is not None:
            obs["images"], obs["actions"], obs["mask"] = self.history.get()
        return obs

    def state_info(self) -> Dict[str, torch.Tensor]:
        """Privileged quantities of the current frame: tilt, setup, path_ref, path_mm,
        path_norm, bounces, stop_reason, step, idle."""
        return {
            "tilt": self.tilt,
            "setup": self.setup,
            "path_ref": self.ref_of(self.setup),
            "path_mm": self.path,
            "path_norm": self._norm(self.path, self.setup),
            "bounces": self.bounces,
            "stop_reason": self.stop,
            "step": self.step_count,
            "idle": self.idle,
        }

    def reset(
        self,
        mask: Optional[torch.Tensor] = None,
        tilts: Optional[torch.Tensor] = None,
        setups: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        """Start new episodes for the masked envs (all if None).

        tilts (B, 2): explicit initial tilts (clamped to the limits); setups (B,):
        explicit setup indices; default: drawn for the env's next episode. Simulates
        only the masked envs (no host sync). Returns obs, or (obs, info) with
        return_info (state_info() plus reset (B,) = mask).
        """
        B = self.num_envs
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=self.device)
        ep = torch.where(mask, self.episode + 1, self.episode)
        d_tilt, d_setup = self._draw(ep)
        new = d_tilt if tilts is None else tilts.to(
            device=self.device, dtype=torch.float32).clamp(-self.limit, self.limit)
        new_s = d_setup if setups is None else setups.to(device=self.device, dtype=torch.long)
        m1 = mask[:, None]
        tilt = torch.where(m1, new, self.tilt)
        setup = torch.where(mask, new_s, self.setup)
        out = self._simulate(tilt, setup, live=mask)
        img = self._render(out)

        self.tilt = tilt
        self.setup = setup
        self.episode = ep
        self.idle = self.idle & ~mask
        self.path = torch.where(mask, out["total_path_length"], self.path)
        self.bounces = torch.where(mask, out["hit_counts"], self.bounces)
        self.stop = torch.where(mask, out["stop_reason"], self.stop)
        self.image = torch.where(mask.view(-1, 1, 1, 1), img, self.image)
        self.prev_action = torch.where(m1, 0.0, self.prev_action)
        self.step_count = torch.where(mask, 0, self.step_count)
        self.ep_return = torch.where(mask, 0.0, self.ep_return)
        self.ep_moves = torch.where(mask, 0, self.ep_moves)
        self.path0 = torch.where(mask, self.path, self.path0)
        if self.history is not None:
            self.history.reset(self.image, mask)
        if return_info:
            info = self.state_info()
            info["reset"] = mask
            return self.observe(), info
        return self.observe()

    def step(self, action: torch.Tensor):
        """Apply (B, 2) ladder indices.

        Returns obs, reward (B,), terminated (B,), truncated (B,), info. info (all
        (B,), computed before the auto-reset):
            final_image (B, 1, H, W) uint8, final_obs (obs dict of that frame, with
            its history stack), applied_action (B, 2), tilt (B, 2) true controlled
            tilts, setup, path_ref (privileged), path_mm, path_norm = path_mm / P_ref
            (path-head label of final_image), bounces, stop_reason, done, idle (env
            was idle: action ignored, reward 0, not done; only without auto-reset);
            ep_return, ep_length, ep_moves, ep_initial_path_mm, ep_final_path_mm,
            ep_final_bounces: valid where done;
            obs_tilt, obs_setup, obs_path_ref, obs_path_norm: the same for the
            returned obs (post-reset).
        """
        c = self.cfg
        B = self.num_envs
        auto = c.auto_reset
        idle = self.idle  # All False with auto-reset
        a = action.to(device=self.device, dtype=torch.long)
        old = self.tilt
        if auto:
            move = (a != 0).any(dim=1)
            new = (old + self.ladder[a]).clamp(-self.limit, self.limit)
            step_count = self.step_count + 1
            terminated = ~move
        else:
            move = (a != 0).any(dim=1) & ~idle
            new = torch.where(move[:, None], (old + self.ladder[a]).clamp(-self.limit, self.limit),
                              old)
            step_count = torch.where(idle, self.step_count, self.step_count + 1)
            terminated = ~move & ~idle
        applied = (new - old) / self.max_step
        truncated = move & (step_count >= c.max_steps)
        done = terminated | truncated

        # One sim + render pass over post-step and (with auto-reset) reset states
        setup = self.setup
        if auto:
            ep_next = self.episode + 1
            reset_tilt, reset_setup = self._draw(ep_next)
            out = self._simulate(
                torch.cat((new, reset_tilt)), torch.cat((setup, reset_setup)),
                live=torch.cat((move, done)),
            )
        else:
            out = self._simulate(new, setup, live=move)
        img = self._render(out)
        path_all = out["total_path_length"]
        count_all = out["hit_counts"]
        stop_all = out["stop_reason"]

        # Post-step state; not move keeps the cached state (no sim needed)
        path = torch.where(move, path_all[:B], self.path)
        bounces = torch.where(move, count_all[:B], self.bounces)
        stop = torch.where(move, stop_all[:B], self.stop)
        image = torch.where(move.view(-1, 1, 1, 1), img[:B], self.image)
        reward = torch.where(move, self._norm(path - self.path, setup) - c.step_cost, 0.0)
        prev_action = applied if auto else torch.where(idle[:, None], self.prev_action, applied)

        ep_return = self.ep_return + reward
        ep_moves = self.ep_moves + move.long()

        final_obs = {"image": image, "action": prev_action}
        if self.history is not None:
            if auto:
                final_obs["images"], final_obs["actions"], final_obs["mask"] = (
                    self.history.peek_with(image, applied)
                )
            else:
                pi, pa, pm = self.history.peek_with(image, applied)
                hi, ha, hm = self.history.get()
                final_obs["images"] = torch.where(idle.view(-1, 1, 1, 1, 1), hi, pi)
                final_obs["actions"] = torch.where(idle.view(-1, 1, 1), ha, pa)
                final_obs["mask"] = torch.where(idle[:, None], hm, pm)
        info = {
            "final_image": image,
            "final_obs": final_obs,
            "applied_action": applied,
            "tilt": new,
            "setup": setup,
            "path_ref": self.ref_of(setup),
            "path_mm": path,
            "path_norm": self._norm(path, setup),
            "bounces": bounces,
            "stop_reason": stop,
            "done": done,
            "idle": idle,
            "ep_return": ep_return,
            "ep_length": step_count,
            "ep_moves": ep_moves,
            "ep_initial_path_mm": self.path0,
            "ep_final_path_mm": path,
            "ep_final_bounces": bounces,
        }

        if auto:
            # Auto-reset of the done envs
            d1 = done[:, None]
            self.tilt = torch.where(d1, reset_tilt, new)
            self.setup = torch.where(done, reset_setup, setup)
            self.episode = torch.where(done, ep_next, self.episode)
            self.path = torch.where(done, path_all[B:], path)
            self.bounces = torch.where(done, count_all[B:], bounces)
            self.stop = torch.where(done, stop_all[B:], stop)
            self.image = torch.where(done.view(-1, 1, 1, 1), img[B:], image)
            self.prev_action = torch.where(d1, 0.0, applied)
            self.step_count = torch.where(done, 0, step_count)
            self.ep_return = torch.where(done, 0.0, ep_return)
            self.ep_moves = torch.where(done, 0, ep_moves)
            self.path0 = torch.where(done, self.path, self.path0)
            if self.history is not None:
                self.history.push(self.image, self.prev_action, done)
        else:
            # Done envs keep their final frame and idle until reset()
            self.tilt = new
            self.path, self.bounces, self.stop, self.image = path, bounces, stop, image
            self.prev_action = prev_action
            self.step_count = step_count
            self.ep_return = ep_return
            self.ep_moves = ep_moves
            self.idle = idle | done
            if self.history is not None:
                self.history.push(image, applied, torch.zeros_like(done), active=~idle)

        info["obs_tilt"] = self.tilt
        info["obs_setup"] = self.setup
        info["obs_path_ref"] = self.ref_of(self.setup)
        info["obs_path_norm"] = self._norm(self.path, self.setup)
        return self.observe(), reward, terminated, truncated, info
