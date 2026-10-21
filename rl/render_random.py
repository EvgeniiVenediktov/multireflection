"""
render_random.py - Randomized camera renderer for encoder pretraining (PLAN phase 5)

RandomRenderer wraps env.SpotRenderer (composition). Called without nuisance
parameters it returns the SpotRenderer image itself (bit-identical). With per-sample
NuisanceParams it renders the same M2 hits through a randomized surface / camera /
pose model. The nuisances act on the input image only; the training targets stay in
the canonical M2 frame (targets.py).

Randomized pipeline per sample (canonical raster = the SpotRenderer pixel grid in
M2-local mm, +lx right, +ly up):
    1. spots: one elliptical Gaussian per M2 hit, peak reflectivity^k (as SpotRenderer,
       no normalization), blur sigma (geometric mean of the two axis sigmas), aspect
       ratio, random orientation; exact evaluation at the pixel centres (dense_spots,
       torch.compile on CUDA; the splat + banded blur of SpotRenderer needs one fixed
       sigma)
    2. scatter halo: isotropic Gaussian per spot, sigma_h, peak halo_amp x spot peak
    3. aperture mask (SpotRenderer's), then a mirror rim (Gaussian ring at the aperture
       edge) and a uniform glow on the mirror face; both in intensity units that give
       the stated brightness at unit gain, so the mirror frame stays visible
    4. camera pose: homography (rotation, shift, scale, corner jitter), applied with
       grid_sample (bilinear, zeros outside the canonical field)
    5. camera: gain x intensity, SpotRenderer response 1 - exp(-g I), gamma, black
       level, shot + read noise, clip to [0, 1], uint8

Held-out ("OOD") ranges for evaluation: OOD_FACTORS / ood_ranges().
"""

import math
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from env import SpotRenderer
from targets import m2_spots

RIM_WIDTH_MM = 0.4  # Gaussian sigma of the rim ring

OOD_FACTORS = ("sigma", "rotation", "read_noise", "gain_low", "gain_high")


# ── Ranges and parameters ───────────────────────────────────────────────────


@dataclass
class NuisanceRanges:
    """Per-sample nuisance ranges, uniform unless noted. Ranges marked 'random sign'
    are magnitudes: |v| uniform in [lo, hi], sign random."""

    # ── Surface finish ──
    sigma: Tuple[float, float] = (0.3, 1.0)  # mm, spot blur (geometric mean of the axes)
    aspect: Tuple[float, float] = (1.0, 1.3)  # Major / minor axis sigma
    halo_sigma: Tuple[float, float] = (2.0, 4.0)  # mm
    halo_amp: Tuple[float, float] = (0.0, 0.1)  # Halo peak / spot peak

    # ── Camera ──
    gain: Tuple[float, float] = (0.6, 1.5)  # Log-uniform; also stands in for laser color
    gamma: Tuple[float, float] = (0.8, 1.2)
    black: Tuple[float, float] = (0.0, 0.05)  # Full scale = 1
    read_noise: Tuple[float, float] = (0.0, 0.02)  # std, full scale = 1
    shot_noise: Tuple[float, float] = (0.0, 1e-3)  # Variance per unit signal

    # ── Camera pose ──
    rotation: Tuple[float, float] = (0.0, 5.0)  # deg, random sign
    shift: Tuple[float, float] = (0.0, 0.5)  # mm per axis, random sign
    scale: Tuple[float, float] = (0.95, 1.05)  # Log-uniform
    corner: float = 0.03  # Max corner displacement per axis, fraction of the field width

    # ── Mirror frame ──
    rim: Tuple[float, float] = (0.0, 0.15)  # Rim brightness at unit gain
    background: Tuple[float, float] = (0.0, 0.05)  # Mirror-face glow at unit gain


def ood_ranges(factor: str, base: Optional[NuisanceRanges] = None) -> NuisanceRanges:
    """Training ranges with one factor moved to its held-out value."""
    base = base or NuisanceRanges()
    if factor == "sigma":
        return replace(base, sigma=(1.2, 1.4))
    if factor == "rotation":
        return replace(base, rotation=(7.0, 9.0))
    if factor == "read_noise":
        return replace(base, read_noise=(0.04, 0.04))
    if factor == "gain_low":
        return replace(base, gain=(0.4, 0.4))
    if factor == "gain_high":
        return replace(base, gain=(1.8, 1.8))
    raise ValueError(f"unknown OOD factor {factor!r}, expected one of {OOD_FACTORS}")


@dataclass
class NuisanceParams:
    """Per-sample nuisance values, all (n,) float32 except warp (n, 3, 3): homography
    from camera to canonical normalized coordinates (grid_sample convention: [-1, 1],
    x right, y down); None = no warp."""

    sigma: torch.Tensor  # mm
    aspect: torch.Tensor
    angle: torch.Tensor  # rad, major axis from +lx
    halo_sigma: torch.Tensor  # mm
    halo_amp: torch.Tensor
    gain: torch.Tensor
    gamma: torch.Tensor
    black: torch.Tensor
    read_noise: torch.Tensor
    shot_noise: torch.Tensor
    rim: torch.Tensor
    background: torch.Tensor
    warp: Optional[torch.Tensor]

    def select(self, mask: torch.Tensor, other: "NuisanceParams") -> "NuisanceParams":
        """Per sample: other where mask, self elsewhere."""
        out = {}
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if a is None or b is None:
                assert a is None and b is None, f.name
                out[f.name] = None
                continue
            out[f.name] = torch.where(mask.view(-1, *([1] * (a.dim() - 1))), b, a)
        return NuisanceParams(**out)

    def scalars(self) -> Dict[str, torch.Tensor]:
        """The (n,) fields (everything but warp)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "warp"}


# ── Sampling helpers ────────────────────────────────────────────────────────


def _uniform(n: int, lo_hi, gen, device) -> torch.Tensor:
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=device)


def _log_uniform(n: int, lo_hi, gen, device) -> torch.Tensor:
    lo, hi = lo_hi
    return torch.exp(_uniform(n, (math.log(lo), math.log(hi)), gen, device))


def _signed(n: int, lo_hi, gen, device) -> torch.Tensor:
    sign = torch.randint(0, 2, (n,), generator=gen, device=device) * 2 - 1
    return sign * _uniform(n, lo_hi, gen, device)


def corner_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """(n, 3, 3) homographies mapping 4 src points to 4 dst points (n, 4, 2); DLT with
    h33 = 1, float64."""
    n = src.shape[0]
    src, dst = src.double(), dst.double()
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z, o = torch.zeros_like(x), torch.ones_like(x)
    rows_u = torch.stack((x, y, o, z, z, z, -u * x, -u * y), -1)
    rows_v = torch.stack((z, z, z, x, y, o, -v * x, -v * y), -1)
    A = torch.cat((rows_u, rows_v), 1)  # (n, 8, 8)
    # solve_ex without the error check: linalg.solve syncs the host to check the result
    h = torch.linalg.solve_ex(A, torch.cat((u, v), 1), check_errors=False).result
    return torch.cat((h, torch.ones(n, 1, dtype=h.dtype, device=h.device)), 1).view(n, 3, 3)


_CORNERS: Dict[torch.device, torch.Tensor] = {}  # Field corners per device (one copy each)


def _field_corners(dev: torch.device) -> torch.Tensor:
    """(4, 2) float64 field corners on dev, created once (a host-to-device copy syncs)."""
    c = _CORNERS.get(dev)
    if c is None:
        c = _CORNERS[dev] = torch.tensor(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64, device=dev)
    return c


def pose_warp(
    rotation_deg: torch.Tensor,
    shift_norm: torch.Tensor,
    scale: torch.Tensor,
    corner_disp: torch.Tensor,
) -> torch.Tensor:
    """(n, 3, 3) camera -> canonical homography in normalized coordinates.

    The camera sees the canonical field through H = T(shift) R(rotation) S(scale) P,
    where P moves the field corners (+-1, +-1) by corner_disp (n, 4, 2); returns H^-1.
    shift_norm (n, 2) is in normalized units (field half-width = 1).
    """
    dev = rotation_deg.device
    n = rotation_deg.shape[0]
    th = torch.deg2rad(rotation_deg.double())
    c, s, sc = torch.cos(th), torch.sin(th), scale.double()
    # Inverse of T R S in closed form: S^-1 R^-1 T^-1
    A_inv = torch.zeros(n, 3, 3, dtype=torch.float64, device=dev)
    A_inv[:, 0, 0] = c / sc
    A_inv[:, 0, 1] = s / sc
    A_inv[:, 1, 0] = -s / sc
    A_inv[:, 1, 1] = c / sc
    t = shift_norm.double()
    A_inv[:, 0, 2] = -(c * t[:, 0] + s * t[:, 1]) / sc
    A_inv[:, 1, 2] = -(-s * t[:, 0] + c * t[:, 1]) / sc
    A_inv[:, 2, 2] = 1.0
    corners = _field_corners(dev).expand(n, 4, 2)
    P_inv = corner_homography(corners + corner_disp.double(), corners)
    return (P_inv @ A_inv).float()


def dense_spots(
    xs: torch.Tensor,
    ys: torch.Tensor,
    pos: torch.Tensor,
    w: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    halo_s2: torch.Tensor,
    halo_amp: torch.Tensor,
) -> torch.Tensor:
    """(n, H, W) sum over spots k of w_k [exp(-q/2) + halo_amp exp(-r^2 / halo_s2)],
    q = A dx^2 + 2 B dx dy + C dy^2 (inverse covariance, per sample (n,)), at the pixel
    centres xs (W,), ys (H,); pos (n, K, 2) mm, w (n, K). Exact, no truncation; meant
    to run under torch.compile (one fused reduction, no (n, K, H, W) buffers)."""
    v = lambda t: t[:, None, None, None]
    dx = xs[None, None, None, :] - pos[:, :, 0, None, None]
    dy = ys[None, None, :, None] - pos[:, :, 1, None, None]
    q = v(A) * dx * dx + 2.0 * v(B) * dx * dy + v(C) * dy * dy
    val = torch.exp(-0.5 * q) + v(halo_amp) * torch.exp(-(dx * dx + dy * dy) / v(halo_s2))
    return (w[:, :, None, None] * val).sum(1)


# ── Renderer ────────────────────────────────────────────────────────────────


class RandomRenderer:
    """SpotRenderer with per-sample nuisances, see the module docstring.

    render(hit_mirror, hit_local_xy)                 -> SpotRenderer output, unchanged
    render(hit_mirror, hit_local_xy, params, gen)    -> randomized image (n, 1, H, W) uint8
    """

    def __init__(self, base: SpotRenderer):
        self.base = b = base
        self.device = b.xs.device
        self.n_spots = (b.hit_intensity.shape[0] + 1) // 2  # M2 hits: even bounce indices
        self._dense = torch.compile(dense_spots) if self.device.type == "cuda" else dense_spots
        r = torch.sqrt(b.xs[None, :] ** 2 + b.ys[:, None] ** 2)  # (H, W) mm
        self.rim_shape = torch.exp(-((r - b.radius) ** 2) / (2 * RIM_WIDTH_MM**2))
        res = b.res
        g = (2.0 * torch.arange(res, device=self.device, dtype=torch.float32) + 1.0) / res - 1.0
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.cam_grid = torch.stack((gx, gy, torch.ones_like(gx)), -1).view(1, res * res, 3)

    # ── Parameters ──

    def sample_params(
        self, n: int, ranges: NuisanceRanges, gen: torch.Generator
    ) -> NuisanceParams:
        """n parameter sets drawn from ranges."""
        r, dev = ranges, self.device
        rot = _signed(n, r.rotation, gen, dev)
        shift_mm = torch.stack((_signed(n, r.shift, gen, dev), _signed(n, r.shift, gen, dev)), 1)
        scale = _log_uniform(n, r.scale, gen, dev)
        corner = 2.0 * r.corner * (2.0 * torch.rand(n, 4, 2, generator=gen, device=dev) - 1.0)
        return NuisanceParams(
            sigma=_uniform(n, r.sigma, gen, dev),
            aspect=_uniform(n, r.aspect, gen, dev),
            angle=math.pi * torch.rand(n, generator=gen, device=dev),
            halo_sigma=_uniform(n, r.halo_sigma, gen, dev),
            halo_amp=_uniform(n, r.halo_amp, gen, dev),
            gain=_log_uniform(n, r.gain, gen, dev),
            gamma=_uniform(n, r.gamma, gen, dev),
            black=_uniform(n, r.black, gen, dev),
            read_noise=_uniform(n, r.read_noise, gen, dev),
            shot_noise=_uniform(n, r.shot_noise, gen, dev),
            rim=_uniform(n, r.rim, gen, dev),
            background=_uniform(n, r.background, gen, dev),
            warp=pose_warp(rot, shift_mm / self.base.radius, scale, corner),
        )

    def sample_ood(
        self, n: int, gen: torch.Generator, base: Optional[NuisanceRanges] = None
    ) -> Tuple[NuisanceParams, torch.Tensor]:
        """Training ranges with one held-out factor per sample (uniform over
        OOD_FACTORS); returns the params and the factor index (n,)."""
        base = base or NuisanceRanges()
        factor = torch.randint(0, len(OOD_FACTORS), (n,), generator=gen, device=self.device)
        params = self.sample_params(n, base, gen)
        for i, name in enumerate(OOD_FACTORS):
            params = params.select(factor == i, self.sample_params(n, ood_ranges(name, base), gen))
        return params, factor

    def identity_params(self, n: int, warp: bool = False) -> NuisanceParams:
        """Parameters that reproduce SpotRenderer up to its splat error; warp=True adds
        an explicit identity homography instead of None."""
        dev = self.device
        z = torch.zeros(n, device=dev)
        o = torch.ones(n, device=dev)
        W = torch.eye(3, device=dev).expand(n, 3, 3).clone() if warp else None
        return NuisanceParams(
            sigma=o * self.base.sigma, aspect=o.clone(), angle=z.clone(),
            halo_sigma=o * 3.0, halo_amp=z.clone(), gain=o.clone(), gamma=o.clone(),
            black=z.clone(), read_noise=z.clone(), shot_noise=z.clone(), rim=z.clone(),
            background=z.clone(), warp=W,
        )

    # ── Stages ──

    def spot_intensity(
        self, pos: torch.Tensor, w: torch.Tensor, p: NuisanceParams
    ) -> torch.Tensor:
        """(n, H, W) elliptical spots + halo on the canonical grid, before the aperture."""
        su2 = p.sigma**2 * p.aspect  # Major-axis variance
        sv2 = p.sigma**2 / p.aspect
        c, s = torch.cos(p.angle), torch.sin(p.angle)
        A = c * c / su2 + s * s / sv2
        C = s * s / su2 + c * c / sv2
        B = c * s * (1.0 / su2 - 1.0 / sv2)
        return self._dense(
            self.base.xs, self.base.ys, pos, w, A, B, C, 2.0 * p.halo_sigma**2, p.halo_amp
        )

    def canonical(
        self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor, p: NuisanceParams
    ) -> torch.Tensor:
        """(n, H, W) canonical intensity: spots, halo, aperture, rim, face glow."""
        b = self.base
        pos, w, _ = m2_spots(hit_mirror, hit_local_xy, b.hit_intensity, self.n_spots)
        img = self.spot_intensity(pos, w, p) * b.aperture
        to_int = lambda v: (-torch.log1p(-v) / b.gain)[:, None, None]  # Brightness -> I
        return img + to_int(p.rim) * self.rim_shape + to_int(p.background) * b.aperture

    def warp(self, img: torch.Tensor, warp: Optional[torch.Tensor]) -> torch.Tensor:
        """(n, H, W) camera view of the canonical image through warp (n, 3, 3)."""
        if warp is None:
            return img
        n, H, W = img.shape
        src = self.cam_grid @ warp.transpose(1, 2)  # (n, H W, 3)
        grid = (src[..., :2] / src[..., 2:]).view(n, H, W, 2)
        return F.grid_sample(
            img[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[:, 0]

    def camera(
        self, img: torch.Tensor, p: NuisanceParams, gen: Optional[torch.Generator]
    ) -> torch.Tensor:
        """Camera response of an (n, H, W) intensity image: (n, 1, H, W) uint8."""
        v = lambda t: t[:, None, None]
        resp = -torch.expm1(-self.base.gain * v(p.gain) * img)
        resp = resp.clamp_min(0.0) ** v(p.gamma)
        std = torch.sqrt(v(p.read_noise) ** 2 + v(p.shot_noise) * resp)
        noise = torch.randn(resp.shape, generator=gen, device=resp.device)
        out = (resp + v(p.black) + std * noise).clamp(0.0, 1.0)
        return (out * 255.0).round().to(torch.uint8).unsqueeze(1)

    # ── Entry point ──

    def render(
        self,
        hit_mirror: torch.Tensor,
        hit_local_xy: torch.Tensor,
        params: Optional[NuisanceParams] = None,
        gen: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """(n, 1, H, W) uint8; params None = SpotRenderer output (randomization off)."""
        if params is None:
            return self.base(hit_mirror, hit_local_xy)
        img = self.canonical(hit_mirror, hit_local_xy, params)
        return self.camera(self.warp(img, params.warp), params, gen)

    __call__ = render
