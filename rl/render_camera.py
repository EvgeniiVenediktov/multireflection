"""
render_camera.py - Camera-frame renderer for real-camera inputs (prototype)

CameraRenderer renders the M2 hits of the sim as a 3-channel camera image (n, 3, H, W)
uint8 through a randomized pose / mirror-face / edge-scatter / light / color / sensor
model. Unlike render_random.py the labels live in the camera frame: the spots where the
camera sees them (pixel positions, count). The camera pose is not a target; it is
left to the history (the dynamics of the actions) to resolve.

Coordinates: canonical normalized (X, Y) = (lx, -ly) / R (the mirror is the unit
disk, y down, as the SpotRenderer raster), camera normalized (u, v) in [-1, 1]
(x right, y down, pixel centres at (2 j + 1) / W - 1). Camera pose: (u, v) =
H (X, Y), H = T(shift) R(rotation) S(scale) P(corner jitter), the parameterization of
render_random.pose_warp (which returns H^-1). scale = mirror diameter / field width.

Layers (intensity before the sensor, per channel c):
    I_c = laser_c mask (spots + scatter)
          + ambient_c illumination (mask face + (1 - mask) surround)
    spots      elliptical Gaussian per M2 hit at its camera position, peak
               laser x reflectivity^k x speckle (log-normal per spot and frame)
    scatter    a beam on the M2 rim is not reflected back properly and lights the
               mirror around that point: a flat-top blob (super-Gaussian, saturating)
               at the rim point, masked by the mirror. Sources: the stored M2 hits and
               the miss point of a ray that ended past the M2 rim (sim miss_mirror /
               miss_local_xy), weight = beam intensity x the share of the beam on the
               rim bevel R +- b (b = EDGE_BEVEL_MM, beam sigma across the rim, d = r - R):
               (erf((b - d) / (sqrt 2 sigma)) + erf((b + d) / (sqrt 2 sigma))) / 2;
               the N_SCATTER strongest are drawn
    face       (mirror frame) level x (1 + radial rings + grain + engraving) + scratches
               (level given as brightness after the sensor, so exposure does not wash it out)
               + dust + edge glow (modulated along the rim)
    illumination  linear gradient x vignetting (camera frame)
Sensor: gain (base gain x exposure x white balance), response 1 - exp(-g I) (as
SpotRenderer), gamma, pixel gain non-uniformity, black level, shot + read noise, row
noise, hot pixels, clip, uint8. Mono cameras: one channel replicated x 3.

Parameters are a dict of per-sample tensors (GROUPS lists the keys per factor); they
hold everything that stays fixed for an episode (pose, textures, dust, hot pixels).
Per-frame randomness (noise, speckle) is drawn in render().
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from env import SpotRenderer
from render_random import _log_uniform, _signed, pose_warp
from targets import M2_ID, m2_spots

N_RING = 3
N_SCRATCH = 6
N_DUST = 12
N_HARM = 4  # Edge-glow harmonics
N_HOT = 6
N_SCATTER = 4  # Edge-scatter sources drawn per frame (the strongest)
EDGE_BEVEL_MM = 0.3  # Rim bevel half-width: the band where a beam scatters
N_TEXT = 48  # Engraving bank
TEXT_RES = 128
TEX_RES = 64  # Grain texture
ENGRAVINGS = ("THORLABS", "CM254-100-E02", "CM254-050-P01", "CM254-075-E03", "LA1608-B")

CameraParams = Dict[str, torch.Tensor]
Miss = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # Sim miss_mirror, miss_local_xy, intensity

GROUPS: Dict[str, Tuple[str, ...]] = {
    "pose": ("warp", "warp_inv", "scale"),
    "spots": ("laser", "sigma", "aspect", "angle", "speckle"),
    "scatter": ("scatter_amp", "scatter_size", "scatter_pow"),
    "color": ("mono", "laser_rgb", "ambient_rgb", "wb"),
    "face": ("face", "ring_r", "ring_w", "ring_a", "grain", "grain_tex", "rim", "rim_w",
             "rim_harm", "surround"),
    "marks": ("engrave_idx", "engrave_amp", "scratch_c", "scratch_dir", "scratch_len",
              "scratch_w", "scratch_amp", "dust_c", "dust_s", "dust_amp"),
    "light": ("gradient", "vignetting"),
    "sensor": ("exposure", "gamma", "black", "read_noise", "shot_noise", "prnu", "prnu_map",
               "row_noise", "hot_idx", "hot_val"),
}


# ── Ranges ──────────────────────────────────────────────────────────────────


@dataclass
class CameraRanges:
    """Per-sample ranges, uniform unless noted; 'random sign' = magnitude range.
    Intensities are in SpotRenderer units (first-spot peak 1 at unit laser)."""

    # ── Pose ──
    rotation: Tuple[float, float] = (0.0, 20.0)  # deg, random sign
    shift: Tuple[float, float] = (0.0, 0.15)  # Normalized (field half-width 1), random sign
    scale: Tuple[float, float] = (0.8, 1.08)  # Mirror diameter / field width, log-uniform
    corner: float = 0.04  # Perspective: max corner displacement, fraction of the field

    # ── Spots ──
    laser: Tuple[float, float] = (0.4, 6.0)  # First-spot peak, log-uniform
    sigma: Tuple[float, float] = (0.3, 1.0)  # mm
    aspect: Tuple[float, float] = (1.0, 1.3)
    speckle: Tuple[float, float] = (0.0, 0.3)  # Log-normal std of the per-spot peak

    # ── Edge scatter ──
    scatter_amp: Tuple[float, float] = (1.0, 20.0)  # Blob peak / beam intensity, log-uniform
    scatter_size: Tuple[float, float] = (0.15, 0.45)  # Mirror radii
    scatter_pow: Tuple[float, float] = (2.0, 5.0)  # Super-Gaussian exponent (flat top)

    # ── Camera type and color ──
    color_p: float = 0.5  # Color cameras; the rest mono (one channel replicated)
    crosstalk: Tuple[float, float] = (0.0, 0.25)  # Laser signal in the two other channels
    mono_qe: Tuple[float, float] = (0.3, 1.0)  # Mono response at the laser line
    tint: float = 0.15  # Ambient color, per-channel log-normal std
    wb: Tuple[float, float] = (0.8, 1.25)  # White balance per channel, log-uniform

    # ── Mirror face ──
    face_p: float = 0.6  # Lit faces; the rest dark (face up to dark_face)
    face: Tuple[float, float] = (0.05, 0.7)  # Brightness after the sensor (any exposure)
    dark_face: float = 0.03
    ring_r: Tuple[float, float] = (0.55, 0.97)  # Radial ring centres, mirror radii
    ring_w: Tuple[float, float] = (0.02, 0.1)
    ring_a: Tuple[float, float] = (-0.4, 0.6)  # Relative to the face level
    grain: Tuple[float, float] = (0.0, 0.3)  # Fractal texture std, relative
    rim: Tuple[float, float] = (0.0, 0.4)  # Edge glow
    rim_w: Tuple[float, float] = (0.02, 0.08)  # Mirror radii
    surround: Tuple[float, float] = (0.0, 0.08)  # Outside the mirror

    # ── Marks ──
    engrave_p: float = 0.5
    engrave_amp: Tuple[float, float] = (-0.5, 0.3)  # Relative to the face level
    scratch_amp: Tuple[float, float] = (0.0, 0.6)  # Random sign, x max(face, 0.1)
    scratch_len: Tuple[float, float] = (0.03, 0.2)  # Half-length, mirror radii
    dust_amp: Tuple[float, float] = (0.0, 1.0)  # Random sign (bright 3 : 1), x max(face, 0.05)
    dust_s: Tuple[float, float] = (0.01, 0.03)  # Mirror radii

    # ── Light ──
    gradient: Tuple[float, float] = (0.0, 0.3)
    vignetting: Tuple[float, float] = (0.0, 0.4)

    # ── Sensor ──
    exposure: Tuple[float, float] = (0.4, 2.5)  # Log-uniform
    gamma: Tuple[float, float] = (0.7, 1.4)
    black: Tuple[float, float] = (0.0, 0.08)
    read_noise: Tuple[float, float] = (0.0, 0.03)
    shot_noise: Tuple[float, float] = (0.0, 3e-3)
    prnu: Tuple[float, float] = (0.0, 0.03)
    row_noise: Tuple[float, float] = (0.0, 0.01)


# ── Engraving bank and textures ─────────────────────────────────────────────


def engraving_bank(n: int, res: int, seed: int) -> torch.Tensor:
    """(n, 1, res, res) float32 in [0, 1]: one or two lines of mirror-mount text at a
    random place and angle over the canonical field (the mirror = inscribed disk)."""
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    rng = np.random.default_rng(seed)
    big = 4 * res
    out = []
    for _ in range(n):
        font = ImageFont.truetype(path, int(big * rng.uniform(0.05, 0.08)))
        lines = list(rng.choice(ENGRAVINGS, size=int(rng.integers(1, 3)), replace=False))
        img = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(img)
        draw.multiline_text((big / 2, big / 2), "\n".join(lines), fill=255, font=font,
                            anchor="mm", align="center", spacing=big // 40)
        r, a = 0.45 * big / 2 * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        img = img.rotate(rng.uniform(0, 360), resample=Image.BILINEAR,
                         translate=(r * math.cos(a), r * math.sin(a)))
        out.append(np.asarray(img.resize((res, res), Image.BOX), dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(out))[:, None]


def fractal_noise(n: int, res: int, gen: torch.Generator, device, beta: float = 0.8) -> torch.Tensor:
    """(n, 1, res, res) zero-mean, unit-std multi-octave noise."""
    tex = torch.zeros(n, 1, res, res, device=device)
    o = 4
    while o <= res:
        z = torch.randn(n, 1, o, o, generator=gen, device=device)
        tex += F.interpolate(z, size=(res, res), mode="bilinear", align_corners=False) * (o / 4) ** -beta
        o *= 2
    tex -= tex.mean((2, 3), keepdim=True)
    return tex / tex.std((2, 3), keepdim=True).clamp_min(1e-6)


def _apply(H: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Homographies (n, 3, 3) applied to points (n, ..., 2)."""
    n = xy.shape[0]
    p = xy.reshape(n, -1, 2)
    q = torch.cat((p, torch.ones_like(p[..., :1])), -1) @ H.transpose(1, 2)
    return (q[..., :2] / q[..., 2:]).view(xy.shape)


def _canonical(lxy: torch.Tensor, radius: float) -> torch.Tensor:
    """M2-local (lx, ly) mm (..., 2) -> canonical normalized (X, Y)."""
    return torch.stack((lxy[..., 0], -lxy[..., 1]), -1) / radius


# ── Renderer ────────────────────────────────────────────────────────────────


class CameraRenderer:
    """Camera-frame renderer, see the module docstring.

    sample_params(n, ranges, gen) -> per-episode parameters
    neutral_params(n)             -> everything off: mono, mirror filling the field, no
                                     noise, no edge scatter (close to SpotRenderer)
    render(hit_mirror, hit_local_xy, params, gen, miss) -> (n, 3, H, W) uint8
    labels(hit_mirror, hit_local_xy, params)            -> camera-frame spots and count
    miss = (miss_mirror, miss_local_xy, intensity) of the sim output, or None (no miss
    point: edge scatter from the stored hits only).
    """

    def __init__(self, base: SpotRenderer, seed: int = 0):
        self.base = b = base
        self.device = dev = b.xs.device
        self.res = res = b.res
        self.n_spots = (b.hit_intensity.shape[0] + 1) // 2  # M2 hits: even bounce indices
        self.px = (2.0 * torch.arange(res, device=dev, dtype=torch.float32) + 1.0) / res - 1.0
        gv, gu = torch.meshgrid(self.px, self.px, indexing="ij")
        self.cam_uv = torch.stack((gu, gv), -1)  # (H, W, 2), camera normalized
        self.text = engraving_bank(N_TEXT, TEXT_RES, seed).to(dev)

    # ── Parameters ──

    def neutral_params(self, n: int) -> CameraParams:
        dev, b = self.device, self.base
        z, o = torch.zeros(n, device=dev), torch.ones(n, device=dev)
        I = torch.eye(3, device=dev).expand(n, 3, 3).clone()
        return {
            "warp": I, "warp_inv": I.clone(), "scale": o.clone(),
            "laser": o.clone(), "sigma": o * b.sigma, "aspect": o.clone(), "angle": z.clone(),
            "speckle": z.clone(),
            "scatter_amp": z.clone(), "scatter_size": o * 0.3, "scatter_pow": o * 2.0,
            "mono": o.clone(), "laser_rgb": torch.ones(n, 3, device=dev),
            "ambient_rgb": torch.ones(n, 3, device=dev), "wb": torch.ones(n, 3, device=dev),
            "face": z.clone(), "ring_r": torch.full((n, N_RING), 0.8, device=dev),
            "ring_w": torch.full((n, N_RING), 0.05, device=dev),
            "ring_a": torch.zeros(n, N_RING, device=dev), "grain": z.clone(),
            "grain_tex": torch.zeros(n, 1, TEX_RES, TEX_RES, device=dev),
            "rim": z.clone(), "rim_w": o * 0.05, "rim_harm": torch.zeros(n, N_HARM, 2, device=dev),
            "surround": z.clone(),
            "engrave_idx": torch.zeros(n, dtype=torch.long, device=dev), "engrave_amp": z.clone(),
            "scratch_c": torch.zeros(n, N_SCRATCH, 2, device=dev),
            "scratch_dir": torch.zeros(n, N_SCRATCH, device=dev),
            "scratch_len": torch.full((n, N_SCRATCH), 0.1, device=dev),
            "scratch_w": torch.full((n, N_SCRATCH), 0.01, device=dev),
            "scratch_amp": torch.zeros(n, N_SCRATCH, device=dev),
            "dust_c": torch.zeros(n, N_DUST, 2, device=dev),
            "dust_s": torch.full((n, N_DUST), 0.02, device=dev),
            "dust_amp": torch.zeros(n, N_DUST, device=dev),
            "gradient": torch.zeros(n, 2, device=dev), "vignetting": z.clone(),
            "exposure": o.clone(), "gamma": o.clone(), "black": z.clone(),
            "read_noise": z.clone(), "shot_noise": z.clone(), "prnu": z.clone(),
            "prnu_map": torch.zeros(n, 1, self.res, self.res, device=dev),
            "row_noise": z.clone(),
            "hot_idx": torch.zeros(n, N_HOT, dtype=torch.long, device=dev),
            "hot_val": torch.zeros(n, N_HOT, device=dev),
        }

    def sample_params(self, n: int, r: CameraRanges, gen: torch.Generator) -> CameraParams:
        dev = self.device
        U = lambda lo_hi, *shape: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(
            n, *shape, generator=gen, device=dev)
        rand = lambda *shape: torch.rand(n, *shape, generator=gen, device=dev)
        sign = lambda *shape: torch.randint(0, 2, (n, *shape), generator=gen, device=dev) * 2.0 - 1.0
        upto = lambda m: torch.arange(m, device=dev)[None] < torch.randint(
            0, m + 1, (n, 1), generator=gen, device=dev)  # (n, m): first k active, k uniform 0..m

        def disk(*shape):  # Uniform points in the unit disk, (n, *shape, 2)
            rr, a = torch.sqrt(rand(*shape)), 2 * math.pi * rand(*shape)
            return torch.stack((rr * torch.cos(a), rr * torch.sin(a)), -1)

        rot = _signed(n, r.rotation, gen, dev)
        shift = torch.stack((_signed(n, r.shift, gen, dev), _signed(n, r.shift, gen, dev)), 1)
        scale = _log_uniform(n, r.scale, gen, dev)
        corner = 2.0 * r.corner * (2.0 * rand(4, 2) - 1.0)
        warp_inv = pose_warp(rot, shift, scale, corner)

        mono = (rand() >= r.color_p).float()
        color = torch.randint(0, 3, (n,), generator=gen, device=dev)
        laser_rgb = torch.where(F.one_hot(color, 3).bool(), 1.0, U(r.crosstalk, 3))
        qe = U(r.mono_qe)[:, None].expand(n, 3)
        tint = torch.exp(r.tint * torch.randn(n, 3, generator=gen, device=dev))
        tint = tint / tint.prod(1, keepdim=True) ** (1 / 3)
        m = mono[:, None] > 0.5
        lit = rand() < r.face_p
        hot_on = upto(N_HOT) & (rand(1) < 0.5)
        return {
            "warp": torch.linalg.inv(warp_inv.double()).float(), "warp_inv": warp_inv,
            "scale": scale,
            "laser": _log_uniform(n, r.laser, gen, dev), "sigma": U(r.sigma), "aspect": U(r.aspect),
            "angle": math.pi * rand(), "speckle": U(r.speckle),
            "scatter_amp": _log_uniform(n, r.scatter_amp, gen, dev),
            "scatter_size": U(r.scatter_size), "scatter_pow": U(r.scatter_pow),
            "mono": mono, "laser_rgb": torch.where(m, qe, laser_rgb),
            "ambient_rgb": torch.where(m, 1.0, tint),
            "wb": torch.where(m, 1.0, torch.exp(U((math.log(r.wb[0]), math.log(r.wb[1])), 3))),
            "face": torch.where(lit, U(r.face), r.dark_face * rand()),
            "ring_r": U(r.ring_r, N_RING), "ring_w": U(r.ring_w, N_RING),
            "ring_a": U(r.ring_a, N_RING), "grain": U(r.grain),
            "grain_tex": fractal_noise(n, TEX_RES, gen, dev),
            "rim": U(r.rim), "rim_w": U(r.rim_w),
            "rim_harm": torch.stack((rand(N_HARM) * 0.8 / torch.arange(1, N_HARM + 1, device=dev),
                                     2 * math.pi * rand(N_HARM)), -1),
            "surround": U(r.surround),
            "engrave_idx": torch.randint(0, N_TEXT, (n,), generator=gen, device=dev),
            "engrave_amp": torch.where(rand() < r.engrave_p, U(r.engrave_amp), 0.0),
            "scratch_c": 0.95 * disk(N_SCRATCH), "scratch_dir": math.pi * rand(N_SCRATCH),
            "scratch_len": U(r.scratch_len, N_SCRATCH), "scratch_w": U((0.006, 0.015), N_SCRATCH),
            "scratch_amp": sign(N_SCRATCH) * U(r.scratch_amp, N_SCRATCH) * upto(N_SCRATCH),
            "dust_c": 0.97 * disk(N_DUST), "dust_s": U(r.dust_s, N_DUST),
            "dust_amp": torch.where(rand(N_DUST) < 0.75, 1.0, -1.0) * U(r.dust_amp, N_DUST)
                        * upto(N_DUST),
            "gradient": U(r.gradient)[:, None] * F.normalize(torch.randn(
                n, 2, generator=gen, device=dev), dim=1),
            "vignetting": U(r.vignetting),
            "exposure": _log_uniform(n, r.exposure, gen, dev), "gamma": U(r.gamma),
            "black": U(r.black), "read_noise": U(r.read_noise), "shot_noise": U(r.shot_noise),
            "prnu": U(r.prnu),
            "prnu_map": torch.randn(n, 1, self.res, self.res, generator=gen, device=dev),
            "row_noise": U(r.row_noise),
            "hot_idx": torch.randint(0, self.res**2, (n, N_HOT), generator=gen, device=dev),
            "hot_val": U((0.4, 1.0), N_HOT) * hot_on,
        }

    # ── Geometry ──

    def spots_camera(self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor,
                     p: CameraParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """M2 spots: camera normalized coordinates (n, K, 2), peak (n, K), valid (n, K)."""
        pos, w, valid = m2_spots(hit_mirror, hit_local_xy, self.base.hit_intensity, self.n_spots)
        return _apply(p["warp"], _canonical(pos, self.base.radius)), w, valid

    def scatter_sources(self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor,
                        miss: Optional[Miss], p: CameraParams) -> Tuple[torch.Tensor, torch.Tensor]:
        """The N_SCATTER strongest edge-scatter sources: rim points in canonical
        normalized coordinates (n, S, 2), weight (n, S) = beam intensity x share of the
        beam on the rim bevel (M2 hits and the M2 miss point)."""
        R = self.base.radius
        pos, w, valid = m2_spots(hit_mirror, hit_local_xy, self.base.hit_intensity, self.n_spots)
        w = w * valid
        if miss is not None:
            miss_mirror, miss_lxy, intensity = miss
            pos = torch.cat((pos, miss_lxy[:, None]), 1)
            w = torch.cat((w, torch.where(miss_mirror == M2_ID, intensity, 0.0)[:, None]), 1)
        d = pos.norm(dim=-1) - R
        s = math.sqrt(2) * p["sigma"][:, None]
        share = 0.5 * (torch.erf((EDGE_BEVEL_MM - d) / s) + torch.erf((EDGE_BEVEL_MM + d) / s))
        weight, idx = torch.topk(w * share, min(N_SCATTER, w.shape[1]), dim=1)
        rim = pos.gather(1, idx[..., None].expand(-1, -1, 2))
        rim = rim / rim.norm(dim=-1, keepdim=True).clamp_min(1e-6) * R  # Onto the rim
        return _canonical(rim, R), weight

    def labels(self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor,
               p: CameraParams) -> Dict[str, torch.Tensor]:
        """Camera-frame targets: spot centres in pixels (n, K, 2) as (col, row), peak
        reflectivity^k (n, K), visible (n, K) (an M2 hit inside the field), count (n,)."""
        uv, w, valid = self.spots_camera(hit_mirror, hit_local_xy, p)
        px = (uv + 1.0) * self.res / 2 - 0.5
        vis = valid & (uv.abs() <= 1.0).all(-1)
        return {"pos_px": px, "peak": w, "visible": vis, "count": vis.sum(1)}

    # ── Layers ──

    def _spot_layer(self, uv: torch.Tensor, w: torch.Tensor, p: CameraParams) -> torch.Tensor:
        """(n, H, W): elliptical spots at camera positions uv (n, K, 2), peaks w (n, K)."""
        v = lambda t: t[:, None, None, None]
        k = p["scale"] / self.base.radius  # mm -> camera normalized
        su2 = (p["sigma"] * k) ** 2 * p["aspect"]
        sv2 = (p["sigma"] * k) ** 2 / p["aspect"]
        c, s = torch.cos(p["angle"]), torch.sin(p["angle"])
        A, B, C = c * c / su2 + s * s / sv2, c * s * (1 / su2 - 1 / sv2), s * s / su2 + c * c / sv2
        dx = self.px[None, None, None, :] - uv[:, :, 0, None, None]
        dy = self.px[None, None, :, None] - uv[:, :, 1, None, None]
        val = torch.exp(-0.5 * (v(A) * dx * dx + 2 * v(B) * dx * dy + v(C) * dy * dy))
        return (w[:, :, None, None] * val).sum(1)

    def _scatter_layer(self, src: torch.Tensor, weight: torch.Tensor,
                       p: CameraParams) -> torch.Tensor:
        """(n, H, W): flat-top blobs at the canonical rim points src (n, S, 2)."""
        v = lambda t: t[:, None, None, None]
        q = _apply(p["warp"], src)
        d = (self.cam_uv[None, None] - q[:, :, None, None]).norm(dim=-1)  # (n, S, H, W)
        blob = torch.exp(-(d / v(p["scatter_size"] * p["scale"])) ** v(p["scatter_pow"]))
        return v(p["scatter_amp"])[:, 0] * (weight[:, :, None, None] * blob).sum(1)

    def _face_layer(self, XY: torch.Tensor, p: CameraParams) -> torch.Tensor:
        """(n, H, W) mirror-face intensity at canonical points XY (n, H, W, 2)."""
        v = lambda t: t[:, None, None]
        X, Y = XY[..., 0], XY[..., 1]
        r = torch.sqrt(X * X + Y * Y)
        face = p["face"]
        rings = (p["ring_a"][:, :, None, None] * torch.exp(
            -(r[:, None] - p["ring_r"][:, :, None, None]) ** 2
            / (2 * p["ring_w"][:, :, None, None] ** 2))).sum(1)
        grain = F.grid_sample(p["grain_tex"], XY, mode="bilinear", padding_mode="reflection",
                              align_corners=False)[:, 0]
        text = F.grid_sample(self.text[p["engrave_idx"]], XY, mode="bilinear",
                             padding_mode="zeros", align_corners=False)[:, 0]
        out = v(face) * (1.0 + rings + v(p["grain"]) * grain + v(p["engrave_amp"]) * text)

        # Scratches: distance to segments
        P = XY[:, None]  # (n, 1, H, W, 2)
        cdir = torch.stack((torch.cos(p["scratch_dir"]), torch.sin(p["scratch_dir"])), -1)
        rel = P - p["scratch_c"][:, :, None, None]
        t = (rel * cdir[:, :, None, None]).sum(-1).clamp(
            -p["scratch_len"][:, :, None, None], p["scratch_len"][:, :, None, None])
        d2 = ((rel - t[..., None] * cdir[:, :, None, None]) ** 2).sum(-1)
        base = face.clamp_min(0.1)
        out = out + v(base) * (p["scratch_amp"][:, :, None, None] * torch.exp(
            -d2 / (2 * p["scratch_w"][:, :, None, None] ** 2))).sum(1)

        # Dust specks
        d2 = ((P - p["dust_c"][:, :, None, None]) ** 2).sum(-1)
        out = out + v(face.clamp_min(0.05)) * (p["dust_amp"][:, :, None, None] * torch.exp(
            -d2 / (2 * p["dust_s"][:, :, None, None] ** 2))).sum(1)

        # Edge glow, modulated along the rim
        th = torch.atan2(Y, X)
        h = torch.arange(1, N_HARM + 1, device=XY.device, dtype=torch.float32)
        mod = (1.0 + (p["rim_harm"][:, :, 0, None, None] * torch.cos(
            h[None, :, None, None] * th[:, None] + p["rim_harm"][:, :, 1, None, None])).sum(1)).clamp_min(0)
        rw = p["rim_w"]
        out = out + v(p["rim"]) * mod * torch.exp(-(r - 1.0 + v(rw)) ** 2 / (2 * v(rw) ** 2))
        return out.clamp_min(0.0)

    # ── Entry point ──

    def render(self, hit_mirror: torch.Tensor, hit_local_xy: torch.Tensor, p: CameraParams,
               gen: torch.Generator, miss: Optional[Miss] = None) -> torch.Tensor:
        """(n, 3, H, W) uint8."""
        n, res, dev = hit_mirror.shape[0], self.res, self.device
        v3 = lambda t: t[:, :, None, None]  # (n, 3) -> (n, 3, 1, 1)
        v1 = lambda t: t[:, None, None, None]  # (n,) -> (n, 1, 1, 1)

        # Canonical point under each camera pixel, mirror mask (anti-aliased edge)
        XY = _apply(p["warp_inv"], self.cam_uv.expand(n, res, res, 2))
        r = XY.norm(dim=-1)
        px_c = 2.0 / res / p["scale"]
        mask = ((1.0 - r) / px_c[:, None, None] + 0.5).clamp(0.0, 1.0)

        g0 = self.base.gain * p["exposure"]
        p = {**p, "face": -torch.log1p(-p["face"].clamp(max=0.99)) / g0}  # Brightness -> I
        uv, w, _ = self.spots_camera(hit_mirror, hit_local_xy, p)
        w = w * torch.exp(p["speckle"][:, None] * torch.randn(w.shape, generator=gen, device=dev))
        src, weight = self.scatter_sources(hit_mirror, hit_local_xy, miss, p)
        laser = v1(p["laser"])[:, 0] * mask * (
            self._spot_layer(uv, w, p) + self._scatter_layer(src, weight, p))

        uvc = self.cam_uv[None]
        illum = (1.0 + (uvc * p["gradient"][:, None, None]).sum(-1)) \
            * (1.0 - p["vignetting"][:, None, None] * (uvc**2).sum(-1) / 2)
        amb = illum.clamp_min(0.0) * (mask * self._face_layer(XY, p)
                                      + (1.0 - mask) * p["surround"][:, None, None])
        I = v3(p["laser_rgb"]) * laser[:, None] + v3(p["ambient_rgb"]) * amb[:, None]  # (n, 3, H, W)

        # Sensor
        g = v1(g0) * v3(p["wb"])
        resp = (-torch.expm1(-g * I)).clamp_min(0.0) ** v1(p["gamma"])
        resp = resp * (1.0 + v1(p["prnu"]) * p["prnu_map"])
        noise = torch.randn(n, 3, res, res, generator=gen, device=dev)
        rows = torch.randn(n, 3, res, 1, generator=gen, device=dev)
        mono = p["mono"] > 0.5
        noise = torch.where(v1(mono), noise[:, :1].expand(-1, 3, -1, -1), noise)
        rows = torch.where(v1(mono), rows[:, :1].expand(-1, 3, -1, -1), rows)
        std = torch.sqrt(v1(p["read_noise"]) ** 2 + v1(p["shot_noise"]) * resp)
        out = resp + v1(p["black"]) + std * noise + v1(p["row_noise"]) * rows
        out = torch.where(v1(mono), out[:, :1].expand(-1, 3, -1, -1), out)
        out = out.reshape(n, 3, res * res).scatter_reduce(
            2, p["hot_idx"][:, None].expand(-1, 3, -1), p["hot_val"][:, None].expand(-1, 3, -1),
            reduce="amax").view(n, 3, res, res)
        return (out.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

    __call__ = render
