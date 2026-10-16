"""
targets.py - Single-frame targets and probe labels from the sim output (PLAN phase 5,
Level 1)

All targets live in the canonical M2 frame (the SpotRenderer pixel grid, M2-local mm,
+lx right, +ly up) and do not see the camera nuisances of render_random.py.

Training targets:
    spot_map   (n, 1, H, W) float32: Gaussian of map_sigma_px (1 px) at each M2 spot
               centre, peak = incident intensity reflectivity^k, overlapping peaks summed
    count      (n,) int64: number of M2 spots, 0-64 (classification)
    ellipse    (n, 5) float32: conic fit to the M2 spots, as the vector
               (cx, cy, mxx, myy, mxy) mm, where M = R diag(a, b) R^T is the shape
               tensor (continuous at a = b, no angle wrap); vec_to_ellipse() gives
               (cx, cy, a, b, phi); ellipse_mask (n,) bool: >= 5 spots and a valid fit

Probe labels (never training targets): bounces, path_mm, stop_reason, region
(REGIONS index); the true M2 tilt is added by the data module (frame_data.py).

Ellipse fit: direct least squares (Fitzgibbon 1999) in the numerically stable form of
Halir and Flusser (1998), on centred and scaled points, float64. The 3 x 3 eigenproblem
runs on the CPU (one host sync per batch; batched cusolver eigen calls are avoided).
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from herriott_sim import STOP_CAP

M2_ID = 2  # hit_mirror value of an M2 hit (env.M2_ID)
REGIONS = ("blind", "31", "57", "89", "trapped", "other")
BLIND_MAX_BOUNCES = 2  # 1-2 bounces: one M2 spot at a fixed place


# ── Spots ───────────────────────────────────────────────────────────────────


def m2_spots(
    hit_mirror: torch.Tensor,
    hit_local_xy: torch.Tensor,
    hit_intensity: torch.Tensor,
    k_max: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First k_max M2 hits in bounce order: positions (n, K, 2) mm, peak intensity
    (n, K) (reflectivity^k, 0 for padding), valid (n, K) bool. No host sync."""
    n, N = hit_mirror.shape
    is_m2 = hit_mirror == M2_ID
    key = torch.where(is_m2, torch.arange(N, 0, -1, device=hit_mirror.device).float(), 0.0)
    K = min(k_max, N)
    _, idx = torch.topk(key, K, dim=1)  # M2 hits first, lowest bounce index first
    valid = is_m2.gather(1, idx)
    w = torch.where(is_m2, hit_intensity[:N], 0.0).gather(1, idx)
    pos = hit_local_xy.gather(1, idx[..., None].expand(-1, -1, 2))
    return torch.where(valid[..., None], pos, 0.0), w, valid


def region_labels(bounces: torch.Tensor, stop_reason: torch.Tensor) -> torch.Tensor:
    """(n,) int64 index into REGIONS."""
    r = torch.full_like(bounces, REGIONS.index("other"))
    r = torch.where(bounces <= BLIND_MAX_BOUNCES, REGIONS.index("blind"), r)
    for name in ("31", "57", "89"):
        r = torch.where(bounces == int(name), REGIONS.index(name), r)
    return torch.where(stop_reason == STOP_CAP, REGIONS.index("trapped"), r)


# ── Ellipse ─────────────────────────────────────────────────────────────────


def ellipse_to_vec(e: torch.Tensor) -> torch.Tensor:
    """(..., 5) (cx, cy, a, b, phi) -> (cx, cy, mxx, myy, mxy)."""
    cx, cy, a, b, phi = e.unbind(-1)
    c, s = torch.cos(phi), torch.sin(phi)
    return torch.stack((cx, cy, a * c * c + b * s * s, a * s * s + b * c * c, (a - b) * c * s), -1)


def vec_to_ellipse(v: torch.Tensor) -> torch.Tensor:
    """(..., 5) (cx, cy, mxx, myy, mxy) -> (cx, cy, a, b, phi), a >= b, phi in [0, pi)."""
    cx, cy, mxx, myy, mxy = v.unbind(-1)
    half_tr = 0.5 * (mxx + myy)
    d = torch.sqrt((0.5 * (mxx - myy)) ** 2 + mxy**2)
    phi = torch.remainder(0.5 * torch.atan2(2.0 * mxy, mxx - myy), math.pi)
    return torch.stack((cx, cy, half_tr + d, half_tr - d, phi), -1)


def fit_ellipse(
    pos: torch.Tensor, valid: torch.Tensor, min_spots: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct least-squares ellipse per sample.

    pos (n, K, 2) mm, valid (n, K). Returns (cx, cy, a, b, phi) (n, 5) float32 on the
    input device (phi: major axis from +lx, [0, pi)), the fit-ok mask (n,) (>= min_spots
    points and an ellipse came out), and the RMS Sampson distance (n,) mm.
    """
    dev = pos.device
    P = pos.detach().double().cpu()
    m = valid.detach().cpu().double()
    n = P.shape[0]
    cnt = m.sum(1)
    enough = cnt >= min_spots
    c = (P * m[..., None]).sum(1) / cnt.clamp_min(1)[:, None]
    d = (P - c[:, None]) * m[..., None]
    s = torch.sqrt((d**2).sum((1, 2)) / cnt.clamp_min(1)).clamp_min(1e-6)
    x, y = d[..., 0] / s[:, None], d[..., 1] / s[:, None]
    D1 = torch.stack((x * x, x * y, y * y), -1) * m[..., None]
    D2 = torch.stack((x, y, torch.ones_like(x)), -1) * m[..., None]
    S1 = D1.transpose(1, 2) @ D1
    S2 = D1.transpose(1, 2) @ D2
    S3 = D2.transpose(1, 2) @ D2
    eye = torch.eye(3, dtype=torch.float64)
    S3 = torch.where(enough[:, None, None], S3, eye) + 1e-12 * eye
    T = -torch.linalg.solve(S3, S2.transpose(1, 2))
    M = S1 + S2 @ T
    C1_inv = torch.tensor([[0.0, 0.0, 0.5], [0.0, -1.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64)
    M = torch.where(enough[:, None, None], C1_inv @ M, eye)
    _, V = torch.linalg.eig(M)
    V = V.real
    cond = 4.0 * V[:, 0] * V[:, 2] - V[:, 1] ** 2  # (n, 3), > 0: ellipse
    j = cond.argmax(1)
    a1 = V.gather(2, j[:, None, None].expand(n, 3, 1))[..., 0]
    a2 = (T @ a1[..., None])[..., 0]
    A, B, C = a1.unbind(-1)
    Dl, E, F = a2.unbind(-1)

    # Centre, then semi-axes from the centred quadratic form u^T Q u = kappa
    den = B * B - 4.0 * A * C
    x0 = (2.0 * C * Dl - B * E) / den
    y0 = (2.0 * A * E - B * Dl) / den
    kappa = -(A * x0 * x0 + B * x0 * y0 + C * y0 * y0 + Dl * x0 + E * y0 + F)
    sign = torch.sign(kappa)
    qa, qb, qc, kappa = A * sign, 0.5 * B * sign, C * sign, kappa * sign
    half_tr = 0.5 * (qa + qc)
    root = torch.sqrt((0.5 * (qa - qc)) ** 2 + qb**2)
    lam_lo, lam_hi = half_tr - root, half_tr + root
    semi_a = torch.sqrt(kappa / lam_lo) * s
    semi_b = torch.sqrt(kappa / lam_hi) * s
    phi = torch.remainder(0.5 * torch.atan2(2.0 * qb, qa - qc) + 0.5 * math.pi, math.pi)

    # Sampson distance of the points (normalized units -> mm)
    f = (
        A[:, None] * x * x + B[:, None] * x * y + C[:, None] * y * y
        + Dl[:, None] * x + E[:, None] * y + F[:, None]
    )
    gx = 2.0 * A[:, None] * x + B[:, None] * y + Dl[:, None]
    gy = B[:, None] * x + 2.0 * C[:, None] * y + E[:, None]
    samp = f.abs() / torch.sqrt(gx * gx + gy * gy).clamp_min(1e-12) * s[:, None]
    rms = torch.sqrt((samp**2 * m).sum(1) / cnt.clamp_min(1))

    out = torch.stack((x0 * s + c[:, 0], y0 * s + c[:, 1], semi_a, semi_b, phi), -1)
    ok = enough & (cond.amax(1) > 0) & (den < 0) & (lam_lo > 0) & torch.isfinite(out).all(1)
    out = torch.where(ok[:, None], out, 0.0)
    rms = torch.where(ok, rms, float("inf"))
    return out.float().to(dev), ok.to(dev), rms.float().to(dev)


# ── Builder ─────────────────────────────────────────────────────────────────


@dataclass
class TargetConfig:
    """TargetBuilder settings."""

    map_sigma_px: float = 1.0  # Spot-map peak width (pixels)
    min_ellipse_spots: int = 5
    max_fit_rms: float = 0.25  # mm, Sampson RMS above which the ellipse is masked
    max_aperture_ratio: float = 2.0  # Semi-axis / aperture radius above which masked


class TargetBuilder:
    """Targets and probe labels of a sim output on a renderer's pixel grid.

    xs, ys: column / row centres (mm) of the image (SpotRenderer.xs / .ys);
    hit_intensity: (N,) reflectivity^k (SpotRenderer.hit_intensity).
    """

    def __init__(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        radius: float,
        hit_intensity: torch.Tensor,
        cfg: TargetConfig = TargetConfig(),
    ):
        self.cfg = cfg
        self.xs, self.ys = xs, ys
        self.radius = radius
        self.pitch = float(xs[1] - xs[0])
        self.hit_intensity = hit_intensity
        self.n_spots = (hit_intensity.shape[0] + 1) // 2  # M2 hits: even bounce indices

    def spot_map(self, pos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(n, 1, H, W) sum of Gaussians (sigma map_sigma_px) with peaks w (n, K)."""
        s2 = 2.0 * (self.cfg.map_sigma_px * self.pitch) ** 2
        gx = torch.exp(-((self.xs[None, None, :] - pos[..., 0, None]) ** 2) / s2)
        gy = torch.exp(-((self.ys[None, None, :] - pos[..., 1, None]) ** 2) / s2)
        return torch.einsum("nk,nki,nkj->nij", w, gy, gx).unsqueeze(1)

    def __call__(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Targets and labels of a sim output dict (herriott_sim.simulate)."""
        c = self.cfg
        pos, w, valid = m2_spots(
            out["hit_mirror"], out["hit_local_xy"], self.hit_intensity, self.n_spots
        )
        count = valid.sum(1)
        ell, ok, rms = fit_ellipse(pos, valid, c.min_ellipse_spots)
        ok = ok & (rms <= c.max_fit_rms) & (ell[:, 2] <= c.max_aperture_ratio * self.radius)
        bounces = out["hit_counts"]
        stop = out["stop_reason"]
        return {
            "spot_map": self.spot_map(pos, w),
            "count": count,
            "ellipse": torch.where(ok[:, None], ellipse_to_vec(ell), 0.0),
            "ellipse_mask": ok,
            "ellipse_rms": rms,
            "spots": pos,
            "spot_w": w,
            "spot_valid": valid,
            "bounces": bounces,
            "path_mm": out["total_path_length"],
            "stop_reason": stop,
            "region": region_labels(bounces, stop),
        }
