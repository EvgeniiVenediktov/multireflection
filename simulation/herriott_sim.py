"""
herriott_sim.py - High-performance Herriott cell simulation for RL
All tensor ops, no Python loops in hot path, fully differentiable

State vector (13 DOF):
    [m1_pitch, m1_yaw, m2_pitch, m2_yaw, separation,
     m1_tx, m1_ty, m2_tx, m2_ty,
     laser_dx, laser_dy, laser_pitch, laser_yaw]

    0  m1_pitch    - Mirror 1 pitch (deg)
    1  m1_yaw      - Mirror 1 yaw (deg)
    2  m2_pitch    - Mirror 2 pitch (deg)
    3  m2_yaw      - Mirror 2 yaw (deg)
    4  separation  - Mirror separation along Z (mm)
    5  m1_tx       - Mirror 1 transverse X offset (mm)
    6  m1_ty       - Mirror 1 transverse Y offset (mm)
    7  m2_tx       - Mirror 2 transverse X offset (mm)
    8  m2_ty       - Mirror 2 transverse Y offset (mm)
    9  laser_dx    - Laser X offset from hole center (mm)
    10 laser_dy    - Laser Y offset from hole center (mm)
    11 laser_pitch - Laser pitch relative to M1 normal (deg)
    12 laser_yaw   - Laser yaw relative to M1 normal (deg)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# State indices
S_M1_PITCH = 0
S_M1_YAW = 1
S_M2_PITCH = 2
S_M2_YAW = 3
S_SEP = 4
S_M1_TX = 5
S_M1_TY = 6
S_M2_TX = 7
S_M2_TY = 8
S_LASER_DX = 9
S_LASER_DY = 10
S_LASER_PITCH = 11
S_LASER_YAW = 12
STATE_DIM = 13


@dataclass
class MirrorConfig:
    """Static mirror configuration."""

    roc: float = 150.0  # Radius of curvature (mm)
    # roc: float = 200.0  # Radius of curvature (mm)
    diameter: float = (
        24.4  # Mirror diameter (mm), FIXME: for sim2real 24.4 instead of usual 25.4
        # 25.4
    )
    hole_radius: float = (
        1.8  # Hole radius (mm), FIXME: for sim2real 1.8 instead of actual 1.5
        # 1.5
    )
    hole_offset_y: float = 7.0  # Hole Y offset (mm).
    reflectivity: float = 0.98


@dataclass
class SimConfig:
    """Simulation parameters."""

    max_bounces: int = 100  # FIXME: for simulation
    # max_bounces: int = 50  # FIXME: for training
    intensity_threshold: float = 1e-4
    device: torch.device = DEVICE
    dtype: torch.dtype = DTYPE


class HerriottSim:
    """
    High-performance Herriott cell simulator with 13 DOF.

    Laser is always mounted on M1, entering through the hole.
    Laser position/angle DOFs represent error/offset from ideal.
    """

    def __init__(
        self,
        m1_cfg: MirrorConfig = None,
        m2_cfg: MirrorConfig = None,
        sim_cfg: SimConfig = None,
    ):
        self.m1_cfg = m1_cfg or MirrorConfig(hole_radius=1.5)
        self.m2_cfg = m2_cfg or MirrorConfig(hole_radius=0)
        self.sim_cfg = sim_cfg or SimConfig()
        self.device = self.sim_cfg.device
        self.dtype = self.sim_cfg.dtype

        self._precompute()

    def _precompute(self):
        """Precompute static values."""
        self.m1_aperture = self.m1_cfg.diameter / 2
        self.m2_aperture = self.m2_cfg.diameter / 2
        self.m1_hole_offset = torch.tensor(
            [0, self.m1_cfg.hole_offset_y], device=self.device, dtype=self.dtype
        )
        self.m1_sag = (
            self.m1_cfg.roc - (self.m1_cfg.roc**2 - self.m1_aperture**2) ** 0.5
        )
        self.m2_sag = (
            self.m2_cfg.roc - (self.m2_cfg.roc**2 - self.m2_aperture**2) ** 0.5
        )

    @staticmethod
    def rotation_matrix(pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Batch rotation matrix from pitch/yaw (degrees). Shape: (B, 3, 3)"""
        px = torch.deg2rad(pitch)
        py = torch.deg2rad(yaw)

        cx, sx = torch.cos(px), torch.sin(px)
        cy, sy = torch.cos(py), torch.sin(py)

        zero = torch.zeros_like(px)
        one = torch.ones_like(px)

        # Ry @ Rx
        r00, r01, r02 = cy, sx * sy, cx * sy
        r10, r11, r12 = zero, cx, -sx
        r20, r21, r22 = -sy, sx * cy, cx * cy

        R = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=-2,
        )
        return R

    def _get_mirror_geometry(
        self,
        pitch: torch.Tensor,
        yaw: torch.Tensor,
        tx: torch.Tensor,
        ty: torch.Tensor,
        base_z: torch.Tensor,
        facing: float,
        roc: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mirror geometry from state, including transverse offsets.

        Returns: (center, normal, rotation, position) - all batched
        """
        B = pitch.shape[0]
        R = self.rotation_matrix(pitch, yaw)  # (B, 3, 3)

        # Base normal
        base_normal = torch.tensor([0, 0, facing], device=self.device, dtype=self.dtype)
        normal = torch.einsum("bij,j->bi", R, base_normal)  # (B, 3)

        # Mirror position: base_z along Z, plus transverse offsets in XY
        position = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        position[:, 0] = tx
        position[:, 1] = ty
        position[:, 2] = base_z

        # Sphere center
        center = position + normal * roc  # (B, 3)

        return center, normal, R, position

    def _ray_sphere_intersect(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        centers: torch.Tensor,
        radius: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched ray-sphere intersection — returns BOTH roots.
        Returns: t_near (B,), valid_near (B,), t_far (B,), valid_far (B,)
        """
        oc = origins - centers
        a = (directions * directions).sum(dim=-1)
        b = 2.0 * (oc * directions).sum(dim=-1)
        c = (oc * oc).sum(dim=-1) - radius**2

        disc = b * b - 4 * a * c
        valid = disc > 0

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=1e-10))
        t1 = (-b - sqrt_disc) / (2 * a + 1e-10)
        t2 = (-b + sqrt_disc) / (2 * a + 1e-10)

        inf = torch.tensor(float("inf"), device=self.device)
        v1 = (t1 > 1e-5) & valid
        v2 = (t2 > 1e-5) & valid
        t_near = torch.where(v1, t1, inf)
        t_far = torch.where(v2, t2, inf)

        return t_near, v1, t_far, v2

    def _check_aperture(
        self,
        hit: torch.Tensor,
        position: torch.Tensor,
        R: torch.Tensor,
        normal: torch.Tensor,
        aperture: float,
        sag: float,
        hole_radius: float = 0,
        hole_offset: torch.Tensor = None,
    ) -> torch.Tensor:
        """Check if hit is within mirror aperture and not in hole."""
        rel = hit - position

        local_x_axis = R[:, :, 0]  # (B, 3)
        local_y_axis = R[:, :, 1]  # (B, 3)

        lx = (rel * local_x_axis).sum(dim=-1)
        ly = (rel * local_y_axis).sum(dim=-1)
        r = torch.sqrt(lx**2 + ly**2)

        dot_normal = (rel * normal).sum(dim=-1)
        valid = (r < aperture) & (dot_normal >= 0) & (dot_normal < sag + 1)

        if hole_radius > 0 and hole_offset is not None:
            hole_dist = torch.sqrt(
                (lx - hole_offset[0]) ** 2 + (ly - hole_offset[1]) ** 2
            )
            valid = valid & (hole_dist >= hole_radius)

        return valid

    def simulate(self, state: torch.Tensor) -> dict:
        """
        Run simulation for given state(s).

        Args:
            state: (B, 13) state vector

        Returns:
            dict with:
                - hit_counts: (B,) number of bounces per config
                - final_positions: (B, 3) last hit position
                - total_path_length: (B,) total optical path
                - hit_sequence: (B, max_bounces, 3) all hit positions (padded)
                - intensity: (B,) remaining beam intensity
        """
        B = state.shape[0]
        assert (
            state.shape[1] == STATE_DIM
        ), f"Expected state dim {STATE_DIM}, got {state.shape[1]}"

        # Parse state
        m1_pitch = state[:, S_M1_PITCH]
        m1_yaw = state[:, S_M1_YAW]
        m2_pitch = state[:, S_M2_PITCH]
        m2_yaw = state[:, S_M2_YAW]
        separation = state[:, S_SEP]
        m1_tx = state[:, S_M1_TX]
        m1_ty = state[:, S_M1_TY]
        m2_tx = state[:, S_M2_TX]
        m2_ty = state[:, S_M2_TY]
        laser_dx = state[:, S_LASER_DX]
        laser_dy = state[:, S_LASER_DY]
        laser_pitch = state[:, S_LASER_PITCH]
        laser_yaw = state[:, S_LASER_YAW]

        # Mirror geometry (now with transverse offsets)
        m1_center, m1_normal, m1_R, m1_pos = self._get_mirror_geometry(
            m1_pitch,
            m1_yaw,
            m1_tx,
            m1_ty,
            torch.zeros(B, device=self.device),
            1.0,
            self.m1_cfg.roc,
        )
        m2_center, m2_normal, m2_R, m2_pos = self._get_mirror_geometry(
            m2_pitch,
            m2_yaw,
            m2_tx,
            m2_ty,
            separation,
            -1.0,
            self.m2_cfg.roc,
        )

        # --- Laser origin ---
        # Hole center in M1 local frame, then offset by laser_dx/dy
        hole_local = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        hole_local[:, 0] = laser_dx  # X error relative to hole center
        hole_local[:, 1] = self.m1_cfg.hole_offset_y + laser_dy  # Y error
        hole_local[:, 2] = -5.0  # Behind mirror surface

        # Transform to world frame (M1 rotation + M1 position)
        origins = torch.einsum("bij,bj->bi", m1_R, hole_local) + m1_pos

        # --- Laser direction ---
        # Laser has its own pitch/yaw *relative* to M1 normal
        # Compose: first rotate by laser_pitch/yaw, then by M1 rotation
        laser_R = self.rotation_matrix(laser_pitch, laser_yaw)  # (B, 3, 3)
        base_dir = torch.tensor([0, 0, 1.0], device=self.device, dtype=self.dtype)
        # Laser dir in M1 local frame
        laser_dir_local = torch.einsum("bij,j->bi", laser_R, base_dir)  # (B, 3)
        # Transform to world frame via M1 rotation
        directions = torch.einsum("bij,bj->bi", m1_R, laser_dir_local)  # (B, 3)
        directions = F.normalize(directions, dim=-1)

        # Tracing storage
        hit_sequence = torch.zeros(
            B, self.sim_cfg.max_bounces, 3, device=self.device, dtype=self.dtype
        )
        hit_counts = torch.zeros(B, dtype=torch.long, device=self.device)
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        intensity = torch.ones(B, device=self.device, dtype=self.dtype)
        path_length = torch.zeros(B, device=self.device, dtype=self.dtype)
        last_hit_mirror = torch.zeros(B, dtype=torch.long, device=self.device)

        for bounce in range(self.sim_cfg.max_bounces):
            if not active.any():
                break

            # Intersect both mirrors (get BOTH roots per sphere)
            t1_near, v1_near, t1_far, v1_far = self._ray_sphere_intersect(
                origins, directions, m1_center, self.m1_cfg.roc
            )
            t2_near, v2_near, t2_far, v2_far = self._ray_sphere_intersect(
                origins, directions, m2_center, self.m2_cfg.roc
            )

            # Compute hit points for all four candidates
            hit1_near = origins + t1_near.unsqueeze(-1) * directions
            hit1_far = origins + t1_far.unsqueeze(-1) * directions
            hit2_near = origins + t2_near.unsqueeze(-1) * directions
            hit2_far = origins + t2_far.unsqueeze(-1) * directions

            # Aperture check each candidate
            v1_near = v1_near & self._check_aperture(
                hit1_near,
                m1_pos,
                m1_R,
                m1_normal,
                self.m1_aperture,
                self.m1_sag,
                self.m1_cfg.hole_radius,
                self.m1_hole_offset,
            )
            v1_far = v1_far & self._check_aperture(
                hit1_far,
                m1_pos,
                m1_R,
                m1_normal,
                self.m1_aperture,
                self.m1_sag,
                self.m1_cfg.hole_radius,
                self.m1_hole_offset,
            )
            v2_near = v2_near & self._check_aperture(
                hit2_near,
                m2_pos,
                m2_R,
                m2_normal,
                self.m2_aperture,
                self.m2_sag,
            )
            v2_far = v2_far & self._check_aperture(
                hit2_far,
                m2_pos,
                m2_R,
                m2_normal,
                self.m2_aperture,
                self.m2_sag,
            )

            # For each mirror, pick the closer valid hit (prefer near, fallback far)
            inf = torch.tensor(float("inf"), device=self.device)
            t1_near = torch.where(v1_near, t1_near, inf)
            t1_far = torch.where(v1_far, t1_far, inf)
            t1 = torch.minimum(t1_near, t1_far)
            v1 = t1 < inf
            hit1 = torch.where((t1 == t1_near).unsqueeze(-1), hit1_near, hit1_far)

            t2_near = torch.where(v2_near, t2_near, inf)
            t2_far = torch.where(v2_far, t2_far, inf)
            t2 = torch.minimum(t2_near, t2_far)
            v2 = t2 < inf
            hit2 = torch.where((t2 == t2_near).unsqueeze(-1), hit2_near, hit2_far)

            # Don't hit the same mirror twice in a row
            v1 = v1 & (last_hit_mirror != 1)
            v2 = v2 & (last_hit_mirror != 2)

            # Pick closer valid hit across mirrors
            t1 = torch.where(v1, t1, inf)
            t2 = torch.where(v2, t2, inf)

            hit_m1 = (t1 <= t2) & v1
            hit_m2 = (t2 < t1) & v2
            any_hit = hit_m1 | hit_m2

            active = active & any_hit
            if not active.any():
                break

            t = torch.where(hit_m1, t1, t2)
            hit_points = torch.where(hit_m1.unsqueeze(-1), hit1, hit2)

            # Normals
            n1 = -(hit1 - m1_center) / self.m1_cfg.roc
            n2 = (hit2 - m2_center) / self.m2_cfg.roc
            normals = torch.where(hit_m1.unsqueeze(-1), n1, n2)

            # Store
            hit_sequence[:, bounce] = torch.where(
                active.unsqueeze(-1), hit_points, hit_sequence[:, bounce]
            )
            hit_counts = torch.where(active, hit_counts + 1, hit_counts)
            path_length = torch.where(active, path_length + t, path_length)
            last_hit_mirror = torch.where(
                active & hit_m1, torch.ones_like(last_hit_mirror), last_hit_mirror
            )
            last_hit_mirror = torch.where(
                active & hit_m2, torch.full_like(last_hit_mirror, 2), last_hit_mirror
            )

            # Reflect
            dot = (directions * normals).sum(dim=-1, keepdim=True)
            directions = F.normalize(directions - 2 * dot * normals, dim=-1)
            origins = hit_points + 1e-4 * directions

            # Attenuate
            refl = torch.where(
                hit_m1, self.m1_cfg.reflectivity, self.m2_cfg.reflectivity
            )
            intensity = intensity * refl
            active = active & (intensity > self.sim_cfg.intensity_threshold)

        # Final positions
        batch_idx = torch.arange(B, device=self.device)
        final_bounce_idx = torch.clamp(hit_counts - 1, min=0)
        final_positions = hit_sequence[batch_idx, final_bounce_idx]

        return {
            "hit_counts": hit_counts,
            "final_positions": final_positions,
            "total_path_length": path_length,
            "hit_sequence": hit_sequence,
            "intensity": intensity,
        }

    def reward(self, state: torch.Tensor, target_bounces: int = 20) -> torch.Tensor:
        """
        Example reward function for RL.
        Rewards configurations that achieve target bounce count.
        """
        result = self.simulate(state)

        bounce_diff = (result["hit_counts"].float() - target_bounces).abs()
        r = -bounce_diff

        # Bonus for path length (longer = better gas absorption)
        r = r + 0.01 * result["total_path_length"]

        return r


def make_state(
    B: int = 1,
    m1_pitch=0.0,
    m1_yaw=0.0,
    m2_pitch=0.0,
    m2_yaw=0.0,
    separation=200.0,
    m1_tx=0.0,
    m1_ty=0.0,
    m2_tx=0.0,
    m2_ty=0.0,
    laser_dx=0.0,
    laser_dy=0.0,
    laser_pitch=0.0,
    laser_yaw=0.0,
    device=DEVICE,
) -> torch.Tensor:
    """Convenience: build a state tensor from named params (broadcasts scalars)."""
    vals = [
        m1_pitch,
        m1_yaw,
        m2_pitch,
        m2_yaw,
        separation,
        m1_tx,
        m1_ty,
        m2_tx,
        m2_ty,
        laser_dx,
        laser_dy,
        laser_pitch,
        laser_yaw,
    ]
    state = torch.zeros(B, STATE_DIM, device=device, dtype=DTYPE)
    for i, v in enumerate(vals):
        state[:, i] = v
    return state


def create_sim(device: str = None) -> HerriottSim:
    """Create simulator with default configs."""
    dev = torch.device(device) if device else DEVICE
    return HerriottSim(
        m1_cfg=MirrorConfig(),
        m2_cfg=MirrorConfig(hole_radius=0),
        sim_cfg=SimConfig(device=dev),
    )


if __name__ == "__main__":
    sim = create_sim()

    # Single config - nominal alignment
    state = make_state(separation=200.0)
    result = sim.simulate(state)
    print(
        f"Nominal: {result['hit_counts'].item()} bounces, "
        f"path={result['total_path_length'].item():.1f} mm"
    )

    # With some misalignment
    state = make_state(
        m1_pitch=0.3,
        m2_pitch=-0.2,
        m1_tx=0.1,
        m2_ty=-0.05,
        laser_dx=0.05,
        laser_pitch=0.1,
        separation=200.0,
    )
    result = sim.simulate(state)
    print(
        f"Misaligned: {result['hit_counts'].item()} bounces, "
        f"path={result['total_path_length'].item():.1f} mm"
    )

    # Batch of 1000 random configs
    states = torch.randn(1000, STATE_DIM, device=DEVICE)
    states[:, S_SEP] = 150 + 100 * torch.rand(1000, device=DEVICE)
    states[:, S_M1_PITCH : S_M2_YAW + 1] *= 2  # angles ±2 deg
    states[:, S_M1_TX : S_M2_TY + 1] *= 0.5  # translations ±0.5 mm
    states[:, S_LASER_DX : S_LASER_DY + 1] *= 0.2  # laser pos error ±0.2 mm
    states[:, S_LASER_PITCH : S_LASER_YAW + 1] *= 1  # laser angle error ±1 deg

    import time

    t0 = time.perf_counter()
    for _ in range(100):
        result = sim.simulate(states)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t1 = time.perf_counter()

    print(f"\nBatch 1000 x 100 iters: {t1-t0:.3f}s ({100000/(t1-t0):.0f} sims/sec)")
    print(
        f"Bounces: min={result['hit_counts'].min().item()}, "
        f"max={result['hit_counts'].max().item()}, "
        f"mean={result['hit_counts'].float().mean():.1f}"
    )
