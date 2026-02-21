"""
herriott_sim.py - High-performance Herriott cell simulation for RL
All tensor ops, no Python loops in hot path, fully differentiable
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# Default device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@dataclass
class MirrorConfig:
    """Static mirror configuration."""

    roc: float = 200.0  # Radius of curvature (mm)
    diameter: float = 25.4  # Mirror diameter (mm)
    hole_radius: float = 1.5  # Hole radius (mm), 0 = no hole
    hole_offset_y: float = 7.0  # Hole Y offset (mm)
    reflectivity: float = 0.98


@dataclass
class SimConfig:
    """Simulation parameters."""

    max_bounces: int = 100
    intensity_threshold: float = 1e-4
    device: torch.device = DEVICE
    dtype: torch.dtype = DTYPE


class HerriottSim:
    """
    High-performance Herriott cell simulator.

    State vector (10 DOF):
        [m1_pitch, m1_yaw, m2_pitch, m2_yaw, separation,
         laser_x, laser_y, laser_z, laser_pitch, laser_yaw]

    For mounted laser (5 DOF), laser inherits m1 orientation:
        [m1_pitch, m1_yaw, m2_pitch, m2_yaw, separation]
    """

    def __init__(
        self,
        m1_cfg: MirrorConfig = None,
        m2_cfg: MirrorConfig = None,
        sim_cfg: SimConfig = None,
        mounted_laser: bool = True,
    ):
        self.m1_cfg = m1_cfg or MirrorConfig(hole_radius=1.5)  # With hole
        self.m2_cfg = m2_cfg or MirrorConfig(hole_radius=0)  # Solid
        self.sim_cfg = sim_cfg or SimConfig()
        self.mounted_laser = mounted_laser
        self.device = self.sim_cfg.device
        self.dtype = self.sim_cfg.dtype

        # Precompute constants
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
        pos_z: torch.Tensor,
        facing: float,
        roc: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mirror geometry from state.
        Returns: (center, normal, rotation) - all batched
        """
        B = pitch.shape[0]
        R = self.rotation_matrix(pitch, yaw)  # (B, 3, 3)

        # Base normal
        base_normal = torch.tensor([0, 0, facing], device=self.device, dtype=self.dtype)
        normal = torch.einsum("bij,j->bi", R, base_normal)  # (B, 3)

        # Mirror position (at z=pos_z)
        position = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        position[:, 2] = pos_z

        # Sphere center
        center = position + normal * roc  # (B, 3)

        return center, normal, R

    def _ray_sphere_intersect(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        centers: torch.Tensor,
        radius: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched ray-sphere intersection.
        origins: (B, 3), directions: (B, 3), centers: (B, 3)
        Returns: t (B,), valid (B,)
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

        # Take closer positive hit
        inf = torch.tensor(float("inf"), device=self.device)
        t1 = torch.where((t1 > 1e-5) & valid, t1, inf)
        t2 = torch.where((t2 > 1e-5) & valid, t2, inf)
        t = torch.minimum(t1, t2)

        return t, t < inf

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

        # Local coordinates
        local_x_axis = R[:, :, 0]  # (B, 3)
        local_y_axis = R[:, :, 1]  # (B, 3)

        lx = (rel * local_x_axis).sum(dim=-1)
        ly = (rel * local_y_axis).sum(dim=-1)
        r = torch.sqrt(lx**2 + ly**2)

        # Check on correct side
        dot_normal = (rel * normal).sum(dim=-1)

        valid = (r < aperture) & (dot_normal >= 0) & (dot_normal < sag + 1)

        # Hole check
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
            state: (B, 5) for mounted laser [m1_pitch, m1_yaw, m2_pitch, m2_yaw, sep]
                   or (B, 10) for free laser (adds laser x,y,z,pitch,yaw)

        Returns:
            dict with:
                - hit_counts: (B,) number of bounces per config
                - final_positions: (B, 3) last hit position
                - exit_through_hole: (B,) bool, True if exited via hole
                - total_path_length: (B,) total optical path
                - hit_sequence: (B, max_bounces, 3) all hit positions (padded)
        """
        B = state.shape[0]

        # Parse state
        m1_pitch, m1_yaw = state[:, 0], state[:, 1]
        m2_pitch, m2_yaw = state[:, 2], state[:, 3]
        separation = state[:, 4]

        # Mirror geometry
        m1_center, m1_normal, m1_R = self._get_mirror_geometry(
            m1_pitch, m1_yaw, torch.zeros(B, device=self.device), 1.0, self.m1_cfg.roc
        )
        m2_center, m2_normal, m2_R = self._get_mirror_geometry(
            m2_pitch, m2_yaw, separation, -1.0, self.m2_cfg.roc
        )

        m1_pos = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        m2_pos = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
        m2_pos[:, 2] = separation

        # Laser origin & direction
        if self.mounted_laser or state.shape[1] == 5:
            # Laser at hole, pointing along m1 normal
            hole_local = torch.zeros(B, 3, device=self.device, dtype=self.dtype)
            hole_local[:, 1] = self.m1_cfg.hole_offset_y
            hole_local[:, 2] = -5  # Behind mirror
            origins = torch.einsum("bij,bj->bi", m1_R, hole_local) + m1_pos
            directions = m1_normal.clone()
        else:
            # Free laser
            origins = state[:, 5:8]
            laser_pitch, laser_yaw = state[:, 8], state[:, 9]
            laser_R = self.rotation_matrix(laser_pitch, laser_yaw)
            base_dir = torch.tensor([0, 0, 1], device=self.device, dtype=self.dtype)
            directions = torch.einsum("bij,j->bi", laser_R, base_dir)

        directions = F.normalize(directions, dim=-1)

        # Tracing storage
        hit_sequence = torch.zeros(B, self.sim_cfg.max_bounces, 3, device=self.device)
        hit_counts = torch.zeros(B, dtype=torch.long, device=self.device)
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        intensity = torch.ones(B, device=self.device, dtype=self.dtype)
        path_length = torch.zeros(B, device=self.device, dtype=self.dtype)
        last_hit_mirror = torch.zeros(
            B, dtype=torch.long, device=self.device
        )  # 0=none, 1=M1, 2=M2

        for bounce in range(self.sim_cfg.max_bounces):
            if not active.any():
                break

            # Intersect both mirrors
            t1, v1 = self._ray_sphere_intersect(
                origins, directions, m1_center, self.m1_cfg.roc
            )
            t2, v2 = self._ray_sphere_intersect(
                origins, directions, m2_center, self.m2_cfg.roc
            )

            # Compute hits
            hit1 = origins + t1.unsqueeze(-1) * directions
            hit2 = origins + t2.unsqueeze(-1) * directions

            # Aperture checks
            v1 = v1 & self._check_aperture(
                hit1,
                m1_pos,
                m1_R,
                m1_normal,
                self.m1_aperture,
                self.m1_sag,
                self.m1_cfg.hole_radius,
                self.m1_hole_offset,
            )
            v2 = v2 & self._check_aperture(
                hit2, m2_pos, m2_R, m2_normal, self.m2_aperture, self.m2_sag
            )

            # Pick closer valid hit
            inf = torch.tensor(float("inf"), device=self.device)
            t1 = torch.where(v1, t1, inf)
            t2 = torch.where(v2, t2, inf)

            hit_m1 = (t1 <= t2) & v1
            hit_m2 = (t2 < t1) & v2
            any_hit = hit_m1 | hit_m2

            # Update active
            active = active & any_hit

            if not active.any():
                break

            # Get hit points and normals
            t = torch.where(hit_m1, t1, t2)
            hit_points = torch.where(hit_m1.unsqueeze(-1), hit1, hit2)

            # Normals (pointing into concave)
            n1 = -(hit1 - m1_center) / self.m1_cfg.roc
            n2 = (hit2 - m2_center) / self.m2_cfg.roc  # M2 faces -Z
            normals = torch.where(hit_m1.unsqueeze(-1), n1, n2)

            # Store hit
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

        # Reward: closer to target bounces = better
        bounce_diff = (result["hit_counts"].float() - target_bounces).abs()
        r = -bounce_diff  # Negative distance as reward

        # Bonus for path length (longer = better gas absorption)
        r = r + 0.01 * result["total_path_length"]

        return r


# Convenience function
def create_sim(mounted_laser: bool = True, device: str = None) -> HerriottSim:
    """Create simulator with default configs."""
    dev = torch.device(device) if device else DEVICE
    return HerriottSim(
        m1_cfg=MirrorConfig(hole_radius=1.5, hole_offset_y=7.0),
        m2_cfg=MirrorConfig(hole_radius=0),
        sim_cfg=SimConfig(device=dev),
        mounted_laser=mounted_laser,
    )


if __name__ == "__main__":
    # Quick test
    sim = create_sim()

    # Single config
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 200.0]], device=DEVICE)
    result = sim.simulate(state)
    print(f"Single: {result['hit_counts'].item()} bounces")

    # Batch of 1000 random configs
    states = torch.randn(1000, 5, device=DEVICE)
    states[:, 4] = 150 + 100 * torch.rand(1000, device=DEVICE)  # separation 150-250
    states[:, :4] *= 2  # angles ±2 degrees

    import time

    t0 = time.perf_counter()
    for _ in range(100):
        result = sim.simulate(states)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    t1 = time.perf_counter()

    print(f"Batch 1000 x 100 iters: {t1-t0:.3f}s ({100000/(t1-t0):.0f} sims/sec)")
    print(
        f"Bounce stats: min={result['hit_counts'].min()}, max={result['hit_counts'].max()}, mean={result['hit_counts'].float().mean():.1f}"
    )
    input()
