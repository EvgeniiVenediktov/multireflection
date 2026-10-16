"""
geometry.py - The measured Herriott cell, single source of truth for RL

Values from tools/zemax_herriott_build.py (the fitted real cell). Angles in degrees,
lengths in mm, in the simulator frame of herriott_sim.py (M1 at z = 0 facing +z,
M2 at z = separation facing -z, laser behind the M1 hole).

At the nominal state the reference simulator gives 31 bounces (16 spots on M2),
a 2733.27 mm path, and the beam leaves through the M1 hole.

SETUPS: the measured cell (first, unchanged) and three more setups found by
find_setups.py (reports/setups.md): same mirrors, laser offset and mirror offsets, other
separation and laser tilt. Each records its nominal M2 tilt (deepest point of its
plateau, the largest hole-exit region); SETUP_INFO holds the regression values at
cap 128 (plateau order and path at the nominal tilt, longest path in the +-2 deg box on
the 0.01 deg grid and the target tilt where it is reached).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from herriott_sim import (
    DEVICE,
    DTYPE,
    STATE_DIM,
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
    HerriottSim,
    MirrorConfig,
    SimConfig,
)

# Regression values of the nominal state (reference simulator)
NOMINAL_BOUNCES = 31
NOMINAL_PATH_MM = 2733.27


@dataclass
class CellGeometry:
    """Measured cell: mirrors, nominal alignment, actuation limits, sim limits."""

    name: str = "measured"

    # ── Mirrors (both) ──
    roc: float = 200.0  # Radius of curvature (mm)
    diameter: float = 25.4  # Mirror diameter (mm)
    reflectivity: float = 0.98

    # ── M1 coupling hole (laser side); M2 has none ──
    m1_hole_radius: float = 1.5  # mm
    m1_hole_offset_y: float = 7.0  # Hole centre, M1-local y (mm)

    # ── Nominal alignment ──
    separation: float = 88.0  # mm
    m1_pitch: float = 0.0  # deg
    m1_yaw: float = 0.0  # deg
    m1_tx: float = 0.0  # mm
    m1_ty: float = 0.0  # mm
    m2_pitch: float = -0.38  # deg, nominal
    m2_yaw: float = 0.27  # deg, nominal
    m2_tx: float = -0.01  # mm
    m2_ty: float = 0.03  # mm
    laser_dx: float = 0.75  # mm, from hole centre
    laser_dy: float = 1.04  # mm, from hole centre
    laser_pitch: float = 4.08  # deg, relative to M1 normal
    laser_yaw: float = 4.82  # deg, relative to M1 normal

    # ── Actuation: absolute M2 tilt range, both axes (deg) ──
    m2_tilt_limit: float = 2.0

    # ── Simulation limits ──
    max_bounces: int = 128  # Bounce cap (decision D1)
    intensity_threshold: float = 1e-4

    def mirror_configs(self) -> Tuple[MirrorConfig, MirrorConfig]:
        """(M1, M2) mirror configs for HerriottSim."""
        m1 = MirrorConfig(
            roc=self.roc,
            diameter=self.diameter,
            hole_radius=self.m1_hole_radius,
            hole_offset_y=self.m1_hole_offset_y,
            reflectivity=self.reflectivity,
        )
        m2 = MirrorConfig(
            roc=self.roc,
            diameter=self.diameter,
            hole_radius=0.0,
            hole_offset_y=0.0,
            reflectivity=self.reflectivity,
        )
        return m1, m2

    def sim_config(
        self, device: Optional[torch.device] = None, max_bounces: Optional[int] = None
    ) -> SimConfig:
        """SimConfig with this cell's limits (max_bounces overrides the cap)."""
        return SimConfig(
            max_bounces=self.max_bounces if max_bounces is None else max_bounces,
            intensity_threshold=self.intensity_threshold,
            device=torch.device(device) if device is not None else DEVICE,
            dtype=DTYPE,
        )

    def make_sim(
        self, device: Optional[torch.device] = None, max_bounces: Optional[int] = None
    ) -> HerriottSim:
        """HerriottSim for this cell."""
        m1, m2 = self.mirror_configs()
        return HerriottSim(m1, m2, self.sim_config(device, max_bounces))

    def base_state(self, B: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """(B, 13) state at the nominal alignment (M2 tilt at nominal)."""
        dev = torch.device(device) if device is not None else DEVICE
        vals = torch.zeros(STATE_DIM, dtype=DTYPE)
        vals[S_M1_PITCH] = self.m1_pitch
        vals[S_M1_YAW] = self.m1_yaw
        vals[S_M2_PITCH] = self.m2_pitch
        vals[S_M2_YAW] = self.m2_yaw
        vals[S_SEP] = self.separation
        vals[S_M1_TX] = self.m1_tx
        vals[S_M1_TY] = self.m1_ty
        vals[S_M2_TX] = self.m2_tx
        vals[S_M2_TY] = self.m2_ty
        vals[S_LASER_DX] = self.laser_dx
        vals[S_LASER_DY] = self.laser_dy
        vals[S_LASER_PITCH] = self.laser_pitch
        vals[S_LASER_YAW] = self.laser_yaw
        return vals.to(dev).expand(B, STATE_DIM).clone()


# ── Setups ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SetupInfo:
    """Regression values of a setup at cap 128 (find_setups.py, reports/setups.md)."""

    plateau_bounces: int  # At the nominal M2 tilt, leaves through the M1 hole
    plateau_path_mm: float
    pmax_mm: float  # Longest path in the M2 box (0.01 deg grid)
    target: Tuple[float, float]  # M2 tilt (deg) in the best region (deepest point)
    target_bounces: int


SETUPS: Tuple[CellGeometry, ...] = (
    CellGeometry(),
    CellGeometry(name="sep116", separation=116.0, laser_pitch=6.35, laser_yaw=0.45,
                 m2_pitch=0.27, m2_yaw=0.15),
    CellGeometry(name="sep144", separation=144.0, laser_pitch=1.75, laser_yaw=3.50,
                 m2_pitch=0.69, m2_yaw=-0.06),
    CellGeometry(name="sep169", separation=169.0, laser_pitch=3.55, laser_yaw=2.50,
                 m2_pitch=0.16, m2_yaw=0.24),
)

SETUP_INFO: Dict[str, SetupInfo] = {
    "measured": SetupInfo(31, 2733.27, 11270.63, (0.09, 1.03), 128),
    "sep116": SetupInfo(21, 2441.57, 13348.60, (0.36, -1.24), 115),
    "sep144": SetupInfo(33, 4757.13, 18437.93, (-0.61, -0.06), 128),
    "sep169": SetupInfo(39, 6596.23, 21637.84, (1.00, 0.72), 128),
}


def setup_by_name(name: str) -> CellGeometry:
    """The SETUPS entry with this name."""
    for g in SETUPS:
        if g.name == name:
            return g
    raise KeyError(f"unknown setup {name!r}, expected one of {[g.name for g in SETUPS]}")
