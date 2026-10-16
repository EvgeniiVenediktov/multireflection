"""
frame_data.py - Online single-frame data for encoder pretraining (PLAN phase 5, Level 1)

Every training batch is new (infinite data), generated on the GPU:
    M2 tilt   cell_frac (50 %) uniform over the 0.01 deg cells with >= 4 bounces (from
              a nominal-geometry scan of the +-2 deg box, computed once), the rest
              uniform in the box
    jitter    physical, per sample: separation +-0.5 mm, laser pitch / yaw +-0.05 deg,
              laser offset and M2 offset +-0.05 mm (both axes); the targets see it
    render    RandomRenderer with the training NuisanceRanges
    targets   TargetBuilder (canonical M2 frame), plus the true M2 tilt (probe label)

Fixed, seeded evaluation sets: one set of states rendered three ways - clean
(SpotRenderer, no nuisances), in-distribution (training ranges) and OOD (one held-out
factor per sample) - and a balanced 89 vs trapped probe set rendered the same three
ways.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from env import SpotRenderer
from geometry import CellGeometry
from herriott_sim import (
    DEVICE,
    S_LASER_DX,
    S_LASER_DY,
    S_LASER_PITCH,
    S_LASER_YAW,
    S_M2_PITCH,
    S_M2_TX,
    S_M2_TY,
    S_M2_YAW,
    S_SEP,
    STOP_CAP,
)
from render_random import NuisanceRanges, RandomRenderer
from targets import REGIONS, TargetBuilder, TargetConfig

MODES = ("clean", "id", "ood")


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class DataConfig:
    """FrameData settings."""

    batch_size: int = 256
    geometry: CellGeometry = field(default_factory=CellGeometry)
    max_bounces: int = 128  # Bounce cap (D1)

    # ── Camera (SpotRenderer, as EnvConfig) ──
    img_res: int = 64
    spot_sigma: float = 0.5  # mm
    gain: float = 1.6

    # ── Tilt sampling ──
    cell_frac: float = 0.5  # Share drawn from the >= min_cell_bounces cells
    min_cell_bounces: int = 4
    scan_step: float = 0.01  # deg

    # ── Physical jitter (half-widths, uniform) ──
    jitter_sep: float = 0.5  # mm
    jitter_laser_tilt: float = 0.05  # deg, pitch and yaw
    jitter_laser_offset: float = 0.05  # mm, dx and dy
    jitter_m2_offset: float = 0.05  # mm, tx and ty

    ranges: NuisanceRanges = field(default_factory=NuisanceRanges)
    targets: TargetConfig = field(default_factory=TargetConfig)


# ── Data ────────────────────────────────────────────────────────────────────


class FrameData:
    """Sim + randomized render + targets for single frames, see the module docstring."""

    def __init__(self, cfg: Optional[DataConfig] = None, device: Optional[torch.device] = None):
        self.cfg = c = cfg or DataConfig()
        self.device = dev = torch.device(device) if device is not None else DEVICE
        g = c.geometry
        self.limit = g.m2_tilt_limit
        self.sim = g.make_sim(dev, max_bounces=c.max_bounces)
        self.base = SpotRenderer(
            c.img_res, g.diameter, c.spot_sigma, c.gain, g.reflectivity, c.max_bounces, 0, dev
        )
        self.renderer = RandomRenderer(self.base)
        self.builder = TargetBuilder(
            self.base.xs, self.base.ys, self.base.radius, self.base.hit_intensity, c.targets
        )
        self._scan()

    def _scan(self):
        """Nominal-geometry scan of the box: cell centres by bounce class."""
        c, dev = self.cfg, self.device
        n = int(round(2 * self.limit / c.scan_step)) + 1
        ax = torch.linspace(-self.limit, self.limit, n, device=dev)
        P, Y = torch.meshgrid(ax, ax, indexing="ij")
        centres = torch.stack((P.reshape(-1), Y.reshape(-1)), 1)
        out = self.sim.simulate(self._states(centres), store_world=False)
        b, stop = out["hit_counts"], out["stop_reason"]
        self.scan_bounces = b
        self.cells = centres[b >= c.min_cell_bounces]
        self.cells_89 = centres[b == 89]
        self.cells_trapped = centres[stop == STOP_CAP]

    # ── States ──

    def _states(self, tilts: torch.Tensor) -> torch.Tensor:
        s = self.cfg.geometry.base_state(tilts.shape[0], self.device)
        s[:, S_M2_PITCH] = tilts[:, 0]
        s[:, S_M2_YAW] = tilts[:, 1]
        return s

    def sample_tilts(self, n: int, gen: torch.Generator, cells: Optional[torch.Tensor] = None,
                     cell_frac: Optional[float] = None) -> torch.Tensor:
        """(n, 2) M2 tilts: cell_frac uniform over cells (default the >= 4 bounce
        cells), the rest uniform in the box."""
        c, dev = self.cfg, self.device
        cells = self.cells if cells is None else cells
        frac = c.cell_frac if cell_frac is None else cell_frac
        box = self.limit * (2.0 * torch.rand(n, 2, generator=gen, device=dev) - 1.0)
        i = torch.randint(0, cells.shape[0], (n,), generator=gen, device=dev)
        inside = c.scan_step * (torch.rand(n, 2, generator=gen, device=dev) - 0.5)
        cell = (cells[i] + inside).clamp(-self.limit, self.limit)
        pick = torch.rand(n, 1, generator=gen, device=dev) < frac
        return torch.where(pick, cell, box)

    def states(self, tilts: torch.Tensor, gen: torch.Generator, jitter: bool = True) -> torch.Tensor:
        """(n, 13) states at the M2 tilts, with the physical jitter."""
        c = self.cfg
        s = self._states(tilts)
        if not jitter:
            return s
        u = lambda: 2.0 * torch.rand(s.shape[0], generator=gen, device=self.device) - 1.0
        s[:, S_SEP] += c.jitter_sep * u()
        s[:, S_LASER_PITCH] += c.jitter_laser_tilt * u()
        s[:, S_LASER_YAW] += c.jitter_laser_tilt * u()
        s[:, S_LASER_DX] += c.jitter_laser_offset * u()
        s[:, S_LASER_DY] += c.jitter_laser_offset * u()
        s[:, S_M2_TX] += c.jitter_m2_offset * u()
        s[:, S_M2_TY] += c.jitter_m2_offset * u()
        return s

    # ── Frames ──

    def simulate(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sim output and targets of (n, 13) states (no image)."""
        out = self.sim.simulate(states, store_world=False)
        d = self.builder(out)
        d["tilt"] = states[:, [S_M2_PITCH, S_M2_YAW]]
        d["state"] = states
        d["_hits"] = (out["hit_mirror"], out["hit_local_xy"])
        return d

    def render(self, d: Dict[str, torch.Tensor], mode: str, gen: torch.Generator) -> Dict[str, torch.Tensor]:
        """Image (n, 1, H, W) uint8 of a simulate() dict: mode clean / id / ood.
        Returns image and, for ood, the factor index per sample."""
        hm, lxy = d["_hits"]
        r = self.renderer
        if mode == "clean":
            return {"image": r(hm, lxy)}
        if mode == "id":
            p = r.sample_params(hm.shape[0], self.cfg.ranges, gen)
            return {"image": r(hm, lxy, p, gen)}
        if mode == "ood":
            p, factor = r.sample_ood(hm.shape[0], gen, self.cfg.ranges)
            return {"image": r(hm, lxy, p, gen), "ood_factor": factor}
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")

    def batch(self, gen: torch.Generator, n: Optional[int] = None, mode: str = "id") -> Dict[str, torch.Tensor]:
        """One training batch: fresh tilts, jitter, render, targets."""
        n = n or self.cfg.batch_size
        d = self.simulate(self.states(self.sample_tilts(n, gen), gen))
        d.update(self.render(d, mode, gen))
        return d

    # ── Fixed evaluation sets ──

    def _render_modes(self, d: Dict[str, torch.Tensor], seed: int, chunk: int) -> Dict[str, torch.Tensor]:
        """Images of d in all MODES (seeded), rendered in chunks; drops the raw hits."""
        hm, lxy = d.pop("_hits")
        n = hm.shape[0]
        for k, mode in enumerate(MODES):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed * 1000 + k)
            parts = [self.render({"_hits": (hm[i:i + chunk], lxy[i:i + chunk])}, mode, gen)
                     for i in range(0, n, chunk)]
            d[f"image_{mode}"] = torch.cat([p["image"] for p in parts])
            if mode == "ood":
                d["ood_factor"] = torch.cat([p["ood_factor"] for p in parts])
        return d

    def eval_set(self, n: int, seed: int, chunk: int = 1024) -> Dict[str, torch.Tensor]:
        """n states from the training distribution (seeded) with targets, labels and
        image_clean / image_id / image_ood."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        states = self.states(self.sample_tilts(n, gen), gen)
        d = self.simulate(states)
        return self._render_modes(d, seed, chunk)

    def probe_set_89_trapped(self, n_per_class: int, seed: int, chunk: int = 1024) -> Dict[str, torch.Tensor]:
        """Balanced 89 vs trapped set: tilts from the nominal 89 / trapped cells plus the
        physical jitter, kept where the jittered sim gives 89 bounces or the cap."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        r89, rtr = REGIONS.index("89"), REGIONS.index("trapped")
        keep = {r89: [], rtr: []}
        cells = {r89: self.cells_89, rtr: self.cells_trapped}
        for _ in range(50):
            for lab in (r89, rtr):
                if sum(t.shape[0] for t in keep[lab]) >= n_per_class:
                    continue
                st = self.states(self.sample_tilts(4 * n_per_class, gen, cells[lab], 1.0), gen)
                reg = self.simulate(st)["region"]
                keep[lab].append(st[reg == lab])
            if all(sum(t.shape[0] for t in v) >= n_per_class for v in keep.values()):
                break
        states = torch.cat([torch.cat(keep[lab])[:n_per_class] for lab in (r89, rtr)])
        perm = torch.randperm(states.shape[0], generator=gen, device=self.device)
        d = self.simulate(states[perm])
        return self._render_modes(d, seed, chunk)
