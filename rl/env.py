"""
env.py - Vectorized GPU environment for M2 alignment (PLAN phase 3: E1-E8, D1-D4)

B independent episodes, all state on the device, no host syncs in step().

    state      13-DOF base state of the measured cell (geometry.CellGeometry); the
               controlled DOFs (default M2 pitch, M2 yaw) move, every other DOF stays
               at the geometry value
    action     (B, 2) int64, per-axis index into the step ladder
               (0, +1, -1, +3, -3, +10, -10, +30, -30, +100, -100) x 0.01 deg;
               index 0 is the zero step, (0, 0) is "not move"
    transition tilt = clamp(tilt + step, -limit, +limit), limit 2 deg (mount range);
               the applied delta (after clamping) / largest ladder step is the action
               that led to the next frame
    reward     move:     (P_t - P_{t-1}) / P_nom - step_cost
               not move: 0, the episode terminates (decision D2)
               P_t = optical path after step t (bounce cap, D1), P_nom = 2733.27 mm;
               the return is (P_final - P_0) / P_nom - step_cost * moves
    truncation the step count reaches max_steps without termination
    auto-reset done envs restart inside step(): obs is the first frame of the new
               episode, info holds the pre-reset quantities (final frame and its
               stack, path, stop reason, true tilt, episode stats)

Observation (dict, see AlignEnv.observe):
    image    (B, 1, H, W) uint8   camera image of the M2 face (SpotRenderer)
    action   (B, 2) float32       applied action that led to the frame (0 at reset)
    images   (B, K, 1, H, W) uint8, actions (B, K, 2), mask (B, K) bool
                                  history stack (HistoryStack), oldest first
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from geometry import NOMINAL_PATH_MM, CellGeometry
from herriott_sim import DEVICE, S_M2_PITCH, S_M2_YAW

M2_ID = 2  # hit_mirror value of an M2 hit
AUTO_SIGMA_PX = 2.5  # Auto oversampling: spot sigma >= this many splat-grid pixels


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

    # ── Actions: per-axis ladder in units of step_unit, index 0 = zero step ──
    step_ladder: Tuple[float, ...] = (0, 1, -1, 3, -3, 10, -10, 30, -30, 100, -100)
    step_unit: float = 0.01  # deg

    # ── Episode and reward (D2) ──
    max_steps: int = 64
    step_cost: float = 0.01

    # ── Reset: initial tilts uniform in center +- radius, clamped to the limits ──
    init_center: Tuple[float, float] = (0.0, 0.0)  # deg
    init_radius: float = 2.0  # deg

    # ── Observation (D3) ──
    img_res: int = 64
    spot_sigma: float = 0.5  # mm
    gain: float = 1.6  # 1 - exp(-gain * I): a unit spot (k = 0) peaks at 0.80
    oversample: int = 0  # Splat grid oversampling; 0 = auto (sigma >= 2.5 grid px)
    history_len: int = 8  # Frames K in the history stack; 0 = no stack


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

    def push(self, image: torch.Tensor, applied_action: torch.Tensor, reset_mask: torch.Tensor):
        """Append a frame (B, 1, H, W) and its action (B, 2); reset_mask (B,) envs restart."""
        keep = ~reset_mask
        action = torch.where(keep[:, None], applied_action, 0.0)
        self.images, self.actions = self._appended(image, action, keep)
        self.count = torch.where(keep, (self.count + 1).clamp(max=self.length), 1)

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


class AlignEnv:
    """Vectorized alignment env, see the module docstring.

    step() simulates and renders one batch of 2B states: the B post-step states and
    B freshly sampled reset states; the reset half is used only where an episode
    ended (no data-dependent indexing, no host syncs).
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = c = cfg or EnvConfig()
        self.device = dev = torch.device(c.device) if c.device is not None else DEVICE
        g = c.geometry
        self.num_envs = B = c.num_envs
        assert len(c.controlled) == 2, "two controlled DOFs"
        assert c.step_ladder[0] == 0, "ladder index 0 must be the zero step"
        self.ctrl = tuple(int(i) for i in c.controlled)
        self.limit = float(g.m2_tilt_limit if c.tilt_limit is None else c.tilt_limit)
        self.p_nom = NOMINAL_PATH_MM

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
        self._base = g.base_state(2 * B, dev)  # Post-step half + reset half

        # ── Reset box ──
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(c.seed)
        self.set_init_box(c.init_center, c.init_radius)

        # ── Per-env state ──
        self.tilt = torch.zeros(B, 2, device=dev)  # Controlled DOFs (deg)
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

    # ── Reset box ──

    def set_init_box(self, center=None, radius=None):
        """Initial-tilt box (curriculum): center (2,) or (B, 2), radius scalar, (2,) or
        (B, 1), deg. Takes effect at the next reset of each env."""
        if center is not None:
            self.init_center = torch.as_tensor(center, dtype=torch.float32).to(self.device)
        if radius is not None:
            self.init_radius = torch.as_tensor(radius, dtype=torch.float32).to(self.device)

    def _sample_tilts(self) -> torch.Tensor:
        """(B, 2) tilts uniform in the init box, clamped to the limits."""
        u = torch.rand(self.num_envs, 2, generator=self.gen, device=self.device)
        t = self.init_center + self.init_radius * (2.0 * u - 1.0)
        return t.clamp(-self.limit, self.limit)

    # ── Sim and render ──

    def _states(self, tilts: torch.Tensor) -> torch.Tensor:
        """(n, 13) states: geometry base with the controlled DOFs set to tilts."""
        n = tilts.shape[0]
        if n <= self._base.shape[0]:
            s = self._base[:n].clone()
        else:
            s = self.cfg.geometry.base_state(n, self.device)
        s[:, self.ctrl[0]] = tilts[:, 0]
        s[:, self.ctrl[1]] = tilts[:, 1]
        return s

    def _simulate(self, tilts: torch.Tensor) -> dict:
        return self.sim.simulate(self._states(tilts), store_world=False)

    def _render(self, out: dict) -> torch.Tensor:
        return self.renderer(out["hit_mirror"], out["hit_local_xy"])

    def render_tilts(self, tilts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Diagnostics: sim + image of arbitrary (n, 2) tilts (no env state change)."""
        tilts = tilts.to(device=self.device, dtype=torch.float32)
        out = self._simulate(tilts)
        inten = self.renderer.intensity(out["hit_mirror"], out["hit_local_xy"])
        return {
            "image": self.renderer.respond(inten),
            "intensity": inten,
            "path_mm": out["total_path_length"],
            "bounces": out["hit_counts"],
            "stop_reason": out["stop_reason"],
            "sim": out,
        }

    # ── Gym-style API ──

    def observe(self) -> Dict[str, torch.Tensor]:
        """Current observation (no state change)."""
        obs = {"image": self.image, "action": self.prev_action}
        if self.history is not None:
            obs["images"], obs["actions"], obs["mask"] = self.history.get()
        return obs

    def reset(
        self, mask: Optional[torch.Tensor] = None, tilts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Start new episodes for the masked envs (all if None).

        tilts (B, 2): explicit initial tilts (clamped to the limits); default: sampled
        in the init box. Simulates all B, updates only the masked envs.
        """
        B = self.num_envs
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=self.device)
        if tilts is None:
            new = self._sample_tilts()
        else:
            new = tilts.to(device=self.device, dtype=torch.float32).clamp(-self.limit, self.limit)
        m1 = mask[:, None]
        tilt = torch.where(m1, new, self.tilt)
        out = self._simulate(tilt)
        img = self._render(out)

        self.tilt = tilt
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
        return self.observe()

    def step(self, action: torch.Tensor):
        """Apply (B, 2) ladder indices.

        Returns obs, reward (B,), terminated (B,), truncated (B,), info. info (all
        (B,), computed before the auto-reset):
            final_image (B, 1, H, W) uint8, final_obs (obs dict of that frame, with
            its history stack), applied_action (B, 2), tilt (B, 2) true controlled
            tilts (privileged), path_mm, path_norm = path_mm / P_nom (path-head label
            of final_image), bounces, stop_reason, done;
            ep_return, ep_length, ep_moves, ep_initial_path_mm, ep_final_path_mm,
            ep_final_bounces: valid where done;
            obs_tilt, obs_path_norm: the same for the returned obs (post-reset).
        """
        c = self.cfg
        B = self.num_envs
        a = action.to(device=self.device, dtype=torch.long)
        move = (a != 0).any(dim=1)
        mv1 = move[:, None]
        old = self.tilt
        new = (old + self.ladder[a]).clamp(-self.limit, self.limit)
        applied = (new - old) / self.max_step

        # One sim + render pass over post-step and reset states
        reset_tilt = self._sample_tilts()
        out = self._simulate(torch.cat((new, reset_tilt)))
        img = self._render(out)
        path_all = out["total_path_length"]
        count_all = out["hit_counts"]
        stop_all = out["stop_reason"]

        # Post-step state; not move keeps the cached state (no sim needed)
        path = torch.where(move, path_all[:B], self.path)
        bounces = torch.where(move, count_all[:B], self.bounces)
        stop = torch.where(move, stop_all[:B], self.stop)
        image = torch.where(move.view(-1, 1, 1, 1), img[:B], self.image)
        reward = torch.where(move, (path - self.path) / self.p_nom - c.step_cost, 0.0)

        step_count = self.step_count + 1
        terminated = ~move
        truncated = move & (step_count >= c.max_steps)
        done = terminated | truncated
        ep_return = self.ep_return + reward
        ep_moves = self.ep_moves + move.long()

        final_obs = {"image": image, "action": applied}
        if self.history is not None:
            final_obs["images"], final_obs["actions"], final_obs["mask"] = (
                self.history.peek_with(image, applied)
            )
        info = {
            "final_image": image,
            "final_obs": final_obs,
            "applied_action": applied,
            "tilt": new,
            "path_mm": path,
            "path_norm": path / self.p_nom,
            "bounces": bounces,
            "stop_reason": stop,
            "done": done,
            "ep_return": ep_return,
            "ep_length": step_count,
            "ep_moves": ep_moves,
            "ep_initial_path_mm": self.path0,
            "ep_final_path_mm": path,
            "ep_final_bounces": bounces,
        }

        # Auto-reset of the done envs
        d1 = done[:, None]
        self.tilt = torch.where(d1, reset_tilt, new)
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

        info["obs_tilt"] = self.tilt
        info["obs_path_norm"] = self.path / self.p_nom
        return self.observe(), reward, terminated, truncated, info
