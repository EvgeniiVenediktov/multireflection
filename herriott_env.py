"""
herriott_env.py - Fully GPU-resident RL environment for Herriott cell
ALL tensors live on CUDA. Zero CPU/RAM usage in the hot path.

Env/train contract:
    env_params:  (m1_pitch, m1_yaw, separation) — fixed per episode, set by sampler
    agent action: (m2_pitch_delta, m2_yaw_delta) — 2 DOF
    obs dict:     {'image': (B,1,H,W), 'state': (B,5)}

Sim contract (unchanged):
    sim.simulate(state: (B,5)) -> dict with hit_counts, hit_sequence, etc.
"""

import torch
import torch.nn.functional as F
from enum import IntEnum
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field

from simulation.herriott_sim import HerriottSim, SimConfig, MirrorConfig, DEVICE, DTYPE


# ── Config ────────────────────────────────────────────────────────────────────


class ObsMirror(IntEnum):
    """Which mirror surface to use as the visual observation source."""

    M1 = 0  # Has hole/window - beam enters and exits here
    M2 = 1  # Solid mirror - no hole


@dataclass
class EnvConfig:
    """Environment configuration — all GPU."""

    # Vectorization
    num_envs: int = 64

    # Observation
    obs_mirror: ObsMirror = ObsMirror.M2
    img_res: int = 64
    img_extent: float = 14.0  # Half-width of rendered view (mm)
    spot_sigma: float = 0.5  # Gaussian spot sigma (mm)
    include_state_obs: bool = True

    # Action — agent controls m2 only
    m2_angle_limit: float = 5.0  # Max m2 tilt (degrees)
    action_scale: Tuple[float, float] = (0.5, 0.5)  # m2_pitch, m2_yaw delta per step

    # Episode
    max_steps: int = 64

    # Sim
    sim: SimConfig = field(default_factory=SimConfig)
    m1: MirrorConfig = field(
        default_factory=lambda: MirrorConfig(hole_radius=1.5, hole_offset_y=7.0)
    )
    m2: MirrorConfig = field(
        default_factory=lambda: MirrorConfig(hole_radius=0.0, hole_offset_y=0.0)
    )

    device: torch.device = DEVICE
    dtype: torch.dtype = DTYPE


# ── Gaussian Renderer (pure GPU) ─────────────────────────────────────────────


class GaussianRenderer:
    """
    Renders hit points as Gaussian blobs on a 2D image tensor.
    Entirely on GPU — no numpy, no matplotlib.

    Output shape: (B, 1, H, W) — single-channel intensity image.
    """

    def __init__(self, cfg: EnvConfig):
        self.res = cfg.img_res
        self.extent = cfg.img_extent
        self.sigma = cfg.spot_sigma
        self.device = cfg.device
        self.dtype = cfg.dtype

        # Pre-compute pixel grid (persistent on GPU)
        lin = torch.linspace(
            -self.extent, self.extent, self.res, device=self.device, dtype=self.dtype
        )
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.gx = gx.reshape(-1)  # (H*W,)
        self.gy = gy.reshape(-1)

        # Mirror aperture mask
        r2 = self.gx**2 + self.gy**2
        self.mirror_mask = (r2 <= (cfg.m1.diameter / 2.0) ** 2).float()

        # Hole mask (only for M1)
        hole_r2 = self.gx**2 + (self.gy - cfg.m1.hole_offset_y) ** 2
        self.hole_mask_m1 = (hole_r2 <= cfg.m1.hole_radius**2).float()

    def render(
        self,
        hit_xy: torch.Tensor,
        hit_intensity: torch.Tensor,
        hit_mask: torch.Tensor,
        obs_mirror: ObsMirror,
    ) -> torch.Tensor:
        """
        Args:
            hit_xy:        (B, N, 2) — local x,y on mirror surface
            hit_intensity: (B, N)    — intensity per hit
            hit_mask:      (B, N)    — 1.0 if valid hit on chosen mirror
        Returns:
            (B, 1, H, W) normalized intensity image on GPU
        """
        B, N, _ = hit_xy.shape

        I = hit_intensity * hit_mask
        cx = hit_xy[..., 0]  # (B, N)
        cy = hit_xy[..., 1]

        # (B, N, HW) distances
        dx = self.gx.unsqueeze(0).unsqueeze(0) - cx.unsqueeze(-1)
        dy = self.gy.unsqueeze(0).unsqueeze(0) - cy.unsqueeze(-1)
        r2 = dx * dx + dy * dy
        gauss = torch.exp(-r2 / (2.0 * self.sigma**2))

        # Weighted sum over spots
        img = (gauss * I.unsqueeze(-1)).sum(dim=1)  # (B, HW)

        # Aperture mask
        img = img * self.mirror_mask.unsqueeze(0)

        # Hole cutout for M1 view
        if obs_mirror == ObsMirror.M1:
            img = img * (1.0 - self.hole_mask_m1.unsqueeze(0))

        # Normalize per batch element
        img = img / img.amax(dim=-1, keepdim=True).clamp(min=1e-8)

        return img.reshape(B, 1, self.res, self.res)


# ── Environment ───────────────────────────────────────────────────────────────


class HerriottEnv:
    """
    Fully GPU-resident vectorized RL environment.

    The sampler sets env_params (m1_pitch, m1_yaw, separation) which are
    fixed for the episode. The agent controls m2_pitch and m2_yaw only.

    Obs dict:
        'image':  (B, 1, H, W)  — Gaussian-rendered mirror view
        'state':  (B, 5)         — [m1_pitch, m1_yaw, m2_pitch, m2_yaw, sep]

    Action: (B, 2) in [-1, 1] — scaled deltas for m2_pitch, m2_yaw
    """

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        c = self.cfg

        self.sim = HerriottSim(
            m1_cfg=c.m1,
            m2_cfg=c.m2,
            sim_cfg=c.sim,
            # mounted_laser=True,
        )
        self.renderer = GaussianRenderer(c)

        self.B = c.num_envs
        self.dev = c.device
        self.dt = c.dtype

        # ── Persistent GPU state ──────────────────────────────────────
        # Fixed per-episode (set by sampler via reset_with_config)
        self.env_params = torch.zeros(self.B, 3, device=self.dev, dtype=self.dt)
        # Agent-controlled
        self.m2_state = torch.zeros(self.B, 2, device=self.dev, dtype=self.dt)
        # Step counter
        self.step_count = torch.zeros(self.B, dtype=torch.long, device=self.dev)

        # Action scaling
        self.action_scale = torch.tensor(c.action_scale, device=self.dev, dtype=self.dt)

        # Pre-compute bounce metadata for sim output conversion
        N = c.sim.max_bounces
        # Mirror pattern: laser enters M1 hole -> first hit is M2, then alternates
        # bounce 0=M2(1), 1=M1(0), 2=M2(1), ...
        self._mirror_pattern = (1 - torch.arange(N, device=self.dev) % 2).long()
        # Per-bounce intensity: reflectivity^(bounce+1)
        refl = min(c.m1.reflectivity, c.m2.reflectivity)  # conservative
        self._intensity_template = refl ** torch.arange(
            1, N + 1, device=self.dev, dtype=self.dt
        )
        self._bounce_idx = torch.arange(N, device=self.dev)

    # ── State assembly ────────────────────────────────────────────────

    def _build_sim_state(self) -> torch.Tensor:
        """Assemble (B, 5) state tensor for the sim."""
        return torch.cat(
            [
                self.env_params[:, :2],  # m1_pitch, m1_yaw
                self.m2_state,  # m2_pitch, m2_yaw
                self.env_params[:, 2:],  # separation
            ],
            dim=-1,
        )

    def _full_state(self) -> torch.Tensor:
        """Full 5D state for observation: [m1p, m1y, m2p, m2y, sep]."""
        return self._build_sim_state()

    # ── Reset ─────────────────────────────────────────────────────────

    def reset_with_config(
        self,
        env_params: torch.Tensor,
        m2_init: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reset environments with specific configuration.

        Args:
            env_params: (3,) or (n, 3) — [m1_pitch, m1_yaw, separation]
            m2_init:    (n, 2)          — [m2_pitch, m2_yaw] initial values
            env_ids:    (n,) optional   — which envs to reset; None = all
        Returns:
            obs dict (all CUDA tensors)
        """
        if env_ids is None:
            env_ids = torch.arange(self.B, device=self.dev)

        n = env_ids.shape[0]

        # Broadcast env_params if needed
        if env_params.dim() == 1:
            ep = env_params.unsqueeze(0).expand(n, -1)
        else:
            ep = env_params
        if ep.shape[0] == 1 and n > 1:
            ep = ep.expand(n, -1)

        self.env_params[env_ids] = ep.to(device=self.dev, dtype=self.dt)
        self.m2_state[env_ids] = m2_init.to(device=self.dev, dtype=self.dt)
        self.step_count[env_ids] = 0

        return self._observe()

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Random reset (for testing / standalone use)."""
        if env_ids is None:
            env_ids = torch.arange(self.B, device=self.dev)
        n = env_ids.shape[0]

        self.env_params[env_ids, 0] = torch.randn(n, device=self.dev, dtype=self.dt)
        self.env_params[env_ids, 1] = torch.randn(n, device=self.dev, dtype=self.dt)
        self.env_params[env_ids, 2] = torch.empty(
            n, device=self.dev, dtype=self.dt
        ).uniform_(100.0, 400.0)
        self.m2_state[env_ids] = torch.randn(n, 2, device=self.dev, dtype=self.dt) * 1.0
        self.step_count[env_ids] = 0

        return self._observe()

    # ── Step ──────────────────────────────────────────────────────────

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Step all environments. No auto-reset (train loop handles it).

        Args:
            actions: (B, 2) in [-1, 1] — m2_pitch, m2_yaw deltas
        Returns:
            obs, reward, done, info — all CUDA tensors
        """
        actions = actions.clamp(-1.0, 1.0)
        delta = actions * self.action_scale
        self.m2_state = self.m2_state + delta
        self.m2_state = self.m2_state.clamp(
            -self.cfg.m2_angle_limit, self.cfg.m2_angle_limit
        )
        self.step_count += 1

        # Simulate + observe
        sim_out = self._simulate()
        obs = self._observe_from_result(sim_out)
        rew = self._compute_reward(sim_out)
        done = self.step_count >= self.cfg.max_steps

        info = {
            "bounce_count": sim_out["bounce_count"],
            "total_path_length": sim_out["path_length"],
            "exit_through_hole": sim_out["exited"],
        }
        return obs, rew, done, info

    # ── Simulation bridge ─────────────────────────────────────────────

    def _simulate(self) -> Dict[str, torch.Tensor]:
        """
        Call sim with (B, 5) state, convert output to env contract.

        Sim returns:
            hit_counts, final_positions, total_path_length,
            hit_sequence (B, N, 3), intensity

        Env needs:
            hit_xy (B,N,2), hit_intensity (B,N), hit_mirrors (B,N),
            hit_valid (B,N), bounce_count, path_length, exited
        """
        state = self._build_sim_state()
        raw = self.sim.simulate(state)
        return self._convert_sim_output(raw)

    def _convert_sim_output(self, raw: dict) -> Dict[str, torch.Tensor]:
        """Bridge sim output dict → env output dict. All GPU, no copies."""
        B = raw["hit_counts"].shape[0]
        N = self.cfg.sim.max_bounces

        # ── hit_valid: bounce b is valid if b < hit_counts[i] ─────────
        hit_valid = self._bounce_idx.unsqueeze(0) < raw["hit_counts"].unsqueeze(1)

        # ── hit_xy: take world x,y from hit_sequence ──────────────────
        # For small tilts this is ≈ local mirror coords
        hit_xy = raw["hit_sequence"][:, :, :2]  # (B, N, 2)

        # ── hit_mirrors: alternating pattern ──────────────────────────
        # Laser enters M1 hole → first hit = M2, then M1, M2, ...
        hit_mirrors = self._mirror_pattern.unsqueeze(0).expand(B, -1)

        # ── hit_intensity: per-bounce attenuation ─────────────────────
        hit_intensity = self._intensity_template.unsqueeze(0).expand(B, -1)
        hit_intensity = hit_intensity * hit_valid.float()

        # ── exited: beam reached hole (stopped early, not from decay) ─
        # If bounce count < max AND final intensity > threshold,
        # the ray likely hit the hole rather than missing the mirror
        final_intensity = self._intensity_template[raw["hit_counts"].clamp(min=1) - 1]
        exited = (
            (raw["hit_counts"] > 0)
            & (raw["hit_counts"] < N)
            & (final_intensity > self.cfg.sim.intensity_threshold)
        )

        return {
            "hit_xy": hit_xy,
            "hit_intensity": hit_intensity,
            "hit_mirrors": hit_mirrors,
            "hit_valid": hit_valid,
            "bounce_count": raw["hit_counts"],
            "path_length": raw["total_path_length"],
            "exited": exited,
        }

    # ── Observation ───────────────────────────────────────────────────

    def _observe(self) -> Dict[str, torch.Tensor]:
        result = self._simulate()
        return self._observe_from_result(result)

    def _observe_from_result(
        self, result: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        mirror_idx = int(self.cfg.obs_mirror)

        hit_mask = (result["hit_mirrors"] == mirror_idx).float()
        hit_mask = hit_mask * result["hit_valid"].float()

        img = self.renderer.render(
            hit_xy=result["hit_xy"],
            hit_intensity=result["hit_intensity"],
            hit_mask=hit_mask,
            obs_mirror=self.cfg.obs_mirror,
        )

        obs = {"image": img}
        if self.cfg.include_state_obs:
            obs["state"] = self._full_state()
        return obs

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self, result: Dict[str, torch.Tensor]) -> torch.Tensor:
        bounces = result["bounce_count"].float()
        exited = result["exited"].float()

        r_bounces = bounces / self.cfg.sim.max_bounces
        r_exit = 0  # exited * 1.0
        r_penalty = (bounces < 4).float() * -0.5

        return r_bounces + r_exit + r_penalty

    # ── Mirror selection ──────────────────────────────────────────────

    def set_obs_mirror(self, mirror: ObsMirror):
        self.cfg.obs_mirror = mirror

    # ── Properties for network construction ───────────────────────────

    @property
    def obs_shape(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {"image": (1, self.cfg.img_res, self.cfg.img_res)}
        if self.cfg.include_state_obs:
            shapes["state"] = (5,)
        return shapes

    @property
    def action_dim(self) -> int:
        return 2

    def get_frame_for_display(self, env_idx: int = 0) -> torch.Tensor:
        """(H, W) float tensor on GPU — call .cpu() only for matplotlib."""
        obs = self._observe()
        return obs["image"][env_idx, 0]


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = EnvConfig(num_envs=8, img_res=64, obs_mirror=ObsMirror.M1)
    env = HerriottEnv(cfg)

    # Test reset_with_config (how train.py uses it)
    env_params = torch.tensor([0.0, 0.0, 200.0], device=cfg.device)
    m2_init = torch.randn(8, 2, device=cfg.device) * 0.5
    obs = env.reset_with_config(env_params=env_params, m2_init=m2_init)
    print(f"Image obs:  {obs['image'].shape} on {obs['image'].device}")
    print(f"State obs:  {obs['state'].shape} on {obs['state'].device}")
    print(f"State[0]:   {obs['state'][0]}")

    # Test step with 2D actions
    for i in range(5):
        actions = torch.randn(8, 2, device=cfg.device).clamp(-1, 1)
        obs, rew, done, info = env.step(actions)
        print(
            f"Step {i+1}: rew={rew.mean():.3f}, "
            f"bounces={info['bounce_count'].float().mean():.1f}, "
            f"exits={info['exit_through_hole'].sum()}, "
            f"m2_state={env.m2_state[0].tolist()}"
        )

    # Test partial reset (like train does for done envs)
    done_ids = torch.tensor([0, 3], device=cfg.device)
    fresh_m2 = torch.randn(2, 2, device=cfg.device) * 0.5
    obs_partial = env.reset_with_config(
        env_params=env_params, m2_init=fresh_m2, env_ids=done_ids
    )
    print(f"\nPartial reset [0,3] — image: {obs_partial['image'].shape}")

    # Test mirror switch
    env.set_obs_mirror(ObsMirror.M2)
    obs = env.reset(env_ids=None)
    print(f"Switched to M2 — image: {obs['image'].shape}")
    print(f"Action dim: {env.action_dim}")
