"""
config_sampler.py - Config sampling for Herriott RL training

Two modes:
  - SimpleConfigSampler: fixed or uniform-random configs (for debugging/testing)
  - BinnedConfigSampler: prioritized bin-based sampling (for full training)
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ConfigRange:
    """Valid ranges for each DoF."""

    m1_pitch: Tuple[float, float] = (-3.0, 3.0)
    m1_yaw: Tuple[float, float] = (-3.0, 3.0)
    sep: Tuple[float, float] = (80.0, 400.0)
    m2_pitch: Tuple[float, float] = (-3.0, 3.0)
    m2_yaw: Tuple[float, float] = (-3.0, 3.0)


@dataclass
class BatchConfig:
    """Config for one training batch."""

    env_params: torch.Tensor  # (3,)   → [M1_pitch, M1_yaw, separation]
    m2_init: torch.Tensor  # (B, 2) → per-worker M2 starting positions
    bin_idx: int = -1  # For binned sampler; -1 = not applicable


# ── Simple Sampler (for testing) ──────────────────────────────────────────────


@dataclass
class SimpleSamplerConfig:
    ranges: ConfigRange = field(default_factory=ConfigRange)
    m2_jitter_std: float = 0.15
    fixed_env_params: Optional[Tuple[float, float, float]] = (
        None  # Set to lock env config
    )
    fixed_m2_init: Optional[Tuple[float, float]] = None  # Set to lock M2 start
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float32


class SimpleConfigSampler:
    """
    Simple uniform or fixed config sampler for debugging.

    Usage:
        # Fully fixed (deterministic env for debugging):
        sampler = SimpleConfigSampler(SimpleSamplerConfig(
            fixed_env_params=(0.0, 0.0, 250.0),
            fixed_m2_init=(0.5, 0.5),
        ))

        # Random env, random M2 start:
        sampler = SimpleConfigSampler(SimpleSamplerConfig())

        config = sampler.sample(num_envs=64)
    """

    def __init__(self, cfg: Optional[SimpleSamplerConfig] = None):
        self.cfg = cfg or SimpleSamplerConfig()

    def sample(self, num_envs: int) -> BatchConfig:
        c = self.cfg

        # Env params
        if c.fixed_env_params is not None:
            env_params = torch.tensor(
                c.fixed_env_params, device=c.device, dtype=c.dtype
            )
        else:
            env_params = torch.tensor(
                [
                    _uniform(*c.ranges.m1_pitch, c.device),
                    _uniform(*c.ranges.m1_yaw, c.device),
                    _uniform(*c.ranges.sep, c.device),
                ],
                device=c.device,
                dtype=c.dtype,
            )

        # M2 init
        if c.fixed_m2_init is not None:
            m2_center = torch.tensor(c.fixed_m2_init, device=c.device, dtype=c.dtype)
        else:
            m2_center = torch.tensor(
                [
                    _uniform(*c.ranges.m2_pitch, c.device),
                    _uniform(*c.ranges.m2_yaw, c.device),
                ],
                device=c.device,
                dtype=c.dtype,
            )

        m2_jitter = (
            torch.randn(num_envs, 2, device=c.device, dtype=c.dtype) * c.m2_jitter_std
        )
        m2_init = m2_center.unsqueeze(0) + m2_jitter
        m2_init[:, 0].clamp_(*c.ranges.m2_pitch)
        m2_init[:, 1].clamp_(*c.ranges.m2_yaw)

        return BatchConfig(env_params=env_params, m2_init=m2_init, bin_idx=-1)

    def update(self, bin_idx: int, mean_return: float):
        """No-op for simple sampler."""
        pass

    def bin_stats(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {"type": "simple"}

    def load_state_dict(self, d: dict):
        pass


# ── Binned Sampler (for full training) ────────────────────────────────────────


@dataclass
class BinnedSamplerConfig:
    ranges: ConfigRange = field(default_factory=ConfigRange)
    bins_per_dim: int = 3  # Start coarse: 3^5 = 243 bins
    m2_jitter_std: float = 0.15
    temperature: float = 1.0
    min_samples_before_priority: int = 10
    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float32


class BinnedConfigSampler:
    """
    Prioritized bin-based sampling over the 5D config space.
    Tracks per-bin performance, oversamples weak regions.
    """

    def __init__(self, cfg: Optional[BinnedSamplerConfig] = None):
        self.cfg = cfg or BinnedSamplerConfig()
        c = self.cfg

        self.ranges = torch.tensor(
            [
                c.ranges.m1_pitch,
                c.ranges.m1_yaw,
                c.ranges.sep,
                c.ranges.m2_pitch,
                c.ranges.m2_yaw,
            ],
            device=c.device,
            dtype=c.dtype,
        )

        self.n_dims = 5
        self.bpd = c.bins_per_dim
        self.total_bins = self.bpd**self.n_dims

        self.bin_return_sum = torch.zeros(
            self.total_bins, device=c.device, dtype=c.dtype
        )
        self.bin_return_sq = torch.zeros(
            self.total_bins, device=c.device, dtype=c.dtype
        )
        self.bin_count = torch.zeros(self.total_bins, device=c.device, dtype=torch.long)

    def sample(self, num_envs: int) -> BatchConfig:
        c = self.cfg
        bin_idx = self._sample_bin()
        multi_idx = self._flat_to_multi(bin_idx)
        center = self._bin_center(multi_idx)

        bin_widths = (self.ranges[:, 1] - self.ranges[:, 0]) / self.bpd
        in_bin_noise = (
            torch.rand(5, device=c.device, dtype=c.dtype) - 0.5
        ) * bin_widths
        config_center = torch.clamp(
            center + in_bin_noise, self.ranges[:, 0], self.ranges[:, 1]
        )

        env_params = config_center[:3]
        m2_center = config_center[3:5]
        m2_jitter = (
            torch.randn(num_envs, 2, device=c.device, dtype=c.dtype) * c.m2_jitter_std
        )
        m2_init = m2_center.unsqueeze(0) + m2_jitter
        m2_init[:, 0].clamp_(*c.ranges.m2_pitch)
        m2_init[:, 1].clamp_(*c.ranges.m2_yaw)

        return BatchConfig(env_params=env_params, m2_init=m2_init, bin_idx=bin_idx)

    def _sample_bin(self) -> int:
        c = self.cfg
        if (self.bin_count < c.min_samples_before_priority).any():
            weights = 1.0 / (self.bin_count.float() + 1.0)
        else:
            mean_ret = self.bin_return_sum / self.bin_count.float().clamp(min=1)
            gap = mean_ret.max() - mean_ret
            weights = torch.softmax(gap / c.temperature, dim=0)
        return torch.multinomial(weights, 1).item()

    def update(self, bin_idx: int, mean_return: float):
        self.bin_return_sum[bin_idx] += mean_return
        self.bin_return_sq[bin_idx] += mean_return**2
        self.bin_count[bin_idx] += 1

    def bin_stats(self) -> dict:
        visited = self.bin_count > 0
        mr = torch.where(
            visited,
            self.bin_return_sum / self.bin_count.float().clamp(min=1),
            torch.zeros_like(self.bin_return_sum),
        )
        return {
            "bins_visited": visited.sum().item(),
            "bins_total": self.total_bins,
            "coverage": visited.sum().item() / self.total_bins,
            "ret_min": mr[visited].min().item() if visited.any() else 0,
            "ret_max": mr[visited].max().item() if visited.any() else 0,
            "ret_avg": mr[visited].mean().item() if visited.any() else 0,
        }

    def state_dict(self) -> dict:
        return {
            "type": "binned",
            "bin_return_sum": self.bin_return_sum.cpu(),
            "bin_return_sq": self.bin_return_sq.cpu(),
            "bin_count": self.bin_count.cpu(),
        }

    def load_state_dict(self, d: dict):
        self.bin_return_sum = d["bin_return_sum"].to(self.cfg.device)
        self.bin_return_sq = d["bin_return_sq"].to(self.cfg.device)
        self.bin_count = d["bin_count"].to(self.cfg.device)

    def _flat_to_multi(self, flat_idx: int) -> torch.Tensor:
        multi = []
        for _ in range(self.n_dims):
            multi.append(flat_idx % self.bpd)
            flat_idx //= self.bpd
        return torch.tensor(multi, device=self.cfg.device, dtype=torch.long)

    def _bin_center(self, multi_idx: torch.Tensor) -> torch.Tensor:
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        return lo + (multi_idx.float() + 0.5) * (hi - lo) / self.bpd


# ── Util ──────────────────────────────────────────────────────────────────────


def _uniform(lo, hi, device):
    return torch.empty(1, device=device).uniform_(lo, hi).item()
