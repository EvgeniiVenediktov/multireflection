"""
policy.py - Recurrent CNN+GRU policy for Herriott cell alignment

Modular encoder for sim-to-real transfer:
  - Train full pipeline in sim
  - Swap encoder, freeze core, fine-tune on real camera data
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Iterator, Optional


# ── Default Encoder ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-activation residual block with optional channel change."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class SpotPatternEncoder(nn.Module):
    """
    Small ResNet encoder for simulated spot pattern images.
    Input:  (B, C, H, W) — e.g. (B, 1, 64, 64) single-channel Gaussian spots
    Output: (B, feat_dim)

    Architecture: stem conv → 4 residual blocks with stride-2 downsampling → global avg pool → linear
    64 → 32 → 16 → 8 → 4 → global pool → feat_dim
    """

    def __init__(self, img_channels: int = 1, img_size: int = 64, feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim

        # Stem: initial conv to get to base channels
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Residual blocks: 32→32 (s2), 32→64 (s2), 64→64 (s2), 64→128 (s2)
        self.blocks = nn.Sequential(
            ResBlock(32, 32, stride=2),    # 64 → 32
            ResBlock(32, 64, stride=2),    # 32 → 16
            ResBlock(64, 64, stride=2),    # 16 → 8
            ResBlock(64, 128, stride=2),   # 8  → 4
        )

        # Global average pool → linear
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).reshape(x.shape[0], -1)  # (B, 128)
        return self.fc(x)


# ── Policy ────────────────────────────────────────────────────────────────────

class RecurrentPolicy(nn.Module):
    """
    CNN encoder + GRU + policy/value heads.

    Encoder is a swappable nn.Module:
      - Default: SpotPatternEncoder (sim)
      - Replace with set_encoder() for real-world camera

    Core (GRU + heads) can be frozen for encoder-only fine-tuning.
    """

    ACTION_DIM = 2  # M2 pitch, M2 yaw

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        hidden_dim: int = 128,
        feat_dim: int = 128,
        img_channels: int = 1,
        img_size: int = 64,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Encoder ───────────────────────────────────────────────────
        if encoder is not None:
            self.encoder = encoder
            feat_dim = encoder.feat_dim
        else:
            self.encoder = SpotPatternEncoder(img_channels, img_size, feat_dim)

        # ── Input projection: (encoder_feat + prev_action) → gru_input ──
        self._gru_input_dim = hidden_dim  # GRU sees hidden_dim-sized input
        self.input_proj = nn.Linear(feat_dim + self.ACTION_DIM, self._gru_input_dim)

        # ── GRU core ──────────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size=self._gru_input_dim,
            hidden_size=hidden_dim,
            batch_first=False,  # We feed single steps: (1, B, dim)
        )

        # ── Policy head (separate from value) ─────────────────────────
        self.pi_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
        )
        self.pi_mean = nn.Linear(64, self.ACTION_DIM)
        self.pi_log_std = nn.Parameter(
            torch.full((self.ACTION_DIM,), log_std_init)
        )

        # ── Value head ────────────────────────────────────────────────
        self.vf_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self._init_weights()

    # ── Forward helpers ───────────────────────────────────────────────

    def _encode_and_step_gru(
        self,
        obs_image: torch.Tensor,   # (B, C, H, W)
        prev_action: torch.Tensor, # (B, 2)
        hidden: torch.Tensor,      # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run encoder → concat prev_action → input_proj → GRU step.
        Returns (gru_output (B, hidden_dim), hidden_new (B, hidden_dim)).
        """
        feat = self.encoder(obs_image)                          # (B, feat_dim)
        x = torch.cat([feat, prev_action], dim=-1)              # (B, feat_dim+2)
        x = self.input_proj(x)                                  # (B, gru_input_dim)

        # GRU expects (seq=1, B, dim), hidden (1, B, dim)
        gru_out, h_new = self.gru(
            x.unsqueeze(0),
            hidden.unsqueeze(0),
        )
        gru_out = gru_out.squeeze(0)  # (B, hidden_dim)
        h_new = h_new.squeeze(0)      # (B, hidden_dim)

        return gru_out, h_new

    def _get_dist(self, gru_out: torch.Tensor) -> Normal:
        """Build squashed Gaussian from GRU output."""
        h = self.pi_net(gru_out)
        mean = self.pi_mean(h)
        std = self.pi_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def _get_value(self, gru_out: torch.Tensor) -> torch.Tensor:
        return self.vf_net(gru_out).squeeze(-1)  # (B,)

    @staticmethod
    def _tanh_log_prob(dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        """Log-prob with tanh squashing correction.
        raw_action is the pre-tanh sample (from dist.rsample()).
        """
        lp = dist.log_prob(raw_action).sum(-1)                    # (B,)
        lp -= (2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))).sum(-1)
        return lp

    @staticmethod
    def _tanh_entropy(dist: Normal) -> torch.Tensor:
        """Approximate entropy of tanh-squashed Gaussian.
        Exact entropy is intractable; use the unsquashed entropy as proxy.
        This is standard practice in continuous PPO.
        """
        return dist.entropy().sum(-1)  # (B,)

    # ── Public API: rollout ───────────────────────────────────────────

    def act(
        self,
        obs_image: torch.Tensor,    # (B, C, H, W)
        prev_action: torch.Tensor,  # (B, 2)
        hidden: torch.Tensor,       # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action during rollout (under torch.no_grad externally).

        Returns:
            action:     (B, 2)          tanh-squashed, in (-1, 1)
            log_prob:   (B,)
            value:      (B,)
            hidden_new: (B, hidden_dim)
        """
        gru_out, h_new = self._encode_and_step_gru(obs_image, prev_action, hidden)

        dist = self._get_dist(gru_out)
        raw = dist.rsample()              # pre-tanh
        action = torch.tanh(raw)          # squashed to (-1, 1)
        log_prob = self._tanh_log_prob(dist, raw)
        value = self._get_value(gru_out)

        return action, log_prob, value, h_new

    # ── Public API: PPO update ────────────────────────────────────────

    def evaluate(
        self,
        obs_image: torch.Tensor,    # (B, C, H, W)
        action: torch.Tensor,       # (B, 2)  — the squashed action from buffer
        prev_action: torch.Tensor,  # (B, 2)
        hidden: torch.Tensor,       # (B, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions for PPO loss.

        Returns:
            log_prob:   (B,)
            value:      (B,)
            entropy:    (B,)
            hidden_new: (B, hidden_dim)
        """
        gru_out, h_new = self._encode_and_step_gru(obs_image, prev_action, hidden)

        dist = self._get_dist(gru_out)

        # Invert tanh to get raw action for log_prob computation
        raw = self._atanh(action)
        log_prob = self._tanh_log_prob(dist, raw)
        entropy = self._tanh_entropy(dist)
        value = self._get_value(gru_out)

        return log_prob, value, entropy, h_new

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable inverse tanh."""
        x = x.clamp(-0.999, 0.999)
        return 0.5 * (x.log1p() - (-x).log1p())

    # ── Modularity API ────────────────────────────────────────────────

    def set_encoder(self, new_encoder: nn.Module):
        """
        Replace the encoder module.
        If feat_dim differs, rebuilds input_proj (adapter layer).
        GRU and heads are untouched.
        """
        old_feat = self.encoder.feat_dim
        new_feat = new_encoder.feat_dim
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        self.encoder = new_encoder.to(device=device, dtype=dtype)

        if new_feat != old_feat:
            self.input_proj = nn.Linear(
                new_feat + self.ACTION_DIM, self._gru_input_dim,
            ).to(device=device, dtype=dtype)
            # Re-init the new projection
            nn.init.orthogonal_(self.input_proj.weight, gain=math.sqrt(2))
            nn.init.zeros_(self.input_proj.bias)

    def freeze_core(self):
        """Freeze GRU + policy head + value head. Encoder and input_proj stay trainable."""
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.pi_net.parameters():
            p.requires_grad = False
        for p in self.pi_mean.parameters():
            p.requires_grad = False
        self.pi_log_std.requires_grad = False
        for p in self.vf_net.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Encoder + input_proj parameters (trainable during fine-tuning)."""
        yield from self.encoder.parameters()
        yield from self.input_proj.parameters()

    def core_parameters(self) -> Iterator[nn.Parameter]:
        """GRU + policy head + value head parameters."""
        yield from self.gru.parameters()
        yield from self.pi_net.parameters()
        yield from self.pi_mean.parameters()
        yield self.pi_log_std
        yield from self.vf_net.parameters()

    # ── Init ──────────────────────────────────────────────────────────

    def _init_weights(self):
        # CNN encoder: Kaiming for conv, standard for BN
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Input projection
        nn.init.orthogonal_(self.input_proj.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.input_proj.bias)

        # GRU: orthogonal recurrent weights
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.orthogonal_(param, gain=math.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Policy head: standard for hidden, small for output
        for m in self.pi_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi_mean.weight, gain=0.01)
        nn.init.zeros_(self.pi_mean.bias)

        # Value head
        for m in self.vf_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)


# ── Tests ─────────────────────────────────────────────────────────────────────

def _test():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 8

    policy = RecurrentPolicy(hidden_dim=128, feat_dim=128).to(dev)
    print(f"Params: {sum(p.numel() for p in policy.parameters()):,}")

    img = torch.randn(B, 1, 64, 64, device=dev)
    pa = torch.zeros(B, 2, device=dev)
    h = torch.zeros(B, policy.hidden_dim, device=dev)

    # ── 1. Shape test ─────────────────────────────────────────────────
    action, lp, val, h_new = policy.act(img, pa, h)
    assert action.shape == (B, 2), f"action {action.shape}"
    assert lp.shape == (B,), f"lp {lp.shape}"
    assert val.shape == (B,), f"val {val.shape}"
    assert h_new.shape == (B, 128), f"h_new {h_new.shape}"
    print("[PASS] 1. act() shapes")

    lp2, val2, ent, h_new2 = policy.evaluate(img, action, pa, h)
    assert lp2.shape == (B,)
    assert val2.shape == (B,)
    assert ent.shape == (B,)
    assert h_new2.shape == (B, 128)
    print("[PASS] 2. evaluate() shapes")

    # ── 2. Action bounds ──────────────────────────────────────────────
    for _ in range(100):
        a, _, _, _ = policy.act(img, pa, h)
        assert (a.abs() < 1.0).all(), f"action out of bounds: {a.min()}, {a.max()}"
    print("[PASS] 3. action bounds (-1, 1)")

    # ── 3. Log-prob finite ────────────────────────────────────────────
    assert torch.isfinite(lp).all(), "lp has NaN/Inf"
    assert torch.isfinite(lp2).all(), "lp2 has NaN/Inf"
    assert torch.isfinite(ent).all(), "ent has NaN/Inf"
    print("[PASS] 4. log_prob / entropy finite")

    # ── 4. Gradient flow ──────────────────────────────────────────────
    loss = lp2.sum() + val2.sum() + ent.sum()
    loss.backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None, f"no grad: {name}"
        assert torch.isfinite(p.grad).all(), f"bad grad: {name}"
    policy.zero_grad()
    print("[PASS] 5. gradient flow through all params")

    # ── 5. GRU state propagation ──────────────────────────────────────
    a1, _, _, h1 = policy.act(img, pa, h)
    a2, _, _, h2 = policy.act(img, pa, h1)  # different hidden
    assert not torch.allclose(h1, h2), "GRU hidden didn't change"
    print("[PASS] 6. GRU state propagation")

    # ── 6. Save/load roundtrip ────────────────────────────────────────
    sd = policy.state_dict()
    policy2 = RecurrentPolicy(hidden_dim=128, feat_dim=128).to(dev)
    policy2.load_state_dict(sd)
    a_orig, _, _, _ = policy.act(img, pa, h)
    a_load, _, _, _ = policy2.act(img, pa, h)
    # Note: rsample is stochastic, so compare deterministic parts
    with torch.no_grad():
        g1, _ = policy._encode_and_step_gru(img, pa, h)
        g2, _ = policy2._encode_and_step_gru(img, pa, h)
    assert torch.allclose(g1, g2, atol=1e-6), "save/load mismatch"
    print("[PASS] 7. save/load roundtrip")

    # ── 7. Encoder swap (same feat_dim) ───────────────────────────────
    core_before = {k: v.clone() for k, v in policy.state_dict().items() if 'encoder' not in k}
    new_enc = SpotPatternEncoder(img_channels=1, img_size=64, feat_dim=128).to(dev)
    policy.set_encoder(new_enc)
    core_after = {k: v for k, v in policy.state_dict().items() if 'encoder' not in k}
    for k in core_before:
        if 'input_proj' not in k:
            assert torch.equal(core_before[k], core_after[k]), f"core changed: {k}"
    print("[PASS] 8. encoder swap (same feat_dim) preserves core")

    # ── 8. Encoder swap (different feat_dim) ──────────────────────────
    big_enc = SpotPatternEncoder(img_channels=1, img_size=64, feat_dim=256).to(dev)
    gru_before = {k: v.clone() for k, v in policy.state_dict().items() if 'gru' in k}
    policy.set_encoder(big_enc)
    gru_after = {k: v for k, v in policy.state_dict().items() if 'gru' in k}
    for k in gru_before:
        assert torch.equal(gru_before[k], gru_after[k]), f"GRU changed: {k}"
    assert policy.input_proj.in_features == 258  # 256 + 2
    # Verify forward still works
    a, _, _, _ = policy.act(img, pa, h)
    assert a.shape == (B, 2)
    print("[PASS] 9. encoder swap (different feat_dim) rebuilds input_proj, GRU intact")

    # ── 9. Freeze / unfreeze ──────────────────────────────────────────
    policy.freeze_core()
    enc_grads = [p.requires_grad for p in policy.encoder_parameters()]
    core_grads = [p.requires_grad for p in policy.core_parameters()]
    assert all(enc_grads), "encoder params should be trainable"
    assert not any(core_grads), "core params should be frozen"

    policy.unfreeze_all()
    all_grads = [p.requires_grad for p in policy.parameters()]
    assert all(all_grads), "all params should be trainable after unfreeze"
    print("[PASS] 10. freeze_core / unfreeze_all")

    # ── 10. Parameter groups disjoint and complete ────────────────────
    enc_ids = {id(p) for p in policy.encoder_parameters()}
    core_ids = {id(p) for p in policy.core_parameters()}
    all_ids = {id(p) for p in policy.parameters()}
    assert enc_ids & core_ids == set(), "param groups overlap"
    assert enc_ids | core_ids == all_ids, "param groups don't cover all params"
    print("[PASS] 11. parameter groups disjoint and complete")

    # ── 11. Gradient isolation when frozen ────────────────────────────
    policy.freeze_core()
    policy.zero_grad()
    lp3, val3, ent3, _ = policy.evaluate(img, action.detach(), pa, h)
    (lp3.sum() + val3.sum() + ent3.sum()).backward()
    for name, p in policy.named_parameters():
        if 'encoder' in name or 'input_proj' in name:
            assert p.grad is not None and (p.grad != 0).any(), f"no grad on trainable: {name}"
        else:
            assert p.grad is None or (p.grad == 0).all(), f"grad leaked to frozen: {name}"
    policy.unfreeze_all()
    print("[PASS] 12. gradient isolation when core frozen")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == '__main__':
    _test()