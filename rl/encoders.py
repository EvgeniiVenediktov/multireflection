"""
encoders.py - Frame encoders and single-frame heads (PLAN phase 5, Level 1)

Both encoders map a (B, 1, 64, 64) uint8 image to an embedding z (B, 256); every head
reads z only (z is the bottleneck the policy will see).

    CNNEncoder   conv stem, pre-activation GroupNorm residual stages down to 8 x 8,
                 spatial-softmax keypoints (expected x, y per keypoint channel)
                 concatenated with mean- and max-pooled features, MLP to z
    ViTEncoder   patch 8 (64 tokens) or patch 4 (256 tokens), pre-LN transformer
                 blocks, fixed 2D sin-cos position embeddings, multi-query attention
                 pooling to z
    FrameHeads   spot-map decoder (linear to 8 x 8, two upsampling conv stages with
                 coordinate channels to 32 x 32, conv to 4 channels and pixel shuffle to
                 64 x 64; output = sqrt of the spot map), count classifier (0-64),
                 ellipse regressor (5, see targets.py)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_COUNT = 65  # M2 spot count classes 0-64
ELL_DIM = 5  # (cx, cy, mxx, myy, mxy)


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class EncoderConfig:
    """Encoder and head settings."""

    arch: str = "cnn"  # cnn | vit
    z_dim: int = 256
    img_res: int = 64

    # ── CNN ──
    cnn_widths: Tuple[int, ...] = (32, 64, 128, 160)  # Stem, then one stage per entry after
    cnn_blocks: Tuple[int, ...] = (1, 1, 2)  # Residual blocks per stage (64 -> 32 -> 16 -> 8)
    n_keypoints: int = 64
    gn_groups: int = 8

    # ── ViT ──
    patch: int = 8
    width: int = 160
    depth: int = 6
    heads: int = 4
    mlp_ratio: float = 2.0
    pool_queries: int = 4

    # ── Heads ──
    dec_width: int = 64  # Channels at 8 x 8 and 16 x 16
    head_hidden: int = 256
    mlp_hidden: int = 384  # CNN keypoint / pooling MLP


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def to_input(img: torch.Tensor) -> torch.Tensor:
    """uint8 (B, 1, H, W) -> float in [0, 1]."""
    return img.float() * (1.0 / 255.0)


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))


# ── CNN ─────────────────────────────────────────────────────────────────────


class ResBlock(nn.Module):
    """Pre-activation residual block: GN, SiLU, conv3x3, twice."""

    def __init__(self, c: int, groups: int):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, c)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class SpatialSoftmax(nn.Module):
    """(B, K, H, W) -> (B, 2K) expected (x, y) in [-1, 1] per channel; learned
    temperature."""

    def __init__(self, h: int, w: int):
        super().__init__()
        ys = torch.linspace(-1.0, 1.0, h)
        xs = torch.linspace(-1.0, 1.0, w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("pos", torch.stack((gx.reshape(-1), gy.reshape(-1)), 1))  # (HW, 2)
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K = x.shape[:2]
        att = torch.softmax(x.flatten(2).float() * torch.exp(-self.log_temp), dim=-1)
        return (att @ self.pos).reshape(B, 2 * K)


class CNNEncoder(nn.Module):
    """See the module docstring."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        w, g = cfg.cnn_widths, cfg.gn_groups
        assert len(w) == len(cfg.cnn_blocks) + 1
        self.stem = nn.Conv2d(1, w[0], 3, padding=1)
        stages = []
        for i, nb in enumerate(cfg.cnn_blocks):
            layers = [nn.GroupNorm(g, w[i]), nn.SiLU(), nn.Conv2d(w[i], w[i + 1], 3, stride=2, padding=1)]
            layers += [ResBlock(w[i + 1], g) for _ in range(nb)]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.out_norm = nn.GroupNorm(g, w[-1])
        res = cfg.img_res // 2 ** len(cfg.cnn_blocks)
        self.kp = nn.Conv2d(w[-1], cfg.n_keypoints, 1)
        self.softmax = SpatialSoftmax(res, res)
        d_in = 2 * cfg.n_keypoints + 2 * w[-1]
        self.mlp = _mlp(d_in, cfg.mlp_hidden, cfg.z_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.out_norm(self.stages(self.stem(to_input(img)))))  # (B, C, 8, 8)
        kp = self.softmax(self.kp(h))
        pooled = torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1)
        return self.mlp(torch.cat((kp.to(pooled.dtype), pooled), 1))


# ── ViT ─────────────────────────────────────────────────────────────────────


def sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """(h w, dim) fixed 2D sin-cos position embedding (MAE style)."""
    assert dim % 4 == 0
    q = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float64) / q))
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
    )
    ox = gx.reshape(-1, 1) * omega[None]
    oy = gy.reshape(-1, 1) * omega[None]
    return torch.cat((ox.sin(), ox.cos(), oy.sin(), oy.cos()), 1).float()


class Attention(nn.Module):
    """Multi-head attention; self-attention when kv is None."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor = None) -> torch.Tensor:
        kv = x if kv is None else kv
        B, Tq, D = x.shape
        Tk = kv.shape[1]
        H = self.heads
        q = self.q(x).view(B, Tq, H, D // H).transpose(1, 2)
        k, v = self.kv(kv).view(B, Tk, 2, H, D // H).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, Tq, D))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViTEncoder(nn.Module):
    """See the module docstring."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        p, d = cfg.patch, cfg.width
        g = cfg.img_res // p
        self.embed = nn.Conv2d(1, d, p, stride=p)
        self.register_buffer("pos", sincos_2d(g, g, d)[None])
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.zeros(1, cfg.pool_queries, d))
        self.pool = Attention(d, cfg.heads)
        self.out = nn.Linear(cfg.pool_queries * d, cfg.z_dim)
        self.apply(self._init)
        nn.init.trunc_normal_(self.queries, std=0.02)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.embed(to_input(img)).flatten(2).transpose(1, 2) + self.pos  # (B, T, d)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = self.pool(self.queries.expand(x.shape[0], -1, -1), x)  # (B, Q, d)
        return self.out(pooled.flatten(1))


# ── Heads ───────────────────────────────────────────────────────────────────


def _coords(x: torch.Tensor) -> torch.Tensor:
    """x with two coordinate channels (x, y in [-1, 1]) appended."""
    B, _, H, W = x.shape
    ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.cat((x, gx.expand(B, 1, H, W), gy.expand(B, 1, H, W)), 1)


class UpStage(nn.Module):
    """Nearest x2 upsampling, coordinate channels, two conv3x3 + GN + SiLU."""

    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        super().__init__()
        self.c1 = nn.Conv2d(c_in + 2, c_out, 3, padding=1)
        self.n1 = nn.GroupNorm(groups, c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.n2 = nn.GroupNorm(groups, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.n1(self.c1(_coords(x))))
        return F.silu(self.n2(self.c2(x)))


class FrameHeads(nn.Module):
    """Single-frame heads on z: map (B, 1, H, W) in sqrt units, count logits (B, 65),
    ellipse (B, 5) in units of ELL_SCALE mm."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c = cfg.dec_width
        self.base_res = cfg.img_res // 8
        self.c0 = c
        self.map_fc = nn.Linear(cfg.z_dim, c * self.base_res**2)
        self.up = nn.Sequential(UpStage(c, c), UpStage(c, 32))
        self.map_out = nn.Sequential(nn.Conv2d(32 + 2, 4, 3, padding=1), nn.PixelShuffle(2))
        self.count = _mlp(cfg.z_dim, cfg.head_hidden, N_COUNT)
        self.ellipse = _mlp(cfg.z_dim, cfg.head_hidden, ELL_DIM)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        r = self.base_res
        h = F.silu(self.map_fc(z)).view(-1, self.c0, r, r)
        return {
            "map": self.map_out(_coords(self.up(h))),
            "count": self.count(z),
            "ellipse": self.ellipse(z),
        }


class FrameModel(nn.Module):
    """Encoder + heads; forward(img) -> dict with z and the head outputs."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.arch == "cnn":
            self.encoder = CNNEncoder(cfg)
        elif cfg.arch == "vit":
            self.encoder = ViTEncoder(cfg)
        else:
            raise ValueError(f"unknown arch {cfg.arch!r}")
        self.heads = FrameHeads(cfg)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(img)
        out = self.heads(z)
        out["z"] = z
        return out


def model_name(cfg: EncoderConfig) -> str:
    """Short architecture tag: cnn, vit-p8, vit-p4."""
    return "cnn" if cfg.arch == "cnn" else f"vit-p{cfg.patch}"
