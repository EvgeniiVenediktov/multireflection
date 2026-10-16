"""
tests/test_encoder.py - RandomRenderer, targets and frame encoder checks

Run from rl/:  uv run python tests/test_encoder.py

    1. RandomRenderer without parameters == SpotRenderer (bit-identical); identity
       parameters through the randomized path: exact Gaussians (vs a dense fp64
       reference), within the SpotRenderer splat error in uint8; identity warp == no warp
    2. camera pose direction: a +0.5 mm x shift moves the spot centroid by +0.5 mm
    3. target spot map of the nominal state: 16 peaks, each at the pixel of its M2 spot
       centre, peak heights reflectivity^k; nominal ellipse fit; a synthetic ellipse
       recovered exactly
    4. encoder output shapes (CNN, ViT-p8, ViT-p4) and parameter counts
    5. 50-step overfit smoke test on a fixed batch of 16 per encoder
    6. four-setup frame data: setups uniform per sample, each sample's state carries
       its setup's geometry, cell-drawn tilts lie in that setup's >= 4 bounce cells
"""

import math
import os
import sys

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from encoders import ELL_DIM, N_COUNT, EncoderConfig, FrameModel, count_params, model_name  # noqa: E402
from frame_data import DataConfig, FrameData  # noqa: E402
from geometry import SETUPS  # noqa: E402
from herriott_sim import S_LASER_PITCH, S_SEP  # noqa: E402
from pretrain_encoder import frame_loss  # noqa: E402
from render_random import pose_warp  # noqa: E402
from targets import ellipse_to_vec, fit_ellipse, vec_to_ellipse  # noqa: E402

DEV = torch.device("cuda")
NOMINAL = (-0.38, 0.27)
STATES = {
    "nominal": NOMINAL,
    "57 island": (-0.55, 0.62),
    "89 island": (0.17, 0.97),
    "trapped": (0.10, 1.04),
    "1 bounce": (-1.8, -1.8),
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def tilts(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32, device=DEV)


def test_states(data: FrameData, n_rand: int = 59):
    """Named tilts (no jitter) + random training-distribution states (with jitter)."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(3)
    named = data.states(tilts(*STATES.values()), gen, jitter=False)
    rand = data.states(data.sample_tilts(n_rand, gen), gen)
    return torch.cat((named, rand))


def dense_reference(data: FrameData, out: dict) -> torch.Tensor:
    """Exact isotropic Gaussian sum per pixel (fp64), SpotRenderer sigma, before aperture."""
    b = data.base
    hm, lxy = out["hit_mirror"], out["hit_local_xy"].double()
    k = torch.arange(hm.shape[1], device=DEV, dtype=torch.float64)
    w = torch.where(hm == 2, data.cfg.geometry.reflectivity**k, 0.0)
    dx2 = (b.xs.double()[None, None, :] - lxy[..., 0, None]) ** 2
    dy2 = (b.ys.double()[None, None, :] - lxy[..., 1, None]) ** 2
    s2 = 2 * b.sigma**2
    return torch.einsum("nk,nki,nkj->nij", w, torch.exp(-dy2 / s2), torch.exp(-dx2 / s2))


# ── Tests ───────────────────────────────────────────────────────────────────


def test_randomization_off(data: FrameData):
    print("\n== 1. randomization off == SpotRenderer ==")
    r = data.renderer
    out = data.sim.simulate(test_states(data), store_world=False)
    hm, lxy = out["hit_mirror"], out["hit_local_xy"]
    n = hm.shape[0]
    base = data.base(hm, lxy)
    off = r(hm, lxy)
    assert off.dtype == torch.uint8 and off.shape == base.shape
    assert torch.equal(off, base)
    print(f"  {n} states (5 named, {n - 5} jittered): RandomRenderer(params=None) bit-identical")

    ident = r.identity_params(n)
    inten = r.canonical(hm, lxy, ident)
    ref = dense_reference(data, out) * data.base.aperture.double()
    err = float((inten.double() - ref).abs().max() / ref.amax((1, 2)).clamp_min(1e-12).max())
    assert err < 1e-5, err
    img_id = r(hm, lxy, ident)
    diff = (img_id.float() - base.float()).abs()
    print(f"  identity parameters: max |dense - fp64 reference| / peak {err:.1e}; uint8 vs "
          f"SpotRenderer max {int(diff.max())} LSB, mean {float(diff.mean()):.3f} LSB "
          f"(SpotRenderer splat error)")
    assert int(diff.max()) <= 6 and float(diff.mean()) < 0.5
    img_w = r(hm, lxy, r.identity_params(n, warp=True))
    assert torch.equal(img_w, img_id)
    print("  explicit identity warp (grid_sample) == no warp, bit-identical")


def test_pose_direction(data: FrameData):
    print("\n== 2. camera pose direction ==")
    r = data.renderer
    out = data.sim.simulate(data.states(tilts(NOMINAL), None, jitter=False), store_world=False)
    hm, lxy = out["hit_mirror"], out["hit_local_xy"]
    b = data.base
    p0 = r.identity_params(1)
    shift = 0.5  # mm
    p1 = r.identity_params(1, warp=True)
    p1.warp = pose_warp(torch.zeros(1, device=DEV), torch.tensor([[shift / b.radius, 0.0]], device=DEV),
                        torch.ones(1, device=DEV), torch.zeros(1, 4, 2, device=DEV))

    def centroid(params):
        img = r.warp(r.canonical(hm, lxy, params), params.warp)[0]
        return float((img.sum(0) * b.xs).sum() / img.sum()), float((img.sum(1) * b.ys).sum() / img.sum())

    (x0, y0), (x1, y1) = centroid(p0), centroid(p1)
    print(f"  centroid ({x0:.3f}, {y0:.3f}) -> ({x1:.3f}, {y1:.3f}) mm for a +{shift} mm x shift")
    assert abs((x1 - x0) - shift) < 0.02 and abs(y1 - y0) < 0.02


def test_targets(data: FrameData):
    print("\n== 3. targets ==")
    d = data.simulate(data.states(tilts(NOMINAL), None, jitter=False))
    b = data.base
    m = d["spot_map"][0, 0]
    pooled = F.max_pool2d(m[None, None], 3, stride=1, padding=1)[0, 0]
    peaks = (m == pooled) & (m > 0.05)
    assert int(d["count"][0]) == 16 and int(peaks.sum()) == 16, (int(d["count"][0]), int(peaks.sum()))
    pos, w = d["spots"][0, :16], d["spot_w"][0, :16]
    j = ((pos[:, 0] + b.radius) / b.pitch).floor().long()
    i = ((b.radius - pos[:, 1]) / b.pitch).floor().long()
    assert bool(peaks[i, j].all()), "a peak is not at its spot pixel"
    k = torch.arange(0, 32, 2, device=DEV, dtype=torch.float32)
    assert torch.allclose(w, 0.98**k, rtol=1e-5)
    ratio = m[i, j] / w
    print(f"  nominal: 16 spots, 16 local maxima, each at the pixel of its centre; peak / "
          f"0.98^k {float(ratio.min()):.3f}-{float(ratio.max()):.3f} (>= 0.78 at a half-pixel "
          f"offset, sigma 1 px)")
    assert float(ratio.min()) > 0.77 and float(ratio.max()) < 1.001

    e = vec_to_ellipse(d["ellipse"][0])
    print(f"  nominal ellipse: centre ({float(e[0]):.2f}, {float(e[1]):.2f}) mm, semi-axes "
          f"{float(e[2]):.2f} / {float(e[3]):.2f} mm, angle {math.degrees(float(e[4])):.1f} deg, "
          f"Sampson RMS {float(d['ellipse_rms'][0]):.3f} mm, mask {bool(d['ellipse_mask'][0])}")
    assert bool(d["ellipse_mask"][0]) and float(d["ellipse_rms"][0]) < 0.25

    # Synthetic ellipses, partial arcs included
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    n, K = 64, 12
    truth = torch.stack((
        4 * torch.rand(n, generator=gen, device=DEV) - 2, 4 * torch.rand(n, generator=gen, device=DEV) - 2,
        8 + 3 * torch.rand(n, generator=gen, device=DEV), 3 + 4 * torch.rand(n, generator=gen, device=DEV),
        math.pi * torch.rand(n, generator=gen, device=DEV)), 1)
    span = 1.0 + 5.0 * torch.rand(n, 1, generator=gen, device=DEV)  # 1 rad to a full turn
    t = span * torch.rand(n, K, generator=gen, device=DEV)
    cx, cy, a, bb, phi = truth.unbind(1)
    u, v = a[:, None] * torch.cos(t), bb[:, None] * torch.sin(t)
    c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
    pts = torch.stack((cx[:, None] + c * u - s * v, cy[:, None] + s * u + c * v), -1)
    valid = torch.ones(n, K, dtype=torch.bool, device=DEV)
    valid[:8, 5:] = False  # 5 points: the minimum
    fit, ok, rms = fit_ellipse(pts, valid)
    dv = (ellipse_to_vec(fit) - ellipse_to_vec(truth)).abs().max()
    print(f"  synthetic: {n} ellipses (5-12 points, arcs 1 rad to full): all valid {bool(ok.all())}, "
          f"max |vec error| {float(dv):.1e} mm, max RMS {float(rms.max()):.1e} mm")
    assert bool(ok.all()) and float(dv) < 1e-3
    valid[:4, 4:] = False  # 4 points: masked
    _, ok4, _ = fit_ellipse(pts, valid)
    assert not bool(ok4[:4].any()) and bool(ok4[4:].all())


def configs():
    return [EncoderConfig(arch="cnn"), EncoderConfig(arch="vit", patch=8), EncoderConfig(arch="vit", patch=4)]


def test_shapes():
    print("\n== 4. encoder output shapes ==")
    x = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8, device=DEV)
    for cfg in configs():
        m = FrameModel(cfg).to(DEV)
        o = m(x)
        assert o["z"].shape == (3, 256)
        assert o["map"].shape == (3, 1, 64, 64)
        assert o["count"].shape == (3, N_COUNT) and o["ellipse"].shape == (3, ELL_DIM)
        print(f"  {model_name(cfg):7s} z {tuple(o['z'].shape)}, map {tuple(o['map'].shape)}, count "
              f"{tuple(o['count'].shape)}, ellipse {tuple(o['ellipse'].shape)}; encoder "
              f"{count_params(m.encoder) / 1e6:.2f} M, heads {count_params(m.heads) / 1e6:.2f} M")


def test_overfit(data: FrameData):
    print("\n== 5. 50-step overfit on a fixed batch of 16 ==")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(11)
    batch = data.batch(gen, n=16)
    for cfg in configs():
        torch.manual_seed(0)
        m = FrameModel(cfg).to(DEV).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        hist = []
        for _ in range(50):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(batch["image"])
            terms = frame_loss(out, batch)
            opt.zero_grad()
            terms["total"].backward()
            opt.step()
            hist.append({k: float(v.detach()) for k, v in terms.items()})
        first, last = hist[0], hist[-1]
        print(f"  {model_name(cfg):7s} " + ", ".join(
            f"{k} {first[k]:.4f} -> {last[k]:.4f}" for k in ("total", "map", "count", "ellipse")))
        assert all(math.isfinite(h["total"]) for h in hist)
        assert last["total"] < 0.3 * first["total"] and last["map"] < 0.5 * first["map"]


def test_multi_setup():
    print("\n== 6. four-setup frame data ==")
    data = FrameData(DataConfig(setups=SETUPS), DEV)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(4)
    n = 8192
    d = data.batch(gen, n=n, mode="clean")
    cnt = torch.bincount(d["setup"], minlength=4).float()
    print(f"  {n} samples, per setup " + ", ".join(
        f"{g.name} {int(c)}" for g, c in zip(SETUPS, cnt)))
    assert float((cnt / n - 0.25).abs().max()) < 0.02
    for k, g in enumerate(SETUPS):
        st = d["state"][d["setup"] == k]
        assert float((st[:, S_SEP] - g.separation).abs().max()) <= data.cfg.jitter_sep + 1e-4
        assert float((st[:, S_LASER_PITCH] - g.laser_pitch).abs().max()) <= (
            data.cfg.jitter_laser_tilt + 1e-4)
    setup = data.sample_setups(4096, gen)
    t = data.sample_tilts(4096, gen, cell_frac=1.0, setup=setup)
    out = data.sim.simulate(data.states(t, gen, jitter=False, setup=setup), store_world=False)
    ok = float((out["hit_counts"] >= data.cfg.min_cell_bounces).float().mean())
    print(f"  cells per setup {data.cell_count.tolist()}; cell-drawn tilts with >= 4 bounces "
          f"in their own setup (no jitter): {ok:.4f}")
    assert ok > 0.98
    print(f"  max M2 spot count {int(d['count'].max())} (classes 0-{N_COUNT - 1})")
    assert int(d["count"].max()) < N_COUNT


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    data = FrameData(DataConfig(), DEV)
    test_randomization_off(data)
    test_pose_direction(data)
    test_targets(data)
    test_shapes()
    test_overfit(data)
    test_multi_setup()
    print("\nall tests passed")
