"""
tests/test_camera.py - CameraRenderer augmentation config, camera-frame targets and the
3-channel encoder init

Run from rl/:  uv run python tests/test_camera.py

    1. neutral parameters reproduce SpotRenderer inside the mirror (within its splat
       error, 4 / 255), all three channels equal
    2. augmentation probabilities: clean_p = 1 gives the neutral parameters exactly;
       every effect at p = 0 leaves it off; every effect at p = 1 turns it on with its
       parameters inside the configured ranges; the empirical on-rate matches p
    3. edge scatter: its look is redrawn per frame (two renders of the same parameters
       differ only where an edge source lies) and nothing changes with it off
    4. targets: count = all M2 spots, the map peaks at each in-field spot with height
       reflectivity^k (a rotation moves the peaks with the spots)
    5. 3-channel stem: expand_stem on a 1-channel encoder gives the same z for a gray
       image copied x 3; init_agent_encoder from a ppo.py-style checkpoint
    6. camera-mode pretraining smoke run (3 steps, eager)
"""

import math
import os
import subprocess
import sys
import tempfile
from dataclasses import fields, replace

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

from agent import Agent, AgentConfig  # noqa: E402
from encoders import CNNEncoder, EncoderConfig, FrameModel  # noqa: E402
from frame_data import DataConfig, FrameData  # noqa: E402
from geometry import SETUPS  # noqa: E402
from pretrain_encoder import camera_sim, expand_stem, init_agent_encoder  # noqa: E402
from render_camera import CameraRanges, CameraRenderer  # noqa: E402

DEV = torch.device("cuda")


def ranges_all(p: float) -> CameraRanges:
    """Every effect probability set to p, clean_p 0."""
    return replace(CameraRanges(), clean_p=0.0,
                   **{f.name: p for f in fields(CameraRanges) if f.name.endswith("_p")
                      and f.name != "clean_p"})


def test_neutral(data: FrameData, cam: CameraRenderer):
    print("\n== neutral parameters vs SpotRenderer ==")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    d = data.batch(gen, 256, mode="clean")
    hm, lxy = d["_hits"]
    img = cam.render(hm, lxy, cam.neutral_params(256), gen, d["_miss"])
    b = data.base
    r = torch.sqrt(b.xs[None, :] ** 2 + b.ys[:, None] ** 2) / b.radius
    diff = (img[:, 0].int() - d["image"][:, 0].int()).abs()[:, r < 0.97].max().item()
    same = bool((img[:, 0] == img[:, 1]).all() and (img[:, 0] == img[:, 2]).all())
    print(f"  max |diff| inside r < 0.97 R: {diff} / 255, channels equal {same}")
    assert diff <= 4 and same


def test_probabilities(cam: CameraRenderer):
    print("\n== augmentation probabilities ==")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    n = 4096
    neutral = cam.neutral_params(n)
    p = cam.sample_params(n, gen, replace(CameraRanges(), clean_p=1.0))
    for k, v in p.items():
        assert torch.equal(v, neutral[k]), f"clean_p = 1: {k} not neutral"
    print("  clean_p = 1: neutral parameters")

    p = cam.sample_params(n, gen, ranges_all(0.0))
    eye = torch.eye(3, device=DEV)
    assert bool(torch.allclose(p["warp"], eye.expand(n, 3, 3), atol=1e-6))
    for k, off in (("laser", 1.0), ("aspect", 1.0), ("speckle", 0.0), ("scatter_on", 0.0),
                   ("mono", 1.0), ("grain", 0.0), ("rim", 0.0), ("surround", 0.0),
                   ("engrave_amp", 0.0), ("scratch_amp", 0.0), ("dust_amp", 0.0),
                   ("vignetting", 0.0), ("gradient", 0.0), ("exposure", 1.0), ("gamma", 1.0),
                   ("black", 0.0), ("read_noise", 0.0), ("shot_noise", 0.0), ("prnu", 0.0),
                   ("row_noise", 0.0), ("hot_val", 0.0), ("ring_a", 0.0)):
        assert bool((p[k] == off).all()), f"p = 0: {k} on"
    assert bool((p["sigma"] == cam.base.sigma).all())
    assert bool((p["face"] <= CameraRanges().dark_face).all())
    print("  every p = 0: every effect off (face dark)")

    r = CameraRanges()
    p = cam.sample_params(n, gen, ranges_all(1.0))
    inside = lambda t, lo_hi: bool(((t >= lo_hi[0] - 1e-6) & (t <= lo_hi[1] + 1e-6)).all())
    assert not bool(torch.isclose(p["warp"], eye.expand(n, 3, 3), atol=1e-6).all((1, 2)).any())
    for k in ("sigma", "aspect", "speckle", "laser", "face", "grain", "rim", "surround", "vignetting",
              "exposure", "gamma", "black", "read_noise", "shot_noise", "prnu", "row_noise"):
        assert inside(p[k], getattr(r, k)), f"p = 1: {k} out of range"
    assert inside(p["engrave_amp"], r.engrave_amp)
    assert bool((p["scatter_on"] == 1).all()) and bool((p["mono"] == 0).all())
    assert bool((p["hot_val"][:, 0] > 0).all()) and bool((p["dust_amp"][:, 0] != 0).all())
    assert bool((p["scratch_amp"][:, 0] != 0).all()) and bool((p["ring_a"][:, 0] != 0).all())
    print("  every p = 1: every effect on, parameters inside the ranges")

    p = cam.sample_params(n, gen, replace(CameraRanges(), clean_p=0.0))
    rates = {"rotation": (p["warp"][:, 0, 1].abs() > 1e-6).float().mean().item(),
             "color": (p["mono"] == 0).float().mean().item(),
             "scatter": p["scatter_on"].mean().item(),
             "hot": (p["hot_val"][:, 0] > 0).float().mean().item()}
    want = {"rotation": None, "color": r.color_p, "scatter": r.scatter_p, "hot": r.hot_p}
    for k, v in rates.items():
        if want[k] is not None:
            assert abs(v - want[k]) < 0.03, f"{k} rate {v:.3f} vs p {want[k]}"
    # Off-diagonal warp terms: rotation or perspective (each p = 0.5) -> 1 - 0.25
    assert abs(rates["rotation"] - 0.75) < 0.03, rates
    print("  on-rates: " + ", ".join(f"{k} {v:.3f}" for k, v in rates.items()))


def test_scatter_per_frame(data: FrameData, cam: CameraRenderer):
    print("\n== edge scatter per frame ==")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(2)
    out = camera_sim(data, gen, 2048)
    hm, lxy = out["hit_mirror"], out["hit_local_xy"]
    miss = (out["miss_mirror"], out["miss_local_xy"], out["intensity"])
    p = cam.neutral_params(2048)
    _, w = cam.scatter_sources(hm, lxy, miss, p)
    edge = w[:, 0] > 0.1
    renders = []
    for on in (1.0, 0.0):
        q = {**p, "scatter_on": torch.full_like(p["scatter_on"], on)}
        for seed in (3, 4):
            g = torch.Generator(device=DEV)
            g.manual_seed(seed)
            renders.append(cam.render(hm, lxy, q, g, miss).int())
    changed = (renders[0] != renders[1]).flatten(1).any(1)
    print(f"  frames with an edge source: {int(edge.sum())}; scatter on: changed between "
          f"renders {int((changed & edge).sum())} of them, {int((changed & ~edge).sum())} others")
    assert int(edge.sum()) > 20 and bool(changed[edge].float().mean() > 0.9)
    assert torch.equal(renders[2], renders[3]), "scatter off: renders differ"
    print("  scatter off: identical renders")


def test_targets(data: FrameData, cam: CameraRenderer):
    print("\n== camera-frame targets ==")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(5)
    out = camera_sim(data, gen, 512)
    hm, lxy = out["hit_mirror"], out["hit_local_xy"]
    for name, p in (("neutral", cam.neutral_params(512)),
                    ("rotated", cam.sample_params(512, gen, replace(
                        ranges_all(0.0), rotation_p=1.0, rotation=(15.0, 20.0))))):
        t = cam.targets(hm, lxy, p)
        lab = cam.labels(hm, lxy, p)
        _, w, valid = cam.spots_camera(hm, lxy, p)
        assert torch.equal(t["count"], valid.sum(1))
        # Peaks: at each isolated in-field spot the map's nearest pixel is ~ its peak
        px = lab["pos_px"].round().long().clamp(0, cam.res - 1)
        val = t["spot_map"][:, 0].flatten(1).gather(1, px[..., 1] * cam.res + px[..., 0])
        d = torch.cdist(lab["pos_px"], lab["pos_px"]) + 1e3 * torch.eye(w.shape[1], device=DEV)
        d = d.masked_fill(~valid[:, None, :], 1e3)
        iso = lab["visible"] & (d.min(2).values > 4) & ((lab["pos_px"] - px).abs().max(-1).values < 0.25)
        rel = (val[iso] / w[iso])
        print(f"  {name}: {int(iso.sum())} isolated spots, map / peak at the spot pixel "
              f"{rel.min().item():.3f} - {rel.max().item():.3f}")
        assert int(iso.sum()) > 100 and bool(((rel > 0.85) & (rel < 1.02)).all())


def test_stem_and_agent_init():
    print("\n== 3-channel stem ==")
    torch.backends.cudnn.allow_tf32 = False  # Exact fp32 convolutions for the comparison
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    cfg1, cfg3 = EncoderConfig(), EncoderConfig(in_ch=3)
    e1 = CNNEncoder(cfg1).to(DEV).eval()
    e3 = CNNEncoder(cfg3).to(DEV).eval()
    e3.load_state_dict(expand_stem(e1.state_dict(), 3))
    img = torch.randint(0, 256, (8, 1, 64, 64), dtype=torch.uint8, device=DEV)
    with torch.no_grad():
        z1, z3 = e1(img), e3(img.expand(-1, 3, -1, -1).contiguous())
    err = (z1 - z3).abs().max().item()
    print(f"  expand_stem: max |z diff| {err:.2e} (gray x 3 vs gray)")
    assert err < 1e-4

    agent = Agent(AgentConfig()).to(DEV)
    ck = {"agent": agent.state_dict(), "agent_cfg": agent.cfg.to_dict(), "name": "test",
          "iteration": 7, "env_steps": 70}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "best.pt")
        torch.save(ck, path)
        model = FrameModel(cfg3).to(DEV).eval()
        info = init_agent_encoder(model, path)
    agent.encoder.eval()
    with torch.no_grad():
        za = agent.encoder(img)
        zm = model.encoder(img.expand(-1, 3, -1, -1).contiguous())
    err = (za - zm).abs().max().item()
    print(f"  init_agent_encoder: {info['tensors']} tensors, max |z diff| {err:.2e}")
    assert err < 1e-4


def test_pretrain_smoke():
    print("\n== camera-mode pretraining smoke run ==")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, os.path.join(RL_DIR, "pretrain_encoder.py"), "--camera",
               "--steps", "3", "--batch", "32", "--val-n", "64", "--val-every", "3",
               "--log-every", "1", "--no-compile", "--render-chunk", "32", "--runs-dir", tmp]
        res = subprocess.run(cmd, cwd=RL_DIR, capture_output=True, text=True, timeout=300)
        tail = "\n".join(res.stdout.strip().splitlines()[-3:])
        print("  " + tail.replace("\n", "\n  "))
        assert res.returncode == 0, res.stderr[-2000:]
        runs = os.listdir(tmp)
        assert len(runs) == 1 and runs[0].endswith("_cam"), runs
        ck = torch.load(os.path.join(tmp, runs[0], "model.pt"), map_location="cpu", weights_only=False)
        assert ck["cfg"]["in_ch"] == 3 and ck["camera"] is not None
        assert tuple(ck["model"]["encoder.stem.weight"].shape)[1] == 3
        print(f"  run {runs[0]}: model.pt with in_ch 3 and the camera config")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    data = FrameData(DataConfig(setups=SETUPS), DEV)
    cam = CameraRenderer(data.base)
    test_neutral(data, cam)
    test_probabilities(cam)
    test_scatter_per_frame(data, cam)
    test_targets(data, cam)
    test_stem_and_agent_init()
    test_pretrain_smoke()
    print("\nall tests passed")
