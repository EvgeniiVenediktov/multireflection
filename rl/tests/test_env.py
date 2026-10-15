"""
tests/test_env.py - AlignEnv, SpotRenderer and HistoryStack checks

Run from rl/:  uv run python tests/test_env.py

    1. action ladder; (0, 0) terminates with reward 0 and leaves the tilt unchanged
    2. clamping at +-2 deg and the recorded applied delta after clamping
    3. reward telescoping over full episodes with random actions
    4. truncation at max_steps; termination and truncation mutually exclusive
    5. auto-reset touches only done envs; reset tilts inside the configured box
    6. info is pre-reset (final frame, path, stats of the finished episode)
    7. renderer vs a dense Gaussian reference; orientation; 16 spots on a ring at nominal
    8. HistoryStack ordering, masking after reset, peek_with without mutation
    9. no host syncs in step() / reset() (torch.cuda.set_sync_debug_mode("error"))
   10. figure figures/env_renders.png (nominal, 57, 89, trapped, 1-3 bounce states)
"""

import math
import os
import sys

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from env import AlignEnv, EnvConfig, HistoryStack  # noqa: E402
from geometry import NOMINAL_PATH_MM, CellGeometry  # noqa: E402
from herriott_sim import STOP_CAP, STOP_HOLE, STOP_NAMES  # noqa: E402

DEV = torch.device("cuda")

# Named M2 tilts (deg) of the landscape (README)
STATES = {
    "nominal": (-0.38, 0.27),
    "57 island": (-0.55, 0.62),
    "89 island": (0.17, 0.97),
    "trapped": (0.10, 1.04),
    "1-3 bounce": (-1.8, -1.8),
}


# ── Helpers ─────────────────────────────────────────────────────────────────


def tilts(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32, device=DEV)


def random_actions(env: AlignEnv, gen: torch.Generator, p_stop: float) -> torch.Tensor:
    """Uniform ladder index per axis; (0, 0) forced with probability p_stop."""
    B = env.num_envs
    a = torch.randint(0, env.n_bins, (B, 2), generator=gen, device=DEV)
    stop = torch.rand(B, generator=gen, device=DEV) < p_stop
    return torch.where(stop[:, None], 0, a)


def dense_reference(env: AlignEnv, out: dict) -> torch.Tensor:
    """Exact Gaussian sum per pixel (fp64), weight reflectivity^k on M2 hits, aperture."""
    r = env.renderer
    hm, lxy = out["hit_mirror"], out["hit_local_xy"].double()
    k = torch.arange(hm.shape[1], device=DEV, dtype=torch.float64)
    w = torch.where(hm == 2, env.cfg.geometry.reflectivity**k, 0.0)
    xs, ys = r.xs.double(), r.ys.double()
    dx2 = (xs[None, None, :] - lxy[..., 0, None]) ** 2  # (n, N, W)
    dy2 = (ys[None, None, :] - lxy[..., 1, None]) ** 2  # (n, N, H)
    s2 = 2 * r.sigma**2
    img = torch.einsum("nk,nki,nkj->nij", w, torch.exp(-dy2 / s2), torch.exp(-dx2 / s2))
    return img * r.aperture.double()


# ── Tests ───────────────────────────────────────────────────────────────────


def test_ladder():
    print("\n== 1. action ladder, not move ==")
    env = AlignEnv(EnvConfig(num_envs=22, seed=1, history_len=4))
    want = tilts(0, 0.01, -0.01, 0.03, -0.03, 0.1, -0.1, 0.3, -0.3, 1.0, -1.0)
    assert env.n_bins == 11 and torch.allclose(env.ladder, want), env.ladder
    assert env.max_step == 1.0

    start = tilts([0.2, -0.3]).expand(22, 2)
    obs0 = env.reset(tilts=start)
    path0 = env.path.clone()
    i = torch.arange(11, device=DEV)
    a = torch.zeros(22, 2, dtype=torch.long, device=DEV)
    a[:11, 0] = i  # pitch index i, yaw 0
    a[11:, 1] = i  # pitch 0, yaw index i
    _, rew, term, trunc, info = env.step(a)

    delta = info["tilt"] - start
    want_delta = torch.zeros(22, 2, device=DEV)
    want_delta[:11, 0] = want
    want_delta[11:, 1] = want
    assert torch.allclose(delta, want_delta, atol=1e-6), delta
    assert torch.allclose(info["applied_action"], want_delta / env.max_step, atol=1e-6)

    still = torch.tensor([0, 11], device=DEV)
    moved = torch.ones(22, dtype=torch.bool, device=DEV)
    moved[still] = False
    assert bool(term[still].all()) and not bool(term[moved].any())
    assert not bool(trunc.any())
    assert bool((rew[still] == 0).all()), rew[still]  # exactly 0
    assert torch.equal(info["tilt"][still], start[still])  # bit-identical
    assert torch.equal(info["path_mm"][still], path0[still])
    assert torch.equal(info["final_image"][still], obs0["image"][still])
    assert torch.allclose(rew[moved], (info["path_mm"] - path0)[moved] / NOMINAL_PATH_MM - 0.01)
    assert bool((env.step_count[still] == 0).all()) and bool((env.step_count[moved] == 1).all())
    print("  ladder (deg):", [round(v, 2) for v in env.ladder.tolist()])
    print("  per-axis deltas and applied actions match; (0, 0): terminated, reward 0.0, "
          "tilt / path / image unchanged")


def test_clamp():
    print("\n== 2. clamping at the tilt limits ==")
    env = AlignEnv(EnvConfig(num_envs=3, history_len=0))
    assert env.limit == 2.0
    env.reset(tilts=tilts([1.95, -1.95], [1.95, -1.95], [-1.999, 0.0]))
    a = torch.tensor([[9, 10], [3, 2], [10, 0]], device=DEV)  # (+1, -1), (+0.03, -0.01), (-1, 0)
    _, rew, term, _, info = env.step(a)
    want_tilt = tilts([2.0, -2.0], [1.98, -1.96], [-2.0, 0.0])
    want_applied = tilts([0.05, -0.05], [0.03, -0.01], [-0.001, 0.0])
    assert torch.allclose(info["tilt"], want_tilt, atol=1e-6), info["tilt"]
    assert torch.allclose(info["applied_action"], want_applied, atol=1e-6), info["applied_action"]
    assert not bool(term.any())

    # Pushing further against the limit: no motion, a move all the same (cost, no stop)
    path_before = env.path.clone()
    _, rew, term, trunc, info = env.step(torch.tensor([[9, 10], [0, 0], [10, 0]], device=DEV))
    assert torch.equal(info["tilt"][[0, 2]], want_tilt[[0, 2]])
    assert bool((info["applied_action"][[0, 2]] == 0).all())
    assert torch.equal(info["path_mm"][[0, 2]], path_before[[0, 2]])
    assert torch.allclose(rew[[0, 2]], torch.full((2,), -0.01, device=DEV))
    assert not bool(term[[0, 2]].any()) and bool(term[1])
    print("  (1.95, -1.95) + (1, -1) -> (2.0, -2.0), applied (0.05, -0.05); "
          "(-1.999) - 1 -> -2.0, applied -0.001")
    print("  stepping against the limit: tilt and path unchanged, reward -step_cost, "
          "not terminated")


def test_telescoping():
    print("\n== 3. reward telescoping ==")
    for name, center, radius in [("full box", (0.0, 0.0), 2.0), ("near nominal", (-0.38, 0.27), 0.6)]:
        env = AlignEnv(EnvConfig(num_envs=256, seed=3, init_center=center, init_radius=radius))
        gen = torch.Generator(device=DEV)
        gen.manual_seed(7)
        c = env.cfg.step_cost
        p0 = env.path.clone()
        ret = torch.zeros(256, device=DEV, dtype=torch.float64)
        moves = torch.zeros(256, dtype=torch.long, device=DEV)
        errs, n_ep, n_term, n_trunc, max_gain = [], 0, 0, 0, 0.0
        for _ in range(300):
            a = random_actions(env, gen, p_stop=0.03)
            _, rew, term, trunc, info = env.step(a)
            assert not bool((term & trunc).any()), "terminated and truncated together"
            ret += rew.double()
            moves += (a != 0).any(1).long()
            done = term | trunc
            if bool(done.any()):
                d = done
                tele = (info["ep_final_path_mm"][d].double() - p0[d].double()) / NOMINAL_PATH_MM
                tele = tele - c * moves[d].double()
                errs.append((ret[d] - tele).abs().max().item())
                errs.append((info["ep_return"][d].double() - tele).abs().max().item())
                assert torch.equal(info["ep_initial_path_mm"][d], p0[d])
                assert torch.equal(info["ep_moves"][d], moves[d])
                assert torch.equal(info["ep_length"][d], moves[d] + term[d].long())
                n_ep += int(d.sum())
                n_term += int(term.sum())
                n_trunc += int(trunc.sum())
                max_gain = max(max_gain, float(tele.max()))
            ret = torch.where(done, 0.0, ret)
            moves = torch.where(done, 0, moves)
            p0 = torch.where(done, env.path, p0)
        err = max(errs)
        print(f"  {name}: {n_ep} episodes ({n_term} terminated, {n_trunc} truncated), "
              f"max |sum r - telescoped| {err:.2e}, best return {max_gain:+.3f}")
        assert err < 1e-4, err


def test_truncation():
    print("\n== 4. truncation ==")
    env = AlignEnv(EnvConfig(num_envs=8, max_steps=5, history_len=3))
    env.reset(tilts=tilts([-0.38, 0.27]).expand(8, 2))
    up = torch.tensor([[1, 0]], device=DEV).expand(8, 2)  # +0.01 deg pitch
    for t in range(1, 5):
        _, _, term, trunc, info = env.step(up)
        assert not bool((term | trunc).any()) and bool((env.step_count == t).all())
    # Step 5: envs 0-3 move (truncated), envs 4-7 not move (terminated, not truncated)
    a = up.clone()
    a[4:] = 0
    _, rew, term, trunc, info = env.step(a)
    assert bool(trunc[:4].all()) and not bool(term[:4].any())
    assert bool(term[4:].all()) and not bool(trunc[4:].any())
    assert bool((info["ep_length"] == 5).all()) and bool((env.step_count == 0).all())
    assert bool((rew[4:] == 0).all())
    print("  step 5 of 5: move -> truncated, (0, 0) -> terminated only; counters reset")


def test_auto_reset():
    print("\n== 5. auto-reset isolation and reset box ==")
    B = 64
    env = AlignEnv(EnvConfig(num_envs=B, seed=5, history_len=4))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(11)
    nonzero = lambda: torch.randint(1, env.n_bins, (B, 2), generator=gen, device=DEV)  # noqa: E731
    for _ in range(3):
        env.step(nonzero())
    for center, radius in [((0.5, -0.5), 0.3), ((1.9, -1.9), 0.3)]:
        env.set_init_box(center, radius)
        before = {k: getattr(env, k).clone() for k in ("tilt", "step_count", "path0", "ep_return")}
        a = nonzero()
        stop = torch.arange(B, device=DEV) % 2 == 0
        a[stop] = 0
        obs, rew, term, trunc, info = env.step(a)
        done = term | trunc
        assert torch.equal(done, stop)
        keep = ~done
        want = (before["tilt"] + env.ladder[a]).clamp(-2, 2)
        assert torch.equal(env.tilt[keep], want[keep])
        assert torch.equal(env.step_count[keep], before["step_count"][keep] + 1)
        assert torch.equal(env.path0[keep], before["path0"][keep])
        assert torch.allclose(env.ep_return[keep], (before["ep_return"] + rew)[keep])
        assert torch.equal(obs["image"][keep], info["final_image"][keep])
        assert torch.equal(obs["images"][keep], info["final_obs"]["images"][keep])
        # Reset envs: in the box (clamped to the limits), fresh counters and history
        lo = (torch.tensor(center, device=DEV) - radius).clamp(-2, 2)
        hi = (torch.tensor(center, device=DEV) + radius).clamp(-2, 2)
        t = env.tilt[done]
        assert bool(((t >= lo) & (t <= hi)).all()), (t.min(0), t.max(0))
        assert bool((env.step_count[done] == 0).all()) and bool((env.ep_return[done] == 0).all())
        assert torch.equal(env.path0[done], env.path[done])
        assert bool((obs["action"][done] == 0).all())
        assert bool((obs["mask"][done].sum(1) == 1).all()) and bool(obs["mask"][done][:, -1].all())
        assert bool((obs["images"][done][:, :-1] == 0).all())
        print(f"  box center {center} radius {radius}: {int(keep.sum())} running envs "
              f"untouched, reset tilts in [{t.min(0).values.tolist()}, "
              f"{t.max(0).values.tolist()}]")


def test_info_pre_reset():
    print("\n== 6. info is pre-reset ==")
    B = 256
    env = AlignEnv(EnvConfig(num_envs=B, seed=9, max_steps=8, init_center=(-0.38, 0.27),
                             init_radius=0.8))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(13)
    n_done, max_img, n_img_diff, n_pix = 0, 0, 0, 0
    for _ in range(40):
        obs, _, term, trunc, info = env.step(random_actions(env, gen, p_stop=0.2))
        done = term | trunc
        ref = env.render_tilts(info["tilt"])
        assert torch.equal(info["path_mm"], ref["path_mm"])
        assert torch.equal(info["bounces"], ref["bounces"])
        assert torch.equal(info["stop_reason"], ref["stop_reason"])
        diff = (info["final_image"].int() - ref["image"].int()).abs()
        max_img = max(max_img, int(diff.max()))
        n_img_diff += int((diff > 0).sum())
        n_pix += diff.numel()
        assert torch.allclose(info["path_norm"], info["path_mm"] / NOMINAL_PATH_MM)
        assert torch.equal(info["ep_final_path_mm"], info["path_mm"])
        assert torch.equal(info["ep_final_bounces"], info["bounces"])
        fo = info["final_obs"]
        assert torch.equal(fo["images"][:, -1], info["final_image"])
        assert torch.equal(fo["actions"][:, -1], info["applied_action"])
        assert bool(fo["mask"][:, -1].all())
        # Returned obs: the new episode's first frame where done
        new = env.render_tilts(env.tilt)
        assert int((obs["image"].int() - new["image"].int()).abs().max()) <= 1
        assert torch.equal(info["obs_path_norm"], new["path_mm"] / NOMINAL_PATH_MM)
        assert torch.equal(info["obs_tilt"], env.tilt)
        if bool(done.any()):
            assert not torch.equal(info["tilt"][done], env.tilt[done])
        n_done += int(done.sum())
    print(f"  40 steps, {n_done} episode ends: path / bounces / stop reason identical to a "
          f"fresh sim of info['tilt']; final_image vs fresh render: max |diff| {max_img} "
          f"({n_img_diff} of {n_pix} pixels differ, atomic-add order)")
    assert max_img <= 1


def test_renderer(geom: CellGeometry):
    print("\n== 7. renderer vs dense Gaussian reference ==")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(17)
    named = tilts(*STATES.values())
    rand = torch.tensor([-0.38, 0.27], device=DEV) + 0.8 * (
        2 * torch.rand(59, 2, generator=gen, device=DEV) - 1
    )
    all_tilts = torch.cat((named, rand))
    for res in (64, 128):
        env = AlignEnv(EnvConfig(num_envs=4, img_res=res, history_len=0))
        r = env.renderer
        out = env.render_tilts(all_tilts)
        ref = dense_reference(env, out["sim"])
        peak = ref.amax((1, 2))
        has = peak > 0
        err = ((out["intensity"].double() - ref).abs().amax((1, 2)) / peak)[has]
        spots = (out["sim"]["hit_mirror"] == 2).sum(1)
        print(f"  {res} px (oversample {r.oversample}, grid {r.n_fine}, kernel "
              f"{2 * r.half + 1} taps): {int(has.sum())} images with spots, M2 spots per "
              f"image {int(spots.min())}-{int(spots.max())}; max |splat+blur - dense| / peak "
              f"{float(err.max()):.4f} (mean {float(err.mean()):.4f})")
        assert float(err.max()) < 0.05, float(err.max())
        want = torch.round(255 * (1 - torch.exp(-env.cfg.gain * out["intensity"])))
        assert int((out["image"][:, 0].float() - want).abs().max()) <= 1

        # Orientation: brightest pixel of the nominal image = first M2 hit (k = 0)
        lx, ly = out["sim"]["hit_local_xy"][0, 0].tolist()
        img = out["intensity"][0]
        i, j = divmod(int(img.argmax()), res)
        want_j = int((lx + r.radius) / r.pitch)
        want_i = int((r.radius - ly) / r.pitch)
        assert abs(i - want_i) <= 1 and abs(j - want_j) <= 1, (i, j, want_i, want_j)
        print(f"    nominal first spot at local ({lx:.2f}, {ly:.2f}) mm -> row {i}, col {j} "
              f"(expected {want_i}, {want_j}); uint8 peak {int(out['image'][0].max())}")

        # 16 distinct spots on a ring at the nominal state
        pooled = F.max_pool2d(img[None, None], 3, stride=1, padding=1)[0, 0]
        peaks = (img == pooled) & (img > 0.3 * float(img.max()))
        ii, jj = torch.nonzero(peaks, as_tuple=True)
        rad = torch.sqrt(r.xs[jj] ** 2 + r.ys[ii] ** 2)
        print(f"    nominal: {int(peaks.sum())} local maxima, radii "
              f"{float(rad.min()):.2f}-{float(rad.max()):.2f} mm")
        assert int(peaks.sum()) == 16
        assert float(rad.min()) > 7.0 and float(rad.max()) < 11.0


def test_history():
    print("\n== 8. HistoryStack ==")
    h = HistoryStack(2, 3, (1, 2, 2), DEV)
    frame = lambda v: torch.full((2, 1, 2, 2), v, dtype=torch.uint8, device=DEV)  # noqa: E731
    act = lambda v: torch.tensor([[v, -v]] * 2, dtype=torch.float32, device=DEV)  # noqa: E731
    no = torch.zeros(2, dtype=torch.bool, device=DEV)

    h.reset(frame(1))
    im, ac, m = h.get()
    assert m.tolist() == [[False, False, True]] * 2
    assert im[:, :, 0, 0, 0].tolist() == [[0, 0, 1]] * 2 and bool((ac == 0).all())
    h.push(frame(2), act(2), no)
    assert h.get()[2].tolist() == [[False, True, True]] * 2
    h.push(frame(3), act(3), no)
    h.push(frame(4), act(4), no)
    im, ac, m = h.get()
    assert im[:, :, 0, 0, 0].tolist() == [[2, 3, 4]] * 2  # oldest first
    assert ac[:, :, 0].tolist() == [[2, 3, 4]] * 2 and ac[:, :, 1].tolist() == [[-2, -3, -4]] * 2
    assert bool(m.all())

    held = [x.clone() for x in (im, ac, m)]
    h.push(frame(5), act(5), torch.tensor([True, False], device=DEV))
    assert all(torch.equal(a, b) for a, b in zip(held, (im, ac, m))), "get() result mutated"
    im, ac, m = h.get()
    assert im[:, :, 0, 0, 0].tolist() == [[0, 0, 5], [3, 4, 5]]
    assert ac[:, :, 0].tolist() == [[0, 0, 0], [3, 4, 5]]
    assert m.tolist() == [[False, False, True], [True, True, True]]

    before = [x.clone() for x in h.get()]
    pi, pa, pm = h.peek_with(frame(6), act(6))
    assert pi[:, :, 0, 0, 0].tolist() == [[0, 5, 6], [4, 5, 6]]
    assert pa[:, :, 0].tolist() == [[0, 0, 6], [4, 5, 6]]
    assert pm.tolist() == [[False, True, True], [True, True, True]]
    assert all(torch.equal(a, b) for a, b in zip(before, h.get())), "peek_with mutated"

    h.reset(frame(7), torch.tensor([False, True], device=DEV))
    im, ac, m = h.get()
    assert im[:, :, 0, 0, 0].tolist() == [[0, 0, 5], [0, 0, 7]]
    assert m.tolist() == [[False, False, True], [False, False, True]]
    print("  oldest-first order, reset clears and masks, push / peek_with leave earlier "
          "results unchanged, partial reset")


def test_no_sync():
    print("\n== 9. no host syncs ==")
    env = AlignEnv(EnvConfig(num_envs=256, seed=21))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(23)
    acts = [random_actions(env, gen, p_stop=0.1) for _ in range(13)]
    mask = torch.rand(256, generator=gen, device=DEV) < 0.5
    for a in acts[:3]:
        env.step(a)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for a in acts[3:]:
            obs, rew, term, trunc, info = env.step(a)
        env.reset(mask)
        env.step(acts[0])
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    print("  11 steps and a masked reset under set_sync_debug_mode('error'): no sync")


def save_figure(path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    print(f"\n== 10. figure {os.path.relpath(path, RL_DIR)} ==")
    names = list(STATES)
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 7.6),
                             layout="constrained")
    for row, res in enumerate((64, 128)):
        env = AlignEnv(EnvConfig(num_envs=4, img_res=res, history_len=0))
        out = env.render_tilts(tilts(*STATES.values()))
        R = env.renderer.radius
        for col, name in enumerate(names):
            ax = axes[row, col]
            p, y = STATES[name]
            n = int(out["bounces"][col])
            stop = STOP_NAMES[int(out["stop_reason"][col])]
            ax.imshow(out["image"][col, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=255,
                      origin="upper", extent=(-R, R, -R, R), interpolation="nearest")
            ax.add_patch(Circle((0, 0), R, fill=False, color="0.45", lw=0.6))
            ax.set_title(f"{name}, {res} px\nM2 ({p:+.2f}, {y:+.2f}) deg\n"
                         f"{n} bounces, {float(out['path_mm'][col]) / 1000:.2f} m, {stop}",
                         fontsize=8.5)
            ax.tick_params(labelsize=7)
            if row == 1:
                ax.set_xlabel("M2 local x (mm)", fontsize=8)
            if col == 0:
                ax.set_ylabel("M2 local y (mm)", fontsize=8)
            print(f"  {res} px {name}: {n} bounces, {float(out['path_mm'][col]):.1f} mm, {stop}")
            if row == 0 and name == "nominal":
                assert n == 31 and int(out["stop_reason"][col]) == STOP_HOLE
            if row == 0 and name == "trapped":
                assert n == 128 and int(out["stop_reason"][col]) == STOP_CAP
    fig.suptitle("AlignEnv observations: M2 face seen from behind (+x right, +y up), "
                 "uint8, gain 1.6", fontsize=10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    geom = CellGeometry()
    print(f"device {torch.cuda.get_device_name()}")
    test_ladder()
    test_clamp()
    test_telescoping()
    test_truncation()
    test_auto_reset()
    test_info_pre_reset()
    test_renderer(geom)
    test_history()
    test_no_sync()
    save_figure(os.path.join(RL_DIR, "figures", "env_renders.png"))
    print("\nall tests passed")
