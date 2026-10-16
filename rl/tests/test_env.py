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
   11. geometry.SETUPS: measured first and unchanged, nominal / target regression values
   12. multi-setup sampling per env (frequencies, weights), per-env base states
   13. per-setup normalization (P_ref = P_max per setup) and reward telescoping
   14. info["setup"] / obs_setup around auto-resets; reset draws deterministic per
       (seed, env, episode), Triton hash = torch reference
   15. auto_reset=False (idle envs), reset(return_info=True), partial reset isolation
   16. no host syncs with 4 setups and without auto-reset
"""

import math
import os
import sys

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from env import AlignEnv, EnvConfig, HistoryStack, uniforms, uniforms_torch  # noqa: E402
from geometry import NOMINAL_PATH_MM, SETUP_INFO, SETUPS, CellGeometry  # noqa: E402
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


# ── Multi-setup tests ───────────────────────────────────────────────────────


def multi_env(**kw) -> AlignEnv:
    return AlignEnv(EnvConfig(setups=SETUPS, **kw))


def test_setups():
    print("\n== 11. geometry.SETUPS ==")
    assert SETUPS[0] == CellGeometry() and SETUPS[0].name == "measured"
    assert len({g.name for g in SETUPS}) == len(SETUPS) == 4
    env = multi_env(num_envs=4, history_len=0)
    assert env.n_setups == 4 and env.path_ref_mode == "max"
    s = torch.arange(4, device=DEV)
    nom = tilts(*[(g.m2_pitch, g.m2_yaw) for g in SETUPS])
    tgt = tilts(*[SETUP_INFO[g.name].target for g in SETUPS])
    on, ot = env.render_tilts(nom, s), env.render_tilts(tgt, s)
    pmax = env.path_max()
    assert torch.equal(env.path_refs, pmax)
    for i, g in enumerate(SETUPS):
        ref = SETUP_INFO[g.name]
        assert int(on["bounces"][i]) == ref.plateau_bounces, (g.name, int(on["bounces"][i]))
        assert int(on["stop_reason"][i]) == STOP_HOLE
        assert abs(float(on["path_mm"][i]) - ref.plateau_path_mm) < 0.01
        assert int(ot["bounces"][i]) == ref.target_bounces
        assert abs(float(pmax[i]) - ref.pmax_mm) < 0.01, (g.name, float(pmax[i]))
        assert float(ot["path_mm"][i]) <= float(pmax[i])
        print(f"  {g.name:8s} {g.separation:5.0f} mm, laser ({g.laser_pitch:.2f}, "
              f"{g.laser_yaw:.2f}): nominal ({g.m2_pitch:+.2f}, {g.m2_yaw:+.2f}) "
              f"{int(on['bounces'][i])} bounces {float(on['path_mm'][i]) / 1000:.2f} m (hole); "
              f"P_max {float(pmax[i]) / 1000:.2f} m, target {ref.target} "
              f"{int(ot['bounces'][i])} bounces")
    # path_ref "nominal" with several setups: the nominal paths, exact P_nom for measured
    envn = multi_env(num_envs=4, history_len=0, path_ref="nominal")
    assert envn.p_nom == NOMINAL_PATH_MM and float(envn.path_refs[0]) == float(
        torch.tensor(NOMINAL_PATH_MM, dtype=torch.float32))
    assert torch.allclose(envn.path_refs[1:], on["path_mm"][1:])
    # One setup with path_ref "max": P_ref = P_max of the measured cell
    env1 = AlignEnv(EnvConfig(num_envs=4, history_len=0, path_ref="max"))
    assert env1.p_nom == float(pmax[0])
    print(f"  path_ref nominal: {[round(v, 1) for v in envn.path_refs.tolist()]} mm; one setup "
          f"with path_ref max: P_ref {env1.p_nom:.1f} mm")


def test_setup_sampling():
    print("\n== 12. setup sampling per env ==")
    B = 4096
    env = multi_env(num_envs=B, history_len=0, seed=31)
    counts = torch.bincount(env.setup, minlength=4).double()
    sd = math.sqrt(B * 0.25 * 0.75)
    assert bool(((counts - B / 4).abs() < 4 * sd).all()), counts
    t = env.tilt
    assert bool((t.abs() <= 2.0).all()) and bool((t.mean(0).abs() < 0.1).all())
    # Per-env base states: env path = a fresh sim of (tilt, setup)
    ref = env.render_tilts(env.tilt, env.setup)
    assert torch.equal(env.path, ref["path_mm"]) and torch.equal(env.bounces, ref["bounces"])
    print(f"  uniform: counts {counts.long().tolist()} (expected {B // 4} +- {sd:.0f}); "
          f"tilt mean {[round(v, 3) for v in t.mean(0).tolist()]}; path = fresh sim of "
          "(tilt, setup) for every env")

    envw = multi_env(num_envs=B, history_len=0, seed=32, setup_weights=(0.0, 1.0, 0.0, 3.0))
    cw = torch.bincount(envw.setup, minlength=4)
    assert int(cw[0]) == 0 and int(cw[2]) == 0
    f3 = float(cw[3]) / B
    assert abs(f3 - 0.75) < 4 * math.sqrt(0.75 * 0.25 / B), f3
    print(f"  weights (0, 1, 0, 3): counts {cw.tolist()}")

    # Setups change per env at auto-reset: run with stops and count episodes per setup
    env = multi_env(num_envs=256, history_len=0, seed=33)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(34)
    n_change, n_done = 0, 0
    for _ in range(40):
        before = env.setup.clone()
        _, _, term, trunc, info = env.step(random_actions(env, gen, p_stop=0.3))
        done = term | trunc
        assert torch.equal(info["setup"], before), "info['setup'] is the pre-reset setup"
        assert torch.equal(env.setup[~done], before[~done]), "setup changes only at reset"
        assert torch.equal(info["obs_setup"], env.setup)
        n_change += int((env.setup != before).sum())
        n_done += int(done.sum())
        ref = env.render_tilts(info["tilt"], info["setup"])
        assert torch.equal(info["path_mm"], ref["path_mm"])
        new = env.render_tilts(env.tilt, env.setup)
        assert torch.equal(env.path, new["path_mm"])
    assert n_change > 0.6 * n_done, (n_change, n_done)
    print(f"  40 steps x 256 envs: {n_done} resets, setup changed in {n_change} "
          f"(expected ~75 %); post-step and post-reset paths = fresh sim of (tilt, setup)")


def test_multi_telescoping():
    print("\n== 13. per-setup normalization and telescoping ==")
    env = multi_env(num_envs=512, seed=41)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(42)
    c = env.cfg.step_cost
    refs = env.path_refs
    p0, s0 = env.path.clone(), env.setup.clone()
    ret = torch.zeros(512, device=DEV, dtype=torch.float64)
    moves = torch.zeros(512, dtype=torch.long, device=DEV)
    errs, n_ep, max_norm = [], 0, 0.0
    per_setup = torch.zeros(4, dtype=torch.long)
    for _ in range(200):
        a = random_actions(env, gen, p_stop=0.05)
        _, rew, term, trunc, info = env.step(a)
        assert torch.equal(info["path_ref"], refs[info["setup"]])
        assert torch.equal(info["path_norm"], info["path_mm"] / info["path_ref"])
        assert torch.equal(info["obs_path_norm"], env.path / refs[env.setup])
        mv = (a != 0).any(1)
        max_norm = max(max_norm, float(info["path_norm"].max()))
        ret += rew.double()
        moves += mv.long()
        done = term | trunc
        if bool(done.any()):
            d = done
            assert torch.equal(info["setup"][d], s0[d])
            tele = (info["ep_final_path_mm"][d].double() - p0[d].double()) / refs[s0[d]].double()
            tele = tele - c * moves[d].double()
            errs.append(float((ret[d] - tele).abs().max()))
            errs.append(float((info["ep_return"][d].double() - tele).abs().max()))
            per_setup += torch.bincount(s0[d], minlength=4).cpu()
            n_ep += int(d.sum())
        ret = torch.where(done, 0.0, ret)
        moves = torch.where(done, 0, moves)
        p0 = torch.where(done, env.path, p0)
        s0 = torch.where(done, env.setup, s0)
    err = max(errs)
    print(f"  {n_ep} episodes (per setup {per_setup.tolist()}): max |sum r - (P_final - P_0) "
          f"/ P_max(setup) + c moves| {err:.2e}; max path / P_ref over visited states "
          f"{max_norm:.5f}")
    assert err < 1e-4 and max_norm < 1.001


def test_reset_draws():
    print("\n== 14. reset draws ==")
    key = torch.randint(0, 2**32, (10000,), device=DEV)
    ep = torch.randint(-1, 10**7, (10000,), device=DEV)
    assert torch.equal(uniforms(key, ep), uniforms_torch(key, ep))
    assert torch.equal(uniforms(key.cpu(), ep.cpu()), uniforms_torch(key, ep).cpu())

    # Episode j of env i starts at the same (tilt, setup) whatever the other envs do
    starts = []
    for run, p_stop in enumerate((0.5, 0.1)):
        env = multi_env(num_envs=16, seed=51, history_len=0)
        gen = torch.Generator(device=DEV)
        gen.manual_seed(60 + run)
        rec = {(i, 0): (tuple(env.tilt[i].tolist()), int(env.setup[i])) for i in range(16)}
        for _ in range(60):
            env.step(random_actions(env, gen, p_stop=p_stop))
            for i in range(16):
                rec.setdefault((i, int(env.episode[i])), (tuple(env.tilt[i].tolist()),
                                                            int(env.setup[i])))
        starts.append(rec)
    common = set(starts[0]) & set(starts[1])
    assert len(common) > 16 and all(starts[0][k] == starts[1][k] for k in common)
    # Same as an explicit reset of all envs at that episode index
    env = multi_env(num_envs=16, seed=51, history_len=0)
    t1, s1 = env._draw(torch.full((16,), 3, dtype=torch.long, device=DEV))
    for i in range(16):
        if (i, 3) in starts[0]:
            assert starts[0][(i, 3)] == (tuple(t1[i].tolist()), int(s1[i]))
    other = multi_env(num_envs=16, seed=52, history_len=0)
    assert not torch.equal(other.tilt, multi_env(num_envs=16, seed=51, history_len=0).tilt)
    print(f"  Triton hash = torch reference (10k keys); {len(common)} (env, episode) starts "
          "identical across two runs with different stop rates; seed changes the draws")


def test_no_auto_reset():
    print("\n== 15. auto_reset=False, reset info, partial reset ==")
    B = 64
    env = multi_env(num_envs=B, seed=61, auto_reset=False, history_len=4)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(62)
    obs, info = env.reset(return_info=True)
    assert bool(info["reset"].all()) and torch.equal(info["setup"], env.setup)
    assert torch.equal(info["path_norm"], env.path / env.path_refs[env.setup])
    assert bool((info["step"] == 0).all()) and not bool(info["idle"].any())
    ever_done = torch.zeros(B, dtype=torch.bool, device=DEV)
    for t in range(12):
        a = random_actions(env, gen, p_stop=0.06)
        was_idle = env.idle.clone()
        snap = {k: getattr(env, k).clone() for k in ("tilt", "path", "image", "step_count",
                                                       "ep_return")}
        hist = env.history.get()[0].clone()
        obs, rew, term, trunc, info = env.step(a)
        done = term | trunc
        assert torch.equal(info["idle"], was_idle)
        assert not bool(done[was_idle].any()) and bool((rew[was_idle] == 0).all())
        for k, v in snap.items():
            assert torch.equal(getattr(env, k)[was_idle], v[was_idle]), k
        assert torch.equal(env.history.get()[0][was_idle], hist[was_idle])
        # Done envs: state is the final frame, obs = final obs
        assert torch.equal(obs["image"][done], info["final_image"][done])
        assert torch.equal(obs["images"][done], info["final_obs"]["images"][done])
        assert torch.equal(env.idle, was_idle | done)
        ever_done |= done
    n_idle = int(env.idle.sum())
    assert 8 <= n_idle <= B - 8, n_idle
    # Partial reset: only the idle envs restart; the running ones are untouched
    run = ~env.idle
    snap = {k: getattr(env, k).clone() for k in ("tilt", "setup", "path", "image", "episode")}
    hist = env.history.get()[0].clone()
    obs, info = env.reset(env.idle.clone(), return_info=True)
    for k, v in snap.items():
        assert torch.equal(getattr(env, k)[run], v[run]), k
    assert torch.equal(env.history.get()[0][run], hist[run])
    m = info["reset"]
    assert not bool(env.idle.any()) and bool((env.step_count[m] == 0).all())
    assert torch.equal(env.episode[m], snap["episode"][m] + 1)
    fresh = env.render_tilts(env.tilt, env.setup)
    assert torch.equal(env.path, fresh["path_mm"])
    assert int((obs["image"][m].int() - fresh["image"][m].int()).abs().max()) <= 1
    print(f"  12 steps of {B} envs: {int(ever_done.sum())} episodes ended and idled (action "
          f"ignored, reward 0, state, history frozen); reset of the {n_idle} idle envs leaves "
          f"the {int(run.sum())} running ones untouched")


def test_no_sync_multi():
    print("\n== 16. no host syncs (4 setups, without auto-reset) ==")
    for kw in ({}, {"auto_reset": False}):
        env = multi_env(num_envs=256, seed=71, **kw)
        gen = torch.Generator(device=DEV)
        gen.manual_seed(72)
        acts = [random_actions(env, gen, p_stop=0.1) for _ in range(8)]
        mask = torch.rand(256, generator=gen, device=DEV) < 0.5
        setups = torch.randint(0, 4, (256,), generator=gen, device=DEV)
        for a in acts[:2]:
            env.step(a)
        env.reset(mask, return_info=True)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for a in acts[2:]:
                env.step(a)
            env.reset(mask, return_info=True)
            env.reset(mask, tilts=env.tilt, setups=setups)
            env.step(acts[0])
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
    print("  steps, masked resets with drawn and explicit setups: no sync (auto-reset on "
          "and off)")


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
    test_setups()
    test_setup_sampling()
    test_multi_telescoping()
    test_reset_draws()
    test_no_auto_reset()
    test_no_sync_multi()
    print("\nall tests passed")
