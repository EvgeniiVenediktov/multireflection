"""
tests/test_scenarios.py - Stage 2: scenario atlas, sampler, scenario env, init-from,
schedules, generalized oracle, unfrozen encoder path

Run from rl/:  uv run python tests/test_scenarios.py

    1. atlas build on a tiny sub-grid (18 grid points + the 4 named setups) = direct
       per-scenario sim scans (P_max, its tilt and bounces, plateau, best, near, blind,
       first hit), P_max of the named setups = AlignEnv.path_max = SETUP_INFO; chunking
       does not change the atlas; P_floor / good / P_ref; load_or_build
    2. sampler: impossible share, fresh only until min_seen good scenarios have a score,
       then fresh vs replay shares; replay distribution = (1 - rho) rank priority + rho
       staleness on hand scores; draw frequencies follow it; score EMA on hand values
       (first visit, second visit, several episodes of one scenario in one step,
       impossible scenarios not scored)
    3. env with the scenario table and the sampler: per-scenario P_ref (P_floor for the
       impossible ones), info scenario parameters and sources, reward telescoping,
       states = direct sim of (tilt, parameters), named setups = the setups env
    4. no host syncs in step(), reset(), sampler draw / observe / refresh
    5. init-from: a checkpoint with the old 4-way setup head loads every other tensor;
       the setup head is reported as skipped, the scenario head as fresh; outputs of the
       shared parts identical to the source model; value norm and run name (_ft<run>)
    6. learning-rate schedule and entropy target / coefficient on hand values
    7. generalized plan_oracle (explicit scenario parameters) = the setup-index oracle on
       the four setups, and = the oracle in the atlas env on the named indices
    8. unfrozen encoder: gradients of channels_last + checkpointed chunks, of one pass
       without recompute and of the compiled bf16 path vs an eager fp32 reference
    9. full-mode trainer: checkpoint save / resume continues the same rollout (sampler
       and env state)
"""

import math
import os
import shutil
import sys
import tempfile
from dataclasses import replace

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

from agent import Agent, AgentConfig  # noqa: E402
from baselines import plan_oracle  # noqa: E402
from env import AlignEnv, EnvConfig  # noqa: E402
from evaluation import eval_initial_tilts, eval_setups  # noqa: E402
from find_setups import box_grid, candidate_states  # noqa: E402
from geometry import SETUP_INFO, SETUPS, CellGeometry  # noqa: E402
from herriott_sim import STOP_HOLE  # noqa: E402
from ppo import (PPOConfig, Trainer, ValueNorm, ent_coef_step, ent_target_at, lr_at,  # noqa: E402
                 run_name)
from scenarios import (SRC_FRESH, SRC_IMPOSSIBLE, SRC_REPLAY, Atlas, AtlasConfig,  # noqa: E402
                       SamplerConfig, ScenarioSampler, build_atlas, load_or_build, save_atlas)

DEV = torch.device("cuda")
TINY = AtlasConfig(sep=(86.0, 90.0, 4.0), pitch=(0.0, 8.0, 4.0), yaw=(0.0, 8.0, 4.0))


def random_actions(env: AlignEnv, gen: torch.Generator, p_stop: float) -> torch.Tensor:
    B = env.num_envs
    a = torch.randint(0, env.n_bins, (B, 2), generator=gen, device=DEV)
    stop = torch.rand(B, generator=gen, device=DEV) < p_stop
    return torch.where(stop[:, None], 0, a)


def fake_atlas(n: int, good: torch.Tensor) -> Atlas:
    """Atlas of n synthetic scenarios (sampler tests): good mask given, P_max 1000 mm."""
    z = lambda *s, **kw: torch.zeros(*s, **kw)
    d = {k: z(n) for k in ("pmax", "p_floor", "plateau_path")}
    d["pmax"] += 1000.0
    d.update(params=torch.stack((torch.linspace(50, 200, n), torch.full((n,), 4.0),
                                 torch.full((n,), 4.0)), 1),
             pmax_tilt=z(n, 2), plateau_c=z(n, 2), first_hit=z(n, 2), good=good.cpu(),
             path_ref=torch.full((n,), 1000.0))
    for k in ("pmax_bounces", "plateau_order", "best_bounces"):
        d[k] = z(n, dtype=torch.int16)
    for k in ("plateau_cells", "best_cells", "near_cells", "blind_cells"):
        d[k] = z(n, dtype=torch.int32)
    d.update(version=1, key={}, named={}, n_grid=n)
    return Atlas(d, DEV, "fake")


# ── 1. Atlas ────────────────────────────────────────────────────────────────


def test_atlas(atlas: Atlas, d: dict):
    print("\n== 1. atlas on a tiny sub-grid ==")
    geom = CellGeometry()
    sim = geom.make_sim(DEV, max_bounces=128)
    grid, W = box_grid(2.0, 0.01, DEV)
    n_ok = 0
    for i in range(atlas.n):
        prm = atlas.params[i:i + 1]
        out = sim.simulate(candidate_states(geom, prm, grid), store_world=False)
        b, st, p = out["hit_counts"], out["stop_reason"], out["total_path_length"]
        pm, im = p.max(0)
        assert float(atlas.pmax[i]) == float(pm), i
        assert torch.equal(atlas.pmax_tilt[i], grid[im]) and int(atlas.pmax_bounces[i]) == int(b[im])
        hole = st == STOP_HOLE
        h = torch.bincount(b[hole], minlength=129)
        cnt, o = h[20:].max(0)
        po = int(o) + 20 if int(cnt) > 0 else 0
        assert int(atlas.plateau_order[i]) == po and int(atlas.plateau_cells[i]) == int(cnt), i
        if po:
            pmask = hole & (b == po)
            assert torch.allclose(atlas.plateau_c[i], grid[pmask].mean(0), atol=1e-5)
            assert abs(float(atlas.plateau_path[i]) - float(p[pmask].mean())) < 1e-2
        assert int(atlas.best_bounces[i]) == int(b.max())
        assert int(atlas.best_cells[i]) == int((b == b.max()).sum())
        assert int(atlas.near_cells[i]) == int((p >= 0.9 * pm).sum())
        assert int(atlas.blind_cells[i]) == int((b <= 2).sum())
        z = sim.simulate(candidate_states(geom, prm, torch.zeros(1, 2, device=DEV)),
                         store_world=False)
        if int(z["hit_mirror"][0, 0]) == 2:
            assert torch.equal(atlas.first_hit[i], z["hit_local_xy"][0, 0])
        else:
            assert bool(atlas.first_hit[i].isnan().all())
        n_ok += 1
    pf = 20.0 * atlas.params[:, 0]
    assert torch.equal(atlas.p_floor, pf) and torch.equal(atlas.good, atlas.pmax >= pf)
    assert torch.equal(atlas.path_ref, torch.maximum(atlas.pmax, pf))
    # Named setups: P_max = env.path_max of the setups env = SETUP_INFO
    env4 = AlignEnv(EnvConfig(setups=SETUPS, num_envs=4, history_len=0))
    pm4 = env4.path_max()
    for k, g in enumerate(SETUPS):
        i = atlas.named[g.name]
        assert torch.equal(atlas.params[i], env4.setup_params[k, :3])
        assert float(atlas.pmax[i]) == float(pm4[k]), g.name
        assert abs(float(atlas.pmax[i]) - SETUP_INFO[g.name].pmax_mm) < 0.01
    # Chunking: one scenario per sim call gives the same atlas
    d1 = build_atlas(replace(TINY, chunk_states=1), DEV, log=lambda s: None)
    for k in Atlas.TENSORS:
        assert torch.equal(d[k].nan_to_num(-7.0), d1[k].nan_to_num(-7.0)), k
    # load_or_build: builds a missing file, loads an existing one
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "sub", "atlas.pt")
        msgs = []
        a1 = load_or_build(path, DEV, TINY, log=msgs.append)
        a2 = load_or_build(path, DEV, TINY, log=msgs.append)
        assert os.path.exists(path) and any("missing, building" in m for m in msgs)
        assert sum("missing" in m for m in msgs) == 1 and torch.equal(a1.pmax, a2.pmax)
        assert torch.equal(a1.pmax, atlas.pmax)
        a1.check(128, 2.0, 0.01)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  {atlas.n} scenarios ({atlas.n_grid} grid + {atlas.n - atlas.n_grid} named; "
          f"{int(atlas.good.sum())} good, {int((~atlas.good).sum())} impossible): every statistic "
          f"= a direct 0.01 deg scan ({n_ok} checked); named P_max = path_max = SETUP_INFO "
          f"{[round(float(atlas.pmax[atlas.named[g.name]]), 1) for g in SETUPS]}; chunking "
          f"invariant; load_or_build builds a missing file once, then loads it")


# ── 2. Sampler ──────────────────────────────────────────────────────────────


def test_sampler():
    print("\n== 2. sampler ==")
    n = 400
    gen = torch.Generator(device=DEV)
    gen.manual_seed(5)
    good = torch.rand(n, generator=gen, device=DEV) < 0.6
    at = fake_atlas(n, good)
    G = int(good.sum())
    cfg = SamplerConfig(min_seen=50)
    sm = ScenarioSampler(at, cfg)
    u = torch.rand(400_000, generator=gen, device=DEV)
    s, src = sm.draw(u)
    f_imp = float((src == SRC_IMPOSSIBLE).float().mean())
    assert bool((good[s] == (src != SRC_IMPOSSIBLE)).all())
    assert abs(f_imp - 0.1) < 4 * math.sqrt(0.09 / u.numel()), f_imp
    assert not bool((src == SRC_REPLAY).any()), "no replay before min_seen"
    cnt = torch.bincount(s[good[s]], minlength=n)[good].double()
    exp = 0.9 * u.numel() / G
    dev_u = float(((cnt - exp).abs() / math.sqrt(exp)).max())
    assert dev_u < 5.0
    print(f"  before min_seen: impossible share {f_imp:.4f} (0.1), good draws uniform over "
          f"{G} (max |count - mean| / sigma {dev_u:.2f}), no replay")

    # Hand scores: EMA, first visit, several episodes of one scenario in one step
    gi = torch.nonzero(good).squeeze(1)
    ii = torch.nonzero(~good).squeeze(1)
    a, b_, c_ = int(gi[0]), int(gi[1]), int(ii[0])
    scen = torch.tensor([a, a, b_, c_, a], device=DEV)
    final = torch.tensor([600.0, 800.0, 100.0, 0.0, 999.0], device=DEV)  # P_max 1000
    done = torch.tensor([True, True, True, True, False], device=DEV)
    sm.observe(scen, final, done, torch.tensor([1, 1, 2, 0, 1], device=DEV))
    assert abs(float(sm.score[a]) - 0.3) < 1e-6  # Mean of 0.4 and 0.2
    assert abs(float(sm.score[b_]) - 0.9) < 1e-6
    assert not bool(sm.seen[c_]) and float(sm.score[c_]) == 0.0, "impossible: no score"
    sm.observe(scen[:1], torch.tensor([0.0], device=DEV), done[:1])
    assert abs(float(sm.score[a]) - (0.7 * 0.3 + 0.3 * 1.0)) < 1e-6
    assert int(sm.visits[a]) == 3
    st = sm.stats()
    assert st["plr/episodes"] == 4 and abs(st["plr/impossible_share"] - 0.25) < 1e-9
    print(f"  scores: first visit = mean shortfall of the step's episodes (0.3, 0.9), then "
          f"EMA 0.7 x 0.3 + 0.3 x 1.0 = {float(sm.score[a]):.3f}; impossible not scored")

    # Replay distribution on hand scores and staleness
    sm = ScenarioSampler(at, cfg)
    sc = torch.rand(n, generator=gen, device=DEV)
    k_seen = 80
    seen_ids = gi[:k_seen]
    sm.score[seen_ids] = sc[seen_ids]
    sm.seen[seen_ids] = True
    sm.last[seen_ids] = torch.arange(k_seen, device=DEV) % 7
    sm.refresh(10)
    p = sm.p_replay  # Over gi
    sel = sm.seen[gi]
    s_seen = sm.score[gi][sel]
    rank = torch.empty(k_seen, dtype=torch.long, device=DEV)
    rank[torch.argsort(s_seen, descending=True)] = torch.arange(1, k_seen + 1, device=DEV)
    h = rank.double() ** (-1 / cfg.beta)
    stale = (10 - sm.last[gi][sel]).double()
    ref = 0.9 * h / h.sum() + 0.1 * stale / stale.sum()
    assert float((p[sel] - ref).abs().max()) < 1e-12 and float(p[~sel].abs().max()) == 0.0
    top = int(gi[int(p.argmax())])
    assert top == int(seen_ids[int(sc[seen_ids].argmax())]), "top priority = largest shortfall"
    s, src = sm.draw(torch.rand(1_000_000, generator=gen, device=DEV))
    fr = {k: float((src == v).float().mean()) for k, v in (("imp", SRC_IMPOSSIBLE),
                                                           ("fresh", SRC_FRESH),
                                                           ("replay", SRC_REPLAY))}
    assert abs(fr["imp"] - 0.1) < 0.002 and abs(fr["fresh"] - 0.27) < 0.002
    assert abs(fr["replay"] - 0.63) < 0.002
    rep = s[src == SRC_REPLAY]
    freq = torch.bincount(rep, minlength=n)[gi].double() / rep.numel()
    big = p > 1e-3
    err = float(((freq[big] - p[big]).abs() / (p[big] * (1 - p[big]) / rep.numel()).sqrt()).max())
    assert err < 5.0 and not bool(freq[~sel].any())
    print(f"  {k_seen} seen (min_seen {cfg.min_seen}): replay = 0.9 rank^(-1/0.3) + 0.1 "
          f"staleness exactly; draw shares impossible {fr['imp']:.3f} / fresh {fr['fresh']:.3f} "
          f"/ replay {fr['replay']:.3f} (0.1 / 0.27 / 0.63); replay frequencies follow it (max "
          f"{err:.2f} sigma); top priority = largest shortfall, its draw share "
          f"{sm.stats()['plr/top_draw_share']:.3f}")


# ── 3. / 4. Env with scenarios ──────────────────────────────────────────────


def test_env_scenarios(atlas: Atlas):
    print("\n== 3. env with the scenario table ==")
    sm = ScenarioSampler(atlas, SamplerConfig(min_seen=4))
    env = AlignEnv(EnvConfig(num_envs=512, seed=81), scenarios=atlas.table(), sampler=sm)
    assert env.n_setups == atlas.n and env.path_ref_mode == "scenario" and env.setups == ()
    gen = torch.Generator(device=DEV)
    gen.manual_seed(82)
    c = env.cfg.step_cost
    refs = atlas.path_ref
    p0, s0 = env.path.clone(), env.setup.clone()
    ret = torch.zeros(512, device=DEV, dtype=torch.float64)
    moves = torch.zeros(512, dtype=torch.long, device=DEV)
    errs, n_ep, n_imp = [], 0, 0
    for it in range(150):
        if it % 30 == 0:
            sm.refresh(it // 30)
        a = random_actions(env, gen, p_stop=0.05)
        _, rew, term, trunc, info = env.step(a)
        sc = info["setup"]
        assert torch.equal(info["path_ref"], refs[sc])
        assert torch.equal(info["scenario_params"][:, :3], atlas.params[sc])
        assert torch.equal(info["path_norm"], info["path_mm"] / refs[sc])
        assert torch.equal(info["obs_scenario_params"][:, :3], atlas.params[env.setup])
        src = info["scenario_src"]
        assert bool(((src == SRC_IMPOSSIBLE) == ~atlas.good[sc]).all())
        ret += rew.double()
        moves += (a != 0).any(1).long()
        done = term | trunc
        sm.observe(sc, info["ep_final_path_mm"], done, src)
        if bool(done.any()):
            d = done
            tele = (info["ep_final_path_mm"][d].double() - p0[d].double()) / refs[s0[d]].double()
            tele = tele - c * moves[d].double()
            errs.append(float((ret[d] - tele).abs().max()))
            n_ep += int(d.sum())
            n_imp += int((~atlas.good[s0[d]]).sum())
        ret = torch.where(done, 0.0, ret)
        moves = torch.where(done, 0, moves)
        p0 = torch.where(done, env.path, p0)
        s0 = torch.where(done, env.setup, s0)
    fresh = env.sim.simulate(env.scenario_states(env.tilt, atlas.params[env.setup]),
                             store_world=False)
    assert torch.equal(env.path, fresh["total_path_length"])
    imp = ~atlas.good
    assert torch.equal(refs[imp], atlas.p_floor[imp]) and torch.equal(refs[~imp], atlas.pmax[~imp])
    print(f"  {n_ep} episodes ({100 * n_imp / n_ep:.1f} % impossible): max |sum r - (P_final - "
          f"P_0) / P_ref + c moves| {max(errs):.1e}; P_ref = P_max (good) / 20 x separation "
          f"(impossible); info scenario parameters and sources consistent; state = direct sim")
    assert max(errs) < 1e-4 and 0.05 < n_imp / n_ep < 0.2

    # Named setups in the atlas env = the setups env
    env4 = AlignEnv(EnvConfig(setups=SETUPS, num_envs=8, history_len=0))
    enva = AlignEnv(EnvConfig(num_envs=8, history_len=0), scenarios=atlas.table())
    t = eval_initial_tilts(8, 3).to(DEV)
    k = torch.arange(8, device=DEV) % 4
    ia = torch.tensor([atlas.named[g.name] for g in SETUPS], device=DEV)[k]
    r4, ra = env4.render_tilts(t, k), enva.render_tilts(t, ia)
    assert torch.equal(r4["image"], ra["image"]) and torch.equal(r4["path_mm"], ra["path_mm"])
    assert torch.equal(env4.path_refs[k], enva.path_refs[ia])
    print("  named setups in the atlas env: images, paths and P_ref = the setups env")

    print("\n== 4. no host syncs (env with sampler) ==")
    acts = [random_actions(env, gen, p_stop=0.1) for _ in range(6)]
    mask = torch.rand(512, generator=gen, device=DEV) < 0.5
    explicit = torch.randint(0, atlas.n, (512,), generator=gen, device=DEV)
    env.step(acts[0])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        sm.refresh(9)
        for a in acts:
            _, _, term, trunc, info = env.step(a)
            sm.observe(info["setup"], info["ep_final_path_mm"], term | trunc,
                       info["scenario_src"])
        env.reset(mask, return_info=True)
        env.reset(mask, tilts=env.tilt, setups=explicit)
        sm.draw(torch.rand(512, device=DEV))
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert bool((env.scen_src[mask] == -1).all()), "explicit setups: source -1"
    print("  refresh, steps with auto-reset draws, observe, masked resets (drawn and explicit "
          "scenarios): no sync")


# ── 5. Init-from ────────────────────────────────────────────────────────────


def test_init_from():
    print("\n== 5. init-from ==")
    d = tempfile.mkdtemp()
    try:
        cfg = AgentConfig(window=6, d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False)
        torch.manual_seed(1)
        src = Agent(cfg).to(DEV)
        with torch.no_grad():
            for p_ in src.parameters():
                p_.add_(0.01 * torch.randn_like(p_))
        sd = {k: v for k, v in src.state_dict().items() if not k.startswith("scenario.")}
        hid = cfg.head_hidden
        sd.update({"setup.0.weight": torch.randn(hid, cfg.d_model), "setup.0.bias": torch.randn(hid),
                   "setup.2.weight": torch.randn(4, hid), "setup.2.bias": torch.randn(4)})
        vn = ValueNorm(0.99, DEV)
        vn.update(torch.tensor([1.0, 2.0, 4.0], device=DEV))
        path = os.path.join(d, "4398136", "best.pt")
        os.makedirs(os.path.dirname(path))
        torch.save({"agent": sd, "vnorm": vn.state_dict(), "name": "old", "iteration": 600,
                    "env_steps": 78_643_200, "encoder_info": {"n_setups": 4, "name": "enc"}}, path)
        torch.manual_seed(2)
        dst = Agent(cfg).to(DEV)
        info = dst.init_from(path)
        assert sorted(info["skipped"]) == ["setup.0.bias", "setup.0.weight", "setup.2.bias",
                                           "setup.2.weight"]
        assert info["missing"] == ["scenario.0.bias", "scenario.0.weight", "scenario.2.bias",
                                   "scenario.2.weight"]
        img = torch.randint(0, 256, (8, 1, 64, 64), dtype=torch.uint8, device=DEV)
        z0 = torch.zeros(8, 2, device=DEV)
        first = torch.ones(8, dtype=torch.bool, device=DEV)
        with torch.no_grad():
            oa, ra = src.act(img, z0, first, src.init_ring(8, DEV))
            ob, rb = dst.act(img, z0, first, dst.init_ring(8, DEV))
            oa, _ = src.act(img.flip(-1), z0 + 0.3, ~first, ra)
            ob, _ = dst.act(img.flip(-1), z0 + 0.3, ~first, rb)
        for k in ("z", "logits", "value", "path", "tilt_mu", "tilt_logvar"):
            assert torch.equal(oa[k], ob[k]), k
        assert not torch.equal(oa["scen_mu"], ob["scen_mu"])
        print(f"  {len(info['loaded'])} tensors loaded; skipped {sorted(info['skipped'])}; fresh "
              f"{info['missing']}; z / logits / value / path / tilt identical to the source")

        # Trainer: value norm from the checkpoint, run name, encoder info
        tc = PPOConfig(setups="all", num_envs=16, rollout_len=8, max_steps=6, history_len=6,
                       d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False,
                       compile=False, wandb=False, minibatches=4, eval_every=0,
                       init_from=path, total_env_steps=10**6)
        tr = Trainer(tc, os.path.join(d, "run"), build_eval=False)
        m, s = tr.vnorm.stats()
        m0, s0 = vn.stats()
        assert torch.equal(m, m0) and torch.equal(s, s0)
        assert "_ft4398136_" in tr.name and tr.name.startswith("ppo_cnn-pre4_k6_set4"), tr.name
        tr.rollout()
        tr.update()  # Fresh optimizer state, runs
        cf = replace(tc, setups="full", scenarios="atlas", init_from=path, unfreeze_at=0)
        nm = run_name(cf, {"n_setups": 4})
        print(f"  trainer: value norm from the checkpoint (mean {float(m):.3f}, std {float(s):.3f}); "
              f"names {tr.name}, {nm}")
        assert nm.startswith("ppo_cnn-pre4_k6_full-plr_b16_s0_ft4398136_uf0_")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── 6. Schedules ────────────────────────────────────────────────────────────


def test_schedules():
    print("\n== 6. learning rate and entropy schedules ==")
    c = PPOConfig(lr=3e-4, lr_warmup=20, lr_final_frac=0.1, num_envs=1024, rollout_len=128,
                  total_env_steps=1_000_000_000)
    spi = 1024 * 128
    s0 = 20 * spi
    cases = [(0, 0, 1.5e-5), (9, 9 * spi, 1.5e-4), (19, 19 * spi, 3e-4), (20, s0, 3e-4),
             (1000, s0 + (10**9 - s0) // 2, 0.5 * (3e-4 + 3e-5)), (7000, 10**9, 3e-5),
             (9000, 2 * 10**9, 3e-5)]
    for it, steps, want in cases:
        got = lr_at(c, it, steps)
        assert abs(got - want) < 1e-12, (it, steps, got, want)
    q = lr_at(c, 500, s0 + (10**9 - s0) // 4)
    assert abs(q - (3e-5 + 2.7e-4 * 0.5 * (1 + math.cos(math.pi / 4)))) < 1e-10
    for steps, want in ((0, 2.0), (4 * 10**8, 2.0 - 1.3 * 0.5), (8 * 10**8, 0.7), (10**9, 0.7)):
        assert abs(ent_target_at(c, steps) - want) < 1e-12, (steps, ent_target_at(c, steps))
    la = torch.tensor(math.log(0.01), device=DEV)
    up = ent_coef_step(la, torch.tensor(1.5, device=DEV), 2.0, c)  # Below target: up
    dn = ent_coef_step(la, torch.tensor(2.6, device=DEV), 2.0, c)  # Above: down
    assert abs(float(up) - (math.log(0.01) + 0.5e-3)) < 1e-6
    assert abs(float(dn) - (math.log(0.01) - 0.6e-3)) < 1e-6
    hi = ent_coef_step(torch.tensor(math.log(0.0999), device=DEV),
                       torch.tensor(-5.0, device=DEV), 2.0, c)
    lo = ent_coef_step(torch.tensor(math.log(1.0001e-4), device=DEV),
                       torch.tensor(9.0, device=DEV), 2.0, c)
    assert abs(float(hi.exp()) - 0.1) < 1e-7 and abs(float(lo.exp()) - 1e-4) < 1e-9
    n = int(round(math.log(10) / 1e-3))
    print(f"  lr: 1.5e-5 at iteration 0, peak 3e-4 at 19-20, 1.65e-4 at mid-decay, 3e-5 at "
          f"total; entropy target 2.0 -> 1.35 (40 %) -> 0.7 (80 %, then constant); log alpha "
          f"+-1e-3 x (H - target) per minibatch, clamped to [1e-4, 0.1] (a 1-nat deficit "
          f"takes {n} minibatches, {n // 32} iterations at 4 x 8, for x10)")


# ── 7. Generalized oracle ───────────────────────────────────────────────────


def test_oracle(atlas: Atlas):
    print("\n== 7. generalized plan_oracle ==")
    env4 = AlignEnv(EnvConfig(setups=SETUPS, num_envs=16, auto_reset=False, history_len=0))
    n = 24
    t = eval_initial_tilts(n, 9).to(DEV)
    s = eval_setups(n, 4, 9).to(DEV)
    a = plan_oracle(env4, t, setups=s)
    b = plan_oracle(env4, t, setups=s, scenarios=env4.setup_params[s], path_ref=env4.path_refs[s])
    enva = AlignEnv(EnvConfig(num_envs=16, auto_reset=False, history_len=0),
                    scenarios=atlas.table())
    ia = torch.tensor([atlas.named[g.name] for g in SETUPS], device=DEV)[s]
    c = plan_oracle(enva, t, setups=ia)
    for f in ("disp", "moves", "actions", "ret", "path0_mm", "target_path_mm", "target_bounces",
              "target_tilt", "max_path_mm"):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
        assert torch.equal(getattr(a, f), getattr(c, f)), f
    print(f"  {n} episodes on the four setups: explicit-parameter oracle and atlas-env oracle = "
          f"setup-index oracle in every field (mean return {float(a.ret.mean()):.4f})")


# ── 8. Unfrozen encoder path ────────────────────────────────────────────────


def _enc_grads(ag: Agent, images: torch.Tensor, w: torch.Tensor):
    ag.zero_grad(set_to_none=True)
    z = ag.encode(images, grad=True)
    (z * w).sum().backward()
    g = torch.cat([p.grad.flatten() for p in ag.encoder.parameters()])
    return z.detach(), g


def test_unfrozen_encoder():
    print("\n== 8. unfrozen encoder path ==")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device=DEV)
    gen.manual_seed(91)
    imgs = torch.randint(0, 256, (300, 1, 64, 64), generator=gen, dtype=torch.uint8, device=DEV)
    w = torch.randn(300, 256, generator=gen, device=DEV)
    base = dict(window=6, d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0)
    ref = Agent(AgentConfig(amp=False, channels_last=False, enc_checkpoint=False, **base)).to(DEV)
    sd = ref.state_dict()
    zr, gr = _enc_grads(ref, imgs, w)
    res = {}
    for name, kw, comp in (("fp32 channels_last ckpt", dict(amp=False, enc_chunk=128), False),
                           ("fp32 channels_last no-ckpt", dict(amp=False, enc_checkpoint=False), False),
                           ("eager bf16 channels_last ckpt", dict(amp=True, enc_chunk=128), False),
                           ("compiled bf16 channels_last ckpt", dict(amp=True, enc_chunk=128), True),
                           ("compiled bf16 channels_last no-ckpt",
                            dict(amp=True, enc_checkpoint=False), True)):
        ag = Agent(AgentConfig(**base, **kw)).to(DEV)
        ag.load_state_dict(sd)
        assert ag.encoder.stem.weight.is_contiguous(memory_format=torch.channels_last)
        if comp:
            ag.compile()
        for _ in range(2 if comp else 1):  # Compiled: the second call runs the cached graphs
            z, g = _enc_grads(ag, imgs, w)
        dz = float((z - zr).abs().max() / zr.abs().max())
        dg = float((g - gr).norm() / gr.norm())
        cos = float(torch.nn.functional.cosine_similarity(g, gr, dim=0))
        res[name] = (dz, dg, cos)
        print(f"  {name}: max |dz| / max |z| {dz:.1e}, |d grad| / |grad| {dg:.1e}, cos vs fp32 "
              f"{cos:.4f}")
        if kw["amp"]:  # bf16: the compiled paths as close to fp32 as eager bf16
            assert dz < 3e-2 and cos > 0.98, name
        else:
            assert dz < 1e-4 and dg < 1e-4, name
    e = res["eager bf16 channels_last ckpt"]
    for k in ("compiled bf16 channels_last ckpt", "compiled bf16 channels_last no-ckpt"):
        assert res[k][2] > e[2] - 0.01, (k, res[k], e)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


# ── 9. Full-mode resume ─────────────────────────────────────────────────────


def test_full_resume(atlas_path: str):
    print("\n== 9. full-mode trainer: save / resume ==")
    d = tempfile.mkdtemp()
    try:
        cfg = PPOConfig(setups="full", scenarios="atlas", atlas=atlas_path, num_envs=16,
                        rollout_len=24, max_steps=6,
                        history_len=6, d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0,
                        amp=False, compile=False, wandb=False, minibatches=4, eval_every=0,
                        seed=7, total_env_steps=10**6, plr_min_seen=2)
        a = Trainer(cfg, d, build_eval=False)
        for it in range(2):
            a.sampler.refresh(a.iteration)
            a.rollout()
            a.update()
            a.iteration += 1
        a.save("latest.pt")
        a.sampler.refresh(a.iteration)
        a.rollout()
        snap = {k: getattr(a.buf, k).clone() for k in ("frames", "actions", "logp", "rewards",
                                                       "scen", "first")}
        sc = a.sampler.score.clone()
        bt = Trainer(cfg, d, build_eval=False)
        bt.load(os.path.join(d, "latest.pt"))
        bt.sampler.refresh(bt.iteration)
        bt.rollout()
        same = {k: bool(torch.equal(v, getattr(bt.buf, k))) for k, v in snap.items()}
        same["sampler score"] = bool(torch.equal(sc, bt.sampler.score))
        same["log_alpha"] = bool(torch.equal(a.log_alpha, bt.log_alpha))
        print("  next rollout after resume identical: " + ", ".join(f"{k} {v}" for k, v in same.items()))
        assert all(same.values())
        assert int(a.sampler.seen.sum()) > 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    tmp = tempfile.mkdtemp()
    try:
        d = build_atlas(TINY, DEV, log=lambda s: None)
        atlas_path = os.path.join(tmp, "tiny.pt")
        save_atlas(d, atlas_path)
        atlas = Atlas(d, DEV, atlas_path)
        test_atlas(atlas, d)
        test_sampler()
        test_env_scenarios(atlas)
        test_init_from()
        test_schedules()
        test_oracle(atlas)
        test_unfrozen_encoder()
        test_full_resume(atlas_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nall tests passed")
