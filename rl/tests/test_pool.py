"""
tests/test_pool.py - Continuous scenarios (D7): candidate generator, scenario pool, env in
pool mode, eval_cont, init-from with a changed scenario head, encoder data and fine-tuning

Run from rl/:  uv run python tests/test_pool.py

    1. generator: a stream read in any cuts gives the same candidates, streams differ;
       laser offsets strictly inside the 1.5 mm disk (float64 and float32), radius 0 =
       the measured offsets; landing mode: the full sim's first hit lands on the drawn M2
       point; sample_valid_scenarios: every scenario valid under the full sim (first hit
       on M2 at the box centre, no clipping), angles within the bounds, stage counts and
       stream positions consistent with candidate_batch; the 1-bounce check = the full
       sim on every candidate; angle mode
    2. pool fill: good / impossible slot split, draws 90 / 10, every slot valid; slot
       statistics = direct 0.01 deg sim scans; the slots are the first good / impossible
       valid candidates of the stream (independent of the round size)
    3. refresh: the oldest refresh_frac of each slot group (ring order) is replaced by the
       next stream scenarios, their scores reset and generation increments, the other
       slots unchanged; no replacement within an iteration; a full ring after 1 / frac
       refreshes; observe ignores episodes of a replaced slot generation
    4. priorities within the pool: draw distribution = 10 % impossible uniform, fresh
       share uniform over the good slots, replay (1 - rho) rank^(-1 / beta) + rho
       staleness on hand scores; draw frequencies follow it
    5. env in pool mode: per-env copies of parameters / P_ref / generation, reward
       telescoping with the episode's P_ref, state = direct sim, a refresh does not change
       episodes in flight; no host syncs in step / reset (drawn and explicit) / draw /
       observe
    6. sim at 20 and 400 mm: valid scenarios, finite paths and P_max, bounce counts
       within the cap
    7. eval_cont: reproducible from its seed (independent of the round size), cached and
       loaded, other settings refused, 90 / 10 one episode per scenario, rows valid; the
       trainer's cont set (prefix, bins, oracle cached)
    8. init-from: a checkpoint with another scenario normalization (or width) loads every
       other tensor, the scenario head re-initialized and reported; same normalization
       loads it; trainer in pool mode (run name, best metric, eval sets, legacy config)
    9. encoder: --init loads exactly the source weights (identical outputs before the
       first step), run names (_ft-grid16384_cont20-400x16384, cnn-ftcont); continuous
       frame data from sample_valid_scenarios (offsets in the disk, no offset jitter)
   10. pool-mode trainer: save / resume continues the same rollout (pool slots, stream
       position, scores)
"""

import argparse
import dataclasses
import math
import os
import shutil
import sys
import tempfile
from dataclasses import replace

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pretrain_encoder as pe  # noqa: E402
from agent import Agent, AgentConfig  # noqa: E402
from encoders import EncoderConfig, FrameModel  # noqa: E402
from env import AlignEnv, EnvConfig  # noqa: E402
from find_setups import box_grid  # noqa: E402
from frame_data import DataConfig, FrameData  # noqa: E402
from geometry import SETUPS, CellGeometry  # noqa: E402
from herriott_sim import S_LASER_DX, S_LASER_DY, STOP_BLOCKED, STOP_HOLE  # noqa: E402
from ppo import (PPOConfig, Trainer, ValueNorm, config_from_dict, cont_eval_set,  # noqa: E402
                 encoder_origin, eval_set_names, run_name)
from scenarios import (STREAM_ENCODER, STREAM_EVAL, STREAM_POOL, AtlasConfig, ContConfig,  # noqa: E402
                       ContRanges, PoolConfig, SamplerConfig, Scanner, ScenarioPool, box_states,
                       build_cont_table, candidate_batch, cont_sep_bins, cont_table_path,
                       load_or_build_cont, offgrid_eval_scenarios, sample_valid_scenarios,
                       scenario_geometry)

DEV = torch.device("cuda")
GEOM = CellGeometry()
R_AP = GEOM.diameter / 2
NARROW = ContRanges(sep=(80.0, 120.0))


def full_first_hit(sc: Scanner, params: np.ndarray):
    """Full (cap 128) sim at the box centre: first hit on M2, clipped, first-hit (lx, ly)."""
    pt = torch.tensor(params, dtype=torch.float32, device=DEV)
    out = sc.sim.simulate(box_states(sc.geom, pt, torch.zeros(1, 2, device=DEV)), store_world=False)
    on = (out["hit_counts"] >= 1) & (out["hit_mirror"][:, 0] == 2)
    clip = (out["hit_counts"] == 0) & (out["stop_reason"] == STOP_BLOCKED)
    return on.cpu().numpy(), clip.cpu().numpy(), out["hit_local_xy"][:, 0].cpu().numpy()


def make_pool(size=64, frac=0.25, seed=0, ranges=NARROW, round_min=64, round_max=2048,
              **sampler) -> ScenarioPool:
    kw = dict(min_seen=4, beta=1.0, rho=0.1)
    kw.update(sampler)
    return ScenarioPool(PoolConfig(size=size, ranges=ranges, refresh_frac=frac, seed=seed,
                                   round_min=round_min, round_max=round_max),
                        SamplerConfig(**kw), device=DEV, log=lambda s: None)


# ── 1. Generator ────────────────────────────────────────────────────────────


def test_generator(sc: Scanner):
    print("\n== 1. candidate generator ==")
    r = ContRanges()
    a = r.candidates(3, 0, 9000, STREAM_POOL)
    b = np.concatenate([r.candidates(3, 0, 1234, STREAM_POOL), r.candidates(3, 1234, 5000, STREAM_POOL),
                        r.candidates(3, 6234, 2766, STREAM_POOL)])
    assert np.array_equal(a, b), "a stream depends on how it is read"
    assert not np.array_equal(a[:16], r.candidates(3, 0, 16, STREAM_EVAL))
    assert not np.array_equal(a[:16], r.candidates(4, 0, 16, STREAM_POOL))
    for mode in ("landing", "angle"):
        p = replace(r, mode=mode).candidates(0, 0, 50000, STREAM_ENCODER)
        n64 = float(np.hypot(p[:, 3], p[:, 4]).max())
        n32 = float(torch.tensor(p[:, 3:5], dtype=torch.float32).norm(dim=1).max())
        assert n64 < 1.5 and n32 < 1.5, (mode, n64, n32)
        assert p[:, 0].min() >= 20.0 and p[:, 0].max() < 400.0
        if mode == "angle":
            assert np.abs(p[:, 1:3]).max() <= 8.0
    p0 = replace(r, offset_radius=0.0).candidates(0, 0, 100, STREAM_POOL)
    assert np.all(p0[:, 3] == GEOM.laser_dx) and np.all(p0[:, 4] == GEOM.laser_dy)
    assert replace(r, offset_radius=0.0).head_range()[0] == (20.0, -8.0, -8.0)
    assert r.head_range() == ((20.0, -8.0, -8.0, -1.5, -1.5), (400.0, 8.0, 8.0, 1.5, 1.5))

    # Landing mode: the full sim's first hit is the drawn M2 point (where it reaches M2)
    from scenarios import CAND_BLOCK, OFFSET_SHRINK

    n = 4096
    p = r.candidates(5, 0, n, STREAM_POOL)
    u = np.random.default_rng([STREAM_POOL, 5, 0]).random((CAND_BLOCK, 5))[:n]
    rl = (R_AP - r.landing_margin) * OFFSET_SHRINK * np.sqrt(u[:, 1])
    lx, ly = rl * np.cos(2 * np.pi * u[:, 2]), rl * np.sin(2 * np.pi * u[:, 2])
    ok = r.angle_ok(p)
    on, clip, hxy = full_first_hit(sc, p[ok])
    err = np.hypot(hxy[on, 0] - lx[ok][on], hxy[on, 1] - ly[ok][on])
    assert err.max() < 1e-4, err.max()
    assert not bool((~on & ~clip).any()), "a beam inside the bounds missed M2 without clipping"

    # sample_valid_scenarios: valid under the full sim; counts and positions
    prm, info = sample_valid_scenarios(3000, r, seed=1, stream=STREAM_EVAL, scanner=sc, round_n=1024)
    assert prm.shape == (3000, 5) and info["valid"] == 3000
    on, clip, _ = full_first_hit(sc, prm)
    assert bool(on.all()) and not bool(clip.any())
    assert bool(r.angle_ok(prm).all()) and float(np.hypot(prm[:, 3], prm[:, 4]).max()) < 1.5
    assert info["candidates"] == info["angle_rejected"] + info["clipped"] + info["missed"] + info["valid"]
    bb = candidate_batch(r, 1, 0, info["next"], STREAM_EVAL, sc)
    assert np.array_equal(np.nonzero(bb["valid"])[0], info["positions"])
    assert np.array_equal(bb["params"][info["positions"]], prm)
    # The 1-bounce check = the full sim on every in-bounds candidate
    ia = np.nonzero(bb["angle_ok"])[0]
    on, clip, _ = full_first_hit(sc, bb["params"][ia])
    assert np.array_equal(on, bb["valid"][ia]) and np.array_equal(clip, bb["clipped"][ia])
    # Restarting from a position continues the same sequence
    prm2, _ = sample_valid_scenarios(500, r, seed=1, stream=STREAM_EVAL, scanner=sc,
                                     start=int(info["positions"][1000]), round_n=300)
    assert np.array_equal(prm2, prm[1000:1500])
    # Angle mode: uniform angles, more clipped / missed
    pa, ia_ = sample_valid_scenarios(500, replace(r, mode="angle"), seed=1, scanner=sc)
    on, clip, _ = full_first_hit(sc, pa)
    assert bool(on.all()) and ia_["angle_rejected"] == 0
    c = info["candidates"]
    print(f"  stream reads independent of the cut; offsets < 1.5 mm (max {n64:.6f} float64, "
          f"{n32:.6f} float32); landing point = full-sim first hit (max err {err.max():.1e} mm); "
          f"3000 valid of {c} candidates (angles out of bounds {100 * info['angle_rejected'] / c:.1f} %, "
          f"clipped {100 * info['clipped'] / c:.1f} %, missed {100 * info['missed'] / c:.1f} %), all "
          f"valid under the full sim; 1-bounce check = full sim on {ia.size} candidates; angle "
          f"mode: {500 / ia_['candidates']:.2f} valid")


# ── 2. Pool fill ────────────────────────────────────────────────────────────


def direct_scan(sc: Scanner, params: torch.Tensor):
    grid, _ = box_grid(2.0, 0.01, DEV)
    out = sc.sim.simulate(box_states(sc.geom, params[None], grid), store_world=False)
    return out, grid


def stream_groups(pool: ScenarioPool, sc: Scanner, end: int, start: int = 0):
    """Good and impossible valid scenarios of the pool's stream in [start, end) (stream
    order) and their positions relative to start."""
    b = candidate_batch(pool.pcfg.ranges, pool.pcfg.seed, start, end - start, STREAM_POOL, sc)
    vid = np.nonzero(b["valid"])[0]
    d = sc.scan(torch.from_numpy(b["params"][vid]))
    g = d["good"].cpu().numpy()
    return d["params"][torch.from_numpy(g).to(DEV)], d["params"][torch.from_numpy(~g).to(DEV)], vid, g


def test_pool_fill(sc: Scanner):
    print("\n== 2. pool fill ==")
    pool = make_pool()
    N, ng, ni = 64, pool.n_good_slots, pool.n_imp_slots
    at = pool.atlas
    assert (ng, ni) == (58, 6)
    assert bool(at.good[:ng].all()) and not bool(at.good[ng:].any())
    assert torch.equal(pool.good_ids.cpu(), torch.arange(ng)) and torch.equal(pool.impossible_ids.cpu(), torch.arange(ng, N))
    prm = at.params.cpu().numpy().astype(np.float64)
    on, clip, _ = full_first_hit(sc, prm)
    assert bool(on.all()), "a pool slot with an invalid beam"
    assert float(at.params[:, 3:5].norm(dim=1).max()) < 1.5 and bool(NARROW.angle_ok(prm).all())
    # Slot statistics = direct scans
    pf = 20.0 * at.params[:, 0]
    assert torch.equal(at.p_floor, pf) and torch.equal(at.path_ref, torch.maximum(at.pmax, pf))
    for i in (0, 17, ng - 1, ng, N - 1):
        out, grid = direct_scan(sc, at.params[i])
        p, b = out["total_path_length"], out["hit_counts"]
        pm, im = p.max(0)
        assert float(at.pmax[i]) == float(pm) and torch.equal(at.pmax_tilt[i], grid[im])
        assert int(at.pmax_bounces[i]) == int(b[im]) and int(at.best_bounces[i]) == int(b.max())
        assert int(at.near_cells[i]) == int((p >= 0.9 * pm).sum())
        assert int(at.blind_cells[i]) == int((b <= 2).sum())
        hole = out["stop_reason"] == STOP_HOLE
        h = torch.bincount(b[hole], minlength=129)
        cnt, o = h[20:].max(0)
        assert int(at.plateau_order[i]) == (int(o) + 20 if int(cnt) > 0 else 0)
    # The slots are the first valid good / impossible scenarios of the stream
    G, I, vid, g = stream_groups(pool, sc, pool.cursor)
    assert torch.equal(at.params[:ng], G[:ng]) and torch.equal(at.params[ng:], I[:ni])
    last = max(int(vid[g][ng - 1]), int(vid[~g][ni - 1]))
    assert pool.cursor == last + 1, (pool.cursor, last)
    p2 = make_pool(round_min=17, round_max=33)
    assert torch.equal(p2.atlas.params, at.params) and p2.cursor == pool.cursor
    # Draws: 10 % impossible
    pool.refresh(0)
    assert abs(float(pool.cdf[ni - 1]) - 0.1) < 1e-9
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    s, src = pool.draw(torch.rand(200000, generator=gen, device=DEV))
    imp = float((s >= ng).float().mean())
    assert abs(imp - 0.1) < 0.005 and bool(((src == 0) == (s >= ng)).all())
    st = pool.stats()
    print(f"  {N} slots: {ng} good / {ni} impossible, every slot a valid beam, statistics = direct "
          f"scans; slots = the first {ng} good / {ni} impossible valid scenarios of the stream "
          f"(cursor {pool.cursor}, the same with rounds of 17-33); impossible draws {100 * imp:.2f} %; "
          f"fill: {int(st['pool/candidates'])} candidates, valid {100 * st['pool/valid_share']:.1f} %, "
          f"good {100 * st['pool/good_share']:.1f} % of valid")


# ── 3. Refresh ──────────────────────────────────────────────────────────────


def test_refresh(sc: Scanner):
    print("\n== 3. refresh ==")
    pool = make_pool(frac=0.25)
    ng, ni, N = pool.n_good_slots, pool.n_imp_slots, 64
    kg, ki = round(0.25 * ng), round(0.25 * ni)
    pool.refresh(0)
    # Hand scores on every good slot
    slots = torch.arange(ng, device=DEV)
    fp = (1.0 - torch.linspace(0.05, 0.9, ng, device=DEV)) * pool.atlas.pmax[:ng]
    pool.observe(slots, fp, torch.ones(ng, dtype=torch.bool, device=DEV))
    sc0, seen0, prm0, gen0 = (pool.score.clone(), pool.seen.clone(), pool.atlas.params.clone(),
                              pool.slot_gen.clone())
    c0 = pool.cursor
    pool.refresh(0)  # Same iteration: nothing replaced
    assert pool.cursor == c0 and torch.equal(pool.atlas.params, prm0)
    pool.refresh(1)
    rep = torch.zeros(N, dtype=torch.bool, device=DEV)
    rep[:kg] = True
    rep[ng:ng + ki] = True
    assert torch.equal(pool.slot_gen, gen0 + rep.long()) and torch.equal(pool.born, rep.long())
    assert bool((pool.score[rep] == 0).all()) and not bool(pool.seen[rep].any())
    assert bool((pool.visits[rep] == 0).all()) and bool((pool.last[rep] == 1).all())
    assert torch.equal(pool.score[~rep], sc0[~rep]) and torch.equal(pool.seen[~rep], seen0[~rep])
    assert torch.equal(pool.atlas.params[~rep], prm0[~rep])
    assert not bool((pool.atlas.params[rep] == prm0[rep]).all(1).any())
    # The new slots: the first good / impossible valid scenarios of the stream after the
    # previous fill's prefix (candidates skipped inside a prefix are discarded)
    G, I, vid, g = stream_groups(pool, sc, pool.cursor, c0)
    assert torch.equal(pool.atlas.params[:kg], G[:kg])
    assert torch.equal(pool.atlas.params[ng:ng + ki], I[:ki])
    assert pool.cursor == c0 + max(int(vid[g][kg - 1]), int(vid[~g][ki - 1])) + 1
    assert torch.equal(pool.pmax[rep], pool.atlas.pmax[rep].clamp(min=1e-6))
    # Stale generation: an episode of a replaced slot does not score it
    s = torch.tensor([0, kg], device=DEV)
    pool.observe(s, 0.5 * pool.atlas.pmax[s], torch.ones(2, dtype=torch.bool, device=DEV),
                 gen=torch.zeros(2, dtype=torch.long, device=DEV))
    assert not bool(pool.seen[0]) and bool(pool.seen[kg]), "stale episode scored a new slot"
    pool.observe(s[:1], 0.5 * pool.atlas.pmax[s[:1]], torch.ones(1, dtype=torch.bool, device=DEV),
                 gen=pool.slot_gen[s[:1]])
    assert bool(pool.seen[0])
    # The ring wraps: every slot replaced after ceil(slots / replaced) refreshes
    m = max(math.ceil(ng / kg), math.ceil(ni / ki))
    for it in range(2, m + 1):
        pool.refresh(it)
    assert bool((pool.slot_gen[:ng] >= 1).all()) and bool((pool.slot_gen[ng:] >= 1).all())
    assert int(pool.slot_gen[:ng].sum()) == m * kg and pool.ptr == [(m * kg) % ng, (m * ki) % ni]
    st = pool.stats()
    for k in ("pool/refresh_s", "pool/candidates", "pool/angle_rejected_share", "pool/clipped_share",
              "pool/good_share_sep80-100", "pool/hist_sep100-120", "plr/top_laser_dx"):
        assert k in st, k
    print(f"  refresh 1 replaced good slots 0-{kg - 1} and impossible slots {ng}-{ng + ki - 1} "
          f"(oldest first) with the next stream scenarios after the fill's prefix: scores reset, generation + 1, others "
          f"unchanged; nothing within an iteration; stale-generation episodes ignored; after {m} "
          f"refreshes every slot replaced (ptr {pool.ptr})")


# ── 4. Priorities ───────────────────────────────────────────────────────────


def test_priorities():
    print("\n== 4. priorities within the pool ==")
    beta, rho, pf = 1.0, 0.1, 0.3
    pool = make_pool(frac=0.25, beta=beta, rho=rho, p_fresh=pf)
    ng, ni = pool.n_good_slots, pool.n_imp_slots
    gen = torch.Generator(device=DEV)
    gen.manual_seed(4)
    seen = torch.zeros(ng, dtype=torch.bool, device=DEV)
    seen[torch.randperm(ng, generator=gen, device=DEV)[:40]] = True
    score = torch.rand(ng, generator=gen, device=DEV)
    last = torch.randint(0, 3, (ng,), generator=gen, device=DEV)
    pool.seen[:ng], pool.score[:ng], pool.last[:ng] = seen, score, last
    pool.refresh(0)  # No replacement at iteration 0
    pool.iteration = 3
    pool.refresh(0)
    # Expected replay distribution (computed here, not by the sampler)
    pool.iteration = 0
    idx = torch.nonzero(seen).squeeze(1)
    order = idx[torch.argsort(score[idx], descending=True)]
    h = torch.zeros(ng, dtype=torch.float64, device=DEV)
    h[order] = torch.arange(1, order.numel() + 1, device=DEV).double() ** (-1.0 / beta)
    ps = h / h.sum()
    stale = torch.where(seen, (0 - last).double(), 0.0)
    pc = stale / stale.sum() if float(stale.sum()) > 0 else seen.double() / seen.sum()
    want_r = (1 - rho) * ps + rho * pc
    assert torch.allclose(pool.p_replay, want_r, atol=1e-12)
    w = torch.cat((torch.full((ni,), 0.1 / ni, dtype=torch.float64, device=DEV),
                   0.9 * pf / ng + 0.9 * (1 - pf) * want_r))
    s, src = pool.draw(torch.rand(400000, generator=gen, device=DEV))
    freq = torch.bincount(s, minlength=64).double() / s.numel()
    want = torch.cat((w[ni:], w[:ni]))  # Slot order: good slots first
    err = float((freq - want).abs().max())
    assert err < 0.004, err
    top = int(torch.argmax(want_r))
    print(f"  40 of {ng} good slots scored: replay = 0.9 rank^-1 + 0.1 staleness exactly; draw "
          f"frequencies over 400k draws within {err:.4f} of 10 % impossible / 27 % fresh / 63 % "
          f"replay (top slot {top}, {100 * float(want[top]):.1f} %)")


# ── 5. Env in pool mode ─────────────────────────────────────────────────────


def test_env_pool():
    print("\n== 5. env in pool mode ==")
    pool = make_pool(frac=0.25)
    env = AlignEnv(EnvConfig(num_envs=256, history_len=0, seed=5, max_steps=12), pool=pool)
    at = pool.atlas
    assert env.n_setups == 64 and env.path_ref_mode == "scenario"
    assert torch.equal(env.scen_params, at.params[env.setup]) and torch.equal(env.scen_ref, at.path_ref[env.setup])
    gen = torch.Generator(device=DEV)
    gen.manual_seed(6)
    c = env.cfg.step_cost
    p0, ref0 = env.path.clone(), env.scen_ref.clone()
    ret = torch.zeros(256, dtype=torch.float64, device=DEV)
    moves = torch.zeros(256, dtype=torch.long, device=DEV)
    errs, n_ep, kept = [], 0, 0
    for t in range(90):
        if t in (30, 60):
            before, setup_b, gen_b = env.scen_params.clone(), env.setup.clone(), env.scen_gen.clone()
            pool.refresh(t // 30)
            assert torch.equal(env.scen_params, before), "a refresh changed an episode in flight"
        a = torch.randint(0, env.n_bins, (256, 2), generator=gen, device=DEV)
        a = torch.where((torch.rand(256, generator=gen, device=DEV) < 0.08)[:, None], 0, a)
        _, rew, term, trunc, info = env.step(a)
        done = term | trunc
        if t in (30, 60):  # The step after a refresh: in-flight episodes keep their copies
            assert torch.equal(info["scenario_params"], before) and torch.equal(info["scenario_gen"], gen_b)
            stale = info["scenario_gen"] != pool.slot_gen[setup_b]
            kept += int(stale.sum())
        assert torch.equal(info["path_ref"], ref0) and torch.equal(info["path_norm"], info["path_mm"] / ref0)
        pool.observe(info["setup"], info["ep_final_path_mm"], done, info["scenario_src"], info["scenario_gen"])
        ret += rew.double()
        moves += (a != 0).any(1).long()
        if bool(done.any()):
            tele = (info["ep_final_path_mm"][done].double() - p0[done].double()) / ref0[done].double()
            errs.append(float((ret[done] - (tele - c * moves[done].double())).abs().max()))
            n_ep += int(done.sum())
        ret = torch.where(done, 0.0, ret)
        moves = torch.where(done, 0, moves)
        p0 = torch.where(done, env.path, p0)
        ref0 = env.scen_ref.clone()
        fresh = done & (env.scen_gen == pool.slot_gen[env.setup])
        assert torch.equal(env.scen_params[fresh], at.params[env.setup[fresh]])
    out = env.sim.simulate(env.scenario_states(env.tilt, env.scen_params), store_world=False)
    assert torch.equal(env.path, out["total_path_length"])
    assert max(errs) < 1e-4 and kept > 0
    st = env.get_state()
    assert "scen_params" in st and "scen_gen" in st

    # No host syncs
    acts = [torch.randint(0, env.n_bins, (256, 2), generator=gen, device=DEV) for _ in range(5)]
    mask = torch.rand(256, generator=gen, device=DEV) < 0.5
    explicit = torch.randint(0, 64, (256,), generator=gen, device=DEV)
    env.step(acts[0])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for a in acts:
            _, _, term, trunc, info = env.step(a)
            pool.observe(info["setup"], info["ep_final_path_mm"], term | trunc, info["scenario_src"],
                         info["scenario_gen"])
        env.reset(mask, return_info=True)
        env.reset(mask, tilts=env.tilt, setups=explicit)
        pool.draw(torch.rand(256, device=DEV))
        env.state_info()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert torch.equal(env.scen_params[mask], at.params[explicit[mask]])
    print(f"  {n_ep} episodes: max |sum r - telescoped| {max(errs):.1e} with the episode's P_ref; "
          f"state = direct sim of the per-env parameters; two refreshes left {kept} in-flight "
          f"episodes on replaced slots unchanged; steps, resets (drawn / explicit slots), draw, "
          f"observe: no sync")


# ── 6. Sim at 20 and 400 mm ─────────────────────────────────────────────────


def test_sim_extremes(sc: Scanner):
    print("\n== 6. sim at 20 and 400 mm ==")
    msg = []
    for lo, hi in ((20.0, 20.0 + 1e-9), (400.0 - 1e-9, 400.0)):
        r = ContRanges(sep=(lo, hi))
        prm, info = sample_valid_scenarios(48, r, seed=2, scanner=sc, round_n=512)
        d = sc.scan(torch.from_numpy(prm))
        assert bool(torch.isfinite(d["pmax"]).all()) and bool(torch.isfinite(d["path_ref"]).all())
        grid, _ = box_grid(2.0, 0.01, DEV)
        out = sc.sim.simulate(box_states(sc.geom, d["params"][:8], grid), store_world=False,
                              store_hits=False)
        p, b = out["total_path_length"], out["hit_counts"]
        assert bool(torch.isfinite(p).all()) and bool((p >= 0).all())
        assert int(b.min()) >= 0 and int(b.max()) <= 128
        c = info["candidates"]
        msg.append(f"{lo:.0f} mm: valid {48 / c:.2f} of candidates (angles out of bounds "
                   f"{info['angle_rejected'] / c:.2f}, clipped {info['clipped'] / c:.2f}), good "
                   f"{int(d['good'].sum())} / 48, P_max {float(d['pmax'].min()) / 1000:.2f}-"
                   f"{float(d['pmax'].max()) / 1000:.2f} m, max bounces {int(d['best_bounces'].max())}")
    print("  finite paths, bounce counts within the cap; " + "; ".join(msg))


# ── 7. eval_cont ────────────────────────────────────────────────────────────


def test_eval_cont(tmp: str, atlas_path: str):
    print("\n== 7. eval_cont ==")
    ccfg = ContConfig(ranges=ContRanges(sep=(60.0, 140.0)), n=48, seed=5)
    a = build_cont_table(ccfg, device=DEV, log=lambda s: None)
    b = build_cont_table(replace(ccfg, round_n=40), device=DEV, log=lambda s: None)
    for k in ("params", "pmax", "path_ref", "good", "candidate_index", "near_cells"):
        assert torch.equal(a[k], b[k]), k
    c = build_cont_table(replace(ccfg, seed=6), device=DEV, log=lambda s: None)
    assert not torch.equal(a["params"][:4], c["params"][:4])
    info = a["offgrid"]
    assert info["candidates"] == info["angle_rejected"] + info["clipped"] + info["missed"] + info["valid"]
    prm = a["params"].numpy().astype(np.float64)
    on, clip, _ = full_first_hit(Scanner(device=DEV), prm)
    assert bool(on.all()) and float(a["params"][:, 3:5].norm(dim=1).max()) < 1.5
    path = cont_table_path(ccfg, tmp)
    assert os.path.basename(path) == "cont60-140_n48_seed5_v1.pt"
    t = load_or_build_cont(path, ccfg, device=DEV, log=lambda s: None)
    mt = os.path.getmtime(path)
    t2 = load_or_build_cont(path, ccfg, device=DEV, log=lambda s: None)
    assert os.path.getmtime(path) == mt and torch.equal(t.params, t2.params)
    try:
        load_or_build_cont(path, replace(ccfg, impossible_share=0.2), device=DEV, log=lambda s: None)
        raise RuntimeError("other settings accepted")
    except AssertionError:
        pass
    scen = offgrid_eval_scenarios(t, 48, 0, 0.1)
    assert scen.unique().numel() == 48 and int((~t.good[scen.to(DEV)]).sum()) == 5
    assert cont_sep_bins(20, 400) == ((20, 50), (50, 100), (100, 150), (150, 200), (200, 250),
                                      (250, 300), (300, 350), (350, 400))
    # The trainer's set
    cfg = PPOConfig(setups="full", scenarios="pool", sep_range=(60.0, 140.0), eval_n=48,
                    eval_envs=48, eval_seed=5, eval_tables=tmp, history_len=4)
    os.makedirs(os.path.join(tmp, "cache"), exist_ok=True)
    es = cont_eval_set(cfg, os.path.join(tmp, "cache"), DEV, log=lambda s: None)
    assert es.prefix == "eval_cont" and es.bins == ((60.0, 100.0), (100.0, 140.0))
    assert torch.equal(es.table.params, t.params)
    mo = os.path.getmtime(os.path.join(tmp, "cache", "oracle_cont.pt"))
    es2 = cont_eval_set(cfg, os.path.join(tmp, "cache"), DEV, log=lambda s: None)
    assert os.path.getmtime(os.path.join(tmp, "cache", "oracle_cont.pt")) == mo
    assert torch.equal(es.setups, es2.setups)
    print(f"  table of 48 episodes ({info['candidates']} candidates, {info['valid']} valid): the same "
          f"with rounds of 40, another seed differs; every row valid; cached and loaded, other "
          f"settings refused; 43 good + 5 impossible, one episode each; trainer set eval_cont/ "
          f"with bins {es.bins}, oracle cached")


# ── 8. Init-from ────────────────────────────────────────────────────────────


def test_init_from(tmp: str):
    print("\n== 8. init-from with a changed scenario head ==")
    base = dict(window=6, d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False)
    torch.manual_seed(1)
    src = Agent(AgentConfig(**base)).to(DEV)  # Atlas head: 3 parameters, 50-200 mm
    with torch.no_grad():
        for p_ in src.parameters():
            p_.add_(0.01 * torch.randn_like(p_))
    vn = ValueNorm(0.99, DEV)
    vn.update(torch.tensor([1.0, 3.0], device=DEV))
    path = os.path.join(tmp, "4407578", "best.pt")
    os.makedirs(os.path.dirname(path))
    torch.save({"agent": src.state_dict(), "agent_cfg": src.cfg.to_dict(), "vnorm": vn.state_dict(),
                "name": "atlas_run", "iteration": 1200, "env_steps": 157_286_400,
                "encoder_info": {"name": "cnn_w160_lr0.001_b256_16k_s0_grid16384"}}, path)
    lo5, hi5 = ContRanges().head_range()
    img = torch.randint(0, 256, (8, 1, 64, 64), dtype=torch.uint8, device=DEV)
    z0 = torch.zeros(8, 2, device=DEV)
    first = torch.ones(8, dtype=torch.bool, device=DEV)
    scen_keys = ["scenario.0.bias", "scenario.0.weight", "scenario.2.bias", "scenario.2.weight"]
    res = {}
    for label, kw in (("5 parameters, 20-400 mm", dict(scen_lo=lo5, scen_hi=hi5)),
                      ("3 parameters, 20-400 mm", dict(scen_lo=(20.0, 0.0, 0.0), scen_hi=(400.0, 8.0, 8.0))),
                      ("3 parameters, 50-200 mm", {})):
        torch.manual_seed(2)
        dst = Agent(AgentConfig(**base, **kw)).to(DEV)
        info = dst.init_from(path)
        with torch.no_grad():
            oa, _ = src.act(img, z0, first, src.init_ring(8, DEV))
            ob, _ = dst.act(img, z0, first, dst.init_ring(8, DEV))
        for k in ("z", "logits", "value", "path", "tilt_mu", "tilt_logvar"):
            assert torch.equal(oa[k], ob[k]), (label, k)
        if kw:
            assert sorted(info["skipped"]) == scen_keys and info["missing"] == scen_keys, label
            assert info["scen_reinit"] and all("re-initialized" in v for v in info["skipped"].values())
            assert ob["scen_mu"].shape[-1] == len(kw["scen_lo"])
        else:
            assert info["skipped"] == {} and info["missing"] == [] and info["scen_reinit"] is None
            assert torch.equal(oa["scen_mu"], ob["scen_mu"])
        res[label] = len(info["loaded"])
    # Trainer in pool mode from that checkpoint
    cfg = PPOConfig(setups="full", scenarios="pool", sep_range=(80.0, 120.0), pool_size=64,
                    pool_refresh=0.25, num_envs=16, rollout_len=8, max_steps=6, history_len=6,
                    d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False, compile=False,
                    wandb=False, minibatches=4, eval_every=0, total_env_steps=10**6,
                    plr_min_seen=4, init_from=path)
    tr = Trainer(cfg, os.path.join(tmp, "pool_run"), build_eval=False)
    assert tr.agent.n_scen == 5 and tr.agent.cfg.scen_lo == (80.0, -8.0, -8.0, -1.5, -1.5)
    assert tr.init_info["scen_reinit"] and tr.best_metric == "eval_cont/all/pct_oracle"
    assert tr.name.startswith("ppo_cnn-pregrid16384_k6_cont80-120-plr_b16_s0_ft4407578_"), tr.name
    assert eval_set_names(cfg) == ["cont", "full", "offgrid", "extrap_short", "extrap_long", "set4"]
    m, s = tr.vnorm.stats()
    assert torch.equal(m, vn.stats()[0]) and torch.equal(s, vn.stats()[1])
    tr.sampler.refresh(0)
    tr.rollout()
    tr.update()
    assert tr.buf.scen.shape[-1] == 5 and float(tr.buf.scen.abs().max()) <= 1.0 + 1e-6
    legacy = config_from_dict({"setups": "full", "atlas": "x", "init_center": [0.0, 0.0]})
    assert legacy.scenarios == "atlas" and isinstance(legacy.init_center, tuple)
    assert config_from_dict({"setups": "full", "scenarios": "pool"}).scenarios == "pool"
    nm = run_name(replace(cfg, laser_range=(-4.0, 4.0), laser_offset_radius=0.0, laser_mode="angle"), None)
    assert "_cont80-120-lp4-off0-ang-plr_" in nm, nm
    print(f"  {res['5 parameters, 20-400 mm']} tensors loaded with the scenario head re-initialized "
          f"(5 parameters / 3 parameters with another range), all {res['3 parameters, 50-200 mm']} with "
          f"the same normalization; outputs of the shared parts identical; pool trainer {tr.name}, "
          f"scenario labels in [-1, 1], legacy checkpoints -> scenarios atlas")


# ── 9. Encoder ──────────────────────────────────────────────────────────────


def test_encoder(sc: Scanner, tmp: str):
    print("\n== 9. encoder fine-tuning and continuous frame data ==")
    torch.manual_seed(3)
    cfg = EncoderConfig()
    srcm = FrameModel(cfg).to(DEV).eval()
    with torch.no_grad():
        for p_ in srcm.parameters():
            p_.add_(0.01 * torch.randn_like(p_))
    path = os.path.join(tmp, "src_model.pt")
    torch.save({"cfg": dataclasses.asdict(cfg), "model": srcm.state_dict(), "step": 16000,
                "name": "cnn_w160_lr0.001_b256_16k_s0_grid16384"}, path)
    ck = pe.load_init(path)
    dst = FrameModel(pe.init_config(ck)).to(DEV).eval()
    info = pe.init_model(dst, ck, path)
    assert info["skipped"] == {} and info["missing"] == [] and info["loaded"] == len(srcm.state_dict())
    img = torch.randint(0, 256, (16, 1, 64, 64), dtype=torch.uint8, device=DEV)
    with torch.no_grad():
        oa, ob = srcm(img), dst(img)
    for k in oa:
        assert torch.equal(oa[k], ob[k]), k
    args = argparse.Namespace(steps=8000, lr=3e-4, batch=256, seed=0, init=path, scenarios=16384,
                              sep_range=[20.0, 400.0], laser_range=[-8.0, 8.0], laser_offset_radius=1.5,
                              laser_mode="landing", setups="all", tag="")
    nm = pe.run_name(args, cfg, ck["name"])
    assert nm == "cnn_w160_lr0.0003_b256_8k_s0_ft-grid16384_cont20-400x16384", nm
    assert encoder_origin({"name": nm}) == "cnn-ftcont"
    assert encoder_origin({"name": "cnn_w160_lr0.001_b256_16k_s0_cont20-400x16384"}) == "cnn-precont"
    assert encoder_origin({"name": ck["name"]}) == "cnn-pregrid16384"
    assert pe.init_origin("cnn_w160_lr0.001_b256_8k_s0_set4") == "set4"
    assert pe.init_origin("cnn_w160_lr0.001_b256_8k_s0") == "meas"
    # Continuous frame data: the generator's scenarios, no offset jitter
    pts, rej = sample_valid_scenarios(32, ContRanges(), seed=0, stream=STREAM_ENCODER, scanner=sc)
    setups = tuple(SETUPS) + tuple(scenario_geometry(p) for p in pts)
    data = FrameData(DataConfig(batch_size=64, setups=setups, scan_step=0.05, jitter_laser_offset=0.0), DEV)
    base = data.setup_base[len(SETUPS):]
    assert torch.equal(base[:, [S_LASER_DX, S_LASER_DY]], torch.tensor(pts[:, 3:5], dtype=torch.float32, device=DEV))
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    bt = data.batch(gen)
    off = bt["state"][:, [S_LASER_DX, S_LASER_DY]]
    sc_rows = bt["setup"] >= len(SETUPS)
    assert float(off[sc_rows].norm(dim=1).max()) < 1.5 and bt["image"].shape == (64, 1, 64, 64)
    print(f"  --init: {info['loaded']} tensors, outputs identical to the source before any step; "
          f"run name {nm} -> PPO encoder origin cnn-ftcont; 32 continuous scenarios "
          f"({rej['candidates']} candidates) as frame-data setups, batch offsets inside the disk")


# ── 10. Pool-mode resume ────────────────────────────────────────────────────


def test_pool_resume(tmp: str):
    print("\n== 10. pool-mode trainer: save / resume ==")
    d = os.path.join(tmp, "resume_run")
    cfg = PPOConfig(setups="full", scenarios="pool", sep_range=(80.0, 120.0), pool_size=64,
                    pool_refresh=0.25, num_envs=16, rollout_len=24, max_steps=6, history_len=6,
                    d_model=64, n_layers=2, n_heads=4, mlp_ratio=2.0, amp=False, compile=False,
                    wandb=False, minibatches=4, eval_every=0, seed=7, total_env_steps=10**6,
                    plr_min_seen=2)
    a = Trainer(cfg, d, build_eval=False)
    for _ in range(2):
        a.sampler.refresh(a.iteration)
        a.rollout()
        a.update()
        a.iteration += 1
    a.save("latest.pt")
    a.sampler.refresh(a.iteration)
    a.rollout()
    snap = {k: getattr(a.buf, k).clone() for k in ("frames", "actions", "logp", "rewards", "scen",
                                                   "first")}
    sc, prm, cur = a.sampler.score.clone(), a.pool.atlas.params.clone(), a.pool.cursor
    bt = Trainer(cfg, d, build_eval=False)
    bt.load(os.path.join(d, "latest.pt"))
    bt.sampler.refresh(bt.iteration)
    bt.rollout()
    same = {k: bool(torch.equal(v, getattr(bt.buf, k))) for k, v in snap.items()}
    same["pool slots"] = bool(torch.equal(prm, bt.pool.atlas.params))
    same["stream position"] = cur == bt.pool.cursor
    same["scores"] = bool(torch.equal(sc, bt.sampler.score))
    same["slot generations"] = bool(torch.equal(a.pool.slot_gen, bt.pool.slot_gen))
    print("  next rollout after resume identical: " + ", ".join(f"{k} {v}" for k, v in same.items()))
    assert all(same.values()) and int(a.pool.slot_gen.sum()) > 0


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    tmp = tempfile.mkdtemp()
    try:
        from scenarios import build_atlas, save_atlas

        TINY = AtlasConfig(sep=(86.0, 90.0, 4.0), pitch=(0.0, 8.0, 4.0), yaw=(0.0, 8.0, 4.0))
        atlas_path = os.path.join(tmp, "tiny.pt")
        save_atlas(build_atlas(TINY, DEV, log=lambda s: None), atlas_path)
        sc = Scanner(device=DEV)
        test_generator(sc)
        test_pool_fill(sc)
        test_refresh(sc)
        test_priorities()
        test_env_pool()
        test_sim_extremes(sc)
        test_eval_cont(tmp, atlas_path)
        test_init_from(tmp)
        test_encoder(sc, tmp)
        test_pool_resume(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nall tests passed")
