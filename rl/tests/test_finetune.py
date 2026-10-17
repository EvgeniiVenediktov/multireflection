"""
tests/test_finetune.py - Hard-target fine-tuning options: near-best starts, region quotas
of the pool, best.pt on the region-bin mean; defaults bit-identical to the code before them

Run from rl/:  uv run python tests/test_finetune.py

    1. near-best starts (env, pool mode): resets and auto-resets of good scenarios start,
       with the per-episode coin of the second hash, on the ring r in [RMIN, RMAX] around
       the slot's best tilt (pmax_tilt), inside the +-2 deg box (clamped rows on its edge);
       share F of the good-scenario resets; impossible scenarios and the other episodes
       start exactly where the F = 0 env starts (same scenario draws); explicit slots in
       reset(); radius mean and angle uniform; no host syncs in step / reset / draw
    2. region quotas of the pool: region_quotas / parse_region_mix; the good slots hold
       exactly the quotas per target-region bin, they are the first good scenarios of each
       bin in the stream, in stream order, their region sizes = direct 0.01 deg scans;
       a refresh refills each replaced slot with the next stream scenario of its bin (the
       shares stay the quotas over a full ring); stats keys; state dict round trip; a
       checkpoint from before region_mix loads into a default pool, a quota checkpoint
       is refused by it
    3. best metric and trainer: region_bin_mean; --best-metric region needs the region set;
       run-name markers; train() with all three options: best/value of every eval row =
       the mean of the four eval_region/ bins' deterministic % of the oracle, best.pt at
       its maximum; resume: --near-best-frac and --best-metric follow the flags,
       --pool-region-mix stays the checkpoint's, the best value resets with a new metric
    4. defaults bit-identical: in subprocesses, the code at REF_REV (env.py, scenarios.py,
       ppo.py before these options) and the current code give the same pool slots, rollout
       buffers, parameters after three PPO iterations and env trajectories (four setups,
       explicit resets); skipped when git or REF_REV is unavailable
"""

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env import AlignEnv, EnvConfig, uniforms  # noqa: E402
from find_setups import box_grid  # noqa: E402
from ppo import (REGION_BIN_MEAN, PPOConfig, Trainer, region_bin_mean, resume_config,  # noqa: E402
                 run_name)
from scenarios import (REGION_BINS, STREAM_POOL, ContRanges, PoolConfig, SamplerConfig,  # noqa: E402
                       Scanner, ScenarioPool, box_states, candidate_batch, parse_region_mix,
                       region_bin, region_quotas)

DEV = torch.device("cuda")
NARROW = ContRanges(sep=(80.0, 120.0))
REF_REV = "493776e"  # Last commit before the fine-tuning options
MIX = (0.2, 0.2, 0.25, 0.35)
q = lambda s: None  # noqa: E731


def make_pool(size=64, frac=0.25, region_mix=(), seed=0) -> ScenarioPool:
    return ScenarioPool(PoolConfig(size=size, ranges=NARROW, refresh_frac=frac, seed=seed,
                                   round_min=64, round_max=2048, region_mix=region_mix),
                        SamplerConfig(min_seen=4, beta=1.0, rho=0.1), device=DEV, log=q)


def ring_check(tilt, best, near, r0, r1, limit):
    """Near rows: inside the box, distance to the best tilt in [r0, r1] (clamped rows: on
    the box edge, <= r1). Returns distances and angles of the rows whose best tilt lies
    at least r1 inside the box (no clamping possible: unbiased), clamped count."""
    t, b = tilt[near], best[near]
    d = t - b
    r = d.norm(dim=1)
    assert bool((t.abs() <= limit).all()), "near-best start outside the box"
    clamped = (t.abs() >= limit - 1e-6).any(1)
    ok = (r >= r0 - 1e-5) & (r <= r1 + 1e-5)
    assert bool(ok[~clamped].all()), f"near-best start off the ring: {r[~clamped][~ok[~clamped]][:5]}"
    assert bool((r[clamped] <= r1 + 1e-5).all())
    inner = (b.abs() <= limit - r1).all(1)
    assert not bool((clamped & inner).any())
    return r[inner], torch.atan2(d[inner, 1], d[inner, 0]), int(clamped.sum())


# ── 1. Near-best starts ─────────────────────────────────────────────────────


def test_near_best():
    print("\n== 1. near-best starts ==")
    F, R = 0.25, (0.05, 0.5)
    pool = make_pool()
    B = 4096
    env = AlignEnv(EnvConfig(num_envs=B, history_len=0, seed=11, max_steps=8, near_best_frac=F,
                             near_best_radius=R), pool=pool)
    twin = AlignEnv(EnvConfig(num_envs=B, history_len=0, seed=11, max_steps=8), pool=pool)
    at, L = pool.atlas, env.limit
    try:
        AlignEnv(EnvConfig(num_envs=4, near_best_frac=0.1))
        raise RuntimeError("near-best starts without the pool accepted")
    except AssertionError:
        pass
    n_good = n_near = n_clamp = 0
    radii, angles = [], []
    for k in range(4):  # Explicit resets (episodes 0..3)
        if k:
            env.reset()
            twin.reset()
        assert torch.equal(env.setup, twin.setup) and torch.equal(env.scen_params, twin.scen_params)
        good = at.good[env.setup]
        near = (env.tilt != twin.tilt).any(1)
        assert not bool((near & ~good).any()), "an impossible scenario started near its best tilt"
        assert torch.equal(env.tilt[~near], twin.tilt[~near])
        v = uniforms(env._near_key, env.episode)
        assert torch.equal(near, good & (v[:, 0] < F)), "near-best starts != the per-episode coin"
        r, a, nc = ring_check(env.tilt, at.pmax_tilt[env.setup], near, *R, L)
        radii.append(r)
        angles.append(a)
        n_good += int(good.sum())
        n_near += int(near.sum())
        n_clamp += nc
    # Auto-resets inside step()
    gen = torch.Generator(device=DEV)
    gen.manual_seed(2)
    n_done = n_dnear = 0
    for t in range(12):
        a = torch.randint(0, env.n_bins, (B, 2), generator=gen, device=DEV)
        a = torch.where((torch.rand(B, generator=gen, device=DEV) < 0.4)[:, None], 0, a)
        _, _, term, trunc, info = env.step(a)
        done = term | trunc
        u_t, u_s, _ = twin._draw_src(env.episode)  # Uniform starts of the new episodes
        assert torch.equal(env.setup[done], u_s[done])
        good = at.good[env.setup]
        near = done & (env.tilt != u_t).any(1)
        assert not bool((near & ~good).any())
        assert torch.equal(env.tilt[done & ~near], u_t[done & ~near])
        v = uniforms(env._near_key, env.episode)
        assert torch.equal(near, done & good & (v[:, 0] < F))
        assert torch.equal(info["obs_tilt"], env.tilt)
        r, a_, nc = ring_check(env.tilt, at.pmax_tilt[env.setup], near, *R, L)
        radii.append(r)
        angles.append(a_)
        n_good += int((done & good).sum())
        n_near += int(near.sum())
        n_clamp += nc
        n_done += int(done.sum())
        n_dnear += int(near.sum())
    share = n_near / n_good
    assert abs(share - F) < 0.012, share
    r, a = torch.cat(radii), torch.cat(angles)  # Best tilts >= RMAX inside the box: unbiased
    n = r.numel()
    se_r, se_a = (R[1] - R[0]) / math.sqrt(12 * n), math.sqrt(0.5 / n)
    assert n > 500 and abs(float(r.mean()) - sum(R) / 2) < 4 * se_r, (n, float(r.mean()))
    assert abs(float(torch.cos(a).mean())) < 4 * se_a and abs(float(torch.sin(a).mean())) < 4 * se_a
    # Explicit slots in reset(): the ring of the given slot
    mask = torch.rand(B, generator=gen, device=DEV) < 0.5
    explicit = torch.randint(0, pool.atlas.n, (B,), generator=gen, device=DEV)
    ep = torch.where(mask, env.episode + 1, env.episode)
    env.reset(mask, setups=explicit)
    u_t = twin._draw_src(ep)[0]
    assert torch.equal(env.setup[mask], explicit[mask])
    near = mask & (env.tilt != u_t).any(1)
    assert torch.equal(near, mask & at.good[explicit] & (uniforms(env._near_key, ep)[:, 0] < F))
    assert torch.equal(env.tilt[mask & ~near], u_t[mask & ~near])
    ring_check(env.tilt, at.pmax_tilt[explicit], near, *R, L)
    # No host syncs
    acts = [torch.randint(0, env.n_bins, (B, 2), generator=gen, device=DEV) for _ in range(4)]
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for a_ in acts:
            _, _, term, trunc, info = env.step(a_)
            pool.observe(info["setup"], info["ep_final_path_mm"], term | trunc, info["scenario_src"],
                         info["scenario_gen"])
        env.reset(mask, return_info=True)
        env.reset(mask, setups=explicit)
        env.reset(mask, tilts=env.tilt, setups=explicit)
        env._sample_tilts()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    print(f"  {n_good} good-scenario starts ({n_done} auto-resets): {100 * share:.2f} % near the "
          f"best tilt (F {F}), all on the ring {R[0]}-{R[1]} deg inside the box ({n_clamp} "
          f"clamped to its edge), mean r {float(r.mean()):.3f}, angle uniform; impossible and "
          f"other starts = the F = 0 env; explicit slots; steps / resets without host sync")


# ── 2. Region quotas ────────────────────────────────────────────────────────


def stream_rows(pool: ScenarioPool, sc: Scanner, start: int, end: int):
    """Valid scenarios of the pool's stream in [start, end): params, good, region bin,
    stream positions (all in stream order)."""
    b = candidate_batch(pool.pcfg.ranges, pool.pcfg.seed, start, end - start, STREAM_POOL, sc)
    vid = np.nonzero(b["valid"])[0]
    d = sc.scan(torch.from_numpy(b["params"][vid]))
    rb = region_bin(d["near_cells"].double() * pool.cell)
    return d["params"], d["good"], rb, torch.from_numpy(vid + start).to(DEV)


def direct_near(sc: Scanner, params: torch.Tensor) -> int:
    grid, _ = box_grid(2.0, 0.01, DEV)
    out = sc.sim.simulate(box_states(sc.geom, params[None], grid), store_world=False,
                          store_hits=False)
    p = out["total_path_length"]
    return int((p >= 0.9 * p.max()).sum())


def test_region_quotas(sc: Scanner):
    print("\n== 2. region quotas ==")
    assert region_quotas(MIX, 230) == [46, 46, 58, 80]  # 57.5 / 80.5: the tie to the lower bin
    assert region_quotas((0.25,) * 4, 58) == [15, 15, 14, 14] and sum(region_quotas(MIX, 14746)) == 14746
    assert parse_region_mix("") == () and parse_region_mix("0.2,0.2,0.25,0.35") == MIX
    for bad in ("0.5,0.5", "0.3,0.3,0.3,0.3", "-0.1,0.4,0.4,0.3"):
        try:
            parse_region_mix(bad)
            raise RuntimeError(f"region mix {bad} accepted")
        except AssertionError:
            pass
    pool = make_pool(region_mix=MIX)
    at, ng, ni = pool.atlas, pool.n_good_slots, pool.n_imp_slots
    quota = region_quotas(MIX, ng)
    assert pool.quotas == quota
    rb = pool.slot_rbin()
    assert torch.bincount(rb[:ng], minlength=4).tolist() == quota
    assert bool(at.good[:ng].all()) and not bool(at.good[ng:].any())
    # The slots: the first quota[j] good scenarios of bin j and the first impossible ones
    prm, good, srb, pos = stream_rows(pool, sc, 0, pool.cursor)
    take = torch.zeros_like(good)
    for j in range(4):
        take[torch.nonzero(good & (srb == j)).squeeze(1)[:quota[j]]] = True
    imp = torch.nonzero(~good).squeeze(1)[:ni]
    assert torch.equal(at.params[:ng], prm[take]), "good slots != the per-bin stream prefix"
    assert torch.equal(at.params[ng:], prm[imp])
    assert pool.cursor == int(max(pos[take].max(), pos[imp].max())) + 1
    # Region sizes of slots = direct scans
    checked = 0
    for j in range(4):
        for i in torch.nonzero(rb[:ng] == j).squeeze(1)[:2].tolist():
            n = direct_near(sc, at.params[i])
            area = torch.tensor([n * 0.01 ** 2], dtype=torch.float64)
            assert int(at.near_cells[i]) == n and int(region_bin(area)) == j
            checked += 1
    st0 = pool.stats()
    for k in ("pool/region_tiny", "pool/region_large", "pool/rejected", "pool/candidates",
              "pool/refresh_s", "pool/scans"):
        assert k in st0, k
    assert abs(st0["pool/region_tiny"] - quota[0] / ng) < 1e-12 and st0["pool/rejected"] > 0
    # Refresh: each replaced slot gets the next stream scenario of its bin
    kg, ki = round(0.25 * ng), round(0.25 * ni)
    rb0, prm0, c0 = rb.clone(), at.params.clone(), pool.cursor
    pool.refresh(1)
    assert torch.equal(pool.slot_rbin()[:ng], rb0[:ng]), "a refresh changed a slot's region bin"
    sg = torch.arange(kg, device=DEV)
    need = torch.bincount(rb0[sg], minlength=4).tolist()
    prm, good, srb, pos = stream_rows(pool, sc, c0, pool.cursor)
    for j in range(4):
        slots_j = sg[rb0[sg] == j]
        rows = torch.nonzero(good & (srb == j)).squeeze(1)[:need[j]]
        assert torch.equal(at.params[slots_j], prm[rows]), j
    assert torch.equal(at.params[kg:ng], prm0[kg:ng]) and torch.equal(at.params[ng + ki:], prm0[ng + ki:])
    assert torch.equal(at.params[ng:ng + ki], prm[torch.nonzero(~good).squeeze(1)[:ki]])
    refresh_counts = []
    for it in range(2, 2 + math.ceil(ng / kg) + 1):  # A full ring and more
        pool.refresh(it)
        refresh_counts.append(pool.last_fill["candidates"])
        assert torch.bincount(pool.slot_rbin()[:ng], minlength=4).tolist() == quota
    assert int(pool.slot_gen[:ng].min()) >= 1
    # State dict round trip; a checkpoint from before region_mix
    sd = pool.state_dict()
    p2 = make_pool(region_mix=MIX)
    p2.load_state_dict(sd)
    assert torch.equal(p2.atlas.params, at.params) and p2.cursor == pool.cursor
    assert np.array_equal(p2.cum_rbin, pool.cum_rbin) and p2.quotas == quota
    dflt = make_pool()
    old = dflt.state_dict()
    del old["pool_cfg"]["region_mix"], old["cum_rbin"]
    make_pool().load_state_dict(old)
    try:
        make_pool().load_state_dict(sd)
        raise RuntimeError("a quota checkpoint loaded into a default pool")
    except AssertionError:
        pass
    nat = dflt.stats()
    print(f"  {ng} good slots: quotas {quota} (mix {MIX}) realized exactly, the per-bin stream "
          f"prefix in stream order ({st0['pool/candidates']:.0f} candidates, {st0['pool/scans']:.0f} "
          f"box scans, {st0['pool/rejected']:.0f} good discarded; natural pool: "
          f"{nat['pool/candidates']:.0f} / {nat['pool/scans']:.0f}, shares "
          + "/".join(f"{100 * nat[f'pool/region_{b}']:.0f}" for b in REGION_BINS)
          + f" %); {checked} slots' region sizes = direct scans; refreshes refill each slot "
          f"from its bin (candidates per refresh {refresh_counts}), shares fixed over a full "
          f"ring; state dict round trip, pre-region_mix checkpoint loads")


# ── 3. Best metric, trainer ─────────────────────────────────────────────────


def tiny_cfg(**kw) -> PPOConfig:
    base = dict(setups="full", scenarios="pool", sep_range=(80.0, 120.0), pool_size=64,
                pool_refresh=0.25, num_envs=16, rollout_len=8, max_steps=6, history_len=4,
                d_model=32, n_layers=1, n_heads=2, mlp_ratio=2.0, amp=False, compile=False,
                wandb=False, minibatches=2, eval_every=1, eval_n=24, eval_envs=24,
                total_env_steps=3 * 16 * 8, plr_min_seen=2, seed=3)
    base.update(kw)
    return PPOConfig(**base)


def test_best_metric(tmp: str):
    print("\n== 3. best metric, trainer ==")
    ev = {f"eval_region/region_{b}/pct_oracle": v for b, v in zip(REGION_BINS, (40.0, 60.0, 80.0, 100.0))}
    assert region_bin_mean(ev) == 70.0
    ev["eval_region/region_small/pct_oracle"] = math.nan
    assert math.isnan(region_bin_mean(ev))
    tables = os.path.join(tmp, "tables")
    try:
        Trainer(tiny_cfg(best_metric="region", eval_sets="cont,set4", eval_tables=tables),
                os.path.join(tmp, "x"), build_eval=False)
        raise RuntimeError("--best-metric region without the region set accepted")
    except AssertionError:
        pass
    try:
        Trainer(tiny_cfg(scenarios="atlas", near_best_frac=0.2), os.path.join(tmp, "x"), build_eval=False)
        raise RuntimeError("near-best starts without the pool accepted")
    except AssertionError:
        pass
    c = tiny_cfg(near_best_frac=0.25, pool_region_mix="0.2,0.2,0.25,0.35", best_metric="region",
                 eval_tables=tables)
    nm = run_name(c, None)
    assert nm.endswith("_pr0.25_nb0.25_rmix_bmregion_" + os.environ.get("SLURM_JOB_ID", "local")), nm
    assert "_nb0.1r0.1-0.3_" in run_name(tiny_cfg(near_best_frac=0.1, near_best_radius=(0.1, 0.3)), None)
    base = run_name(tiny_cfg(), None)
    assert not any(m in base for m in ("_nb", "_rmix", "_bm")), base
    d = os.path.join(tmp, "ft_run")
    tr = Trainer(c, d)
    assert tr.best_metric == REGION_BIN_MEAN and tr.env.cfg.near_best_frac == 0.25
    assert tr.pool.quotas == region_quotas(MIX, tr.pool.n_good_slots)
    tr.train()
    rows = list(csv.DictReader(open(os.path.join(d, "eval.csv"))))
    assert len(rows) == 3, len(rows)
    vals = []
    for r in rows:
        m = sum(float(r[f"eval_region/region_{b}/pct_oracle"]) for b in REGION_BINS) / 4
        assert abs(float(r["best/value"]) - m) < 1e-9 and abs(float(r[REGION_BIN_MEAN]) - m) < 1e-9
        vals.append(m)
    ck = torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=False)
    assert ck["best_eval"] == max(vals) and ck["best_metric"] == REGION_BIN_MEAN
    assert ck["iteration"] == 1 + vals.index(max(vals))
    trows = list(csv.DictReader(open(os.path.join(d, "train.csv"))))
    shares = [float(trows[-1][f"pool/region_{b}"]) for b in REGION_BINS]
    q_ = region_quotas(MIX, tr.pool.n_good_slots)
    assert all(abs(s - k / tr.pool.n_good_slots) < 1e-9 for s, k in zip(shares, q_))
    # Resume: near-best and best metric follow the flags, the region mix stays
    path = os.path.join(d, "latest.pt")
    cfg = resume_config(path, ["--run-dir", d, "--resume", "--near-best-frac", "0.5",
                               "--near-best-radius", "0.1", "0.2", "--best-metric", "cont",
                               "--pool-region-mix", "0.25,0.25,0.25,0.25",
                               "--total-env-steps", str(5 * 16 * 8)])
    assert (cfg.near_best_frac, cfg.near_best_radius, cfg.best_metric) == (0.5, (0.1, 0.2), "cont")
    assert cfg.pool_region_mix == "0.2,0.2,0.25,0.35"
    t2 = Trainer(cfg, d, load_encoder=False)
    t2.load(path)
    assert t2.best_metric == "eval_cont/all/pct_oracle" and t2.best_eval == -math.inf
    assert t2.env.cfg.near_best_frac == 0.5 and t2.name == tr.name
    t2.train()
    assert t2.iteration == 5
    rows = list(csv.DictReader(open(os.path.join(d, "eval.csv"))))
    assert float(rows[-1]["best/value"]) == float(rows[-1]["eval_cont/all/pct_oracle"])
    print(f"  region_bin_mean; run name ...{nm[-40:]}; 3 evals: best/value = region-bin mean "
          f"{[round(v, 2) for v in vals]}, best.pt at iteration {ck['iteration']}; pool region "
          f"shares {shares}; resume with --near-best-frac 0.5 --best-metric cont (mix kept) to "
          f"iteration {t2.iteration}")


# ── 4. Defaults bit-identical to REF_REV ────────────────────────────────────


DRIVER = r'''
import os, sys
ref, rl, out = sys.argv[1:4]
sys.path[:0] = ([ref] if ref != "-" else []) + [rl]
import torch
import ppo
from env import AlignEnv, EnvConfig
from geometry import SETUPS
assert os.path.dirname(os.path.abspath(ppo.__file__)) == os.path.abspath(ref if ref != "-" else rl)
cfg = ppo.PPOConfig(setups="full", scenarios="pool", sep_range=(80.0, 120.0), pool_size=64,
                    pool_refresh=0.25, num_envs=16, rollout_len=12, max_steps=6, history_len=4,
                    d_model=32, n_layers=1, n_heads=2, mlp_ratio=2.0, amp=False, compile=False,
                    wandb=False, minibatches=2, eval_every=0, seed=3, total_env_steps=10**6,
                    plr_min_seen=2)
tr = ppo.Trainer(cfg, os.path.join(os.path.dirname(out), "run_" + os.path.basename(out)),
                 build_eval=False)
res = {"name": tr.name}
for it in range(3):
    tr.sampler.refresh(tr.iteration)
    tr.rollout()
    for k in ("frames", "actions", "logp", "rewards", "scen", "first"):
        res[f"{k}{it}"] = getattr(tr.buf, k).cpu().clone()
    tr.update()
    tr.iteration += 1
    tr.env_steps += 16 * 12
res["slots"] = tr.pool.atlas.params.cpu()
res["slot_gen"] = tr.pool.slot_gen.cpu()
res["cursor"] = tr.pool.cursor
res["params"] = {k: v.cpu() for k, v in tr.agent.state_dict().items()}
res["env"] = {k: v.cpu() for k, v in tr.env.get_state().items()}
env = AlignEnv(EnvConfig(num_envs=64, setups=SETUPS, history_len=4, seed=9, max_steps=7))
g = torch.Generator(device=env.device)
g.manual_seed(1)
for t in range(20):
    a = torch.randint(0, env.n_bins, (64, 2), generator=g, device=env.device)
    a = torch.where((torch.rand(64, generator=g, device=env.device) < 0.2)[:, None], 0, a)
    env.step(a)
    if t == 10:
        m = torch.rand(64, generator=g, device=env.device) < 0.5
        env.reset(m, setups=torch.randint(0, 4, (64,), generator=g, device=env.device))
    if t == 15:
        env.reset(m)
res["env4"] = {k: v.cpu() for k, v in env.get_state().items()}
torch.save(res, out)
'''


def same(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
    return a == b


def test_defaults_identical(tmp: str):
    print(f"\n== 4. defaults bit-identical to {REF_REV} ==")
    ref = os.path.join(tmp, "ref")
    os.makedirs(ref)
    try:
        for f in ("env.py", "scenarios.py", "ppo.py"):
            src = subprocess.run(["git", "show", f"{REF_REV}:rl/{f}"], cwd=RL_DIR, check=True,
                                 capture_output=True, text=True).stdout
            with open(os.path.join(ref, f), "w") as fh:
                fh.write(src)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  skipped: {REF_REV} not available ({e})")
        return
    drv = os.path.join(tmp, "driver.py")
    with open(drv, "w") as fh:
        fh.write(DRIVER)
    outs = {}
    for tag, r in (("ref", ref), ("new", "-")):
        outs[tag] = os.path.join(tmp, f"{tag}.pt")
        subprocess.run([sys.executable, "-I", drv, r, RL_DIR, outs[tag]], check=True,
                       capture_output=True, text=True)
    a, b = (torch.load(outs[t], weights_only=False) for t in ("ref", "new"))
    diff = sorted(k for k in a if not same(a[k], b[k]))
    assert a.keys() == b.keys() and not diff, f"differs from {REF_REV}: {diff}"
    print(f"  pool slots (fill + 2 refreshes), rollout buffers and parameters after 3 PPO "
          f"iterations, env state, run name {a['name']}, four-setup env with explicit resets: "
          f"identical ({len(a)} entries)")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    tmp = tempfile.mkdtemp()
    try:
        sc = Scanner(device=DEV)
        test_near_best()
        test_region_quotas(sc)
        test_best_metric(tmp)
        test_defaults_identical(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nall tests passed")
