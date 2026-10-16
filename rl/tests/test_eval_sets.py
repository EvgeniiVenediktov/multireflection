"""
tests/test_eval_sets.py - Unseen-scenario evaluation: off-grid tables, grid hold-out,
eval sets of ppo.py and eval_policy.py

Run from rl/:  uv run python tests/test_eval_sets.py

    1. lattices of the default grid: offgrid = midpoints in all three parameters (odd mm
       51-199, k x 0.25 + 0.125 deg), extrap_short / extrap_long = separation 20-49 /
       201-350 mm; no coordinate of any point lies on a training grid axis, no point is
       a scenario of the atlas (grid or named), float32 exact
    2. off-grid table on a small grid: every statistic of sampled rows = a direct 0.01 deg
       scan; the table is the shortest prefix of the seeded lattice order holding the
       good / impossible counts; independent of the round size; load_or_build builds
       once, then loads, a different OffgridConfig is refused; offgrid_eval_scenarios =
       one episode per scenario, the first of each group; an extrapolation table; an
       exhausted lattice (whole lattice, eval set cycles)
    3. hold-out: holdout_mask stratified (round(frac x count) of the good and impossible
       grid points, named setups never), seeded; the sampler never draws a held-out
       scenario over many draws before and after replay starts (also when held-out
       scenarios have scores); frac 0 keeps the draws bit-identical
    4. hold-out in the env and the trainer: auto-reset episodes of AlignEnv with the
       sampler over thousands of draws never start a held-out scenario; the hold-out
       eval set holds only held-out scenarios, the full set none; run name marker
    5. eval_policy.py smoke on tiny sets (full, holdout, offgrid, extrap_short,
       extrap_long, set4) of a tiny checkpoint: JSON and report written; full-set argmax
       metrics = the trainer's evaluate(); the cached oracle is reused
"""

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

import eval_policy  # noqa: E402
from env import AlignEnv, EnvConfig  # noqa: E402
from find_setups import box_grid, candidate_states  # noqa: E402
from geometry import CellGeometry  # noqa: E402
from herriott_sim import STOP_HOLE  # noqa: E402
from ppo import (PPOConfig, Trainer, eval_set_names, full_eval_set,  # noqa: E402
                 holdout_eval_set, run_name)
from scenarios import (TABLE_KINDS, TABLE_SEP, Atlas, AtlasConfig, OffgridConfig,  # noqa: E402
                       SamplerConfig, ScenarioSampler, build_atlas, build_offgrid,
                       holdout_mask, load_or_build_offgrid, offgrid_eval_scenarios,
                       offgrid_lattice, on_grid_axes, save_atlas, scenario_params)

DEV = torch.device("cuda")
TINY = AtlasConfig(sep=(86.0, 90.0, 4.0), pitch=(0.0, 8.0, 4.0), yaw=(0.0, 8.0, 4.0))
SMALL = AtlasConfig(sep=(80.0, 96.0, 4.0), pitch=(0.0, 8.0, 2.0), yaw=(0.0, 8.0, 2.0))


def fake_atlas(n: int, good: torch.Tensor, named: dict) -> Atlas:
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
    d.update(version=1, key={}, named=named, n_grid=n - 1)  # Last row: a named extra
    return Atlas(d, DEV, "fake")


def direct_check(t: Atlas, rows, cfg: AtlasConfig):
    """Statistics of table rows = a direct 0.01 deg scan of each (as tests/test_scenarios)."""
    geom = CellGeometry()
    sim = geom.make_sim(DEV, max_bounces=cfg.cap)
    grid, _ = box_grid(cfg.limit, cfg.step, DEV)
    for i in rows:
        prm = t.params[i:i + 1]
        out = sim.simulate(candidate_states(geom, prm, grid), store_world=False)
        b, st, p = out["hit_counts"], out["stop_reason"], out["total_path_length"]
        pm, im = p.max(0)
        assert float(t.pmax[i]) == float(pm), i
        assert torch.equal(t.pmax_tilt[i], grid[im]) and int(t.pmax_bounces[i]) == int(b[im])
        hole = st == STOP_HOLE
        h = torch.bincount(b[hole], minlength=cfg.cap + 1)
        cnt, o = h[cfg.min_order:].max(0)
        po = int(o) + cfg.min_order if int(cnt) > 0 else 0
        assert int(t.plateau_order[i]) == po and int(t.plateau_cells[i]) == int(cnt), i
        if po:
            pmask = hole & (b == po)
            assert torch.allclose(t.plateau_c[i], grid[pmask].mean(0), atol=1e-5)
            assert abs(float(t.plateau_path[i]) - float(p[pmask].mean())) < 1e-2
        assert int(t.best_bounces[i]) == int(b.max())
        assert int(t.best_cells[i]) == int((b == b.max()).sum())
        assert int(t.near_cells[i]) == int((p >= 0.9 * pm).sum())
        assert int(t.blind_cells[i]) == int((b <= 2).sum())
        z = sim.simulate(candidate_states(geom, prm, torch.zeros(1, 2, device=DEV)),
                         store_world=False)
        if int(z["hit_mirror"][0, 0]) == 2:
            assert torch.equal(t.first_hit[i], z["hit_local_xy"][0, 0])
        else:
            assert bool(t.first_hit[i].isnan().all())
        pf = cfg.floor_bounces * float(t.params[i, 0])
        assert bool(t.good[i]) == (float(pm) >= pf)
        assert float(t.path_ref[i]) == max(float(pm), pf)


# ── 1. Lattices ─────────────────────────────────────────────────────────────


def test_lattices():
    print("\n== 1. lattices of the default grid ==")
    cfg = AtlasConfig()
    atlas_p, named, n_grid = scenario_params(cfg)
    keys = {tuple(np.round(r, 6)) for r in atlas_p}
    sizes = {}
    for kind in TABLE_KINDS:
        lat = offgrid_lattice(cfg, kind)
        sizes[kind] = lat.shape[0]
        sep = np.unique(lat[:, 0])
        lp = np.unique(lat[:, 1])
        assert np.array_equal(lp, np.unique(lat[:, 2]))
        assert np.array_equal(lp, 0.125 + 0.25 * np.arange(32)), "laser midpoints"
        if kind == "offgrid":
            assert np.array_equal(sep, np.arange(51.0, 200.0, 2.0)), "odd mm 51-199"
            assert not on_grid_axes(lat, cfg).any(), "a coordinate on the grid"
        else:
            lo, hi, st = TABLE_SEP[kind]
            assert np.array_equal(sep, np.arange(lo, hi + 0.5 * st, st))
            assert ((sep < 50.0) | (sep > 200.0)).all(), "separation inside the trained range"
            assert not on_grid_axes(lat, cfg)[:, 1:].any()
        assert not any(tuple(np.round(r, 6)) in keys for r in lat), "a lattice point is an atlas scenario"
        assert np.array_equal(lat.astype(np.float32).astype(np.float64), lat), "not float32 exact"
    assert sizes == {"offgrid": 76800, "extrap_short": 30720, "extrap_long": 153600}, sizes
    print(f"  sizes {sizes}; no coordinate on a grid axis (offgrid), separations outside "
          f"50-200 mm (extrap), no point among the {atlas_p.shape[0]} atlas scenarios, float32 exact")


# ── 2. Off-grid table ───────────────────────────────────────────────────────


def test_offgrid_table(tmp: str):
    print("\n== 2. off-grid table on a small grid ==")
    q = lambda s: None
    lat = offgrid_lattice(SMALL)  # 4 x 4 x 4 midpoints
    ocfg = OffgridConfig(n=16, impossible_share=0.25, round_n=7)  # 12 good + 4 impossible
    d = build_offgrid(ocfg, SMALL, DEV, q)
    t = Atlas(d, DEV, "small")
    L = t.n
    order = np.random.default_rng(ocfg.seed).permutation(lat.shape[0])
    assert torch.equal(d["lattice_index"], torch.from_numpy(order[:L]))
    assert torch.equal(t.params.cpu(), torch.tensor(lat[order[:L]], dtype=torch.float32))
    n_good, n_imp = ocfg.counts()
    g = t.good.cpu()
    assert int(g.sum()) >= n_good and int((~g).sum()) >= n_imp
    assert int(g[:L - 1].sum()) < n_good or int((~g[:L - 1]).sum()) < n_imp, "not the shortest prefix"
    rows = sorted(set([0, L - 1] + torch.nonzero(g).squeeze(1)[:3].tolist()
                      + torch.nonzero(~g).squeeze(1)[:2].tolist()))
    direct_check(t, rows, SMALL)
    d2 = build_offgrid(replace(ocfg, round_n=64), SMALL, DEV, q)
    for k in Atlas.TENSORS:
        assert torch.equal(d[k].nan_to_num(-7.0), d2[k].nan_to_num(-7.0)), k
    # load_or_build: builds once, loads, refuses another config
    path = os.path.join(tmp, "tables", "offgrid_v1.pt")
    msgs = []
    a1 = load_or_build_offgrid(path, SMALL, ocfg, DEV, msgs.append)
    a2 = load_or_build_offgrid(path, SMALL, ocfg, DEV, msgs.append)
    assert sum("missing" in m for m in msgs) == 1 and torch.equal(a1.pmax, a2.pmax)
    assert torch.equal(a1.pmax, t.pmax) and a1.offgrid["candidates"] == L
    try:
        load_or_build_offgrid(path, SMALL, replace(ocfg, seed=1), DEV, q)
        raise RuntimeError("a table of another config was accepted")
    except AssertionError:
        pass
    # Eval scenarios: one episode per scenario, the first of each group in table order
    s = offgrid_eval_scenarios(t, 16, 0, 0.25)
    assert torch.unique(s).numel() == 16
    assert torch.equal(torch.sort(s[t.good[s.to(DEV)].cpu()]).values, t.good_ids[:12].cpu())
    assert torch.equal(torch.sort(s[~t.good[s.to(DEV)].cpu()]).values, t.impossible_ids[:4].cpu())
    s2 = offgrid_eval_scenarios(t, 80, 0, 0.25)  # More than the table: cycles
    gi = t.good_ids.cpu()
    assert torch.equal(torch.sort(s2[t.good[s2.to(DEV)].cpu()]).values,
                       torch.sort(gi[torch.arange(60) % gi.numel()]).values)
    # Extrapolation table: separations 20-49 mm, laser midpoints of the grid
    de = build_offgrid(OffgridConfig(kind="extrap_short", n=8, impossible_share=0.25), SMALL,
                       DEV, q)
    te = Atlas(de, DEV, "extrap")
    sep = te.params[:, 0].cpu()
    assert bool(((sep >= 20) & (sep <= 49) & (sep == sep.round())).all())
    assert not on_grid_axes(te.params.cpu().double().numpy(), SMALL)[:, 1:].any()
    direct_check(te, [0, te.n - 1], SMALL)
    # Exhausted lattice: the whole lattice, flagged short; the eval set cycles
    msgs = []
    dx = build_offgrid(OffgridConfig(n=4096), TINY, DEV, msgs.append)
    assert dx["offgrid"]["short"] and dx["offgrid"]["candidates"] == offgrid_lattice(TINY).shape[0]
    assert any("WARNING" in m for m in msgs)
    tx = Atlas(dx, DEV, "tiny")
    sx = offgrid_eval_scenarios(tx, 64, 0, 0.1)
    assert sx.numel() == 64 and int(sx.max()) < tx.n
    print(f"  {L} candidates (12 good + 4 impossible asked; shortest prefix of the seeded "
          f"order), {len(rows)} rows = direct scans, round size 7 = 64, cached build once / "
          f"load / refuse another seed; eval set: 16 distinct, the first of each group; "
          f"extrap_short table ({te.n} rows, sep {sorted(set(sep.tolist()))[:4]}...); an "
          f"exhausted lattice ({tx.n} points) gives a short table and a cycling eval set")


# ── 3. Hold-out mask and sampler ────────────────────────────────────────────


def test_holdout_sampler():
    print("\n== 3. hold-out mask and sampler ==")
    n = 3000
    gen = torch.Generator(device=DEV)
    gen.manual_seed(3)
    good = torch.rand(n, generator=gen, device=DEV) < 0.3
    named = {"a": 5, "b": 17, "c": n - 1}  # Two on the grid, one extra row
    at = fake_atlas(n, good, named)
    frac = 0.1
    ho = holdout_mask(at, frac, 0)
    free = torch.arange(n, device=DEV) < at.n_grid
    free[[5, 17, n - 1]] = False
    for grp in (good, ~good):
        assert int((ho & grp).sum()) == round(frac * int((grp & free).sum()))
    assert not bool(ho[[5, 17, n - 1]].any()) and not bool(ho[n - 1])
    assert torch.equal(ho, holdout_mask(at, frac, 0)) and not torch.equal(ho, holdout_mask(at, frac, 1))
    assert not bool(holdout_mask(at, 0.0).any())

    cfg = SamplerConfig(min_seen=50, holdout_frac=frac, holdout_seed=0)
    sm = ScenarioSampler(at, cfg)
    assert torch.equal(sm.holdout, ho)
    train = ~ho
    u = torch.rand(2_000_000, generator=gen, device=DEV)
    s, src = sm.draw(u)
    assert not bool(ho[s].any()), "a held-out scenario was drawn (fresh)"
    cover = torch.zeros(n, dtype=torch.bool, device=DEV)
    cover[s] = True
    assert bool((cover == train).all()), "every training scenario is drawn, no other"
    # Replay: scores on every good scenario, held-out ones included (an eval might observe
    # them); draws still never return them
    gi = torch.nonzero(good).squeeze(1)
    sm.observe(gi, torch.rand(gi.numel(), generator=gen, device=DEV) * 1000.0,
               torch.ones(gi.numel(), dtype=torch.bool, device=DEV))
    sm.refresh(5)
    assert abs(float(sm.p_fresh_eff) - cfg.p_fresh) < 1e-6, "replay not started"
    n_draw = 0
    for _ in range(5):
        s, src = sm.draw(torch.rand(2_000_000, generator=gen, device=DEV))
        assert not bool(ho[s].any()), "a held-out scenario was drawn (replay)"
        n_draw += s.numel()
    rep = float((src == 2).float().mean())
    assert rep > 0.5
    # frac 0: draws bit-identical to a default sampler
    a0, b0 = ScenarioSampler(at, SamplerConfig(min_seen=50)), ScenarioSampler(
        at, SamplerConfig(min_seen=50, holdout_frac=0.0, holdout_seed=9))
    u = torch.rand(100_000, generator=gen, device=DEV)
    assert torch.equal(a0.draw(u)[0], b0.draw(u)[0]) and torch.equal(a0.ids, torch.cat(
        (at.impossible_ids, at.good_ids, at.good_ids)))
    print(f"  mask: round(0.1 x count) of {int((good & free).sum())} good / "
          f"{int((~good & free).sum())} impossible grid points ({int(ho.sum())} held out), named "
          f"never, seeded; {2_000_000 + n_draw} draws (fresh only, then replay share {rep:.2f} "
          f"with scored held-out scenarios): none held out, every training scenario drawn; "
          f"frac 0 = default draws")


# ── 4. Hold-out in the env and the trainer ──────────────────────────────────


def test_holdout_env_trainer(atlas: Atlas, atlas_path: str, tmp: str):
    print("\n== 4. hold-out in the env and the trainer ==")
    frac = 0.5
    sm = ScenarioSampler(atlas, SamplerConfig(min_seen=2, holdout_frac=frac, holdout_seed=0))
    ho = sm.holdout
    assert int(ho.sum()) > 0
    env = AlignEnv(EnvConfig(num_envs=256, history_len=0, max_steps=4, seed=1),
                   scenarios=atlas.table(), sampler=sm)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    starts = 0
    used = torch.zeros(atlas.n, dtype=torch.bool, device=DEV)
    assert not bool(ho[env.setup].any())
    for t in range(300):
        a = torch.randint(0, env.n_bins, (env.num_envs, 2), generator=gen, device=DEV)
        a = torch.where((torch.rand(env.num_envs, generator=gen, device=DEV) < 0.3)[:, None], 0, a)
        _, _, term, trunc, info = env.step(a)
        done = term | trunc
        sm.observe(info["setup"], info["ep_final_path_mm"], done, info["scenario_src"])
        if t % 50 == 49:
            sm.refresh(t)
        assert not bool(ho[env.setup].any()), "an episode started on a held-out scenario"
        used[env.setup] = True
        starts += int(done.sum())
    assert bool((used == ~ho).all()), "every training scenario started, no held-out one"
    # Trainer: the sampler's hold-out, the eval sets, the run name
    cfg = PPOConfig(setups="full", scenarios="atlas", atlas=atlas_path, num_envs=16,
                    rollout_len=8, max_steps=4,
                    history_len=4, d_model=32, n_layers=1, n_heads=2, mlp_ratio=2.0, amp=False,
                    compile=False, wandb=False, minibatches=2, eval_every=0, eval_n=64,
                    eval_envs=64, holdout_frac=frac, total_env_steps=10**6, plr_min_seen=2)
    tr = Trainer(cfg, os.path.join(tmp, "ho_run"), build_eval=False)
    assert torch.equal(tr.sampler.holdout, holdout_mask(atlas, frac, 0))
    assert "-ho50_" in tr.name, tr.name
    assert eval_set_names(cfg)[:2] == ["full", "holdout"]
    hs = holdout_eval_set(cfg, atlas, tr.run_dir, DEV)
    fs = full_eval_set(cfg, atlas, tr.run_dir, DEV)
    assert bool(ho[hs.setups].all()), "a hold-out episode on a training scenario"
    assert not bool(ho[fs.setups].any()), "a full-set episode on a held-out scenario"
    assert bool((~atlas.good[hs.setups]).any()) and bool(atlas.good[hs.setups].any())
    assert "-ho" not in run_name(replace(cfg, holdout_frac=0.0), None)
    print(f"  env: {starts} episodes over {env.num_envs} envs with the sampler, none on the "
          f"{int(ho.sum())} held-out scenarios, every training scenario used; trainer "
          f"{tr.name}: hold-out set {hs.setups.numel()} episodes on "
          f"{torch.unique(hs.setups).numel()} held-out scenarios only, full set none")


# ── 5. eval_policy.py smoke ─────────────────────────────────────────────────


def test_eval_policy(atlas_path: str, tmp: str):
    print("\n== 5. eval_policy.py smoke ==")
    run_dir = os.path.join(tmp, "smoke_run")
    tables = os.path.join(tmp, "smoke_tables")
    cfg = PPOConfig(setups="full", scenarios="atlas", atlas=atlas_path, num_envs=16,
                    rollout_len=8, max_steps=6,
                    history_len=4, d_model=32, n_layers=1, n_heads=2, mlp_ratio=2.0, amp=False,
                    compile=False, wandb=False, minibatches=2, eval_every=0, eval_n=24,
                    eval_envs=24, holdout_frac=0.3, total_env_steps=10**6, plr_min_seen=2,
                    eval_tables=tables, seed=3)
    tr = Trainer(cfg, run_dir, build_eval=False)
    tr.sampler.refresh(0)
    tr.rollout()
    tr.update()  # Non-trivial weights
    tr.save("best.pt")
    tr.setup_eval()
    names = [es.name for es in tr.eval_sets]
    assert names == ["full", "holdout", "offgrid", "extrap_short", "extrap_long", "set4"], names
    ev = tr.evaluate()
    assert "eval_extrap_long/all/pct_oracle" in ev and "eval_offgrid_samp/sep50-80/n" in ev
    mt = os.path.getmtime(os.path.join(run_dir, "oracle_full.pt"))
    out = os.path.join(tmp, "smoke_out")
    res = eval_policy.main(["--ckpt", os.path.join(run_dir, "best.pt"), "--sets",
                            "full,holdout,offgrid,extrap_short,extrap_long,set4", "--n", "24",
                            "--out", out])
    assert os.path.getmtime(os.path.join(run_dir, "oracle_full.pt")) == mt, "oracle recomputed"
    stem = os.path.join(out, "eval_smoke_run_best")
    assert os.path.exists(stem + ".json") and os.path.exists(stem + ".md")
    md = open(stem + ".md").read()
    for h in ("## Seen grid vs unseen off-grid", "## Extrapolation (separation outside",
              "## Head errors", "## Scenario tables", "### extrap_long"):
        assert h in md, h
    assert "| sep20-30 |" in md and "| sep300-350 |" in md
    same = []
    for nm, pre in (("full", "eval_full"), ("offgrid", "eval_offgrid"), ("set4", "eval")):
        a = res["sets"][nm]["policies"]["argmax"]["splits"]["all"]
        for k in ("return", "success", "length", "stop_rate"):
            v = ev[f"{pre}/all/{k}"]
            assert (a[k] is None and math.isnan(v)) or abs(a[k] - v) < 1e-9, (nm, k, a[k], v)
        same.append(nm)
    for nm in ("full", "holdout", "offgrid", "extrap_short", "extrap_long", "set4"):
        sp = res["sets"][nm]["policies"]["sampled"]["splits"]
        assert sp["all"]["n"] == 24 and "head" in sp["all"]
    print(f"  sets {names}; report and JSON written ({len(md.splitlines())} lines); argmax "
          f"metrics = Trainer.evaluate on {same}; cached oracle_full.pt reused")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    tmp = tempfile.mkdtemp()
    try:
        d = build_atlas(TINY, DEV, log=lambda s: None)
        atlas_path = os.path.join(tmp, "tiny.pt")
        save_atlas(d, atlas_path)
        atlas = Atlas(d, DEV, atlas_path)
        test_lattices()
        test_offgrid_table(tmp)
        test_holdout_sampler()
        test_holdout_env_trainer(atlas, atlas_path, tmp)
        test_eval_policy(atlas_path, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nall tests passed")
