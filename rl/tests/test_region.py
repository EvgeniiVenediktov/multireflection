"""
tests/test_region.py - Target-region size, the eval_region set, the eval-set selection
(--eval-sets) and checkpoints from before it

Run from rl/:  uv run python tests/test_region.py

    1. target-region size: region_area / best_area / best_bounces of table rows (a region
       table, a cont table, pool slots) = a direct count from a 0.01 deg box scan (cells
       with path >= 0.9 P_max, cells at the highest bounce count); AlignEnv.box_stats of
       the four setups = a direct scan = the atlas's named setups, P_max = path_max;
       NEAR_FRAC = SUCCESS_FRAC = AtlasConfig.near_frac; bins [lo, hi) at the edges
    2. region table: n / 4 per bin, every row good and in its bin; the rows are the
       first good scenarios of each bin in the seeded stream (recomputed from
       sample_valid_scenarios + box scans), the candidate sample = every good scenario of
       the used prefix; independent of the round size, another seed differs; built once,
       loaded, other settings refused
    3. eval sets: pruned defaults (pool runs cont, cont_samp, region, set4; atlas runs
       and --setups all / measured unchanged), --eval-sets parsing, --no-eval-sampled,
       the set of best.pt required; region_eval_set: one episode per scenario, n / 4 per
       bin, splits; the region splits of cont (good rows) and set4 (the setups' bins)
    4. old checkpoint: a pool checkpoint whose config predates --eval-sets (eval_offgrid /
       eval_extrap / eval_set4) resumes (resume_config + Trainer.load) with the sets it
       evaluated; --eval-sets on the resume replaces them; the resumed trainer evaluates
       the pruned list (no training)
    5. eval_policy.py on that checkpoint: default sets cont, region, set4, --name,
       --compare; target-region section; argmax metrics = Trainer.evaluate; cached oracles
       reused (explicit legacy sets: tests/test_eval_sets.py, 5.)
"""

import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, replace

RL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RL_DIR)

import torch  # noqa: E402

import eval_policy  # noqa: E402
from env import NEAR_FRAC, AlignEnv, EnvConfig  # noqa: E402
from evaluation import SUCCESS_FRAC  # noqa: E402
from find_setups import box_grid  # noqa: E402
from geometry import SETUPS  # noqa: E402
from ppo import (POOL_EVAL_SETS, PPOConfig, Trainer, config_from_dict, cont_eval_set,  # noqa: E402
                 default_eval_sets, eval_entries, eval_set_names, legacy_eval_sets,
                 parse_eval_sets, region_eval_set, resume_config, setup_eval_set)
from scenarios import (REGION_BINS, REGION_EDGES, STREAM_REGION, Atlas, AtlasConfig,  # noqa: E402
                       ContConfig, ContRanges, PoolConfig, RegionConfig, SamplerConfig, Scanner,
                       ScenarioPool, best_area, box_states, build_atlas, build_cont_table,
                       build_region_table, load_or_build_region, region_area, region_bin,
                       region_bin_masks, region_table_path, sample_valid_scenarios)

DEV = torch.device("cuda")
TINY = AtlasConfig(sep=(86.0, 90.0, 4.0), pitch=(0.0, 8.0, 4.0), yaw=(0.0, 8.0, 4.0))
NARROW = ContRanges(sep=(80.0, 120.0))
q = lambda s: None  # noqa: E731


def direct_counts(sc: Scanner, params: torch.Tensor):
    """Direct 0.01 deg box scan of one scenario (params (3 | 5,)): cells with path >= 0.9
    P_max, cells at the highest bounce count, that count, P_max."""
    grid, _ = box_grid(2.0, 0.01, DEV)
    out = sc.sim.simulate(box_states(sc.geom, params[None].to(DEV), grid), store_world=False,
                          store_hits=False)
    p, b = out["total_path_length"], out["hit_counts"]
    pm = p.max()
    return int((p >= 0.9 * pm).sum()), int((b == b.max()).sum()), int(b.max()), float(pm)


def check_rows(sc: Scanner, t: Atlas, rows) -> int:
    ra, ba = region_area(t), best_area(t)
    for i in rows:
        near, best, bb, pm = direct_counts(sc, t.params[i])
        assert float(t.pmax[i]) == pm, i
        assert int(t.near_cells[i]) == near and float(ra[i]) == near * 0.01 ** 2, i
        assert int(t.best_cells[i]) == best and float(ba[i]) == best * 0.01 ** 2, i
        assert int(t.best_bounces[i]) == bb, i
    return len(rows)


# ── 1. Target-region size ───────────────────────────────────────────────────


def test_region_size(sc: Scanner, tmp: str):
    print("\n== 1. target-region size = direct box scans ==")
    assert NEAR_FRAC == SUCCESS_FRAC == AtlasConfig().near_frac
    # Bins [lo, hi) at the edges
    e = torch.tensor([0.0, 0.0199, 0.02, 0.05, 0.1, 0.4999, 0.5, 9.0], dtype=torch.float64)
    assert region_bin(e).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    m = region_bin_masks(e)
    assert list(m) == [f"region_{nm}" for nm in REGION_BINS] and REGION_EDGES == (0.02, 0.1, 0.5)
    assert sum(int(v.sum()) for v in m.values()) == e.numel()
    # Region table, cont table, pool slots
    n = 0
    rt = Atlas(build_region_table(RegionConfig(n=16, seed=1), device=DEV, log=q), DEV, "r")
    b = region_bin(region_area(rt))
    n += check_rows(sc, rt, [int(torch.nonzero(b == j)[0]) for j in range(4)])
    ct = Atlas(build_cont_table(ContConfig(ranges=NARROW, n=24, seed=2), device=DEV, log=q), DEV, "c")
    n += check_rows(sc, ct, [0, int(ct.good_ids[0]), int(ct.impossible_ids[0]), ct.n - 1])
    pool = ScenarioPool(PoolConfig(size=32, ranges=NARROW, refresh_frac=0.25, round_min=64),
                        SamplerConfig(min_seen=4), device=DEV, log=q)
    n += check_rows(sc, pool.atlas, [0, 7, 31])
    # Named setups: box_stats of the setups env (the P_max scan) = direct = the atlas
    env = AlignEnv(EnvConfig(setups=SETUPS, num_envs=4, history_len=0, path_ref="max"))
    bx = env.box_stats()
    assert torch.equal(env.path_max(), bx["pmax"]) and torch.equal(env.path_refs, bx["pmax"])
    atlas = Atlas(build_atlas(TINY, DEV, log=q), DEV, "tiny")
    sizes = {}
    for k, g in enumerate(SETUPS):
        near, best, bb, pm = direct_counts(sc, env.setup_params[k])
        assert (int(bx["near_cells"][k]), int(bx["best_cells"][k]), int(bx["best_bounces"][k])) == \
            (near, best, bb), g.name
        assert float(bx["pmax"][k]) == pm, g.name
        i = atlas.named[g.name]
        assert (int(atlas.near_cells[i]), int(atlas.best_cells[i]), int(atlas.best_bounces[i])) == \
            (near, best, bb), g.name
        sizes[g.name] = near * 1e-4
    assert region_bin(torch.tensor(list(sizes.values()), dtype=torch.float64)).tolist() == [0, 2, 2, 0]
    print(f"  bins [lo, hi) at {REGION_EDGES}; {n} table rows (region, cont, pool slots): target "
          f"region, max-bounce area and count = direct scans; named setups (box_stats = direct = "
          f"atlas): " + ", ".join(f"{k} {v:.4f}" for k, v in sizes.items()) + " deg^2")


# ── 2. Region table ─────────────────────────────────────────────────────────


def test_region_table(sc: Scanner, tmp: str):
    print("\n== 2. region table ==")
    rcfg = RegionConfig(n=24, seed=5, round_n=40)
    d = build_region_table(rcfg, device=DEV, log=q)
    t = Atlas(d, DEV, "region")
    area = region_area(t)
    b = region_bin(area).cpu()
    assert t.n == 24 and bool(t.good.all()) and rcfg.counts() == [6, 6, 6, 6]
    assert torch.equal(torch.bincount(b, minlength=4), torch.full((4,), 6))
    assert torch.equal(d["region_bin"].long(), b)
    pos = d["candidate_index"]
    assert bool((pos[1:] > pos[:-1]).all()), "rows not in stream order"
    # Recomputed: the first good scenarios of each bin among the first valid ones
    info = d["offgrid"]
    L = info["candidates"]
    prm, si = sample_valid_scenarios(info["valid"], rcfg.ranges, rcfg.seed, STREAM_REGION, scanner=sc)
    assert si["next"] == L and int(si["positions"][-1]) == int(pos[-1])
    s = sc.scan(torch.from_numpy(prm))
    a2 = s["near_cells"].double().cpu() * 1e-4
    g = s["good"].cpu()
    b2 = region_bin(a2)
    want = torch.cat([torch.nonzero(g & (b2 == j)).squeeze(1)[:6] for j in range(4)]).sort().values
    assert torch.equal(torch.from_numpy(si["positions"])[want], pos)
    assert torch.equal(s["params"].cpu()[want], d["params"])
    assert torch.equal(d["region_sample"], a2[g].float()) and info["good"] == int(g.sum())
    assert abs(info["region"]["bin_share"]["region_tiny"] - float((b2[g] == 0).double().mean())) < 1e-12
    # Round size, seed
    d2 = build_region_table(replace(rcfg, round_n=300), device=DEV, log=q)
    for k in Atlas.TENSORS + ("candidate_index", "region_bin", "region_sample"):
        assert torch.equal(d[k].nan_to_num(-7.0), d2[k].nan_to_num(-7.0)), k
    d3 = build_region_table(replace(rcfg, seed=6), device=DEV, log=q)
    assert not torch.equal(d3["params"][:4], d["params"][:4])
    # Cache: built once, loaded, other settings refused
    path = region_table_path(rcfg, tmp)
    assert os.path.basename(path) == "cont20-400_region0.02-0.1-0.5_n24_seed5_v1.pt"
    msgs = []
    t1 = load_or_build_region(path, rcfg, device=DEV, log=msgs.append)
    mt = os.path.getmtime(path)
    t2 = load_or_build_region(path, rcfg, device=DEV, log=msgs.append)
    assert os.path.getmtime(path) == mt and sum("missing" in m for m in msgs) == 1
    assert torch.equal(t1.params, t.params) and torch.equal(t2.near_cells, t.near_cells)
    for other in (replace(rcfg, edges=(0.05, 0.5, 2.0)), replace(rcfg, ranges=NARROW)):
        try:
            load_or_build_region(path, other, device=DEV, log=q)
            raise RuntimeError("other settings accepted")
        except AssertionError:
            pass
    print(f"  24 rows, 6 per bin, all good, stream order; = the first good ones per bin of the "
          f"first {info['valid']} valid scenarios ({L} candidates, {info['good']} good: the "
          f"candidate sample); rounds 40 = 300, another seed differs; cached once, other edges / "
          f"ranges refused")


# ── 3. Eval sets ────────────────────────────────────────────────────────────


def tiny_cfg(**kw) -> PPOConfig:
    base = dict(setups="full", scenarios="pool", sep_range=(80.0, 120.0), pool_size=64,
                pool_refresh=0.25, num_envs=16, rollout_len=8, max_steps=6, history_len=4,
                d_model=32, n_layers=1, n_heads=2, mlp_ratio=2.0, amp=False, compile=False,
                wandb=False, minibatches=2, eval_every=10, eval_n=24, eval_envs=24,
                total_env_steps=10**6, plr_min_seen=2, seed=3)
    base.update(kw)
    return PPOConfig(**base)


def fake_ep(es):
    """Per-episode tensors of a set as the splits read them (no rollout)."""
    n = es.setups.numel()
    return {"setup": es.setups, "init_bounces": torch.arange(n, device=DEV) % 5}


def test_eval_sets(tmp: str):
    print("\n== 3. eval sets ==")
    c = PPOConfig()
    assert eval_entries(c) == ["cont", "cont_samp", "region", "set4"] == list(POOL_EVAL_SETS)
    assert eval_set_names(c) == ["cont", "region", "set4"]
    assert eval_entries(replace(c, eval_sampled=False)) == ["cont", "region", "set4"]
    grid = ["full", "offgrid", "extrap_short", "extrap_long", "set4"]
    at = replace(c, scenarios="atlas")
    assert eval_set_names(at) == grid and eval_entries(at)[:2] == ["full", "full_samp"]
    assert eval_set_names(replace(at, holdout_frac=0.1))[:2] == ["full", "holdout"]
    assert eval_entries(replace(c, setups="all")) == ["set4", "set4_samp"]
    assert eval_entries(replace(c, setups="measured")) == ["measured", "measured_samp"]
    full = "cont,cont_samp,region,set4,full,offgrid,extrap_short,extrap_long"
    assert eval_set_names(replace(c, eval_sets=full)) == ["cont", "region", "set4", "full", "offgrid",
                                                           "extrap_short", "extrap_long"]
    assert default_eval_sets(c) == list(POOL_EVAL_SETS)
    assert parse_eval_sets(" cont, region_samp,cont ") == ["cont", "region_samp"]
    for bad in ("foo", "cont_sampled", "set5_samp"):
        try:
            parse_eval_sets(bad)
            raise RuntimeError(f"{bad} accepted")
        except AssertionError:
            pass
    try:  # best.pt is selected on eval_cont (deterministic)
        Trainer(tiny_cfg(eval_sets="cont_samp,region,set4"), os.path.join(tmp, "x"), build_eval=False)
        raise RuntimeError("a pool run without cont was accepted")
    except AssertionError:
        pass
    Trainer(tiny_cfg(eval_sets="region", eval_every=0), os.path.join(tmp, "y"), build_eval=False)
    # Region set: one episode per scenario, n / 4 per bin
    cfg = tiny_cfg(eval_tables=os.path.join(tmp, "tables"))
    cache = os.path.join(tmp, "cache")
    os.makedirs(cache, exist_ok=True)
    es = region_eval_set(cfg, cache, DEV, log=q)
    assert es.prefix == "eval_region" and es.prefix_samp == "eval_region_samp"
    assert torch.equal(es.setups.sort().values.cpu(), torch.arange(24))
    assert torch.equal(es.region, region_area(es.table)) and bool(es.table.good.all())
    sp = es.splits(fake_ep(es))
    assert list(sp) == ["all", "blind", "informative"] + [f"region_{nm}" for nm in REGION_BINS]
    assert [int(sp[f"region_{nm}"].sum()) for nm in REGION_BINS] == [6, 6, 6, 6]
    mo = os.path.getmtime(os.path.join(cache, "oracle_region.pt"))
    es2 = region_eval_set(cfg, cache, DEV, log=q)
    assert os.path.getmtime(os.path.join(cache, "oracle_region.pt")) == mo
    assert torch.equal(es2.setups, es.setups)
    # Cont: region bins of the good rows
    ec = cont_eval_set(cfg, cache, DEV, log=q)
    ep = fake_ep(ec)
    sc_ = ec.splits(ep)
    rb = region_bin(region_area(ec.table))[ep["setup"]]
    good = ec.table.good[ep["setup"]]
    for j, nm in enumerate(REGION_BINS):
        assert torch.equal(sc_[f"region_{nm}"], (rb == j) & good), nm
    assert sum(int(sc_[f"region_{nm}"].sum()) for nm in REGION_BINS) == int(good.sum())
    # Set4: the bins of the setups (measured, sep169 tiny; sep116, sep144 medium)
    e4 = setup_eval_set(cfg, "all", cache, DEV)
    s4 = e4.splits(fake_ep(e4))
    setup = e4.setups
    assert torch.equal(s4["region_tiny"], (setup == 0) | (setup == 3))
    assert torch.equal(s4["region_medium"], (setup == 1) | (setup == 2))
    assert not bool(s4["region_small"].any()) and not bool(s4["region_large"].any())
    assert torch.equal(e4.region, e4.env.box_stats()["near_cells"].double() * 1e-4)
    print(f"  defaults: pool {eval_entries(c)}, atlas {eval_set_names(at)} (+ _samp), --setups "
          f"all / measured their set; parsing, --no-eval-sampled, best set required; region set "
          f"24 episodes on 24 scenarios, 6 per bin, oracle cached; cont region splits on "
          f"{int(good.sum())} good episodes; set4 bins tiny = measured + sep169, medium = "
          f"sep116 + sep144")


# ── 4. Old checkpoint ───────────────────────────────────────────────────────


def test_old_checkpoint(tmp: str) -> str:
    print("\n== 4. checkpoint from before --eval-sets ==")
    d = os.path.join(tmp, "old_run")
    tables = os.path.join(tmp, "tables")
    a = Trainer(tiny_cfg(eval_tables=tables), d, build_eval=False)
    a.iteration, a.env_steps, a.best_eval = 7, 7 * 16 * 8, 41.5
    a.save("latest.pt")
    path = os.path.join(d, "latest.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old = dict(ck["cfg"])
    del old["eval_sets"]
    old.update(eval_offgrid=True, eval_extrap=False, eval_set4=True)  # The old flags
    ck["cfg"] = old
    torch.save(ck, path)
    leg = "cont,cont_samp,full,full_samp,offgrid,offgrid_samp,set4,set4_samp"
    assert legacy_eval_sets(old) == leg and config_from_dict(old).eval_sets == leg
    assert config_from_dict({"setups": "full"}).eval_sets == \
        "full,full_samp,offgrid,offgrid_samp,extrap_short,extrap_short_samp,extrap_long," \
        "extrap_long_samp,set4,set4_samp"  # An atlas run (LEGACY_DEFAULTS scenarios atlas)
    assert config_from_dict({"setups": "all"}).eval_sets == "set4,set4_samp"
    # Resume as ppo.main does it: the old sets
    cfg = resume_config(path, ["--run-dir", d, "--resume"])
    assert cfg.eval_sets == leg and "eval_offgrid" not in asdict(cfg)
    tr = Trainer(cfg, d, build_eval=False, load_encoder=False)
    tr.load(path)
    assert (tr.iteration, tr.best_eval, tr.best_metric) == (7, 41.5, "eval_cont/all/pct_oracle")
    assert eval_set_names(tr.cfg) == ["cont", "full", "offgrid", "set4"]
    # --eval-sets on the resume: the pruned list, evaluated (inference only)
    cfg2 = resume_config(path, ["--run-dir", d, "--resume", "--eval-sets", "cont,cont_samp,region,set4",
                                "--sep-range", "20", "400"])
    assert cfg2.eval_sets == "cont,cont_samp,region,set4" and cfg2.sep_range == (80.0, 120.0)
    tr2 = Trainer(cfg2, d, load_encoder=False)
    tr2.load(path)
    assert tr2.atlas is None, "the pruned list needs no atlas"
    ev = tr2.evaluate()
    pres = sorted({k.split("/")[0] for k in ev})
    assert pres == ["eval", "eval_cont", "eval_cont_samp", "eval_region"], pres
    for k in ("eval_cont/region_tiny/success", "eval_region/region_large/stop_low",
              "eval/region_medium/n", "eval/measured/stop_low_path_frac", tr2.best_metric):
        assert k in ev, k
    assert ev["eval_region/region_tiny/n"] == 6 and math.isnan(ev["eval/region_small/n"])
    for pre in ("eval_cont", "eval_region", "eval"):
        a_ = ev[f"{pre}/all/success"]
        lo = ev[f"{pre}/all/stop_low"]
        assert 0.0 <= lo <= 1.0 - a_ + 1e-9, pre
    tr2.save("best.pt")
    print(f"  old config -> eval_sets {leg}; resumed at iteration {tr.iteration} (best "
          f"{tr.best_eval}, {tr.best_metric}); --eval-sets on the resume: {pres} "
          f"({len(ev)} keys), no atlas")
    return d


# ── 5. eval_policy.py ───────────────────────────────────────────────────────


def test_eval_policy(run_dir: str, tmp: str):
    print("\n== 5. eval_policy.py defaults ==")
    ck = os.path.join(run_dir, "best.pt")
    out = os.path.join(tmp, "out")
    tables = os.path.join(tmp, "tables")
    mt = {f: os.path.getmtime(os.path.join(run_dir, f))
          for f in ("oracle_cont.pt", "oracle_region.pt", "oracle_eval.pt")}
    res = eval_policy.main(["--ckpt", ck, "--n", "24", "--out", out, "--tables", tables,
                            "--name", "eval_region_a"])
    assert list(res["sets"]) == ["cont", "region", "set4"]
    for f, v in mt.items():
        assert os.path.getmtime(os.path.join(run_dir, f)) == v, f"{f} recomputed"
    # Argmax = the trainer's evaluation of the same checkpoint
    tr = Trainer(resume_config(ck, ["--run-dir", run_dir, "--resume"]), run_dir, load_encoder=False)
    tr.load(ck)
    ev = tr.evaluate()
    for nm, pre in (("cont", "eval_cont"), ("region", "eval_region"), ("set4", "eval")):
        sp = res["sets"][nm]["policies"]["argmax"]["splits"]
        for split in ("all", "region_tiny", "region_large"):
            for k in ("return", "success", "stop_low", "length"):
                v, w = sp[split].get(k), ev[f"{pre}/{split}/{k}"]
                assert (v is None and math.isnan(w)) or abs(v - w) < 1e-9, (nm, split, k, v, w)
    st = res["sets"]["set4"]["setups"]
    assert [g["name"] for g in st] == [g.name for g in SETUPS]
    assert [g["region_bin"] for g in st] == ["tiny", "medium", "medium", "tiny"]
    b = res["sets"]["region"]["tables"]["build"]
    assert b["kind"] == "region" and b["n_per_bin"] == [6, 6, 6, 6]
    rs = res["sets"]["region"]["policies"]["sampled"]["splits"]
    assert [rs[f"region_{nm}"]["n"] for nm in REGION_BINS] == [6, 6, 6, 6]
    md = open(os.path.join(out, "eval_region_a.md")).read()
    for h in ("## Target-region size", "### region", "### cont", "### set4", "| measured |",
              "stop < 0.9 P_ref %", "| region_tiny | < 0.02 |", "## Scenario tables"):
        assert h in md, h
    assert os.path.exists(os.path.join(out, "eval_region_a.json"))
    # --compare: the other run's bins next to these
    res2 = eval_policy.main(["--ckpt", ck, "--n", "24", "--out", out, "--tables", tables, "--name",
                             "eval_region_b", "--policy", "argmax", "--compare",
                             os.path.join(out, "eval_region_a.json")])
    md2 = open(os.path.join(out, "eval_region_b.md")).read()
    assert "### Comparison with run old_run (best.pt" in md2
    a2 = res2["sets"]["region"]["policies"]["argmax"]["splits"]["region_tiny"]["pct_oracle"]
    a1 = res["sets"]["region"]["policies"]["argmax"]["splits"]["region_tiny"]["pct_oracle"]
    assert (a1 is None and a2 is None) or abs(a1 - a2) < 1e-9
    print(f"  default sets {list(res['sets'])} (argmax + sampled), report {len(md.splitlines())} "
          f"lines; argmax = Trainer.evaluate on cont / region / set4 (all, tiny, large); cached "
          f"oracles reused; --name, --compare")


if __name__ == "__main__":
    print(f"device {torch.cuda.get_device_name()}")
    tmp = tempfile.mkdtemp()
    try:
        sc = Scanner(device=DEV)
        test_region_size(sc, tmp)
        test_region_table(sc, tmp)
        test_eval_sets(tmp)
        run = test_old_checkpoint(tmp)
        test_eval_policy(run, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nall tests passed")
