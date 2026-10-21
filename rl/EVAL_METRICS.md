# PPO eval metrics (eval.csv, W&B)

Every eval (`ppo.py`, every `--eval-every` iterations) logs one row to `eval.csv` and W&B.
Keys have the form

    eval_<set>[_samp]/<split>/<metric>

e.g. `eval_cont/region_medium/pct_oracle`: the cont set, the episodes whose target region
is medium-sized, the return as a percentage of the oracle's. Each set has `--eval-n`
(4096) episodes; every set uses the same fixed initial M2 tilts (seeded by `--eval-seed`).
Code: `ppo.py` (sets, splits, `FLAT_METRICS`, `EvalPolicy.diagnostics`), `evaluation.py`
(`split_metrics`), `scenarios.py` (eval tables, region bins), `baselines.plan_oracle`.

## Sets

| prefix | set |
|---|---|
| `eval_cont` | continuous scenarios of the training ranges (separation 20-400 mm, laser +-8 deg, laser offset in the M1 hole), the training filter and a 90 / 10 good / impossible mix, one episode per scenario; a seeded table cached in `--eval-tables` |
| `eval_region` | good continuous scenarios, a quarter per target-region bin (so the small-region cases are not a minority) |
| `eval` | "set4": the four named setups (`geometry.SETUPS`); the prefix keeps the keys of the first run (r1) |
| `_samp` | the same set with sampled actions (as in training); without it the policy takes its argmax |

Atlas runs also have `eval_full`, `eval_holdout`, `eval_offgrid`, `eval_extrap_short`,
`eval_extrap_long` (see the `ppo.py` docstring, `--eval-sets`).

## Splits

| split | episodes |
|---|---|
| `all` | every episode of the set |
| `blind` | start frame with 1-2 bounces (one M2 spot at a fixed place: no hint where to go; `evaluation.BLIND_MAX_BOUNCES`) |
| `informative` | start frame with >= 3 bounces |
| `good` / `impossible` | (cont) impossible: P_max below the floor (floor bounces x separation), the right behavior is to stop quickly; good: the rest |
| `region_tiny` / `region_small` / `region_medium` / `region_large` | target-region size: area of the M2 box cells with path >= 0.9 P_max; tiny < 0.02, small 0.02-0.1, medium 0.1-0.5, large >= 0.5 deg^2 (`scenarios.REGION_EDGES`); smaller is harder to find and hold |
| `sep20-50` ... `sep350-400` | (cont) separation bins, mm |
| `measured`, `sep116`, `sep144`, `sep169` | (set4) the four setups: the measured cell and the cells at 116 / 144 / 169 mm separation |

`eval_region/bin_mean/pct_oracle` is not a split: the mean of `pct_oracle` over the four
region bins (each bin weighted equally). With `--best-metric region` (r5 on) `best.pt` is
chosen on it; `best/value` is the value of the row, `best/best` the best so far.

## Metrics per split (`FLAT_METRICS`)

P_ref = max(P_max, P_floor) of the episode's scenario. The oracle knows the scenario and
plans the best ladder-move sequence from the same start (`baselines.plan_oracle`).

| metric | meaning |
|---|---|
| `return` | mean episode return: (P_final - P_start) / P_ref - 0.01 x moves |
| `oracle_return` | the oracle's mean return on the same episodes |
| `pct_oracle` | 100 x return / oracle_return (ratio of the means); the main score; negative = worse than not moving |
| `regret` | mean of (oracle return - return) |
| `path_frac` | mean P_final / P_ref |
| `success` | share ending at P_final >= 0.9 P_ref (`evaluation.SUCCESS_FRAC`) |
| `stop_rate` | share ended by the agent's "not move" (stop) action, not by the 64-step limit |
| `stop_first` | share stopping at the first step |
| `stop_low` | share stopped below 0.9 P_ref: premature stop (the main failure mode) |
| `stop_low_path_frac` | mean P / P_ref of those premature stops |
| `length` | mean episode length, steps |
| `end_cap` | share ending at the sim's bounce cap |
| `n` | episodes in the split |

## Per set (not per split)

| key | meaning |
|---|---|
| `wall_s` | eval time |
| `head_path_mae` | P / P_ref head, mean absolute error |
| `head_logpath_rel_err`, `head_logpath_log_mae` | absolute-path head (`--log-path-head`): mean relative error of P, mean error of log P |
| `head_tilt_mae_deg` | current-M2-tilt head, mean absolute error (deg); `_ge3`: frames with >= 3 bounces only |
| `head_tilt_std_deg` | the tilt head's own predicted uncertainty |
| `head_sep_mae_mm`, `head_laser_pitch_mae_deg`, `head_laser_yaw_mae_deg`, `head_laser_dx_mae_mm`, `head_laser_dy_mae_mm` | scenario head, per parameter |
| `..._step0` | the same error on the first frame of each episode only |

The heads are diagnostics; they do not enter the return.

## What to look at

1. `eval_region/bin_mean/pct_oracle`: model selection, balanced over difficulty.
2. `eval_cont/all/pct_oracle` and `success`: overall quality on the training distribution.
3. `eval_cont/all/stop_low`: how often the agent gives up too early.
4. `eval_region/region_tiny/pct_oracle`: the hardest cases.
5. `eval/all/pct_oracle` (set4): the four fixed setups, comparable back to r1.

W&B view with the `all` split of every set (pct_oracle, path_frac, stop_rate):
[Decision metrics](https://wandb.ai/e-venediktov-university-of-pittsburgh/herriott-rl?nw=94wy9kt38je).
Real-cell evals are separate (`eval_real.py`, reports in `reports/real_eval/`).
