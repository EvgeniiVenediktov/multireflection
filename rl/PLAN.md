# RL plan

Scope: everything RL lives in `rl/`; run scripts from here or from the repo root
(`uv run python rl/<script>.py`), imports are flat sibling imports. Nothing else in the repo
is needed. `rl/herriott_sim.py` is a copy of `simulation/herriott_sim.py`: the original stays
because the stability search, the interactive viewers and `tools/` import it, and the RL copy
will diverge.

Moved here from the repo root (unchanged apart from imports): `herriott_env.py`, `policy.py`,
`train_rl.py`, `config_sampler.py`, `interactive_policy_trial.py`. They are the old
implementation and get replaced by the rewrite.

Review is of the working tree as of 2026-10-15, including the uncommitted changes to those
files. Timings: local RTX A2000 under WSL2, B = 256 envs.

## Review findings

### Simulator - `herriott_sim.py`

| # | Finding | Action |
|---|---|---|
| S1 | Defaults do not match the cell: `MirrorConfig` has ROC 150, diameter 24.4, hole 1.8 ("sim2real" fudges); `HerriottSim()` defaults the M1 hole to 1.5 while `create_sim()` uses 1.8. The measured cell is ROC 200 / 25.4 / 1.5, and ROC 150 does not give the 31-bounce pattern. | One `CellGeometry` with the measured values. Regression test: nominal state gives 31 bounces, 2733 mm, exit through the hole. |
| S2 | **Speed.** About 100 small kernels per bounce plus two `active.any()` host syncs: 3.5 ms per bounce. A step costs 110 ms on the 31-bounce plateau and 1.36 s for a trapped beam at a 400-bounce cap. Large batches amortize it (361k configs in 2 s), RL-sized batches do not. | Fused per-ray Triton kernel (one thread per ray, bounce loop in registers, early exit per ray). Fallback: sync-free fixed-length loop captured in a CUDA graph. Keep the PyTorch version as the reference for tests. |
| S3 | Rays that never find the hole run to `max_bounces` (or the intensity cutoff, ~456 bounces at mirror reflectivity 0.98), so their path is set by the cap. | Needs a reward decision (D1). |
| S4 | Per-hit mirror id and mirror-local (x, y) are computed in `_check_aperture` but not returned; the env reconstructs them (E2). | Return `hit_mirror (B,N)`, `hit_local_xy (B,N,2)` and a stop reason (hole / missed / back side / cap). |
| S5 | Path length includes the 5 mm from the laser to M1 and excludes the last segment to the exit. | Define the reward path explicitly (constant offset otherwise, low priority). |
| S6 | `HerriottSim.reward()` is a leftover example. | Delete. |

### Environment - `herriott_env.py`

| # | Finding | Action |
|---|---|---|
| E1 | Renderer builds a dense `(B, N, H*W)` tensor: 2.0 GB at N = 100 and 8.0 GB at N = 400, roughly half of the 57 / 154 ms step. | Splat valid hits into the pixel grid (bilinear `index_add_`), then a separable Gaussian blur: O(B*N + B*H*W*k). Can be fused into the sim kernel later. |
| E2 | Spot coordinates are world x, y, and the mirror of each hit is assumed from strict M2/M1 alternation. | Use S4 outputs. |
| E3 | Per-image max normalization: one spot and 16 spots look equally bright. | Fixed intensity scale (spot after k reflections drawn at reflectivity^k, optional log). |
| E4 | `obs['state']` exposes the true mirror angles. Unused by the policy, but privileged. | Keep only in `info`, for diagnostics and an optional privileged critic. |
| E5 | Reward is `bounces / max_bounces - 0.5 * [bounces < 4]`; its scale changes with `max_bounces`, and it is not path length. | Path-length reward with "not move" termination, see Decisions. |
| E6 | `reset_with_config(env_ids=subset)` re-simulates and re-renders all B envs. | Partial reset touches only the reset envs; auto-reset inside the env, final obs kept for the truncation bootstrap. |
| E7 | Controlled DOFs are hard-wired to M2; M1 tilt and separation are "env params", the laser is fixed at 0. | Geometry = full 13-DOF base state; controlled DOF indices are config (M2 now, M1 later). |
| E8 | Continuous action, up to 0.5 deg per step, limit +-5 deg. The 89-bounce island is 0.24 deg wide, so a 0.5 deg step jumps across it, and a sampled continuous action is never exactly 0. | Discrete actions per axis including 0 (P1). Limits from the real actuation range. |

### Policy - `policy.py`

| # | Finding | Action |
|---|---|---|
| P1 | Tanh-Gaussian continuous action: no exact "not move". | Factorized categorical per axis over a step ladder, e.g. `{0, +-1, +-3, +-10, +-30, +-100} x 0.01 deg` (11 bins per axis, configurable). (0, 0) = not move. Exact entropy, no tanh corrections. |
| P2 | History through a GRU over (encoding, previous action), trained with BPTT over whole trajectories. The spec is an explicit stack. | Stack of the last K (encoding, applied action) pairs, zero-padded with a mask at episode start; actor and critic MLPs on the stack. No hidden state, no BPTT. |
| P3 | Actor and critic share encoder and GRU; value loss on unnormalized returns flows into the shared trunk. | Shared encoder, separate heads, value normalization (running mean/std). Separate encoders as an option. |
| P4 | Encoder forward is 17 ms for 256 images in fp32. | bf16 autocast, channels_last; size the encoder after the env is fast. |
| P5 | `set_encoder` / `freeze_core` hooks for sim-to-real. | Keep the encoder a separate module with the same idea. |
| P6 | (new) **Path-length head.** Same encoder and stack, extra head that predicts the current light path length (and optionally bounce count / stop reason) from the images. Labels come free from the sim at every step. | Auxiliary supervised loss during PPO: shapes the encoder and gives a camera-only path estimate at deployment, usable as a stop/confidence signal for "not move" and on the real cell. Not the value head: value is discounted future return, this is the current state. |

### Trainer - `train_rl.py`

| # | Finding | Action |
|---|---|---|
| T1 | Rollout length = episode length and all envs reset at each iteration, so the mid-rollout reset path only fires at the last step (one wasted sim + render of all B) and episodes never span iterations. | Rollout length independent of episode length; envs persist across iterations with their history stacks. |
| T2 | All 256 envs share one sampled geometry per iteration. Harmless for stage 1, wrong for stage 2. | Per-env scenario sampling; sampler rewritten, see Stage 2. |
| T3 | "Best" checkpoint = best noisy training return; every improvement writes a new file. | Periodic deterministic eval on a fixed seeded set of initial tilts; keep `latest` and `best` (by eval) only. |
| T4 | No success metrics. | Eval: final path, fraction ending at >= 31 / 57 / 89 bounces, steps to first reach each, not-move rate, path-head error. |
| T5 | Per-env `.item()` loop on episode end; hyperparameters edited in `__main__`; W&B run name `None`. | Vectorized episode stats; CLI arguments; W&B name built at start from the config. |
| T6 | No KL early stop. | Target-KL early stop per epoch. |

The tanh log-prob math, GAE with truncation bootstrap and GroupNorm in the encoder are
correct; they go away only because the action space changes (P1) and the GRU is dropped (P2).

## Phases

1. **Baseline commit.** Done, `bb31707`.
2. **Simulator** (S1-S6). Done, `5bb72c2`: fused Triton kernel, 0.09-0.72 ms per call at
   B = 256, bit-identical to the PyTorch reference.
3. **Environment** (E1-E8). Done, `76aaabe`: `env.py`, 3 ms per step at B = 256, 64 px.
4. **Baselines.** Done: `evaluation.py`, `baselines.py`, `reports/baselines.md`,
   `reports/baselines_multisetup.md`. Oracle mean return 3.61 (single setup) / 0.862 (four
   setups, per-setup scale); setup-blind homing 32 % of the oracle on four setups.
5. **Encoder and history pretraining.** Level 1 benchmark done (`reports/encoder_benchmark.md`):
   CNN with spatial-softmax keypoints beats ViT-p8 / p4 on every metric; ViT trained stably but
   learned ~3x slower and missed the rare 89 / trapped frames. Level 2 targets go into phase 6
   as auxiliary heads.
6. **Policy and PPO** `agent.py`, `ppo.py`, `cluster/ppo_l40s.slurm` (P1-P6, T1-T6). Sanity order: one fixed initial
   tilt (must solve), a small radius around nominal, then the full range (curriculum on the
   initial-tilt radius). Path head and the other auxiliary heads trained from the start.
7. **Viewer** - rewrite `interactive_policy_trial.py` on the new env and policy, then delete
   the old files.

Stages 2-4 from README follow once stage 1 trains reliably.

## Phase 5 - encoder and history pretraining

**Identifiability.** In 41.5 % of the +-2 deg box (1-2 bounces) the M2 image is one spot at
(8.6, 1.4) mm, constant to 0.05 mm (0.1 px at 64 px): the first hit, set by the laser, not by
M2. The 1-bounce part (34.7 %) is blank on M1 as well. One frame cannot give the tilt there;
the history gives only displacement since the episode start. The policy has to search and
remember where it has looked, so K = 8 may be too short: try 16-32 with a causal transformer
over time. (Edge scattering on the real cell may make part of this region visible.)

**One network:** frame encoder -> token (frame embedding, applied action) per step -> causal
transformer over the last K steps -> heads: policy (incl. not move), value, path, auxiliary.
"Pretraining" is a training signal for this network, not a separate model.

**Level 1 - frame encoder, single-image targets only** (what is in the picture):
- spot map: narrow peak at each M2 spot centre, peak height = incident intensity
  (reflectivity^k); order-free, handles overlap, forces a position-preserving encoder (no
  global average pooling);
- spot count, classified into bins (overlap makes it noisy);
- ellipse fitted to the spots (centre, axes, angle).
Not targets here: tilt, laser tilt, separation, path in mm.

Nuisance randomization on the input only, targets unchanged: blur sigma and shape plus a
scatter halo (surface finish), gain, gamma, black level, noise (camera), channel response
(laser color), camera pose (homography; render the mirror rim so mirror-frame targets stay
identifiable), small geometry jitter. Added to `SpotRenderer` as a randomized mode.

Candidates, compared on this supervised benchmark before any RL:
- CNN with spatial-softmax keypoints (or flatten) instead of global pooling;
- small ViT: 4-6 layers, width 128-192, pre-LN, 4-8 px patches, 2D position embeddings.
Metrics: spot-map and count error, 89 vs trapped discrimination, robustness on held-out
render parameters, inference speed (deployment hardware undecided; minimizing model size comes later).

**Level 2 - history targets** (same network, need several steps):
- M2 tilt as mean + variance (Gaussian NLL), so the model can say "unknown" while the
  history is blind; stage 2: laser tilt and separation (RMA-style inference from history);
- forward dynamics: next spot map given the next action; inverse dynamics: action between
  two frames;
- current path / bounces (path head).
Trained on simulated trajectories with exploratory actions (random and scripted search).

**Order.** Default: one phase, these heads as auxiliary losses from the first PPO update.
Fallback (PPO slow to start, or ViT unstable): supervised warm-up of the same network, then
PPO with the encoder frozen, unfrozen later at ~10x lower learning rate.

**Real data (stage 4).** Spot pseudo-labels from the classical extraction in
`tools/count_reflections.py` / `tools/fit_extract_spots.py`; the 228k real raster (0.01 deg,
known motor positions, M1 tilt) as real trajectories for the dynamics targets; MAE on mixed
sim and real images. None of this needs laser-tilt labels.

## Stage 2 - scenario sampling (replaces `config_sampler.py`)

Problem: many (separation, laser tilt) pairs give no pattern, or only 1-2 bounces, anywhere
in the M2 box. Uniform sampling makes the reward scarce and fills batches with scenarios
that have nothing to learn. The old `BinnedConfigSampler` (one config per batch, priority
from raw return) does not address this; it is replaced.

Prior work weights scenarios by learning potential: Goal GAN (goals of intermediate
difficulty), Teacher-Student curriculum (learning progress), PAIRED / PLR / Robust PLR /
ACCEL (regret-based level generation and replay), ADR (widen randomization ranges while
performance holds), reverse curriculum (start near the goal); PopArt for per-task scale,
RMA for inferring the scenario from history. They estimate the achievable return; with the
Triton kernel we compute it exactly.

1. **Scenario atlas** (`scenarios.py`, built offline, cached to disk): grid over separation x
   laser pitch / yaw (laser offset and mirror offsets later). Per scenario, scan the M2 box
   (0.01 deg, 2-4 ms each): best path P_max and its tilt, area of the box above fractions of
   P_max, bounce histogram.
2. **Per-scenario reward scale:** `r = dP / P_max(scenario) - step_cost`, replacing P_nom.
   Best possible return is about `1 - P_0 / P_max`.
3. **Impossible scenarios** (P_max below a threshold, e.g. 10 bounces) stay at a small share
   (~10 %). The right behavior is to stop at once; on the real cell that is the signal that
   separation or laser need a human.
4. **Sampler**, per env (not one scenario per batch):
   - ADR-style range: start at the stage-1 geometry; widen each parameter's range when
     performance at its boundary stays above a threshold, shrink it when below;
   - inside the range, PLR-style replay prioritized by exact regret `P_max - P_final`
     (rank-based, with a staleness term), plus a share of fresh uniform samples;
   - the initial-tilt radius curriculum from stage 1 stays.
5. **Scenario inference:** the agent has to infer the scenario from its image/action
   history (implicit system identification, as in RMA); the optional privileged critic
   would get the scenario parameters.
6. **Metrics:** normalized final path `P_final / P_max` per scenario, success rate at
   >= 0.9 P_max, stop rate on impossible scenarios, regret distribution over the atlas.
7. **Bounce cap:** the intensity cutoff (~456) once separation varies (see D1).

**Implemented (2026-10-16), first full-range run:** `scenarios.py` (atlas: grid above plus
the four SETUPS, 0.01 deg scans at cap 128, P_floor = 20 x separation, P_ref =
max(P_max, P_floor); sampler: 10 % impossible, PLR replay of the good set with shortfall
`1 - P_final / P_max` as the score, fresh share 0.3), `AlignEnv(cfg, scenarios, sampler)`,
scenario head (separation, laser tilt; Gaussian NLL) instead of the setup head,
`ppo.py --setups full --init-from`, lr warmup + cosine, adaptive entropy, evals "full"
(atlas, 10 % impossible) and "set4" (deterministic and sampled). Not implemented: ADR
range widening, the initial-tilt curriculum, the intensity-cutoff cap (cap stays 128).
Encoder check (`eval_encoder.py --scenarios 4096`, `reports/encoder_tables_full.md`): the
four-setup encoder does not transfer (in-distribution recall 0.078 vs 0.825, count exact
0.187 vs 0.999, tilt probe RMSE 0.54 vs 0.09 deg), so the run trains it from the start
(`--unfreeze-at 0 --encoder-lr-scale 0.1`).

**Unseen-scenario evaluation (2026-10-16):** the "full" set draws from the training grid,
so it is in-distribution only. Added: off-grid eval tables (`scenarios.py offgrid`;
`runs/eval_sets/<kind>_v1.pt`): `offgrid` = midpoints between grid points in all three
parameters (interpolation), `extrap_short` / `extrap_long` = separation 20-49 / 201-350 mm
(extrapolation); 4096 episodes each, one per scenario, 90 / 10 good / impossible. `ppo.py`
evaluates them by default in full mode (`eval_offgrid/`, `eval_extrap_short/`,
`eval_extrap_long/`; `--no-eval-offgrid`, `--no-eval-extrap`) and can hold out a seeded,
stratified share of the grid (`--holdout-frac`, `--holdout-seed`; `eval_holdout/`, never
drawn by the sampler). `eval_policy.py` (`cluster/eval_l40s.slurm`) evaluates any
checkpoint. Run 4407578 best.pt (`reports/eval_4407578_best.md`), argmax: full 93.9 % of the
oracle, offgrid 63.8 %, extrap_short 39.8 %, extrap_long -46 % (wanders, stops in 37 %).
Off-grid the tilt and scenario heads are as accurate as on the grid at the first frame;
the path head (P / P_ref) is not, episodes get 3x longer and the stop rate drops. A one-off
check (1024 grid scenarios shifted off the grid) puts the loss on separation: laser tilt
+0.125 deg costs ~2 points of % oracle, separation +0.5 or +1 mm ~30; between neighbouring
grid separations (2 mm) only 9 % of the scenarios keep their plateau order and the P_max
tilt moves by a median 0.6 deg (laser +0.25 deg: 89 %, 0.2 deg). The 2 mm separation grid
undersamples the landscape, which suggests drawing separation continuously (or on a much
finer grid) in the next run.

**Implemented (2026-10-16), continuous scenarios (D7):** `scenarios.py` ContRanges (5D
scenario, constructive laser angles), `sample_valid_scenarios` / `candidate_batch` (the one
generator: seeded streams, angle bounds, 1-bounce beam check before any box scan),
`Scanner`, `ScenarioPool` (16,384 slots, 1/8 rescanned per iteration, PLR per slot with
generation-checked scores), the continuous eval table (`scenarios.py cont`);
`AlignEnv(cfg, pool=...)` with per-env copies of parameters / P_ref / slot generation;
5-parameter scenario head (re-initialized on `--init-from` when the normalization
changes); `ppo.py --scenarios pool --sep-range --laser-range --laser-offset-radius
--laser-mode --pool-size --pool-refresh` (default for `--setups full`), `eval_cont/`
(best.pt), the grid sets kept as fixed benchmarks; `pretrain_encoder.py --init` (fine-tune)
with continuous scenarios, `eval_encoder.py` per separation bin; tests `tests/test_pool.py`.

**Implemented (2026-10-16), target-region evaluation:** target-region size of a scenario =
area of the M2 box cells with path >= 0.9 P_max (where an episode counts as success), from
the P_max scan (tables, pool slots; `AlignEnv.box_stats` for the named setups); bins tiny
< 0.02, small 0.02-0.1, medium 0.1-0.5, large >= 0.5 deg^2 (training distribution: 6.3 /
8.9 / 19.3 / 65.5 %, median 1.18 deg^2; measured cell 0.016 and sep169 0.011 tiny, sep116
0.23 and sep144 0.20 medium). New set `eval_region/` (4096 good continuous scenarios, 1024
per bin, `scenarios.py region`), region splits in `eval_cont/` and `eval/`, early-stop
metrics (stop below 0.9 P_ref and its final P / P_ref). Pool runs evaluate by default
`cont`, `cont_samp`, `region`, `set4` (`--eval-sets`; the grid sets on request, older
checkpoints keep their sets). Run 4410759 at iteration 1600 (argmax, region set): success
31 / 56 / 75 / 94 % per bin; nearly every failure is a stop on a lower-order pattern
(final P / P_max 0.55-0.65, step-limit failures < 2 %); set4 17 % success (measured cell
0 %, stops at 0.43 P_max). Report `reports/eval_region_r3_it1600.md`.

## Decisions (2026-10-15)

- **D1 - trapped beams: bounce cap, default 128.** A 0.001 deg scan around the 89 island
  finds only 31, 55 (edge miss), 57, 89 and trapped (runs to the 456-bounce intensity cutoff);
  nothing between 89 and trapped. Any cap above 89 keeps the order; 128 puts the trapped
  path at ~11.3 m (~4.1x nominal) and bounds sim cost. Intensity weighting (segment after k reflections counted at reflectivity^k) was rejected: it is
  bounded (at most segment / (1 - reflectivity) = 4.4 m; 31 -> 2.05, 57 -> 3.02, 89 -> 3.68,
  trapped -> 4.41 m), so not unstable, but it compresses the top end and still ranks trapped
  first, so it buys nothing over the cap.
  The 128 value is specific to 88 mm: with the same laser tilt, 120 mm has a 129-bounce
  hole-exit pattern and 60 mm has patterns up to 403. When separation varies (stage 2), cap
  at the intensity cutoff (~456) instead.
- **D2 - not move terminates the episode.** The agent learns to recognize that it reached
  the longest path and stop. Reward with P_t the path after step t (capped), P_nom = 2733 mm:
  - move: `r_t = (P_t - P_{t-1}) / P_nom - c`, step cost `c` (start at 0.01);
  - not move: episode ends, `r_t = 0`;
  - time limit: truncation, bootstrap from V.

  The return telescopes to `(P_final - P_0) / P_nom - c * moves`: dense feedback on every
  move, final path is what counts, and moving on only pays when the expected gain exceeds
  `c`. The path head (P6) predicts P_t, which is exactly the quantity the stop decision
  depends on. Eval adds: stop rate, final path at stop vs the oracle's best reachable, steps.
- **D3 - camera view: M2 face** (the mirror without the laser window), plain mirror-plane
  view in M2-local coordinates. A real camera model comes with sim-to-real (stage 4).
- **D4 - action ladder: `{0, +-1, +-3, +-10, +-30, +-100} x 0.01 deg` per axis.** M2 tilts
  +-2 deg on both axes (mount limit, absolute angles in the sim frame); initial tilts uniform
  in that box. Nominal (-0.38, 0.27) and the 57 / 89 / trapped regions all lie inside it.

- **D5 - four setups in stage 1** (2026-10-15). With one setup the absolute tilt solves the
  task (homing via the mount limit reaches 99 % of the oracle). Setups: measured 88 mm plus
  116 / 144 / 169 mm with their own laser tilts, one per episode, reward scaled by each setup's
  best path. Setup-blind homing drops to 32 % of the oracle.
- **D6 - encoder: CNN with spatial-softmax keypoints** (benchmark above); ViT revisited only
  with longer training and oversampled rare patterns.
- **D7 - continuous scenarios, separation 20-400 mm** (2026-10-16). Run 4407578 trained
  on the grid (separation 50-200 mm step 2, laser pitch / yaw 0-8 deg step 0.25): 93.9 % of
  the oracle on grid scenarios, 63.8 % on off-grid midpoints. Shifting separation by
  0.5 mm costs 30 points (laser shifts 1-2); between neighbouring 2 mm separations only
  9 % of the scenarios keep their plateau order and the best tilt moves a median 0.6 deg:
  the policy memorized grid-specific targets (`reports/eval_4407578_best.md`). Decision:
  - no grid: scenarios drawn continuously, separation 20-400 mm (400 = 2 R, the
    concentric stability limit of the ROC 200 mm mirrors);
  - scenario = (separation, laser pitch, laser yaw, laser dx, laser dy): laser offset
    uniform over the M1 hole disk (|offset| < 1.5 mm, around the hole centre; radius 0 =
    the measured 0.75, 1.04 mm), laser angles -8..8 deg on both axes (a check at
    50-200 mm, 600 scenarios per quadrant: good shares pitch+ / yaw+ 31.8 %, + / - 35.5 %,
    - / + 4.0 %, - / - 13.0 %);
  - laser angles constructive: the M2 landing point at zero M2 tilt uniform over the
    aperture minus 0.5 mm, pitch / yaw derived from the laser origin (5 mm behind M1) and
    that point, rejected beyond +-8 deg; the useful angle range shrinks with separation by
    construction (valid |pitch| / |yaw| q90: 6.5 / 6.7 deg at 20-50 mm, 3.9 / 2.4 at
    200-250, 2.3 / 1.5 at 350-400). Uniform angles stay selectable (`--laser-mode angle`);
  - validity before any box scan (1-bounce sim at the box centre, exact): the beam must
    pass the M1 hole (not clipped by the back of M1 at the hole edge) and hit M2;
    invalid candidates are discarded and redrawn, they are not "impossible" (= valid,
    P_max < 20 x separation); 90 / 10 good / impossible among the valid ones;
  - candidate sample (12,288, constructive): angles out of bounds 11 %, clipped 8.3 %,
    missed 0, valid 80.7 %, good 46.6 % of the valid; good share of the valid per
    separation band 20-50 / 50-100 / ... / 350-400 mm: 59.5 / 55.0 / 59.1 / 58.7 / 59.2 /
    51.5 / 33.4 / 13.6 %; good slots per band 2.6 / 10.3 / 17.9 / 18.2 / 18.7 / 16.4 / 11.4
    / 4.4 % (short separations are rare: 75 % of the 20-50 mm candidates need more than
    8 deg). Uniform angles: valid 23.8 % (clipped 23.7 %, missed 52.4 %), good slots 24.8 %
    at 20-50 mm and 0.5 % at 350-400 mm;
  - near the limits the sim is finite (no NaN / inf, bounces within the cap): at 20 mm
    the cap binds (128 bounces, P_max up to 2.58 m, 2.5 % of the box at the cap); at
    380-400 mm 95.6 % of the candidates are valid but only 5.4 % good, P_max q50 2.3 m
    (median best pattern 6 bounces), 95 % of the box blind (<= 2 bounces), the good ones
    long (P_max q50 14.4 m, 37 bounces);
  - training: the scenario pool (16,384 slots, 1/8 rescanned per iteration, PLR within
    it: beta 1.0, rho 0.1, p_fresh 0.3), scenario head on the five parameters; eval
    `eval_cont/` (4096 seeded continuous scenarios, best.pt) plus the grid sets of run
    4407578 as fixed benchmarks; the encoder fine-tuned from the grid encoder on 16,384
    continuous scenarios (8k steps, lr 3e-4); transformer and heads from 4407578.

## Future fine-tuning (2026-10-16)

- **Near-best starts** (reverse curriculum), planned as a fine-tuning stage on a trained policy:
  a share of resets (e.g. 20 %) starts within 0.2-0.5 deg of the scenario's best tilt, so the
  agent learns the last precise steps onto small targets and when not to stop. Motivation
  (reports/eval_region_r3_it1600.md): success falls from 94 % on large targets (>= 0.5 deg^2)
  to 31 % on tiny ones (< 0.02 deg^2); the failures are deliberate stops on a lower-order
  pattern at 0.55-0.65 of P_ref after 8-10 moves.
- Option not yet decided: fill the pool stratified by target-region size (tiny / small /
  medium / large 25 % each instead of the natural 6 / 9 / 19 / 66 %).

**Implemented (2026-10-16), fine-tuning options** (all off by default; defaults bit-identical
to 493776e, `tests/test_finetune.py`): `--near-best-frac F --near-best-radius RMIN RMAX`
(pool runs: a share F of the good-scenario episodes starts at the slot's best tilt plus a
ring offset, r uniform in [RMIN, RMAX], clamped to the box; impossible scenarios uniform;
22 % of the good slots have their best tilt on the box edge, so ~16 % of the ring starts
are clamped and ~3.5 % land closer than 0.05 deg); `--pool-region-mix a,b,c,d` (good-slot
shares per target-region bin, rejection with per-bin quotas; a refresh refills each slot
from its own bin, so the shares stay exact); `--best-metric region` (best.pt on the
bin-balanced mean of eval_region/ % of the oracle). Planned run from r4b's best snapshot:
F 0.25, ring 0.05-0.5 deg, mix 20 / 20 / 25 / 35 %, lr 1e-4, 2e8 env steps, entropy target
1.5 -> 0.7 nats (`cluster/ppo_l40s.slurm`).
