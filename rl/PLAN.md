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

1. **Baseline commit.** Commit the move into `rl/` together with the current RL working-tree
   changes, so the rewrite has a reference point.
2. **Simulator** (S1-S6). `geometry.py` with the measured cell; fused kernel; per-hit mirror,
   local coordinates, stop reason. Tests: nominal 31 bounces / 2733 mm / hole exit; fused
   kernel matches the PyTorch reference on the landscape scan in README. Target: < 2 ms
   per step at B = 256 for any bounce count.
3. **Environment** `env.py` (E1-E8). Splat renderer, discrete actions with not move, path
   reward, auto-reset, history stack kept in the env. Target: < 5 ms per step at B = 256.
   Scripted checks: renders of nominal, 57, 89 and 1-3 bounce states.
4. **Baselines** before trusting PPO: random policy, always-not-move, and a sim-access oracle
   (greedy search on true path with the same action ladder) to get the reachable path and the
   step count the agent should match.
5. **Policy and PPO** `policy.py`, `train.py` (P1-P6, T1-T6). Sanity order: one fixed initial
   tilt (must solve), a small radius around nominal, then the full range (curriculum on the
   initial-tilt radius). Path head trained from the start as an auxiliary loss; check its
   error on its own before relying on it.
6. **Viewer** - rewrite `interactive_policy_trial.py` on the new env and policy, then delete
   the old files.

Stages 2-4 from README follow once stage 1 trains reliably.

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
