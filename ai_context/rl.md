# Reinforcement learning branch (dormant)

> **Not a current concern.** The paper being revised covers the supervised MLP/CNN stage only.
> This exists so the RL code is not re-derived; expect drift and rough edges.

Goal: instead of regressing an absolute pose from one frame, learn a closed-loop *policy* that
nudges mirror 2 until the beam makes many bounces. Trained entirely in the GPU ray-trace sim
([simulation.md](simulation.md)), with sim-to-real transfer planned via a swappable encoder.

## Environment - `herriott_env.py`

`HerriottEnv` (`:147`) - vectorized, fully GPU-resident, no gymnasium dependency, no auto-reset
(the train loop resets done envs itself).

- **Split of control**: `env_params = (m1_pitch, m1_yaw, separation)` are fixed per episode by the
  sampler; the agent only moves `m2_pitch, m2_yaw`.
- **Action** `(B, 2)` in [-1, 1], scaled by `action_scale = (0.5, 0.5)` deg per step, accumulated
  into `m2_state`, clamped to `m2_angle_limit = 5.0` deg.
- **Observation** dict: `image (B,1,64,64)` + optional `state (B,5)` = `[m1p, m1y, m2p, m2y, sep]`.
- **`GaussianRenderer`** (`:70`) renders bounce hit points as Gaussian blobs (`spot_sigma = 0.5` mm)
  on a `img_extent = 14` mm half-width grid, with a precomputed mirror-aperture mask and (for M1)
  a hole mask. This is the visual analogue of the real camera view. `obs_mirror` selects M1
  (has the entry hole) or M2 (solid, default).
- **Reward** (`:390`): `bounce_count / max_bounces` minus `0.5` if fewer than 4 bounces.
  The exit-through-hole term is written but disabled (`r_exit = 0`).
- **Episode**: `max_steps = 64`, done purely on step count.
- `reset_with_config(env_params, m2_init, env_ids)` is the API the trainer uses;
  `reset()` is a random-init convenience for standalone testing.
- `_convert_sim_output` (`:322`) bridges the sim dict to the env contract: validity mask from
  `hit_counts`, world x,y taken as approximate local mirror coords, an alternating M2/M1 mirror
  pattern, geometric intensity decay, and a heuristic `exited` flag (stopped early while still
  bright => probably went out the hole).

## Policy - `policy.py`

`RecurrentPolicy` (`:81`): encoder -> concat previous action -> linear projection -> GRU(128) ->
separate policy and value heads. Continuous actions: `Normal(mean, exp(log_std))` squashed with
`tanh`, with the matching log-prob and entropy corrections (`_tanh_log_prob` `:179`,
`_tanh_entropy` `:188`, `_atanh` `:253`). `act()` `:197` for rollout, `evaluate()` `:224` for PPO.

`SpotPatternEncoder` (`:39`): stem conv -> 4 pre-activation `ResBlock`s with stride-2
(64 -> 32 -> 16 -> 8 -> 4) -> global average pool -> `feat_dim = 128`.

**Sim-to-real hooks** - the reason the encoder is a separate module:
`set_encoder(new_encoder)` (`:260`) swaps in a real-camera encoder, `freeze_core()` (`:281`) freezes
GRU + heads for encoder-only fine-tuning, `unfreeze_all()` (`:293`), and
`encoder_parameters()` / `core_parameters()` for per-group learning rates.

## Config sampling - `config_sampler.py`

`ConfigRange` (`:15`): m1_pitch/yaw and m2_pitch/yaw in [-3, 3] deg, separation in [80, 400] mm.
A batch shares one `env_params` (so the GRU sees a consistent environment) and gets per-env
`m2_init`. `BatchConfig` (`:26`) carries `env_params (3,)`, `m2_init (B,2)`, `bin_idx`.

- `SimpleConfigSampler` (`:49`) - fixed or uniform-random; `fixed_env_params` /
  `fixed_m2_init` lock the environment for debugging. `m2_jitter_std = 0.15`.
- `BinnedConfigSampler` (`:138`) - discretizes the config space into bins and samples them with a
  priority derived from observed returns (`update(bin_idx, mean_return)`, `_sample_bin` `:197`),
  i.e. a curriculum over hard configurations. `state_dict` / `load_state_dict` so priorities
  survive checkpoint resume.

## Training - `train_rl.py`

Recurrent PPO, single file.

- `TrainConfig` (`:36`): 256 envs x 64 steps per iteration, 4 PPO epochs, mini-batch of 64 *envs*
  (whole trajectories, to keep GRU state contiguous), clip 0.2, vf 0.5, ent 0.01, grad-norm 0.5,
  gamma 0.99, GAE lambda 0.95, Adam lr 3e-4, 50k iterations.
- Rollout buffers are preallocated on GPU: obs, action, logprob, reward, done, value, hidden,
  prev_action, all `(T, B, ...)`.
- Done envs are reset mid-rollout to the *same* env config with fresh M2 jitter, and their GRU
  hidden state is zeroed.
- `compute_gae` (`:195`), `ppo_update` (`:213`), `train` (`:286`).
- `CheckpointManager` (`:81`) writes latest/best under `run_dir = runs/herriott`, storing policy,
  optimizer, sampler state, iteration and return; `resume_last` is on by default.
- `Logger` (`:148`) wraps W&B (`herriott-rl` project), optional.

`interactive_policy_trial.py` loads a checkpoint into the matplotlib viewer for qualitative
inspection.

## Known drift - fix before running

- `HerriottEnv._build_sim_state` produces a **(B, 5)** tensor, but `HerriottSim.simulate` asserts
  `STATE_DIM == 13`. The env needs to pad the 8 offset/laser-error DOFs with zeros.
- `interactive_policy_trial.py` calls `create_sim(mounted_laser=True)` and `herriott_env.py` has a
  commented `mounted_laser=True` argument, but neither `create_sim` nor `HerriottSim.__init__`
  accepts that keyword any more.
- `EnvConfig.sim.max_bounces` defaults to 100 (the search setting); the training comment in
  `herriott_sim.py:73` suggests 50. The reward is normalized by whichever is active, so the reward
  scale silently changes with this value.
- `_convert_sim_output` treats world x,y as mirror-local coordinates - only valid for small tilts.

## Possible improvements (noted, not done)

- **Data/observation path**: rendering is recomputed from the full `hit_sequence` every step; only
  the last-bounce set changes when M2 moves slightly. Caching or incremental rendering would cut
  the per-step cost.
- The env re-simulates the entire bounce cascade on every step even though `env_params` are fixed
  for the episode; a warm start from the previous trajectory is possible.
- No replay/dataset abstraction at all - rollouts live only in the preallocated buffers. If
  off-policy methods or offline pretraining on the real dataset are wanted, a JPEG-folder loader
  like the supervised one (`DirectImageDataset` in [training.md](training.md)) would be the natural
  bridge, and would also be the vehicle for the planned encoder swap.
- Reward is bounce-count only; path length and exit-through-hole are computed but unused.
