"""
train.py - Recurrent PPO training loop for Herriott cell alignment

Features:
  - Save/load checkpoints: stop and resume anytime
  - W&B logging (optional, on by default)
  - Same-config batching with GRU-based policy
  - Simple or binned config sampling
"""

import torch
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from policy import RecurrentPolicy
from herriott_env import HerriottEnv, EnvConfig

# from herriott_env import HerriottEnv, EnvConfig
# from policy import RecurrentPolicy
from config_sampler import (
    SimpleConfigSampler,
    SimpleSamplerConfig,
    BinnedConfigSampler,
    BinnedSamplerConfig,
    ConfigRange,
)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    # Env
    num_envs: int = 256
    max_steps: int = 64

    # PPO
    ppo_epochs: int = 4
    mini_batch_envs: int = 64
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4

    # Training
    total_iterations: int = 50_000
    log_interval: int = 50
    save_interval: int = 500

    # Checkpoints
    run_dir: str = "runs/herriott"
    resume_last: bool = True
    resume_best: bool = False

    # Logging
    use_wandb: bool = True
    wandb_project: str = "herriott-rl"
    wandb_run_name: Optional[str] = None  # Auto-generated if None

    # Sampler mode: 'simple' for testing, 'binned' for full training
    sampler_mode: str = "simple"

    # Fixed config for simple sampler (None = random each iteration)
    fixed_env_params: Optional[tuple] = (0.0, 0.0, 250.0)
    fixed_m2_init: Optional[tuple] = (0.5, 0.5)

    device: torch.device = torch.device("cuda")
    dtype: torch.dtype = torch.float32


# ── Checkpoint ────────────────────────────────────────────────────────────────


class CheckpointManager:
    """Handles saving and loading full training state."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

    def save(
        self,
        iteration: int,
        policy,
        optimizer,
        sampler,
        extra: dict = None,
        best: bool = False,
        ret: float = None,
    ) -> Path:
        state = {
            "iteration": iteration,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "sampler_state_dict": sampler.state_dict(),
            "extra": extra or {},
            "return": ret or None,
        }
        path = self.ckpt_dir / f"ckpt_{iteration:07d}_{'best' if best else ''}.pt"
        torch.save(state, path)

        # Write latest pointer (Windows-compatible, no symlink)
        pointer = self.ckpt_dir / "latest.txt"
        pointer.write_text(path.name)

        if best:
            pointer = self.ckpt_dir / "best.txt"
            pointer.write_text(path.name)

        return path

    def load_latest(self) -> Optional[dict]:
        pointer = self.ckpt_dir / "latest.txt"
        if not pointer.exists():
            return None
        ckpt_name = pointer.read_text().strip()
        ckpt_path = self.ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            return None
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def load_best(self) -> Optional[dict]:
        pointer = self.ckpt_dir / "best.txt"
        if not pointer.exists():
            return None
        ckpt_name = pointer.read_text().strip()
        ckpt_path = self.ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            return None
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def list_checkpoints(self) -> list:
        return sorted(self.ckpt_dir.glob("ckpt_*.pt"))


# ── Logging ───────────────────────────────────────────────────────────────────


class Logger:
    """Thin wrapper: prints to console, optionally logs to W&B."""

    def __init__(self, cfg: TrainConfig):
        self.use_wandb = cfg.use_wandb
        self.wb_run = None

        if self.use_wandb:
            try:
                import wandb

                self.wb_run = wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    config={
                        "num_envs": cfg.num_envs,
                        "max_steps": cfg.max_steps,
                        "lr": cfg.lr,
                        "ppo_epochs": cfg.ppo_epochs,
                        "clip_eps": cfg.clip_eps,
                        "gamma": cfg.gamma,
                        "gae_lambda": cfg.gae_lambda,
                        "sampler_mode": cfg.sampler_mode,
                    },
                    resume="allow",
                    dir=cfg.run_dir,
                )
            except ImportError:
                print("[WARN] wandb not installed, logging to console only")
                self.use_wandb = False

    def log(self, data: dict, step: int):
        if self.use_wandb and self.wb_run is not None:
            import wandb

            wandb.log(data, step=step)

    def finish(self):
        if self.use_wandb and self.wb_run is not None:
            import wandb

            wandb.finish()


# ── GAE ───────────────────────────────────────────────────────────────────────


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        not_done = (~dones[t]).float()
        delta = rewards[t] + gamma * nv * not_done - values[t]
        last_adv = delta + gamma * lam * not_done * last_adv
        advantages[t] = last_adv

    return advantages, advantages + values


# ── PPO Update ────────────────────────────────────────────────────────────────


def ppo_update(
    policy,
    optimizer,
    cfg,
    obs,
    actions,
    old_lp,
    old_val,
    stored_hidden,
    stored_prev_act,
    advantages,
    returns,
):
    T, B = advantages.shape
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_pg = 0.0
    total_vf = 0.0
    total_ent = 0.0
    n = 0

    for epoch in range(cfg.ppo_epochs):
        perm = torch.randperm(B, device=cfg.device)

        for s in range(0, B, cfg.mini_batch_envs):
            idx = perm[s : s + cfg.mini_batch_envs]
            mb_B = idx.shape[0]

            mb_obs = obs[:, idx]
            mb_act = actions[:, idx]
            mb_olp = old_lp[:, idx]
            mb_adv = adv[:, idx]
            mb_ret = returns[:, idx]
            mb_pa = stored_prev_act[:, idx]

            h = torch.zeros(mb_B, policy.hidden_dim, device=cfg.device, dtype=cfg.dtype)

            new_lp, new_val, ent = [], [], []
            for t in range(T):
                lp, v, e, h = policy.evaluate(mb_obs[t], mb_act[t], mb_pa[t], h)
                new_lp.append(lp)
                new_val.append(v)
                ent.append(e)

            new_lp = torch.stack(new_lp)
            new_val = torch.stack(new_val)
            ent = torch.stack(ent)

            ratio = (new_lp - mb_olp).exp()
            s1 = ratio * mb_adv
            s2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            pg_loss = -torch.min(s1, s2).mean()
            vf_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
            ent_loss = -ent.mean()

            loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += ent_loss.item()
            n += 1

    return total_pg / n, total_vf / n, total_ent / n


# ── Main ──────────────────────────────────────────────────────────────────────


def train(
    cfg: Optional[TrainConfig] = None,
    sampler_cfg: Optional[SimpleSamplerConfig | BinnedSamplerConfig] = None,
):
    cfg = cfg or TrainConfig()
    dev = cfg.device

    # ── Init components ───────────────────────────────────────────────
    env = HerriottEnv(EnvConfig(num_envs=cfg.num_envs))
    policy = RecurrentPolicy().to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    if cfg.sampler_mode == "simple":
        if sampler_cfg is None:
            sampler_cfg = SimpleSamplerConfig(
                fixed_env_params=cfg.fixed_env_params,
                fixed_m2_init=cfg.fixed_m2_init,
            )
        sampler = SimpleConfigSampler(sampler_cfg)
    else:
        if sampler_cfg is None:
            sampler_cfg = BinnedSamplerConfig()
        sampler = BinnedConfigSampler(sampler_cfg)

    ckpt_mgr = CheckpointManager(cfg.run_dir)
    logger = Logger(cfg)

    B = cfg.num_envs
    T = cfg.max_steps
    start_iter = 0
    best_return = float("-inf")

    # ── Resume from checkpoint ────────────────────────────────────────
    if cfg.resume_last or cfg.resume_best:

        ckpt = ckpt_mgr.load_latest() if cfg.resume_last else None
        ckpt = ckpt_mgr.load_best() if cfg.resume_best else None
        if ckpt is not None:
            policy.load_state_dict(ckpt["policy_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            sampler.load_state_dict(ckpt["sampler_state_dict"])
            start_iter = ckpt["iteration"] + 1
            best_return = ckpt["return"] if ckpt["return"] is not None else best_return
            print(f"[RESUME] Loaded checkpoint at iteration {ckpt['iteration']}")
        else:
            print("[START] No checkpoint found, training from scratch")

    # ── Pre-allocate rollout buffers (GPU) ────────────────────────────
    img_res = env.cfg.img_res  # e.g. 64
    buf_obs = torch.empty(T, B, 1, img_res, img_res, device=dev)
    buf_action = torch.empty(T, B, 2, device=dev)
    buf_logprob = torch.empty(T, B, device=dev)
    buf_reward = torch.empty(T, B, device=dev)
    buf_done = torch.empty(T, B, dtype=torch.bool, device=dev)
    buf_value = torch.empty(T, B, device=dev)
    buf_hidden = torch.empty(T, B, policy.hidden_dim, device=dev)
    buf_prev_act = torch.empty(T, B, 2, device=dev)

    # ── Training loop ─────────────────────────────────────────────────
    for iteration in range(start_iter, cfg.total_iterations):
        t_start = time.time()

        # 1. Sample config
        batch_config = sampler.sample(num_envs=B)

        # 2. Reset all envs
        obs = env.reset_with_config(
            env_params=batch_config.env_params,
            m2_init=batch_config.m2_init,
        )
        hidden = torch.zeros(B, policy.hidden_dim, device=dev, dtype=cfg.dtype)
        prev_action = torch.zeros(B, 2, device=dev, dtype=cfg.dtype)

        ep_returns = torch.zeros(B, device=dev)
        ep_lengths = torch.zeros(B, device=dev)
        completed_returns = []
        completed_lengths = []
        completed_bounces = []

        # 3. Collect rollout
        with torch.no_grad():
            for t in range(T):
                buf_obs[t] = obs["image"]
                buf_hidden[t] = hidden
                buf_prev_act[t] = prev_action

                action, log_prob, value, h_new = policy.act(
                    obs["image"], prev_action, hidden
                )

                obs_next, reward, done, info = env.step(action)

                buf_action[t] = action
                buf_logprob[t] = log_prob
                buf_reward[t] = reward
                buf_done[t] = done
                buf_value[t] = value

                ep_returns += reward
                ep_lengths += 1

                # Track completed episodes
                if done.any():
                    d_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    for di in d_ids:
                        completed_returns.append(ep_returns[di].item())
                        completed_lengths.append(ep_lengths[di].item())
                        if "bounce_count" in info:
                            completed_bounces.append(info["bounce_count"][di].item())

                    ep_returns[done] = 0.0
                    ep_lengths[done] = 0.0

                    # Reset done envs — same config, fresh M2 jitter
                    fresh_m2 = batch_config.m2_init[d_ids] + (
                        torch.randn(d_ids.shape[0], 2, device=dev)
                        * sampler.cfg.m2_jitter_std
                        if hasattr(sampler, "cfg")
                        and hasattr(sampler.cfg, "m2_jitter_std")
                        else torch.randn(d_ids.shape[0], 2, device=dev) * 0.15
                    )
                    obs_next_partial = env.reset_with_config(
                        env_params=batch_config.env_params,
                        m2_init=fresh_m2,
                        env_ids=d_ids,
                    )
                    # Merge reset obs into obs_next
                    for k in obs_next:
                        obs_next[k][d_ids] = obs_next_partial[k]

                    h_new[done] = 0.0

                obs = obs_next
                hidden = h_new
                prev_action = action

            # Bootstrap
            _, _, last_val, _ = policy.act(obs["image"], prev_action, hidden)

        # 4. GAE
        advantages, returns = compute_gae(
            buf_reward,
            buf_value,
            buf_done,
            last_val,
            cfg.gamma,
            cfg.gae_lambda,
        )

        # 5. PPO update
        pg_loss, vf_loss, ent_loss = ppo_update(
            policy,
            optimizer,
            cfg,
            buf_obs,
            buf_action,
            buf_logprob,
            buf_value,
            buf_hidden,
            buf_prev_act,
            advantages,
            returns,
        )

        # 6. Report to sampler
        iter_return = buf_reward.sum(dim=0).mean().item()
        sampler.update(batch_config.bin_idx, iter_return)

        t_elapsed = time.time() - t_start
        fps = (B * T) / t_elapsed

        # 7. Logging
        if iteration % cfg.log_interval == 0:
            log_data = {
                "train/iteration": iteration,
                "train/iter_return": iter_return,
                "train/pg_loss": pg_loss,
                "train/vf_loss": vf_loss,
                "train/ent_loss": ent_loss,
                "train/fps": fps,
                "train/elapsed_s": t_elapsed,
                "config/m1_pitch": batch_config.env_params[0].item(),
                "config/m1_yaw": batch_config.env_params[1].item(),
                "config/sep": batch_config.env_params[2].item(),
            }

            if completed_returns:
                log_data["episode/mean_return"] = sum(completed_returns) / len(
                    completed_returns
                )
                log_data["episode/mean_length"] = sum(completed_lengths) / len(
                    completed_lengths
                )
                log_data["episode/completed"] = len(completed_returns)
            if completed_bounces:
                log_data["episode/mean_bounces"] = sum(completed_bounces) / len(
                    completed_bounces
                )
                log_data["episode/max_bounces"] = max(completed_bounces)

            bin_stats = sampler.bin_stats()
            if bin_stats:
                for k, v in bin_stats.items():
                    log_data[f"sampler/{k}"] = v

            logger.log(log_data, step=iteration)

            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"iter {iteration:6d} | "
                f"ret {iter_return:7.2f} | "
                f"pg {pg_loss:.4f} vf {vf_loss:.4f} ent {ent_loss:.4f} | "
                f"fps {fps:.0f} | "
                f"eps {len(completed_returns)}"
            )

        if iter_return > best_return:
            best_return = iter_return
            ckpt_mgr.save(
                iteration,
                policy,
                optimizer,
                sampler,
                extra={"iter_return": iter_return},
                best=True,
                ret=best_return,
            )
            print(f"[NEW BEST] {best_return:.2f} at iteration {iteration}")

        # 8. Checkpoint
        if iteration % cfg.save_interval == 0 and iteration > 0:
            path = ckpt_mgr.save(
                iteration,
                policy,
                optimizer,
                sampler,
                extra={"iter_return": iter_return},
            )
            print(f"[SAVE] {path}")

    # Final save
    ckpt_mgr.save(cfg.total_iterations, policy, optimizer, sampler)
    logger.finish()
    print("[DONE]")


if __name__ == "__main__":
    train(
        cfg=TrainConfig(
            num_envs=256,
            max_steps=16,
            total_iterations=25_000,
            ent_coef=0.05,
            mini_batch_envs=256,
            resume_last=False,
        ),
        sampler_cfg=SimpleSamplerConfig(
            ranges=ConfigRange(
                sep=(93, 94),
                m1_pitch=(-0.1, 0.1),
                m1_yaw=(-0.1, 0.1),
                m2_pitch=(-5, 5),
                m2_yaw=(-5, 5),
            )
        ),
    )
