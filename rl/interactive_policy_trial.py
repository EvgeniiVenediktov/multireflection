"""
Interactive Herriott Cell visualization with trained policy integration.
Uses herriott_sim.py as backend and policy.py for the trained agent.

Controls:
  - Sliders: manually set M1/M2 pitch/yaw and separation
  - "Run Policy": auto-step the policy with animation
  - "Step":       single policy step
  - "Reset":      reset policy hidden state and prev_action
  - "Load Ckpt":  load a policy checkpoint file
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from herriott_sim import DEVICE
from herriott_env import HerriottEnv, EnvConfig, ObsMirror
from policy import RecurrentPolicy
import torch

# ── Constants (drawing only; observation and action come from EnvConfig) ──────
DIAMETER = 25.4
HOLE_DIAMETER = 3.0
HOLE_OFFSET = 7.0


class InteractiveHerriottCell:
    def __init__(
        self, checkpoint_path: str = None, obs_mirror="M2", policy_step_limit=16
    ):
        # Single-env HerriottEnv: same sim, renderer and action scaling as training
        self.env = HerriottEnv(
            EnvConfig(num_envs=1, obs_mirror=ObsMirror[obs_mirror], device=DEVICE)
        )
        self.sim = self.env.sim

        # ── Policy state ──────────────────────────────────────────────
        self.policy = None
        self.hidden = None
        self.prev_action = None
        self.step_count = 0
        self.running = False
        self.timer = None
        self.obs_mirror = obs_mirror
        self.policy_step_limit = policy_step_limit
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            self._load_policy(checkpoint_path)

        self.setup_plot()

    # ── Policy management ─────────────────────────────────────────────

    def _load_policy(self, path: str):
        """Load a trained policy checkpoint (matches train.py CheckpointManager format)."""
        self.policy = RecurrentPolicy(hidden_dim=128, feat_dim=128).to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        # train.py CheckpointManager saves:
        #   { "iteration", "policy_state_dict", "optimizer_state_dict", "sampler_state_dict", "extra" }
        if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
            sd = ckpt["policy_state_dict"]
            print(f"[INFO] Checkpoint from iteration {ckpt.get('iteration', '?')}")
        elif isinstance(ckpt, dict) and any(k.startswith("encoder.") for k in ckpt):
            # Raw state_dict saved directly
            sd = ckpt
        else:
            raise ValueError(
                f"Unrecognized checkpoint format. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )

        self.policy.load_state_dict(sd)
        self.policy.eval()
        self._reset_policy_state()
        print(f"[INFO] Policy loaded from {path}")

    def _reset_policy_state(self):
        """Reset GRU hidden state and prev_action."""
        self.hidden = torch.zeros(1, self.policy.hidden_dim, device=DEVICE)
        self.prev_action = torch.zeros(1, 2, device=DEVICE)
        self.step_count = 0

    # ── Observation rendering ─────────────────────────────────────────

    def _render_obs(self, result) -> torch.Tensor:
        """
        Render the policy observation (1, 1, img_res, img_res) from a sim result
        through the env's own conversion and GaussianRenderer.
        """
        env_out = self.env._convert_sim_output(result)
        return self.env._observe_from_result(env_out)["image"]

    # ── Policy stepping ───────────────────────────────────────────────

    def _policy_step(self):
        """Execute one policy step: observe → act → apply → redraw."""
        if self.policy is None:
            print("[WARN] No policy loaded.")
            return

        # 1. Get current sim state and hits
        state = self._current_state_tensor()
        result = self.sim.simulate(state)
        hits = self._result_to_hits(result)

        # 2. Render observation
        obs = self._render_obs(result)

        # 3. Policy forward pass (deterministic: tanh of the mean)
        with torch.no_grad():
            action, _, log_prob, value, h_new = self.policy.act(
                obs, self.prev_action, self.hidden, deterministic=True
            )

        # 4. Scale and apply action (adjust M2 pitch/yaw), as HerriottEnv.step
        cfg = self.env.cfg
        delta_pitch = action[0, 0].item() * cfg.action_scale[0]
        delta_yaw = action[0, 1].item() * cfg.action_scale[1]

        lim = cfg.m2_angle_limit
        new_m2_pitch = np.clip(self.s_m2_pitch.val + delta_pitch, -lim, lim)
        new_m2_yaw = np.clip(self.s_m2_yaw.val + delta_yaw, -lim, lim)

        # Update sliders (this triggers self.update via on_changed)
        self.s_m2_pitch.set_val(new_m2_pitch)
        self.s_m2_yaw.set_val(new_m2_yaw)

        # 5. Update policy state
        self.hidden = h_new
        self.prev_action = action
        self.step_count += 1

        if self.step_count == self.policy_step_limit:
            self._on_run(None)  # stop if we hit the step limit

        # 6. Update info display
        self._update_info(action, log_prob, value)

    def _update_info(self, action, log_prob, value):
        """Update the policy info text on the figure."""
        self.info_text.set_text(
            f"Step: {self.step_count}  |  "
            f"Action: [{action[0,0]:.3f}, {action[0,1]:.3f}]  |  "
            f"V: {value[0]:.3f}  |  "
            f"log π: {log_prob[0]:.3f}"
        )
        self.fig.canvas.draw_idle()

    # ── Button callbacks ──────────────────────────────────────────────

    def _on_step(self, event):
        """Single policy step."""
        self._policy_step()

    def _on_run(self, event):
        """Toggle continuous policy stepping."""
        if self.running:
            self.running = False
            if self.timer is not None:
                self.timer.stop()
                self.timer = None
            self.btn_run.label.set_text("Run Policy")
        else:
            if self.policy is None:
                print("[WARN] No policy loaded.")
                return
            self.running = True
            self.btn_run.label.set_text("Stop")
            self.timer = self.fig.canvas.new_timer(interval=500)  # ms between steps
            self.timer.add_callback(self._timer_step)
            self.timer.start()
        self.fig.canvas.draw_idle()

    def _timer_step(self):
        """Called by the animation timer."""
        if not self.running:
            return
        self._policy_step()

    def _on_reset(self, event):
        """Reset policy hidden state (keep current slider positions)."""
        if self.running:
            self._on_run(None)  # stop first
        if self.policy is not None:
            self._reset_policy_state()
            self.info_text.set_text("Policy state reset.")
            self.fig.canvas.draw_idle()

    def _on_load(self, text):
        """Load a checkpoint from the text box path."""
        path = text.strip()
        if not path:
            return
        try:
            self._load_policy(path)
            self.info_text.set_text(f"Loaded: {path}")
        except Exception as e:
            self.info_text.set_text(f"Load failed: {e}")
        self.fig.canvas.draw_idle()

    # ── Sim helpers ───────────────────────────────────────────────────

    def _current_state_tensor(self):
        """(1, 13) sim state from the sliders, built by the env (zero offsets)."""
        self.env.env_params[0] = torch.tensor(
            [self.s_m1_pitch.val, self.s_m1_yaw.val, self.s_sep.val]
        )
        self.env.m2_state[0] = torch.tensor([self.s_m2_pitch.val, self.s_m2_yaw.val])
        return self.env._build_sim_state()

    def _result_to_hits(self, result):
        # Mirror by bounce parity as in the env: first hit M2, then alternating
        n = result["hit_counts"][0].item()
        seq = result["hit_sequence"][0, :n].cpu().numpy()
        return [
            {"point": pt, "mirror": "M2" if i % 2 == 0 else "M1"}
            for i, pt in enumerate(seq)
        ]

    def _rotation_matrix(self, pitch, yaw):
        px, py = np.radians(pitch), np.radians(yaw)
        cx, sx, cy, sy = np.cos(px), np.sin(px), np.cos(py), np.sin(py)
        return np.array([[cy, sx * sy, cx * sy], [0, cx, -sx], [-sy, sx * cy, cx * cy]])

    def _get_laser_pos(self, state):
        R = self._rotation_matrix(state[0, 0].item(), state[0, 1].item())
        return R @ np.array([0, HOLE_OFFSET, -5])

    # ── Plot setup ────────────────────────────────────────────────────

    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 9))

        # Main axes: 3D, schematic, realistic, observation
        self.ax3d = self.fig.add_subplot(141, projection="3d")
        self.ax2d = self.fig.add_subplot(142)
        self.ax2d_real = self.fig.add_subplot(143)
        self.ax_obs = self.fig.add_subplot(144)

        plt.subplots_adjust(bottom=0.35, top=0.93, left=0.02, right=0.98, wspace=0.05)

        # ── Sliders ──────────────────────────────────────────────────
        ax_m1_pitch = plt.axes([0.10, 0.24, 0.25, 0.025])
        ax_m1_yaw = plt.axes([0.10, 0.20, 0.25, 0.025])
        ax_m2_pitch = plt.axes([0.45, 0.24, 0.25, 0.025])
        ax_m2_yaw = plt.axes([0.45, 0.20, 0.25, 0.025])
        ax_sep = plt.axes([0.10, 0.14, 0.60, 0.025])

        self.s_m1_pitch = Slider(ax_m1_pitch, "M1 Pitch", -5, 5, valinit=0)
        self.s_m1_yaw = Slider(ax_m1_yaw, "M1 Yaw", -5, 5, valinit=0)
        self.s_m2_pitch = Slider(ax_m2_pitch, "M2 Pitch", -5, 5, valinit=0)
        self.s_m2_yaw = Slider(ax_m2_yaw, "M2 Yaw", -5, 5, valinit=0)
        self.s_sep = Slider(ax_sep, "Separation", 50, 400, valinit=100)

        for s in [
            self.s_m1_pitch,
            self.s_m1_yaw,
            self.s_m2_pitch,
            self.s_m2_yaw,
            self.s_sep,
        ]:
            s.on_changed(self.update)

        # ── Policy buttons ───────────────────────────────────────────
        ax_step = plt.axes([0.10, 0.06, 0.08, 0.04])
        ax_run = plt.axes([0.20, 0.06, 0.08, 0.04])
        ax_reset = plt.axes([0.30, 0.06, 0.08, 0.04])

        self.btn_step = Button(ax_step, "Step")
        self.btn_run = Button(ax_run, "Run Policy")
        self.btn_reset = Button(ax_reset, "Reset")

        self.btn_step.on_clicked(self._on_step)
        self.btn_run.on_clicked(self._on_run)
        self.btn_reset.on_clicked(self._on_reset)

        # ── Checkpoint path text box ─────────────────────────────────
        ax_path = plt.axes([0.52, 0.06, 0.35, 0.04])
        self.textbox = TextBox(
            ax_path,
            "Ckpt: ",
            initial=(self.checkpoint_path if self.checkpoint_path else ""),
        )
        self.textbox.on_submit(self._on_load)

        # ── Info text ────────────────────────────────────────────────
        self.info_text = self.fig.text(
            0.50,
            0.01,
            (
                "No policy loaded. Enter checkpoint path and press Enter."
                if not self.checkpoint_path
                else ""
            ),
            ha="center",
            fontsize=9,
            family="monospace",
        )

        self.update(None)
        plt.show()

    # ── Drawing ───────────────────────────────────────────────────────

    def update(self, val):
        state = self._current_state_tensor()
        result = self.sim.simulate(state)
        hits = self._result_to_hits(result)
        self.draw(hits, state, result)

    def _render_gaussian_spots(self, ax, hits, spot_sigma=0.5):
        res, extent = 256, 15
        x = np.linspace(-extent, extent, res)
        X, Y = np.meshgrid(x, x)

        img = np.zeros((res, res), dtype=np.float32)
        m_hits = [
            (h["point"][0], h["point"][1], 0.98 ** (i + 1))
            for i, h in enumerate(hits)
            if h["mirror"] == self.obs_mirror
        ]
        for px, py, intensity in m_hits:
            img += intensity * np.exp(
                -((X - px) ** 2 + (Y - py) ** 2) / (2 * spot_sigma**2)
            )

        R2 = X**2 + Y**2
        mirror_mask = R2 <= (DIAMETER / 2) ** 2

        rgb = np.zeros((res, res, 3), dtype=np.float32)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = img, img * 0.2, img * 0.1

        bg = np.zeros((res, res, 3), dtype=np.float32)
        bg[mirror_mask] = [0.1, 0.10, 0.1]

        if self.obs_mirror == "M1":
            hole_mask = X**2 + (Y - HOLE_OFFSET) ** 2 <= (HOLE_DIAMETER / 2) ** 2
            bg[hole_mask] = 0

        ax.imshow(
            np.clip(bg + rgb * 2, 0, 1),
            extent=[-extent, extent, -extent, extent],
            origin="lower",
        )
        ax.set_facecolor("black")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")

    def draw(self, hits, state, result):
        self.ax3d.cla()
        self.ax2d.cla()
        self.ax2d_real.cla()
        self.ax_obs.cla()

        sep = self.s_sep.val
        u = np.linspace(0, 2 * np.pi, 30)

        # ── 3D: mirrors ──────────────────────────────────────────────
        for z, pitch, yaw, c in [
            (0, self.s_m1_pitch.val, self.s_m1_yaw.val, "blue"),
            (sep, self.s_m2_pitch.val, self.s_m2_yaw.val, "red"),
        ]:
            R = self._rotation_matrix(pitch, yaw)
            pos = np.array([0, 0, z])
            for r in np.linspace(0, DIAMETER / 2, 5):
                local = np.stack(
                    [r * np.cos(u), r * np.sin(u), np.zeros_like(u)], axis=1
                )
                world = (R @ local.T).T + pos
                self.ax3d.plot(
                    world[:, 0], world[:, 1], world[:, 2], color=c, alpha=0.3, lw=0.5
                )
            edge = np.stack(
                [DIAMETER / 2 * np.cos(u), DIAMETER / 2 * np.sin(u), np.zeros_like(u)],
                axis=1,
            )
            self.ax3d.plot(*((R @ edge.T) + pos.reshape(3, 1)), color=c, lw=2)

        # Hole
        R1 = self._rotation_matrix(self.s_m1_pitch.val, self.s_m1_yaw.val)
        hole = np.stack(
            [
                HOLE_DIAMETER / 2 * np.cos(u),
                HOLE_OFFSET + HOLE_DIAMETER / 2 * np.sin(u),
                np.zeros_like(u),
            ],
            axis=1,
        )
        self.ax3d.plot(*((R1 @ hole.T)), "yellow", lw=2)

        # Laser & path
        lp = self._get_laser_pos(state)
        self.ax3d.scatter(*lp, c="green", s=80, marker="^")
        if hits:
            path = np.array([lp] + [h["point"] for h in hits])
            self.ax3d.plot(path[:, 0], path[:, 1], path[:, 2], "g-", lw=1.5)
            for h in hits:
                self.ax3d.scatter(
                    *h["point"], c="blue" if h["mirror"] == "M1" else "red", s=15
                )

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title(f"Bounces: {len(hits)}")

        # ── 2D schematic ─────────────────────────────────────────────
        self.ax2d.add_patch(plt.Circle((0, 0), DIAMETER / 2, fill=False, color="red"))

        if self.obs_mirror == "M1":
            self.ax2d.add_patch(
                plt.Circle(
                    (0, HOLE_OFFSET), HOLE_DIAMETER / 2, color="yellow", alpha=0.5
                )
            )
        for i, h in enumerate(hits):
            if h["mirror"] == self.obs_mirror:
                self.ax2d.scatter(h["point"][0], h["point"][1], c="red", s=30)
                self.ax2d.annotate(str(i + 1), h["point"][:2], fontsize=7)
        self.ax2d.set_xlim(-15, 15)
        self.ax2d.set_ylim(-15, 15)
        self.ax2d.set_xticks([])
        self.ax2d.set_yticks([])
        self.ax2d.set_aspect("equal")
        self.ax2d.set_title(f"{self.obs_mirror} Schematic")

        # ── 2D realistic ─────────────────────────────────────────────
        self._render_gaussian_spots(self.ax2d_real, hits)
        self.ax2d_real.set_title(f"{self.obs_mirror} Realistic")
        self.ax2d_real.set_xticks([])
        self.ax2d_real.set_yticks([])

        # ── Policy observation (what the agent sees) ──────────────────
        obs = self._render_obs(result)
        obs_np = obs[0, 0].cpu().numpy()
        ext = self.env.cfg.img_extent
        res = self.env.cfg.img_res
        self.ax_obs.imshow(
            obs_np,
            cmap="hot",
            origin="lower",
            extent=[-ext, ext, -ext, ext],
        )
        self.ax_obs.set_title(f"Policy Obs ({res}×{res})  Step {self.step_count}")
        self.ax_obs.set_aspect("equal")
        self.ax_obs.set_xticks([])
        self.ax_obs.set_yticks([])

        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    import sys

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    InteractiveHerriottCell(checkpoint_path=ckpt)
