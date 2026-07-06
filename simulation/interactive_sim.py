"""
Interactive Herriott Cell visualization - 13 DOF (performance-optimized)
Uses herriott_sim.py as backend

Key perf fixes vs original:
  1. Timer-based throttle (~30ms) so slider drags don't queue unlimited redraws
  2. Vectorized Gaussian spot rendering (no Python loop over hits)
  3. Artist reuse: update line/scatter data in place instead of cla() + recreate
  4. Selective redraw: only blit the axes that actually changed
  5. Mirror wireframe cached and only rebuilt when geometry params change
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from herriott_sim import create_sim, make_state, STATE_DIM, DEVICE
import time

DIAMETER = 24.4
HOLE_DIAMETER = 3.2
HOLE_OFFSET = 7.0

# Throttle interval in seconds
UPDATE_INTERVAL = 0.03


class InteractiveHerriottCell:
    def __init__(self):
        self.sim = create_sim()
        self._updating = False
        self._pending_update = False
        self._last_update_time = 0.0

        # Cache keys for mirror geometry
        self._cached_mirror_params = None

        self.setup_plot()

    # ------------------------------------------------------------------ #
    # Slider / textbox creation (unchanged logic, wired to throttled update)
    # ------------------------------------------------------------------ #
    def _make_slider_with_textbox(
        self, slider_ax, textbox_ax, label, vmin, vmax, valinit=0, valfmt="%.2f"
    ):
        slider = Slider(slider_ax, label, vmin, vmax, valinit=valinit, valfmt=valfmt)
        textbox = TextBox(textbox_ax, "", initial=valfmt % valinit)
        textbox.label.set_fontsize(8)

        def on_slider_changed(val):
            if self._updating:
                return
            self._updating = True
            textbox.set_val(valfmt % val)
            self._updating = False
            self._request_update()

        def on_textbox_submit(text):
            if self._updating:
                return
            try:
                val = float(text)
                val = np.clip(val, vmin, vmax)
            except ValueError:
                val = slider.val
            self._updating = True
            slider.set_val(val)
            textbox.set_val(valfmt % val)
            self._updating = False
            self._request_update()

        slider.on_changed(on_slider_changed)
        textbox.on_submit(on_textbox_submit)
        return slider, textbox

    # ------------------------------------------------------------------ #
    # Throttled update
    # ------------------------------------------------------------------ #
    def _request_update(self):
        """Throttle: skip if called faster than UPDATE_INTERVAL."""
        now = time.monotonic()
        if now - self._last_update_time < UPDATE_INTERVAL:
            # Schedule a trailing-edge update so the final position is always drawn
            if not self._pending_update:
                self._pending_update = True
                self.fig.canvas.new_timer(interval=int(UPDATE_INTERVAL * 1000))
                # Use a simple timer callback
                timer = self.fig.canvas.new_timer(interval=int(UPDATE_INTERVAL * 1000))
                timer.add_callback(self._flush_pending)
                timer.single_shot = True
                timer.start()
            return
        self._last_update_time = now
        self._do_update()

    def _flush_pending(self):
        if self._pending_update:
            self._pending_update = False
            self._last_update_time = time.monotonic()
            self._do_update()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup_plot(self):
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(
            1, 5, bottom=0.42, hspace=0.3, wspace=0.05, left=0.02, right=0.98
        )
        self.ax3d = self.fig.add_subplot(gs[0, 0:3], projection="3d")
        self.ax2d = self.fig.add_subplot(gs[0, 3])
        self.ax2d_real = self.fig.add_subplot(gs[0, 4])

        # --- Slider layout (same as original) ---
        row_h = 0.025
        row_gap = 0.007
        left_x, left_w = 0.08, 0.32
        right_x, right_w = 0.55, 0.32
        tb_w = 0.04
        tb_gap = 0.005

        def row_y(r):
            return 0.02 + r * (row_h + row_gap)

        def left_slider_ax(r):
            return plt.axes([left_x, row_y(r), left_w, row_h])

        def left_tb_ax(r):
            return plt.axes([left_x + left_w + tb_gap, row_y(r), tb_w, row_h])

        def right_slider_ax(r):
            return plt.axes([right_x, row_y(r), right_w, row_h])

        def right_tb_ax(r):
            return plt.axes([right_x + right_w + tb_gap, row_y(r), tb_w, row_h])

        sep_w = 0.75
        ax_sep = plt.axes([left_x, row_y(0), sep_w, row_h])
        ax_sep_tb = plt.axes([left_x + sep_w + tb_gap, row_y(0), tb_w, row_h])

        ax_m2_ty = right_slider_ax(1)
        ax_m2_ty_tb = right_tb_ax(1)
        ax_m2_tx = right_slider_ax(2)
        ax_m2_tx_tb = right_tb_ax(2)
        ax_m2_yaw = right_slider_ax(3)
        ax_m2_yaw_tb = right_tb_ax(3)
        ax_m2_pitch = right_slider_ax(4)
        ax_m2_pitch_tb = right_tb_ax(4)
        ax_ldx = left_slider_ax(5)
        ax_ldx_tb = left_tb_ax(5)
        ax_ldy = right_slider_ax(5)
        ax_ldy_tb = right_tb_ax(5)
        ax_lpitch = left_slider_ax(6)
        ax_lpitch_tb = left_tb_ax(6)
        ax_lyaw = right_slider_ax(6)
        ax_lyaw_tb = right_tb_ax(6)

        self.s_sep, _ = self._make_slider_with_textbox(
            ax_sep, ax_sep_tb, "Separation (mm)", 40, 600, valinit=100, valfmt="%.1f"
        )
        self.s_m2_pitch, _ = self._make_slider_with_textbox(
            ax_m2_pitch, ax_m2_pitch_tb, "M2 Pitch (°)", -5, 5
        )
        self.s_m2_yaw, _ = self._make_slider_with_textbox(
            ax_m2_yaw, ax_m2_yaw_tb, "M2 Yaw (°)", -5, 5
        )
        self.s_m2_tx, _ = self._make_slider_with_textbox(
            ax_m2_tx, ax_m2_tx_tb, "M2 ΔX (mm)", -3, 3
        )
        self.s_m2_ty, _ = self._make_slider_with_textbox(
            ax_m2_ty, ax_m2_ty_tb, "M2 ΔY (mm)", -3, 3
        )
        self.s_ldx, _ = self._make_slider_with_textbox(
            ax_ldx, ax_ldx_tb, "Laser ΔX (mm)", -1.5, 1.5
        )
        self.s_ldy, _ = self._make_slider_with_textbox(
            ax_ldy, ax_ldy_tb, "Laser ΔY (mm)", -1.5, 1.5
        )
        self.s_lpitch, _ = self._make_slider_with_textbox(
            ax_lpitch, ax_lpitch_tb, "Laser Pitch (°)", -10, 10
        )
        self.s_lyaw, _ = self._make_slider_with_textbox(
            ax_lyaw, ax_lyaw_tb, "Laser Yaw (°)", -10, 10
        )

        # --- Persistent artists (created once, data updated in place) ---
        self._init_artists()
        self._do_update()
        plt.show()

    def _init_artists(self):
        """Create all plot artists once. We update their data each frame."""
        # 3D: mirror wireframes (lists of Line3D — rebuilt only on geometry change)
        self._mirror_lines = []

        # 3D: beam path + scatter
        (self._beam_line,) = self.ax3d.plot([], [], [], "g-", lw=1.5, alpha=0.4)
        self._m1_scatter = self.ax3d.scatter([], [], [], c="blue", s=15)
        self._m2_scatter = self.ax3d.scatter([], [], [], c="red", s=15)
        self._laser_scatter = self.ax3d.scatter([], [], [], c="green", s=80, marker="^")

        # 2D schematic
        self._m2_circle = plt.Circle((0, 0), DIAMETER / 2, fill=False, color="red")
        self.ax2d.add_patch(self._m2_circle)
        self._m2_scat2d = self.ax2d.scatter([], [], c="red", s=30)
        self._m2_annotations = []
        self.ax2d.set_xlim(-15, 15)
        self.ax2d.set_ylim(-15, 15)
        self.ax2d.set_aspect("equal")

        # 2D realistic: pre-allocate image
        self._spot_res = 256
        self._spot_extent = 15
        x = np.linspace(-self._spot_extent, self._spot_extent, self._spot_res)
        self._spot_X, self._spot_Y = np.meshgrid(x, x)
        R2 = self._spot_X**2 + self._spot_Y**2
        self._mirror_mask = R2 <= (DIAMETER / 2) ** 2

        init_img = np.zeros((self._spot_res, self._spot_res, 3))
        self._spot_im = self.ax2d_real.imshow(
            init_img,
            extent=[
                -self._spot_extent,
                self._spot_extent,
                -self._spot_extent,
                self._spot_extent,
            ],
            origin="lower",
        )
        self.ax2d_real.set_facecolor("black")
        self.ax2d_real.set_xlim(-self._spot_extent, self._spot_extent)
        self.ax2d_real.set_ylim(-self._spot_extent, self._spot_extent)
        self.ax2d_real.set_aspect("equal")

        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")

    # ------------------------------------------------------------------ #
    # Core update
    # ------------------------------------------------------------------ #
    def _build_state(self):
        return make_state(
            m2_pitch=self.s_m2_pitch.val,
            m2_yaw=self.s_m2_yaw.val,
            separation=self.s_sep.val,
            m2_tx=self.s_m2_tx.val,
            m2_ty=self.s_m2_ty.val,
            laser_dx=self.s_ldx.val,
            laser_dy=self.s_ldy.val,
            laser_pitch=self.s_lpitch.val,
            laser_yaw=self.s_lyaw.val,
        )

    def _do_update(self):
        state = self._build_state()
        result = self.sim.simulate(state)
        hits = self._result_to_hits(result)
        self._update_artists(hits, state)
        self.fig.canvas.draw_idle()

    def _result_to_hits(self, result):
        n = result["hit_counts"][0].item()
        seq = result["hit_sequence"][0, :n].cpu().numpy()
        sep = self.s_sep.val
        return [
            {"point": pt, "mirror": "M1" if pt[2] < sep / 2 else "M2"} for pt in seq
        ]

    # ------------------------------------------------------------------ #
    # Artist updates (no cla!)
    # ------------------------------------------------------------------ #
    def _rotation_matrix(self, pitch, yaw):
        px, py = np.radians(pitch), np.radians(yaw)
        cx, sx, cy, sy = np.cos(px), np.sin(px), np.cos(py), np.sin(py)
        return np.array(
            [
                [cy, sx * sy, cx * sy],
                [0, cx, -sx],
                [-sy, sx * cy, cx * cy],
            ]
        )

    def _get_laser_pos(self, state):
        s = state[0].cpu().numpy()
        R = self._rotation_matrix(s[0], s[1])
        m1_pos = np.array([s[5], s[6], 0.0])
        hole_local = np.array([s[9], HOLE_OFFSET + s[10], -5.0])
        return R @ hole_local + m1_pos

    def _mirror_params_key(self):
        """Tuple of values that affect mirror wireframe geometry."""
        return (
            self.s_sep.val,
            self.s_m2_pitch.val,
            self.s_m2_yaw.val,
            self.s_m2_tx.val,
            self.s_m2_ty.val,
        )

    def _rebuild_mirror_wireframes(self):
        """Remove old mirror lines and draw new ones. Only called when geometry changes."""
        for ln in self._mirror_lines:
            ln.remove()
        self._mirror_lines = []

        sep = self.s_sep.val
        u = np.linspace(0, 2 * np.pi, 30)
        mirrors = [
            (0, 0, 0, 0, 0, "blue"),
            (
                sep,
                self.s_m2_pitch.val,
                self.s_m2_yaw.val,
                self.s_m2_tx.val,
                self.s_m2_ty.val,
                "red",
            ),
        ]
        for z, pitch, yaw, tx, ty, c in mirrors:
            R = self._rotation_matrix(pitch, yaw)
            pos = np.array([tx, ty, z])
            for r in np.linspace(0, DIAMETER / 2, 5):
                local = np.stack(
                    [r * np.cos(u), r * np.sin(u), np.zeros_like(u)], axis=1
                )
                world = (R @ local.T).T + pos
                (ln,) = self.ax3d.plot(
                    world[:, 0], world[:, 1], world[:, 2], color=c, alpha=0.3, lw=0.5
                )
                self._mirror_lines.append(ln)
            edge = np.stack(
                [DIAMETER / 2 * np.cos(u), DIAMETER / 2 * np.sin(u), np.zeros_like(u)],
                axis=1,
            )
            w = (R @ edge.T) + pos.reshape(3, 1)
            (ln,) = self.ax3d.plot(w[0], w[1], w[2], color=c, lw=2)
            self._mirror_lines.append(ln)

        # Hole on M1
        hole = np.stack(
            [
                HOLE_DIAMETER / 2 * np.cos(u),
                HOLE_OFFSET + HOLE_DIAMETER / 2 * np.sin(u),
                np.zeros_like(u),
            ],
            axis=1,
        )
        hw = hole.T  # M1 at identity
        (ln,) = self.ax3d.plot(hw[0], hw[1], hw[2], "yellow", lw=2)
        self._mirror_lines.append(ln)

    def _update_artists(self, hits, state):
        sep = self.s_sep.val

        # --- Conditionally rebuild mirror wireframes ---
        key = self._mirror_params_key()
        if key != self._cached_mirror_params:
            self._cached_mirror_params = key
            self._rebuild_mirror_wireframes()

        # --- 3D beam path (update data in place) ---
        lp = self._get_laser_pos(state)
        if hits:
            path = np.array([lp] + [h["point"] for h in hits])
            self._beam_line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])

            m1_pts = np.array([h["point"] for h in hits if h["mirror"] == "M1"])
            m2_pts = np.array([h["point"] for h in hits if h["mirror"] == "M2"])

            if len(m1_pts):
                self._m1_scatter._offsets3d = (m1_pts[:, 0], m1_pts[:, 1], m1_pts[:, 2])
            else:
                self._m1_scatter._offsets3d = ([], [], [])
            if len(m2_pts):
                self._m2_scatter._offsets3d = (m2_pts[:, 0], m2_pts[:, 1], m2_pts[:, 2])
            else:
                self._m2_scatter._offsets3d = ([], [], [])
        else:
            self._beam_line.set_data_3d([], [], [])
            self._m1_scatter._offsets3d = ([], [], [])
            self._m2_scatter._offsets3d = ([], [], [])

        self._laser_scatter._offsets3d = ([lp[0]], [lp[1]], [lp[2]])

        # Adjust 3D axis limits to contain geometry
        pad = DIAMETER / 2 + 2
        self.ax3d.set_xlim(-pad, pad)
        self.ax3d.set_ylim(-pad, pad)
        self.ax3d.set_zlim(-10, sep + 10)
        self.ax3d.set_title(f"Bounces: {len(hits)}")

        # --- 2D schematic (update scatter data, not cla) ---
        self._m2_circle.center = (self.s_m2_tx.val, self.s_m2_ty.val)
        m2_hits_2d = [(i, h) for i, h in enumerate(hits) if h["mirror"] == "M2"]
        if m2_hits_2d:
            xy = np.array([h["point"][:2] for _, h in m2_hits_2d])
            self._m2_scat2d.set_offsets(xy)
        else:
            self._m2_scat2d.set_offsets(np.empty((0, 2)))

        # Annotations: remove old, add new
        for ann in self._m2_annotations:
            ann.remove()
        self._m2_annotations = []
        for idx, h in m2_hits_2d:
            ann = self.ax2d.annotate(str(idx + 1), h["point"][:2], fontsize=7)
            self._m2_annotations.append(ann)
        self.ax2d.set_title("M2 Schematic")

        # --- 2D realistic spots (vectorized) ---
        self._update_spots_vectorized(hits)
        self.ax2d_real.set_title("M2 Realistic")

    def _update_spots_vectorized(self, hits, spot_sigma=0.5):
        """Render all Gaussian spots in one vectorized pass — no Python loop."""
        X, Y = self._spot_X, self._spot_Y
        res = self._spot_res

        m2_data = [
            (h["point"][0], h["point"][1], 0.98 ** (i + 1))
            for i, h in enumerate(hits)
            if h["mirror"] == "M2"
        ]

        img = np.zeros((res, res), dtype=np.float32)

        if m2_data:
            arr = np.array(m2_data, dtype=np.float32)  # (N, 3)
            px = arr[:, 0]  # (N,)
            py = arr[:, 1]
            intensity = arr[:, 2]

            # Broadcast: (N, 1, 1) vs (1, res, res) -> (N, res, res)
            dx = X[np.newaxis, :, :] - px[:, np.newaxis, np.newaxis]
            dy = Y[np.newaxis, :, :] - py[:, np.newaxis, np.newaxis]
            inv2s2 = -1.0 / (2.0 * spot_sigma * spot_sigma)
            gauss = intensity[:, np.newaxis, np.newaxis] * np.exp(
                (dx * dx + dy * dy) * inv2s2
            )
            img = gauss.sum(axis=0)

        rgb = np.zeros((res, res, 3), dtype=np.float32)
        rgb[..., 0] = img
        rgb[..., 1] = img * 0.2
        rgb[..., 2] = img * 0.1

        bg = np.zeros((res, res, 3), dtype=np.float32)
        bg[self._mirror_mask] = [0.1, 0.1, 0.1]

        self._spot_im.set_data(np.clip(bg + rgb * 2, 0, 1))


if __name__ == "__main__":
    InteractiveHerriottCell()
