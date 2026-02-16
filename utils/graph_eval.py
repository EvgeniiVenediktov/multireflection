import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import griddata
import sys
from datetime import datetime

fname = "eval.log"
threshold = 0.97

if len(sys.argv) > 1:
    fname = sys.argv[1]
if len(sys.argv) > 2:
    threshold = float(sys.argv[2])

assert fname[-4:] == ".log"

print(f"Reading {fname}, threshold={threshold}")

# Collect all data points, grouped by origin
origins = {}  # (origin_x, origin_y) -> [(timestamp, adj_n, sim_index, pos_x, pos_y), ...]

with open(fname, "r") as f:
    for s in f.readlines():
        timestamp_str = s.split(" - INFO - ")[0]
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        s = s.split(" - INFO - ")[1]
        vals = s.split(",")
        assert len(vals) == 8
        origin_x  = float(vals[0].split(':')[1]) 
        origin_y  = float(vals[1].split(':')[1]) 
        adj_n     = int(float(vals[2].split(':')[1]))
        pos_x     = float(vals[5].split(':')[1])
        pos_y     = float(vals[6].split(':')[1])
        sim_index = float(vals[7].split(':')[1])
        
        key = (origin_x, origin_y)
        if key not in origins:
            origins[key] = []
        origins[key].append((timestamp, adj_n, sim_index, pos_x, pos_y))

# Build arrays
x = []
y = []
adj = []
pos_errors = []  # Position error (distance from zero) at last adjustment
adj_times = []   # Time from first to last adjustment
end_ssims = []   # SSIM at last adjustment
end_angular_errors = []  # Angular error at last adjustment

for (origin_x, origin_y), history in origins.items():
    x.append(origin_x)
    y.append(origin_y)
    
    # Find first adj_n where sim_index >= threshold
    final_idx = len(history) - 1
    for i, (timestamp, adj_n, sim_index, pos_x, pos_y) in enumerate(history):
        if sim_index >= threshold:
            final_idx = i
            break
    
    final_adj = history[final_idx][1]
    final_pos_x = history[final_idx][3]
    final_pos_y = history[final_idx][4]
    final_ssim = history[final_idx][2]
    
    # Adjustment time (seconds)
    adj_time = (history[final_idx][0] - history[0][0]).total_seconds()
    adj_times.append(adj_time)
    
    adj.append(final_adj)
    end_ssims.append(final_ssim)
    
    # Calculate position error as Euclidean distance from (0, 0)
    pos_error = np.sqrt(final_pos_x**2 + final_pos_y**2)
    pos_errors.append(pos_error)
    
    # Angular error (assuming pos_x, pos_y are in degrees)
    angular_error = np.sqrt(final_pos_x**2 + final_pos_y**2)
    end_angular_errors.append(angular_error)

x = np.array(x)
y = np.array(y)
adj = np.array(adj, dtype=np.int32)
pos_errors = np.array(pos_errors)
adj_times = np.array(adj_times)
end_ssims = np.array(end_ssims)
end_angular_errors = np.array(end_angular_errors)

print(f"Adjustments: min={np.min(adj)}, max={np.max(adj)}, mean={np.mean(adj):.2f}, std={np.std(adj):.2f}, median={np.median(adj)}")
print(f"Position error from zero: min={np.min(pos_errors)}, max={np.max(pos_errors)}, mean={np.mean(pos_errors):.4f}, std={np.std(pos_errors):.4f}")
print(f"Adjustment time (s): min={np.min(adj_times)}, max={np.max(adj_times)}, mean={np.mean(adj_times):.2f}, std={np.std(adj_times):.2f}, median={np.median(adj_times):.2f}")
print(f"End SSIM: min={np.min(end_ssims)}, max={np.max(end_ssims)}, mean={np.mean(end_ssims):.4f}, std={np.std(end_ssims):.4f}, median={np.median(end_ssims):.4f}")
print(f"End angular error: min={np.min(end_angular_errors)}, max={np.max(end_angular_errors)}, mean={np.mean(end_angular_errors):.4f}, std={np.std(end_angular_errors):.4f}, median={np.median(end_angular_errors):.4f}")
# === 2D Heatmap ===
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((x, y), adj, (grid_x, grid_y), method='nearest')

plt.figure()
plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)),
           origin='lower', cmap='viridis', aspect='auto')
cbar = plt.colorbar(label="Number of Adjustments")
ticks = np.arange(np.min(adj), np.max(adj)+1, 1, dtype=np.int32)
cbar.set_ticks(ticks)

plt.xlabel("X origin (°)")
plt.ylabel("Y origin (°)")

plt.savefig(f"./graphs/eval/{fname.split('.log')[0]}_thresh{threshold}_heatmap.png", dpi=300, transparent=True)
plt.show()

# # === 2D Heatmap Without Interpolation ===
# # Create unique sorted coordinates
# x_unique = np.sort(np.unique(x))
# y_unique = np.sort(np.unique(y))

# # Create a 2D grid of shape (len(y), len(x)) filled with adj values
# adj_grid = np.full((len(y_unique), len(x_unique)), np.nan)
# for xi, yi, zi in zip(x, y, adj):
#     ix = np.where(x_unique == xi)[0][0]
#     iy = np.where(y_unique == yi)[0][0]
#     adj_grid[iy, ix] = zi

# plt.figure()
# plt.imshow(adj_grid, extent=(x_unique[0], x_unique[-1], y_unique[0], y_unique[-1]),
#            origin='lower', cmap='viridis', aspect='equal')

# cbar = plt.colorbar(label="Adjustments #")
# ticks = np.arange(np.floor(np.nanmin(adj)), np.ceil(np.nanmax(adj))).astype(int)
# cbar.set_ticks(ticks)

# plt.xlabel("X origin")
# plt.ylabel("Y origin")
# plt.title("Heatmap of Adjustments")

# # plt.savefig(fname.split('.log')[0] + "_heatmap.png", dpi=300, transparent=True)
# plt.show()



# # === 3D Plot ===
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# ax.view_init(elev=16, azim=-130)

# surf = ax.plot_trisurf(x, y, adj, cmap='viridis', edgecolor='none')

# ax.xaxis._axinfo['grid'].update(color=(0, 0, 0, 0.5))
# ax.yaxis._axinfo['grid'].update(color=(0, 0, 0, 0.5))
# ax.zaxis._axinfo['grid'].update(color=(0, 0, 0, 0.5))
# ax.xaxis.pane.set_alpha(0.5)
# ax.yaxis.pane.set_alpha(0.5)
# ax.zaxis.pane.set_alpha(0.5)

# ax.set_zlim(top=8)

# ax.set_xlabel("X origin")
# ax.set_ylabel("Y origin")
# ax.set_zlabel("Adjustments #")

# cbar = plt.colorbar(surf)
# fig.savefig("./graphs/eval/"+fname.split('.log')[0] + "_3d.png", dpi=300, transparent=True)

# cbar.ax.set_visible(True)
# plt.show()
