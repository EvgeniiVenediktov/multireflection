import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import griddata
import sys
import matplotlib.ticker as ticker  # add to your imports


sample = "2025-06-03 15:10:24,583 - INFO - origin_x:4.0,origin_y:2.0,adj_n:6,pred_x:0.03757834434509277,pred_y:0.038370609283447266,pos_x:-0.02,pos_y:-0.12,sim_index:0.92"

fname = "eval.log"
if len(sys.argv) > 1:
    fname = sys.argv[1]
assert(fname[-4::] == ".log")

print(f"Reading {fname}")

x = []
y = []
adj = []

lx = -5
ly = -5
la = -1
with open(fname, "r") as f:
    for s in f.readlines():
        s = s.split(" - INFO - ")[1]
        vals = s.split(",")
        assert(len(vals) == 8)
        origin_x  = float(vals[0].split(':')[1]) 
        origin_y  = float(vals[1].split(':')[1]) 
        adj_n     = float(vals[2].split(':')[1])    
        pred_x    = float(vals[3].split(':')[1])    
        pred_y    = float(vals[4].split(':')[1])   
        pos_x     = float(vals[5].split(':')[1])     
        pos_y     = float(vals[6].split(':')[1])  
        sim_index = float(vals[7].split(':')[1]) 

        if origin_x == lx and origin_y == ly:
            la = adj_n
        else:
            x.append(origin_x)
            y.append(origin_y)
            lx = origin_x
            ly = origin_y
            if la == -1:
                la = adj_n
            adj.append(la)
            la = adj_n
    x.append(origin_x)
    y.append(origin_y)
    adj.append(la)

x = np.array(x)
y = np.array(y)
adj = np.array(adj, dtype=np.int32) # + 1

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

# === 2D Heatmap ===
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
# grid_z = griddata((x, y), adj, (grid_x, grid_y), method='linear')
grid_z = griddata((x, y), adj, (grid_x, grid_y), method='nearest')
# grid_z = griddata((x, y), adj, (grid_x, grid_y), method='cubic')

plt.figure()
plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)),
           origin='lower', cmap='viridis', aspect='auto')
# print("max", np.max(adj))
cbar = plt.colorbar(label="Number of Adjustments")
ticks = np.arange(np.min(adj), np.max(adj)+1, 1, dtype=np.int32)
cbar.set_ticks(ticks)

plt.xlabel("X origin (°)")
plt.ylabel("Y origin (°)")
# plt.title("Heatmap of Adjustments")

plt.savefig("./graphs/eval/kkk"+fname.split('.log')[0] + "_heatmap.png", dpi=300, transparent=True)
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