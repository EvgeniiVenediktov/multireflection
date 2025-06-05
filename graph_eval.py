import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

sample = "2025-06-03 15:10:24,583 - INFO - origin_x:4.0,origin_y:2.0,adj_n:6,pred_x:0.03757834434509277,pred_y:0.038370609283447266,pos_x:-0.02,pos_y:-0.12,sim_index:0.92"

fname = "eval.log"
if len(sys.argv) > 1:
    fname = sys.argv[1]
assert(fname[-4::] == ".log")

print(f"Reading {fname}")

x = []
y = []
adj = []

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

        x.append(origin_x)
        y.append(origin_y)
        adj.append(adj_n)

x = np.array(x)
y = np.array(y)
adj = np.array(adj)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.view_init(elev=16, azim=-130)

surf = ax.plot_trisurf(x, y, adj, cmap='viridis', edgecolor='none')

ax.xaxis._axinfo['grid'].update(color=(0, 0, 0, 0.5))
ax.yaxis._axinfo['grid'].update(color=(0, 0, 0, 0.5))
ax.zaxis._axinfo['grid'].update(color=(0, 0, 0, 0.5))
ax.xaxis.pane.set_alpha(0.5)
ax.yaxis.pane.set_alpha(0.5)
ax.zaxis.pane.set_alpha(0.5)

ax.set_zlim(top=8)

ax.set_xlabel("X origin")
ax.set_ylabel("Y origin")
ax.set_zlabel("Adjustments #")

cbar=plt.colorbar(surf)
# cbar.ax.set_visible(False)
fig.savefig(fname.split('.log')[0]+".png", dpi=300, transparent=True) #, labelpad=10

cbar.ax.set_visible(True)
plt.show()

        
