# Path probe: 4456000 iteration 2650

New head (log P, trained on the frozen core features) vs the existing path head (P / P_ref x true P_ref) on 102400 test frames (training: {'converged': True, 'epochs': 45, 'val_loss_std': 0.049859460443258286, 'lr_reductions': 3}).

| frames | n | new head: median rel. err | old head: median rel. err | new: mean rel. err | old: mean rel. err | new: MAE (P_ref units) | old: MAE (P_ref units) | constant: median rel. err |
|---|---|---|---|---|---|---|---|---|
| all | 102400 | 10.7 % | 35.7 % | 20.6 % | 54.2 % | 0.059 | 0.124 | 85.2 % |
| b0-2 | 30766 | 14.5 % | 45.8 % | 30.3 % | 78.7 % | 0.010 | 0.017 | 367.5 % |
| b3-30 | 49208 | 9.1 % | 39.3 % | 16.1 % | 50.5 % | 0.051 | 0.156 | 58.6 % |
| b31-31 | 1589 | 8.9 % | 30.6 % | 14.1 % | 39.2 % | 0.109 | 0.255 | 73.8 % |
| b32-56 | 13308 | 8.8 % | 23.8 % | 14.6 % | 32.2 % | 0.111 | 0.216 | 82.1 % |
| b57-57 | 927 | 10.3 % | 18.7 % | 16.4 % | 24.3 % | 0.132 | 0.176 | 87.4 % |
| b58-88 | 3941 | 12.6 % | 14.7 % | 18.4 % | 21.1 % | 0.148 | 0.155 | 89.6 % |
| b89+ | 2661 | 25.4 % | 11.2 % | 31.8 % | 16.0 % | 0.305 | 0.152 | 93.9 % |
