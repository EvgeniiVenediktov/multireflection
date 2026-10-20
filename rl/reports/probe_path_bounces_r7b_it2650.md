# Path probe: 4456000 iteration 2650

New heads on the frozen core features vs the existing path head (P / P_ref x true P_ref) on 102400 test frames (409600 train). Training: {'logP': {'converged': True, 'epochs': 45, 'val_loss': 0.049859460443258286, 'lr_reductions': 3}, 'logP+bounces': {'converged': True, 'epochs': 77, 'val_loss': 0.14544802904129028, 'lr_reductions': 3}}.

### Path (relative error of the absolute path; MAE in P_ref units)

| frames | n | logP: median | logP+bounces: median | old head: median | logP: mean | logP+bounces: mean | old head: mean | logP: MAE P_ref | logP+bounces: MAE P_ref | old head: MAE P_ref | constant: median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 102400 | 10.7 % | 10.9 % | 35.7 % | 20.6 % | 20.9 % | 54.2 % | 0.059 | 0.060 | 0.124 | 85.2 % |
| b0-2 | 30766 | 14.5 % | 14.4 % | 45.8 % | 30.3 % | 30.5 % | 78.7 % | 0.010 | 0.010 | 0.017 | 367.5 % |
| b3-30 | 49208 | 9.1 % | 9.4 % | 39.3 % | 16.1 % | 16.3 % | 50.5 % | 0.051 | 0.053 | 0.156 | 58.6 % |
| b31-31 | 1589 | 8.9 % | 9.5 % | 30.6 % | 14.1 % | 14.3 % | 39.2 % | 0.109 | 0.110 | 0.255 | 73.8 % |
| b32-56 | 13308 | 8.8 % | 9.4 % | 23.8 % | 14.6 % | 15.2 % | 32.2 % | 0.111 | 0.116 | 0.216 | 82.1 % |
| b57-57 | 927 | 10.3 % | 11.0 % | 18.7 % | 16.4 % | 17.2 % | 24.3 % | 0.132 | 0.138 | 0.176 | 87.4 % |
| b58-88 | 3941 | 12.6 % | 12.5 % | 14.7 % | 18.4 % | 18.2 % | 21.1 % | 0.148 | 0.147 | 0.155 | 89.6 % |
| b89+ | 2661 | 25.4 % | 25.7 % | 11.2 % | 31.8 % | 31.6 % | 16.0 % | 0.305 | 0.303 | 0.152 | 93.9 % |

### Bounce count (logP+bounces head)

| frames | exact | within 2 | MAE (bounces) |
|---|---|---|---|
| all | 70.7 % | 83.1 % | 2.70 |
| b0-2 | 90.8 % | 99.6 % | 0.12 |
| b3-30 | 75.7 % | 88.7 % | 1.01 |
| b31-31 | 43.4 % | 76.2 % | 3.03 |
| b32-56 | 35.9 % | 55.4 % | 4.95 |
| b57-57 | 31.7 % | 37.5 % | 9.56 |
| b58-88 | 14.1 % | 23.5 % | 17.81 |
| b89+ | 34.2 % | 35.1 % | 27.86 |
