| model | seeds | encoder params | train min | fwd ms (bf16 eager / fp32 / compiled) |
|---|---|---|---|---|
| cnn-pre4 | 1 | 1.85 M | 12.0 | - |

**in-distribution** (n = 4096)

| model | precision | recall | loc err (mm) | count exact | count +-1 | ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |
|---|---|---|---|---|---|---|---|---|
| cnn-pre4 | 0.028 | 0.078 | 0.321 | 0.187 | 0.684 | 0.967 | 0.931 / 0.842 | 11.35 |

**clean** (n = 4096)

| model | precision | recall | loc err (mm) | count exact | count +-1 | ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |
|---|---|---|---|---|---|---|---|---|
| cnn-pre4 | 0.025 | 0.076 | 0.318 | 0.192 | 0.310 | 0.959 | 1.029 / 0.830 | 12.35 |

**OOD** (n = 4096)

| model | precision | recall | loc err (mm) | count exact | count +-1 | ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |
|---|---|---|---|---|---|---|---|---|
| cnn-pre4 | 0.028 | 0.077 | 0.317 | 0.183 | 0.675 | 0.966 | 0.930 / 0.846 | 11.87 |

Detector on the target map (ceiling): precision 1.000, recall 0.754, loc err 0.022 mm.

**OOD by held-out factor** (recall / count exact)

| model | sigma | rotation | read_noise | gain_low | gain_high |
|---|---|---|---|---|---|
| cnn-pre4 | 0.077 / 0.162 | 0.063 / 0.192 | 0.084 / 0.165 | 0.088 / 0.189 | 0.073 / 0.208 |

**In-distribution by region** (recall / count exact)

| model | blind (n=2301) | 31 (n=26) | 57 (n=19) | 89 (n=2) | trapped (n=27) | other (n=1721) |
|---|---|---|---|---|---|---|
| cnn-pre4 | 0.031 / 0.170 | 0.084 / 0.154 | 0.123 / 0.053 | 0.011 / 0.000 | 0.045 / 0.000 | 0.082 / 0.214 |

**89 vs trapped probe set** (1024 each; precision / recall / count exact; detector ceiling recall 89 0.644, trapped 0.453)

| model | set | 89 | trapped |
|---|---|---|---|
| cnn-pre4 | clean | 0.960 / 0.567 / 1.000 | 0.991 / 0.458 / 1.000 |
| cnn-pre4 | id | 0.984 / 0.589 / 0.978 | 0.994 / 0.460 / 0.897 |
| cnn-pre4 | ood | 0.978 / 0.598 / 0.731 | 0.992 / 0.460 / 0.715 |

**Linear probes on z** (test half; tilt on >= 3 bounces)

| model | set | tilt R^2 | tilt RMSE (deg) | bounces R^2 | bounces MAE | 89 vs trapped acc |
|---|---|---|---|---|---|---|
| cnn-pre4 | clean | 0.723 | 0.510 | 0.683 | 4.32 | 1.000 |
| cnn-pre4 | id | 0.695 | 0.537 | 0.636 | 5.78 | 0.999 |
| cnn-pre4 | ood | 0.680 | 0.550 | 0.625 | 5.74 | 0.993 |
| pixels (32 x 32) | clean | 0.484 | 0.695 | 0.799 | 4.08 | 1.000 |
| pixels (32 x 32) | id | 0.411 | 0.741 | 0.623 | 5.99 | 0.939 |
| pixels (32 x 32) | ood | 0.401 | 0.750 | 0.474 | 6.72 | 0.854 |

**Linear scenario probe on z** (ridge, test half; R^2 / RMSE, all frames and >= 3 bounces; separation mm, laser pitch / yaw deg)

| model | set | separation | laser pitch | laser yaw | separation >= 3 | pitch >= 3 | yaw >= 3 |
|---|---|---|---|---|---|---|---|
| cnn-pre4 | clean | 0.249 / 38.28 | 0.332 / 1.92 | 0.713 / 1.29 | 0.292 / 35.22 | 0.640 / 1.29 | 0.805 / 0.76 |
| cnn-pre4 | id | 0.226 / 38.88 | 0.310 / 1.95 | 0.685 / 1.35 | 0.269 / 35.78 | 0.614 / 1.33 | 0.783 / 0.80 |
| cnn-pre4 | ood | 0.212 / 39.21 | 0.279 / 1.99 | 0.679 / 1.36 | 0.207 / 37.26 | 0.586 / 1.38 | 0.761 / 0.84 |
| pixels (32 x 32) | clean | 0.075 / 42.50 | 0.213 / 2.08 | 0.293 / 2.02 | 0.004 / 41.77 | 0.462 / 1.57 | 0.463 / 1.26 |
| pixels (32 x 32) | id | 0.097 / 41.99 | 0.196 / 2.10 | 0.289 / 2.03 | 0.062 / 40.54 | 0.435 / 1.61 | 0.451 / 1.27 |
| pixels (32 x 32) | ood | 0.062 / 42.79 | 0.178 / 2.13 | 0.268 / 2.06 | 0.022 / 41.38 | 0.403 / 1.66 | 0.411 / 1.32 |
