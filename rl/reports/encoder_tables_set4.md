| model | seeds | encoder params | train min | fwd ms (bf16 eager / fp32 / compiled) |
|---|---|---|---|---|
| cnn | 1 | 1.85 M | 13.4 | 66.1 / 33.4 / 12.0 |
| cnn-pre4 | 1 | 1.85 M | 12.0 | 30.8 / 33.6 / 27.4 |

**in-distribution** (n = 8192)

| model | precision | recall | loc err (mm) | count exact | count +-1 | ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |
|---|---|---|---|---|---|---|---|---|
| cnn | 0.372 | 0.322 | 0.104 | 0.425 | 0.597 | 0.649 | 0.558 / 1.690 | 25.96 |
| cnn-pre4 | 0.977 | 0.825 | 0.106 | 0.999 | 0.999 | 0.045 | 0.047 / 0.035 | 0.47 |

**clean** (n = 8192)

| model | precision | recall | loc err (mm) | count exact | count +-1 | ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |
|---|---|---|---|---|---|---|---|---|
| cnn | 0.379 | 0.314 | 0.173 | 0.435 | 0.605 | 0.710 | 0.554 / 1.826 | 25.81 |
| cnn-pre4 | 0.965 | 0.821 | 0.152 | 1.000 | 1.000 | 0.069 | 0.092 / 0.059 | 0.72 |

**OOD** (n = 8192)

| model | precision | recall | loc err (mm) | count exact | count +-1 | ell centre (mm) | ell a / b MAE (mm) | ell angle (deg) |
|---|---|---|---|---|---|---|---|---|
| cnn | 0.374 | 0.323 | 0.123 | 0.423 | 0.591 | 0.653 | 0.563 / 1.700 | 25.97 |
| cnn-pre4 | 0.965 | 0.818 | 0.128 | 0.981 | 0.983 | 0.055 | 0.057 / 0.053 | 0.58 |

Detector on the target map (ceiling): precision 1.000, recall 0.847, loc err 0.018 mm.

**OOD by held-out factor** (recall / count exact)

| model | sigma | rotation | read_noise | gain_low | gain_high |
|---|---|---|---|---|---|
| cnn | 0.333 / 0.416 | 0.311 / 0.401 | 0.303 / 0.410 | 0.339 / 0.443 | 0.328 / 0.444 |
| cnn-pre4 | 0.812 / 0.917 | 0.827 / 0.998 | 0.801 / 0.996 | 0.829 / 0.999 | 0.822 / 0.995 |

**In-distribution by region** (recall / count exact)

| model | blind (n=1861) | 31 (n=775) | 57 (n=381) | 89 (n=3) | trapped (n=21) | other (n=5151) |
|---|---|---|---|---|---|---|
| cnn | 0.228 / 0.718 | 0.970 / 0.996 | 0.867 / 1.000 | 0.593 / 1.000 | 0.039 / 0.143 | 0.115 / 0.192 |
| cnn-pre4 | 1.000 / 1.000 | 0.972 / 1.000 | 0.852 / 1.000 | 0.541 / 1.000 | 0.342 / 0.952 | 0.798 / 0.998 |

**89 vs trapped probe set** (1024 each; precision / recall / count exact; detector ceiling recall 89 0.644, trapped 0.453)

| model | set | 89 | trapped |
|---|---|---|---|
| cnn | clean | 0.981 / 0.645 / 1.000 | 0.995 / 0.460 / 0.999 |
| cnn | id | 0.983 / 0.650 / 0.990 | 0.994 / 0.461 / 0.955 |
| cnn | ood | 0.983 / 0.649 / 0.829 | 0.995 / 0.461 / 0.827 |
| cnn-pre4 | clean | 0.962 / 0.568 / 1.000 | 0.992 / 0.457 / 1.000 |
| cnn-pre4 | id | 0.983 / 0.590 / 0.981 | 0.995 / 0.459 / 0.899 |
| cnn-pre4 | ood | 0.978 / 0.598 / 0.768 | 0.993 / 0.459 / 0.732 |

**Linear probes on z** (test half; tilt on >= 3 bounces)

| model | set | tilt R^2 | tilt RMSE (deg) | bounces R^2 | bounces MAE | 89 vs trapped acc |
|---|---|---|---|---|---|---|
| cnn | clean | 0.977 | 0.128 | 0.869 | 4.66 | 1.000 |
| cnn | id | 0.940 | 0.209 | 0.794 | 6.42 | 1.000 |
| cnn | ood | 0.930 | 0.226 | 0.774 | 6.68 | 0.993 |
| cnn-pre4 | clean | 0.996 | 0.056 | 0.984 | 1.74 | 1.000 |
| cnn-pre4 | id | 0.989 | 0.093 | 0.962 | 2.79 | 0.998 |
| cnn-pre4 | ood | 0.983 | 0.113 | 0.930 | 3.40 | 0.985 |
| pixels (32 x 32) | clean | 0.967 | 0.154 | 0.967 | 2.20 | 1.000 |
| pixels (32 x 32) | id | 0.680 | 0.486 | 0.702 | 8.25 | 0.933 |
| pixels (32 x 32) | ood | 0.621 | 0.528 | 0.600 | 9.71 | 0.855 |

**Linear setup probe on z** (4-way, test half; all frames / >= 3 bounces / blind)

| model | set | all | >= 3 bounces | blind |
|---|---|---|---|---|
| cnn | clean | 1.000 | 0.999 | 1.000 |
| cnn | id | 0.999 | 0.999 | 0.999 |
| cnn | ood | 0.997 | 1.000 | 0.989 |
| cnn-pre4 | clean | 1.000 | 1.000 | 1.000 |
| cnn-pre4 | id | 1.000 | 1.000 | 1.000 |
| cnn-pre4 | ood | 1.000 | 1.000 | 0.999 |
| pixels (32 x 32) | clean | 1.000 | 1.000 | 1.000 |
| pixels (32 x 32) | id | 0.972 | 0.971 | 0.977 |
| pixels (32 x 32) | ood | 0.951 | 0.950 | 0.951 |
