# Frame data timing (NVIDIA L40S)

median of 10 calls after 3 warm-up calls; peak = memory above the start of the timed calls; first = first call (compile included)

| entry | B | chunk | ms | us / frame | peak MiB | first ms |
|---|---|---|---|---|---|---|
| sim store_hits=on | 256 |  | 0.26 | 1.01 | 0 | 0 |
| sim store_hits=off | 256 |  | 0.25 | 0.98 | 0 | 567 |
| current: sim + targets | 256 |  | 2.62 | 10.22 | 17 | 329 |
| current: RandomRenderer 1ch | 256 |  | 0.67 | 2.61 | 33 | 7140 |
| camera: sample_params | 256 |  | 2.27 | 8.86 | 17 | 811 |
| camera: labels | 256 |  | 0.26 | 1.00 | 1 | 1 |
| camera: map target (1 px spots) | 256 |  | 5.38 | 21.03 | 524 | 6 |
| camera: render 3ch (eager) | 256 | 128 | 8.38 | 32.75 | 274 | 54 |
| camera: render 3ch (compiled) | 256 | 128 | 3.22 | 12.57 | 142 | 7644 |
| sim store_hits=on | 1024 |  | 0.26 | 0.25 | 1 | 0 |
| sim store_hits=off | 1024 |  | 0.25 | 0.25 | 0 | 0 |
| current: sim + targets | 1024 |  | 4.65 | 4.55 | 66 | 7 |
| current: RandomRenderer 1ch | 1024 |  | 1.75 | 1.71 | 128 | 1500 |
| camera: sample_params | 1024 |  | 3.20 | 3.12 | 64 | 4 |
| camera: labels | 1024 |  | 0.25 | 0.25 | 3 | 1 |
| camera: map target (1 px spots) | 1024 |  | 20.08 | 19.61 | 2097 | 20 |
| camera: render 3ch (eager) | 1024 | 128 | 31.56 | 30.82 | 287 | 32 |
| camera: render 3ch (compiled) | 1024 | 128 | 12.57 | 12.28 | 155 | 13 |
| camera: render 3ch (eager) | 1024 | 512 | 35.12 | 34.30 | 1095 | 35 |
| camera: render 3ch (compiled) | 1024 | 512 | 6.70 | 6.55 | 567 | 7204 |
| sim store_hits=on | 4096 |  | 0.26 | 0.06 | 5 | 0 |
| sim store_hits=off | 4096 |  | 0.25 | 0.06 | 0 | 0 |
| current: sim + targets | 4096 |  | 13.34 | 3.26 | 265 | 24 |
| current: RandomRenderer 1ch | 4096 |  | 8.10 | 1.98 | 512 | 8 |
| camera: sample_params | 4096 |  | 9.17 | 2.24 | 257 | 10 |
| camera: labels | 4096 |  | 0.27 | 0.07 | 11 | 1 |
| camera: map target (1 px spots) | 4096 |  | 79.38 | 19.38 | 8388 | 80 |
| camera: render 3ch (eager) | 4096 | 128 | 123.45 | 30.14 | 335 | 124 |
| camera: render 3ch (compiled) | 4096 | 128 | 50.74 | 12.39 | 203 | 51 |
| camera: render 3ch (eager) | 4096 | 512 | 139.32 | 34.01 | 1131 | 140 |
| camera: render 3ch (compiled) | 4096 | 512 | 26.05 | 6.36 | 603 | 26 |
| camera: render 3ch (eager) | 4096 | 2048 | 165.19 | 40.33 | 4378 | 165 |
| camera: render 3ch (compiled) | 4096 | 2048 | 38.09 | 9.30 | 2266 | 7675 |
| encoder train step (eager, bf16) | 256 |  | 27.58 | 107.73 | 1554 | 1747 |
| encoder train step (eager, bf16) | 1024 |  | 139.05 | 135.79 | 6196 | 140 |
