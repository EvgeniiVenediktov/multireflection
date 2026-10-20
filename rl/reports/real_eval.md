# Real-raster evaluation (eval_real.py)

Episodes on the real 572 x 400 tilt grid (1024 random starts, seed 0, argmax policy, max 64
steps); after each action the frame of the new tilt is taken from the dataset. Result: the
pattern the policy ends on. The real cell's maximum is 16 reflections on M2 (counts above
16 are counter artifacts of scatter blobs); frames of the bright band (mean grey > 40) are
left out of the counts.

## Baseline: r6 best (job 4423352, iteration 7630; clean sim renders only), eval job 4455742

Encoder trained without camera augmentation; the reference for what PPO on the new renders
adds (r7 / r8 below, same settings).

| axis map | start >= 16 sp | end >= 16 sp | end in bright band | stop rate | mean length |
|---|---|---|---|---|---|
| x,y | 27.8 % | 23.1 % | 19.7 % | 96.7 % | 6.2 |
| x,-y | 27.8 % | 19.3 % | 22.9 % | 97.9 % | 5.3 |
| -x,y | 27.8 % | 26.6 % | 20.2 % | 93.0 % | 8.9 |
| -x,-y | 27.8 % | 23.1 % | 20.1 % | 93.1 % | 8.7 |
| y,x | 27.8 % | 23.8 % | 20.8 % | 96.0 % | 6.4 |
| y,-x | 27.8 % | 29.3 % | 20.7 % | 81.8 % | 13.5 |
| -y,x | 27.8 % | 19.1 % | 20.3 % | 98.5 % | 4.9 |
| -y,-x | 27.8 % | 20.6 % | 22.6 % | 91.9 % | 9.1 |

Files: r6_it7630.json, r6_it7630_final_frames.png (start / end frames under map y,-x).

## r7b best (job 4456000, iteration 2650, 347M env steps; camera renders, frozen r6 policy), local eval

Frozen r6 policy + camera-trained encoder (converged supervised encoder of job 4455999, then
PPO with the policy frozen). Same starts as the baseline (start: 27.8 % of the episodes on
>= 16 spots, 15.5 % in the bright band).

| axis map | end >= 16 sp: r6 | r7b | end in bright band: r6 | r7b | improved count: r6 | r7b | stop rate: r6 | r7b | mean length: r6 | r7b |
|---|---|---|---|---|---|---|---|---|---|---|
| x,y | 23.1 % | 42.4 % | 19.7 % | 9.2 % | 19.6 % | 53.9 % | 96.7 % | 94.6 % | 6.2 | 16.0 |
| x,-y | 19.3 % | 43.8 % | 22.9 % | 7.3 % | 15.7 % | 57.8 % | 97.9 % | 89.8 % | 5.3 | 16.7 |
| -x,y | 26.6 % | 31.9 % | 20.2 % | 9.0 % | 17.6 % | 44.9 % | 93.0 % | 84.9 % | 8.9 | 21.3 |
| -x,-y | 23.1 % | 37.7 % | 20.1 % | 8.3 % | 16.0 % | 49.7 % | 93.1 % | 81.5 % | 8.7 | 20.5 |
| y,x | 23.8 % | 41.0 % | 20.8 % | 12.8 % | 20.5 % | 49.8 % | 96.0 % | 92.1 % | 6.4 | 18.2 |
| y,-x | 29.3 % | 44.5 % | 20.7 % | 5.5 % | 21.9 % | 45.4 % | 81.8 % | 77.3 % | 13.5 | 24.3 |
| -y,x | 19.1 % | 39.6 % | 20.3 % | 4.0 % | 14.7 % | 49.7 % | 98.5 % | 90.5 % | 4.9 | 14.1 |
| -y,-x | 20.6 % | 26.4 % | 22.6 % | 10.5 % | 13.2 % | 42.0 % | 91.9 % | 89.8 % | 9.1 | 19.9 |

Files: r7b_it2650.json, r7b_it2650_final_frames.png (start / end frames under map y,-x).
