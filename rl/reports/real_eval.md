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
