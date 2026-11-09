# Hyperparameter Experiments

**Member Name:** Merveille

**Focus:** batch size, and how it works together with learning rate

Every experiment trained for 300,000 steps on Pong. Anything not listed in a row stayed at the baseline: learning rate 0.0001, gamma 0.99, batch size 32, epsilon from 1.0 to 0.05 over 100,000 steps. "Score" is the average reward over the last 10 episodes. A random agent scores -19.8 and the baseline (Michael's experiment 1) scored -15.6.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 16 | 1.0 | 0.05 | 100,000 | Score -20.6. Barely learnt. With only 16 samples per update, each update was too noisy to point in a useful direction. |
| 2 | 0.0001 | 0.99 | 64 | 1.0 | 0.05 | 100,000 | Score -17.9, with a best 10 game stretch of -14.8. Clearly learning, and episodes grew to about 2,100 steps. |
| 3 | 0.0001 | 0.99 | 128 | 1.0 | 0.05 | 100,000 | Best run of the whole group. Score -9.4, best game -3. Flat until about 80 episodes, then climbed all the way to the end. Episodes about 2,800 steps, so the agent was returning the ball a lot. |
| 4 | 0.0001 | 0.99 | 256 | 1.0 | 0.05 | 100,000 | Score -11.1, best game -4. Second best in the group. Slightly behind 128, probably because each update takes longer, so fewer updates fit in the same time. |
| 5 | 0.00005 | 0.99 | 16 | 1.0 | 0.05 | 100,000 | Worst run (tied). Score -21.0. Small noisy batches plus tiny learning steps meant almost nothing was learnt. |
| 6 | 0.00005 | 0.99 | 64 | 1.0 | 0.05 | 100,000 | Score -17.3. Still learning, but slower than experiment 2 because of the smaller learning rate. |
| 7 | 0.00025 | 0.99 | 64 | 1.0 | 0.05 | 100,000 | Score -20.8. A bigger learning rate broke learning, the same thing Michael found. Episodes shrank to about 820 steps. |
| 8 | 0.00025 | 0.99 | 128 | 1.0 | 0.05 | 100,000 | Score -20.7. Even the best batch size could not save a learning rate that was too high. |
| 9 | 0.0005 | 0.99 | 128 | 1.0 | 0.05 | 100,000 | Score -20.9. Same failure as experiment 8, a little worse. |
| 10 | 0.0005 | 0.99 | 256 | 1.0 | 0.05 | 100,000 | Worst run (tied). Score -21.0. The largest batch with a high learning rate still failed. |

## Summary

**Best:** experiment 3 (batch size 128, learning rate 0.0001), score -9.4. This was the best of all 30 experiments.

**Worst:** experiments 5 and 10, both -21.0. One was too small and too slow (batch 16, learning rate 0.00005), the other too fast (batch 256, learning rate 0.0005).

**What helped and what hurt:**

- At learning rate 0.0001, a bigger batch helped a lot: 16 scored -20.6, 32 scored -15.6, 64 scored -17.9, 128 scored -9.4 and 256 scored -11.1. A bigger batch averages over more past experiences, so each update is steadier.
- Past 128 the gains stopped. 256 was a bit worse, so 128 looks like the sweet spot.
- A big batch could not fix a learning rate that was too high. Every run at 0.00025 or 0.0005 failed, whatever the batch size.
- Each setting was only run once, so small differences may be luck. The gap between batch 16 and batch 128 is far too big to be luck.

**Final choice:** batch size 128 with learning rate 0.0001. This went into the final 1,000,000 step run, which reached a score of +2.6.
