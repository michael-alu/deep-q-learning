# Hyperparameter Experiments

**Member Name:** Ivan

**Focus:** exploration (epsilon start, epsilon end and epsilon decay)

Every experiment trained for 300,000 steps on Pong. Anything not listed in a row stayed at the baseline: learning rate 0.0001, gamma 0.99, batch size 32, epsilon from 1.0 to 0.05 over 100,000 steps. "Epsilon decay" is the number of steps it takes epsilon to drop from start to end, in a straight line. "Score" is the average reward over the last 10 episodes. A random agent scores -19.8 and the baseline (Michael's experiment 1) scored -15.6.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Score -15.8, best game -9. About the same as the baseline. Exploring for half as long did not hurt. |
| 2 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 200,000 | Score -18.8. Worse. The agent kept acting randomly for two thirds of training, so it had little time left to use what it learnt. |
| 3 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 300,000 | Score -19.0. Epsilon only reached 0.05 on the very last step, so the agent was partly random the whole time. Too much exploration. |
| 4 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Score -17.9, but it had a game at -6, one of my best. Ending with almost no randomness let it play its best moves more often. |
| 5 | 0.0001 | 0.99 | 32 | 1.0 | 0.1 | 100,000 | Worst run. Score -19.7. Keeping 10% random moves forever meant it often threw away points it could have won. |
| 6 | 0.0001 | 0.99 | 32 | 1.0 | 0.2 | 100,000 | Score -18.7. Also hurt by constant random moves. It scored a little better than 0.1, which is probably luck from a single run. |
| 7 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 100,000 | Score -19.3. Starting half random meant the early replay buffer was filled by an untrained network, so it saw less variety while learning. |
| 8 | 0.0001 | 0.99 | 32 | 0.5 | 0.05 | 50,000 | Score -18.2. Better than experiment 7 because the short decay gave it more time to exploit, but still behind starting at 1.0. |
| 9 | 0.0001 | 0.99 | 32 | 1.0 | 0.01 | 50,000 | Best run. Score -13.6, best game -8. Short exploration plus almost no randomness at the end gave the agent the most time to practise its best moves. |
| 10 | 0.0001 | 0.99 | 32 | 0.8 | 0.02 | 150,000 | Score -16.5. A middle setting, and it landed in the middle. |

## Summary

**Best:** experiment 9 (epsilon 1.0 to 0.01 over 50,000 steps), score -13.6.

**Worst:** experiment 5 (epsilon ending at 0.1), score -19.7.

**What helped and what hurt:**

- With only 300,000 steps, exploring for a short time worked best. Decays of 200,000 and 300,000 steps left too little time to use what the agent had learnt.
- A low end value helped. 0.01 beat 0.05, and 0.1 and 0.2 were both worse, because random moves late in training throw away points.
- Starting below 1.0 hurt. Full exploration at the start fills the replay buffer with a wide variety of situations, which the network needs to learn from.
- Each setting was only run once, so small differences (like 0.1 vs 0.2) may be luck. The trend across all ten runs is clear though.

**Final choice:** epsilon from 1.0 to 0.01 over 50,000 steps. This went into the final 1,000,000 step run, which reached a score of +2.6.
