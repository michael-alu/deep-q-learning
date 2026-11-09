# Hyperparameter Experiments

**Member Name:** Michael

**Focus:** learning rate and gamma

Every experiment trained for 300,000 steps on Pong. Anything not listed in a row stayed at the baseline: learning rate 0.0001, gamma 0.99, batch size 32, epsilon from 1.0 to 0.05 over 100,000 steps. "Score" is the average reward over the last 10 episodes. A random agent scores -19.8 and the worst possible score is -21.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Best run. Score -15.6, best single game -9. Stayed flat near -20 until about 120k steps, then climbed steadily. Episodes grew to about 1,900 steps, so the agent started returning the ball. |
| 2 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Score -20.7. No learning at all. Episodes shrank to about 820 steps, shorter than random play (940), so the agent got stuck on one bad habit. The updates were too big and kept overshooting. |
| 3 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Score -20.8. Same failure as experiment 2 but slightly worse. The highest learning rate I tried, and the agent never got past random play. |
| 4 | 0.00005 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Score -19.1. Learning was stable but slow. It was only starting to improve when training ended (episodes about 1,250 steps). It would likely need many more steps. |
| 5 | 0.00025 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Score -20.8. Even 2.5 times the baseline learning rate was enough to break learning completely. This is where the cliff is. |
| 6 | 0.0001 | 0.95 | 32 | 1.0 | 0.05 | 100,000 | Score -16.7, close to the baseline. Caring a bit less about the future did not hurt much in Pong, because each point is decided within a short time. |
| 7 | 0.0001 | 0.9 | 32 | 1.0 | 0.05 | 100,000 | Score -18.0. Worse than 0.95. The agent became too short-sighted to connect moving the paddle now with winning the point later. |
| 8 | 0.0001 | 0.999 | 32 | 1.0 | 0.05 | 100,000 | Score -18.4, but it had one game at -9. Caring about rewards very far away made the targets noisier and learning less steady. |
| 9 | 0.00025 | 0.95 | 32 | 1.0 | 0.05 | 100,000 | Score -20.8. A lower gamma could not rescue a learning rate that was too high. |
| 10 | 0.00025 | 0.995 | 32 | 1.0 | 0.05 | 100,000 | Worst run. Score -20.9, episodes about 790 steps. A high learning rate with a high gamma was the worst combination. |

## Summary

**Best:** experiment 1, the baseline (learning rate 0.0001, gamma 0.99), score -15.6.

**Worst:** experiment 10 (learning rate 0.00025, gamma 0.995), score -20.9.

**What helped and what hurt:**

- Learning rate mattered most. Every run at 0.00025 or higher failed completely, whatever gamma was set to. 0.0001 was the sweet spot, and 0.00005 was safe but too slow for 300,000 steps.
- Gamma mattered less. 0.95 and 0.99 were close, while 0.9 (too short-sighted) and 0.999 (too noisy) were both worse.
- Each setting was only run once, and luck can move a score by a couple of points. So I treat -15.6 and -16.7 as about the same, but the learning rate failures are clear.

**Final choice:** learning rate 0.0001 and gamma 0.99. Together with Merveille's batch size 128 and Ivan's short exploration, this went into the final 1,000,000 step run, which reached a score of +2.6 and won 16 of its last 20 games.
