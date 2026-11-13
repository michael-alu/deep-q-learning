import gymnasium as gym
import numpy as np

from environment import GAME_ID, make_env


def main() -> None:
    env = make_env()

    observation, info = env.reset()
    print(f"Game: {GAME_ID}")
    print(f"Observation shape: {observation.shape}")
    print(f"Number of actions: {env.action_space.n}")
    print()

    episode_rewards = []
    episode_lengths = []

    for episode in range(10):
        total_reward, steps = _play_one_episode(env)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: reward = {total_reward}, length = {steps} steps")

    env.close()

    print()
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.0f} steps")


def _play_one_episode(env: gym.Env) -> tuple[float, int]:
    env.reset()
    total_reward = 0.0
    steps = 0
    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        episode_over = terminated or truncated

    return total_reward, steps


if __name__ == "__main__":
    main()
