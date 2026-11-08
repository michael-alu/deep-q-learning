# Watches a trained DQN agent play Pong in real time.
#
# Loads the saved weights into the same network that train.py builds, then plays with greedy
# actions only: no exploration and no learning. It only reads the model file and never changes it.
#
# Run from the project root:
#     uv run python play.py
#     uv run python play.py --model models/ivan/experiment_3.pth --episodes 3

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from environment import make_env
from train import DEVICE, build_q_network, to_tensor

default_model_path = "dqn_model.pth"
default_episodes = 3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Pong.")
    parser.add_argument("--model", default=default_model_path, help="Path to the saved .pth weights.")
    parser.add_argument("--episodes", type=int, default=default_episodes)
    return parser.parse_args()


# Rebuilds the training network and loads the weights. Exits with a clear message if the file is missing.
def load_q_network(model_path: Path, number_of_actions: int) -> nn.Sequential:
    if not model_path.exists():
        raise SystemExit(
            f"No model found at {model_path}. Train one first with `uv run python train.py`, "
            "or pass --model with the path to a saved .pth file."
        )

    q_network = build_q_network(number_of_actions).to(DEVICE)
    weights = torch.load(model_path, map_location=DEVICE)
    q_network.load_state_dict(weights)

    # Evaluation mode: the agent is only being watched, so nothing should behave like training.
    q_network.eval()
    return q_network


# The action with the highest predicted Q-value. Never random, unlike epsilon-greedy in training.
def greedy_action(q_network: nn.Sequential, observation: np.ndarray) -> int:
    observation_batch = to_tensor(np.expand_dims(observation, axis=0))
    with torch.no_grad():
        q_values = q_network(observation_batch)
    return int(q_values.argmax(dim=1).item())


# Plays one full game and returns the total reward and the number of steps.
def play_episode(env: gym.Env, q_network: nn.Sequential) -> tuple[float, int]:
    observation, info = env.reset()
    total_reward = 0.0
    steps = 0
    episode_over = False

    while not episode_over:
        action = greedy_action(q_network, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        episode_over = terminated or truncated

    return total_reward, steps


def main() -> None:
    arguments = parse_arguments()

    # Same make_env as training, so the agent sees frames exactly the way it learnt from them.
    env = make_env(render_mode="human")
    number_of_actions = int(env.action_space.n)
    q_network = load_q_network(Path(arguments.model), number_of_actions)

    episode_rewards = []
    for episode in range(1, arguments.episodes + 1):
        total_reward, steps = play_episode(env, q_network)
        episode_rewards.append(total_reward)
        print(f"Episode {episode}: reward {total_reward:+.0f}, {steps} steps")

    env.close()
    print(f"Average reward over {len(episode_rewards)} episodes: {np.mean(episode_rewards):+.2f}")


if __name__ == "__main__":
    main()
