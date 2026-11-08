"""
Trains a DQN agent to play Pong.

1. Fill the replay buffer by playing with a random policy at first.
2. Pick actions with epsilon-greedy, slowly exploring less over time.
3. Every few steps, learn from a random batch of past experiences.
4. Every few thousand steps, copy the online network into the target network.
5. Save the model and a CSV log of every episode.

Run from the project root:
    uv run python train.py
    uv run python train.py --learning-rate 0.0005 --gamma 0.95 --output models/michael/experiment_1.pth
"""

import argparse
import csv
import random
from collections import deque
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import nn

from environment import make_env

TOTAL_STEPS = 200_000
BUFFER_SIZE = 50_000
LEARNING_STARTS = 10_000
TRAIN_EVERY_STEPS = 4
TARGET_UPDATE_STEPS = 1_000

DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_GAMMA = 0.99
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.05
DEFAULT_EPSILON_DECAY_STEPS = 100_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transition(TypedDict):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Pong.")

    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)

    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument("--epsilon-start", type=float, default=DEFAULT_EPSILON_START)

    parser.add_argument("--epsilon-end", type=float, default=DEFAULT_EPSILON_END)

    parser.add_argument("--epsilon-decay-steps", type=int, default=DEFAULT_EPSILON_DECAY_STEPS)

    parser.add_argument("--total-steps", type=int, default=TOTAL_STEPS)

    parser.add_argument("--output", type=str, default="dqn_model.pth")

    return parser.parse_args()


def build_q_network(number_of_actions: int) -> nn.Sequential:
    q_network = nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),

        nn.ReLU(),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),

        nn.ReLU(),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),

        nn.ReLU(),

        nn.Flatten(),

        nn.Linear(in_features=64 * 7 * 7, out_features=512),

        nn.ReLU(),

        nn.Linear(in_features=512, out_features=number_of_actions),
    )

    return q_network


def to_tensor(observations: np.ndarray) -> torch.Tensor:
    pixels = torch.tensor(observations, dtype=torch.float32, device=DEVICE)

    scaled_pixels = pixels / 255.0

    return scaled_pixels


def get_epsilon(step: int, epsilon_start: float, epsilon_end: float, decay_steps: int) -> float:
    progress = min(step / decay_steps, 1.0)

    epsilon_drop = (epsilon_start - epsilon_end) * progress

    return epsilon_start - epsilon_drop


def choose_action(q_network: nn.Sequential, observation: np.ndarray, epsilon: float, number_of_actions: int) -> int:
    should_explore = random.random() < epsilon

    if should_explore:
        return random.randrange(number_of_actions)

    observation_batch = to_tensor(np.expand_dims(observation, axis=0))

    with torch.no_grad():
        q_values = q_network(observation_batch)

    best_action = int(q_values.argmax(dim=1).item())

    return best_action


def train_step(
    online_network: nn.Sequential,
    target_network: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    gamma: float,
) -> float:
    observations = to_tensor(np.array([transition["observation"] for transition in transitions]))

    next_observations = to_tensor(np.array([transition["next_observation"] for transition in transitions]))

    actions = torch.tensor([transition["action"] for transition in transitions], device=DEVICE)

    rewards = torch.tensor([transition["reward"] for transition in transitions], dtype=torch.float32, device=DEVICE)

    dones = torch.tensor([transition["done"] for transition in transitions], dtype=torch.float32, device=DEVICE)

    all_q_values = online_network(observations)

    predicted_q_values = all_q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        best_next_q_values = target_network(next_observations).max(dim=1).values

    future_value = gamma * best_next_q_values * (1 - dones)

    target_q_values = rewards + future_value

    loss = nn.functional.smooth_l1_loss(predicted_q_values, target_q_values)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()


def main() -> None:
    arguments = parse_arguments()

    model_path = Path(arguments.output)

    log_path = model_path.with_suffix(".csv")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env()

    number_of_actions = int(env.action_space.n)

    online_network = build_q_network(number_of_actions).to(DEVICE)

    target_network = build_q_network(number_of_actions).to(DEVICE)

    target_network.load_state_dict(online_network.state_dict())

    optimizer = torch.optim.Adam(online_network.parameters(), lr=arguments.learning_rate)

    replay_buffer: deque[Transition] = deque(maxlen=BUFFER_SIZE)

    log_file = open(log_path, "w", newline="")

    log_writer = csv.writer(log_file)

    log_writer.writerow(["episode", "total_steps", "episode_length", "episode_reward", "epsilon"])

    observation, info = env.reset()

    episode_number = 1

    episode_reward = 0.0

    episode_length = 0

    for step in range(1, arguments.total_steps + 1):
        epsilon = get_epsilon(step, arguments.epsilon_start, arguments.epsilon_end, arguments.epsilon_decay_steps)

        action = choose_action(online_network, observation, epsilon, number_of_actions)

        next_observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        replay_buffer.append(
            {
                "observation": observation,
                "action": action,
                "reward": float(reward),
                "next_observation": next_observation,
                "done": terminated,
            }
        )

        observation = next_observation

        episode_reward += float(reward)

        episode_length += 1

        is_learning_time = step > LEARNING_STARTS and step % TRAIN_EVERY_STEPS == 0

        if is_learning_time:
            transitions = random.sample(replay_buffer, arguments.batch_size)

            train_step(online_network, target_network, optimizer, transitions, arguments.gamma)

        if step % TARGET_UPDATE_STEPS == 0:
            target_network.load_state_dict(online_network.state_dict())

        if done:
            log_writer.writerow([episode_number, step, episode_length, episode_reward, round(epsilon, 3)])

            log_file.flush()

            print(f"Episode {episode_number} | step {step} | length {episode_length} | reward {episode_reward} | epsilon {epsilon:.3f}")

            observation, info = env.reset()

            episode_number += 1

            episode_reward = 0.0

            episode_length = 0

    log_file.close()

    env.close()

    torch.save(online_network.state_dict(), model_path)

    print(f"Saved model to {model_path} and log to {log_path}")


if __name__ == "__main__":
    main()
