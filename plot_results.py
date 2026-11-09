"""
Draws the training plots from the CSV logs that train.py writes.

1. Training performance of one run: reward, episode length and total steps per episode.
2. One comparison chart per member: the best run in blue, the worst in orange, the rest in gray.

Run from the project root:
    uv run python plot_results.py
    uv run python plot_results.py --log models/merveille/experiment_3.csv
"""

import argparse
import csv
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MEMBERS = ["michael", "merveille", "ivan"]
MODELS_FOLDER = Path("models")
PLOTS_FOLDER = Path("plots")
MOVING_AVERAGE_WINDOW = 10

BEST_COLOR = "#2a78d6"
WORST_COLOR = "#eb6834"
CONTEXT_COLOR = "#c9c8c2"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e8e7e3"
BACKGROUND_COLOR = "#fcfcfb"


class TrainingLog(TypedDict):
    episodes: list[int]
    total_steps: list[int]
    episode_lengths: list[int]
    episode_rewards: list[float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw training plots from the CSV logs.")

    parser.add_argument("--log", type=str, default="dqn_model.csv")

    return parser.parse_args()


def read_log(log_path: Path) -> TrainingLog:
    log_file = open(log_path, newline="")

    rows = list(csv.DictReader(log_file))

    log_file.close()

    training_log: TrainingLog = {
        "episodes": [int(row["episode"]) for row in rows],

        "total_steps": [int(row["total_steps"]) for row in rows],

        "episode_lengths": [int(row["episode_length"]) for row in rows],

        "episode_rewards": [float(row["episode_reward"]) for row in rows],
    }

    return training_log


def moving_average(values: list[float], window: int) -> list[float]:
    averages = []

    for index in range(len(values)):
        start = max(0, index - window + 1)

        recent_values = values[start : index + 1]

        averages.append(sum(recent_values) / len(recent_values))

    return averages


def final_score(log_path: Path) -> float:
    rewards = read_log(log_path)["episode_rewards"]

    last_rewards = rewards[-MOVING_AVERAGE_WINDOW:]

    return sum(last_rewards) / len(last_rewards)


def style_axis(axis: plt.Axes, title: str, y_label: str) -> None:
    axis.set_title(title, loc="left", color=TEXT_COLOR, fontsize=11)

    axis.set_ylabel(y_label, color=TEXT_COLOR)

    axis.set_facecolor(BACKGROUND_COLOR)

    axis.grid(color=GRID_COLOR, linewidth=0.8)

    axis.tick_params(colors=TEXT_COLOR)

    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        axis.spines[side].set_color(GRID_COLOR)


def plot_training_performance(log_path: Path, output_path: Path) -> None:
    training_log = read_log(log_path)

    episodes = training_log["episodes"]

    rewards = training_log["episode_rewards"]

    average_rewards = moving_average(rewards, MOVING_AVERAGE_WINDOW)

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, facecolor=BACKGROUND_COLOR)

    reward_axis, length_axis, steps_axis = axes

    reward_axis.plot(episodes, rewards, color=CONTEXT_COLOR, linewidth=1)

    reward_axis.plot(episodes, average_rewards, color=BEST_COLOR, linewidth=2)

    reward_axis.annotate(
        f"{MOVING_AVERAGE_WINDOW}-episode average",
        xy=(episodes[-1], average_rewards[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=TEXT_COLOR,
        va="center",
    )

    style_axis(reward_axis, "Reward per episode (gray: each episode)", "Reward")

    length_axis.plot(episodes, training_log["episode_lengths"], color=BEST_COLOR, linewidth=1.5)

    style_axis(length_axis, "Episode length", "Steps in episode")

    steps_axis.plot(episodes, training_log["total_steps"], color=BEST_COLOR, linewidth=2)

    style_axis(steps_axis, "Total training steps", "Steps so far")

    steps_axis.ticklabel_format(axis="y", style="plain")

    steps_axis.set_xlabel("Episode", color=TEXT_COLOR)

    figure.suptitle(f"Training performance: {log_path}", x=0.02, ha="left", color=TEXT_COLOR)

    figure.tight_layout()

    figure.savefig(output_path, dpi=150)

    plt.close(figure)


def plot_member_comparison(member: str, output_path: Path) -> None:
    log_paths = sorted((MODELS_FOLDER / member).glob("experiment_*.csv"))

    best_log_path = max(log_paths, key=final_score)

    worst_log_path = min(log_paths, key=final_score)

    figure, axis = plt.subplots(figsize=(10, 5.5), facecolor=BACKGROUND_COLOR)

    for log_path in log_paths:
        training_log = read_log(log_path)

        average_rewards = moving_average(training_log["episode_rewards"], MOVING_AVERAGE_WINDOW)

        is_best = log_path == best_log_path

        is_worst = log_path == worst_log_path

        color = BEST_COLOR if is_best else WORST_COLOR if is_worst else CONTEXT_COLOR

        line_width = 2 if is_best or is_worst else 1

        axis.plot(training_log["total_steps"], average_rewards, color=color, linewidth=line_width, zorder=3 if is_best or is_worst else 2)

        if is_best or is_worst:
            label = f"{log_path.stem} ({'best' if is_best else 'worst'}: {final_score(log_path):.1f})"

            axis.annotate(
                label,
                xy=(training_log["total_steps"][-1], average_rewards[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=TEXT_COLOR,
                va="center",
            )

    style_axis(axis, f"{member.capitalize()}: {MOVING_AVERAGE_WINDOW}-episode average reward for all 10 experiments", "Average reward")

    axis.set_xlabel("Training steps", color=TEXT_COLOR)

    axis.set_ylim(-21.5, 0)

    figure.tight_layout()

    figure.savefig(output_path, dpi=150)

    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()

    PLOTS_FOLDER.mkdir(exist_ok=True)

    log_path = Path(arguments.log)

    performance_plot_path = PLOTS_FOLDER / "training_performance.png"

    plot_training_performance(log_path, performance_plot_path)

    print(f"Saved {performance_plot_path}")

    for member in MEMBERS:
        comparison_plot_path = PLOTS_FOLDER / f"{member}_experiments.png"

        plot_member_comparison(member, comparison_plot_path)

        print(f"Saved {comparison_plot_path}")


if __name__ == "__main__":
    main()
