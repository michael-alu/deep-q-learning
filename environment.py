import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)

GAME_ID = "ALE/Pong-v5"


# Shared by train.py and play.py so both always see the exact same game
def make_env(render_mode: str | None = None) -> gym.Env:
    env = gym.make(GAME_ID, render_mode=render_mode, frameskip=1)

    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        noop_max=30,
    )
    env = FrameStackObservation(env, stack_size=4)

    return env
