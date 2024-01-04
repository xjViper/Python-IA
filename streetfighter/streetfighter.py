import retro
from gymnasium import Env
from gymnasium.spaces import MultiBinary, Box
import numpy as np
import cv2


class StreetFigter(Env):
    def __init__(self):
        super().__init__()

        self.game = retro.make(
            game="StreetFighterIISpecialChampionEdition-Genesis",
            use_restricted_actions=retro.Actions.FILTERED,
        )

        # shape=(200, 256, 3)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self.game.set_seed(seed)
        obs = self.game.reset()
        obs = self.preprocess(obs)

        self.previous_frame = obs

        self.score = 0

        return obs, {}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.game.step(action)
        observation = self.preprocess(observation)

        frame_delta = observation - self.previous_frame
        self.previous_frame = observation

        reward = info["score"] - self.score
        self.score = info["score"]

        return frame_delta, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        self.game.render()

    def preprocess(self, observation):
        if observation is None:
            return np.zeros((84, 84, 1), dtype=np.uint8)

        if isinstance(observation, tuple):
            observation = np.array(observation[0])

        observation = np.uint8(observation)
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))

        return channels

    def close(self):
        self.game.close()


import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

LOG_DIR = "./logs/"
OPT_DIR = "./opt/"
CHECKPOINT_DIR = "./train/"


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)

        return True


callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = StreetFigter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")


model = PPO("CnnPolicy", env, tensorboard_log=LOG_DIR, verbose=0)

model.load(os.path.join(OPT_DIR, "trial_6_best_model.zip"))

model.learn(total_timesteps=20000, callback=callback)
