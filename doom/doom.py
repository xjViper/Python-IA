import vizdoom as doom
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2


class VizDoomGymnasium(Env):
    def __init__(self, render=False):
        super().__init__()
        self.game = doom.DoomGame()
        self.game.load_config("./scenarios/basic.cfg")

        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()

        self.observation_space = Box(
            low=0,
            high=255,
            # shape=(self.game.get_screen_height(), self.game.get_screen_width(), 3),
            shape=(3, 240, 320),
            dtype=np.uint8,
        )
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state() is not None:
            state = self.game.get_state()
            observation = self.grayscale(state)
            # observation = state.screen_buffer.transpose(
            #     1, 2, 0
            # )  # Transpose para (H, W, C)
            # observation = cv2.resize(
            #     observation, (320, 240), interpolation=cv2.INTER_CUBIC
            # )
            # observation = observation.transpose(2, 0, 1)

        else:
            # state = np.zeros(self.observation_space.shape)
            observation = np.zeros(self.observation_space.shape)

        terminated = self.game.is_episode_finished()
        truncated = False

        return observation, reward, terminated, truncated, {}

    def render():
        pass

    def reset(self, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            return state.screen_buffer, {}
        else:
            # Return a valid state if screen_buffer is None
            return np.zeros(self.observation_space.shape), {}

    def grayscale(self, observation):
        transpose = observation.screen_buffer.transpose(
            1, 2, 0
        )  # Transpose para (H, W, C)
        resize = cv2.resize(transpose, (320, 240), interpolation=cv2.INTER_CUBIC)
        state = resize.transpose(2, 0, 1)
        return state

    def close(self):
        self.game.close()


import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


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


CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"


callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGymnasium(render=True)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.0001,
    n_steps=256,
)
model.learn(total_timesteps=10000, callback=callback)
