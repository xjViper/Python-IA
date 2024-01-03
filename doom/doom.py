import vizdoom as doom
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import cv2


class VizDoomGymnasium(Env):
    def __init__(self, render=False, config="./scenarios/deadly_corridor_s1.cfg"):
        super().__init__()
        self.game = doom.DoomGame()
        # Change the Load Config to use the AI in others Levels
        # Search in this repository the avaliable Levels
        # https://github.com/Farama-Foundation/ViZDoom/tree/master/scenarios
        self.game.load_config(config)

        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(3, 240, 320),
            dtype=np.uint8,
        )

        # Number of Available buttons in Config file
        self.action_space = Discrete(7)

        # Game Variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.hitcount = 0
        self.selected_weapon_ammo = 52

    def step(self, action):
        # Number of Available buttons in Config file
        actions = np.identity(7, dtype=np.uint8)
        movement_reward = self.game.make_action(actions[action], 4)

        reward = 0

        if self.game.get_state() is not None:
            state = self.game.get_state()
            observation = self.grayscale(state)

            # Reward Shaping
            game_variables = state.game_variables
            health, damage_taken, hitcount, ammo = game_variables

            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.selected_weapon_ammo
            self.selected_weapon_ammo = ammo

            reward = (
                movement_reward
                + damage_taken_delta * 10
                + hitcount_delta * 200
                + ammo_delta * 5
            )

        else:
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
            return np.zeros(self.observation_space.shape), {}

    def grayscale(self, observation):
        transpose = observation.screen_buffer.transpose(1, 2, 0)
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


# Change the diretory to each Config File
CHECKPOINT_DIR = "./train/"
LOG_DIR = "./logs/"


callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Add here the render and config variables
env = VizDoomGymnasium(render=True)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=0.0001,
    n_steps=4096,
)
model.learn(total_timesteps=20000, callback=callback)
