from stable_baselines3 import PPO

model = PPO.load("./train/best_model_10000.zip")

import gymnasium

env = gymnasium.make("ALE/VideoCube-v5", render_mode="human")

from gymnasium.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

state = env.reset()

while True:
    action, _state = model.predict(state)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
