import gymnasium

env = gymnasium.make("ALE/VideoCube-v5", render_mode="human")

from gymnasium.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order="last")

state = env.reset()
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

# plt.figure(figsize=(10, 8))
# for idx in range(state.shape[3]):
#     plt.subplot(1, 4, idx=1)
#     plt.imshow(state[0][:, :, idx])
# plt.show()

plt.imshow(state)
