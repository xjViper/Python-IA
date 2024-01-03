from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from doom import doom
import time

model = PPO.load("./train/best_model_10000.zip")

env = doom.VizDoomGymnasium(render=True)

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

for episode in range(5):
    total_reward = 0
    state = env.reset()
    terminated = False
    while not terminated:
        action, _state = model.predict(state)
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.05)
        total_reward += reward
    print("Total Reward for episode {} is {}".format(episode, total_reward))
