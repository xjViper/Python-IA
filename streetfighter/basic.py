import retro
import time

env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")

obs = env.reset()

done = False

for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        obs, reward, terminate, truncate, info = env.step(env.action_space.sample())
        time.sleep(0.01)
        if reward != 0:
            print(reward)
        if terminate or truncate:
            break
