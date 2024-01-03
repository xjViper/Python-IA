import vizdoom as doom
import random
import time


game = doom.DoomGame()
game.load_config("./scenarios/basic.cfg")
game.init()


import numpy as np

actions = np.identity(3, dtype=np.uint8)


episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        imgs = state.screen_buffer
        info = state.game_variables
        reward = game.make_action(random.choice(actions))
        print("reward: ", reward)
        time.sleep(0.02)
    print("Result: ", game.get_total_reward())
    time.sleep(2)


game.close()
