import gymnasium as gym
import time

#Creating the environment
env = gym.make("LunarLander-v3", render_mode="human")

env.reset()

for step in range(100):
    env.render()
    env.step(env.action_space.sample())

env.close()