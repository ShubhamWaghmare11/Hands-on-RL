import gymnasium as gym
import time

#Creating the environment
env = gym.make("LunarLander-v3", render_mode="human")


#Sample an action

sample_action = env.action_space.sample()
print("Sample action: ", sample_action)

sample_observation = env.observation_space.sample()
print("Sample observation: ",sample_observation)

