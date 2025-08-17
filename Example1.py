import gymnasium as gym


#Creating the environment
env = gym.make("LunarLander-v3", render_mode="human")

env.reset()

for step in range(10):
    env.render()
    env.step(env.action_space.sample())


env.close()