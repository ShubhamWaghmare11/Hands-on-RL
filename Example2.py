import gymnasium as gym


env = gym.make("LunarLander-v3", render_mode='human')

env.reset()

for i in range(100):
    env.render()
    env.step(env.action_space.sample())

env.close()