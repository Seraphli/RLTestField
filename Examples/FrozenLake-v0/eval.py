import gym
from gym import wrappers
from agent import Agent

env = gym.make('FrozenLake-v0')
env = wrappers.Monitor(env, '/tmp/FrozenLake-experiment')
agent = Agent(1, env.action_space.n)
agent.load()
for i_episode in range(100):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = agent.eval(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
gym.upload('/tmp/FrozenLake-experiment', api_key='sk_Gpmd1ULuQT2E7JLnaWP1tA')
