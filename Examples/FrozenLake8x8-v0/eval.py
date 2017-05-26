import gym
from gym import wrappers
from agent import Agent
import shutil

env_name = 'FrozenLake8x8-v0'
env = gym.make(env_name)
shutil.rmtree('/tmp/' + env_name + '-experiment', ignore_errors=True)
env = wrappers.Monitor(env, '/tmp/' + env_name + '-experiment')
agent = Agent(1, env.action_space.n)
agent.load()
for i_episode in range(3000):
    observation = env.reset()
    for t in range(10000):
        # env.render()
        action = agent.eval(observation)
        observation, reward, done, info = env.step(action)
        # env.render()
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
gym.upload('/tmp/' + env_name + '-experiment', api_key='sk_Gpmd1ULuQT2E7JLnaWP1tA')
