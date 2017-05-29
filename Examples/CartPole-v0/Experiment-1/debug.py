import gym, numpy as np
from agent import Agent

env = gym.make("CartPole-v0")
agent = Agent(env)