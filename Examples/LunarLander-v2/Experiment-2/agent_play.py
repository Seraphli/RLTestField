from sequence import Sequence
from myqueue import PriorityQueue
import gym, pickle, numpy as np
from utility.utility import init_logger, get_path
from agent import Agent
from tqdm import trange

env_name = 'LunarLander-v2'
env = gym.make(env_name)
logger = init_logger(env_name)
agent = Agent(32, env.action_space.n, logger)
agent.load_session(get_path('tmp/' + env_name))
hist_len = 4


def play():
    seq = Sequence()
    s = env.reset()
    hist = []
    i = -1
    while True:
        i += 1
        if i < hist_len:
            hist.append(s)
            if i == hist_len - 1:
                state = np.array(hist).flatten()
            continue
        a = agent.take_action(state, 0.05)
        s_, r, t, info = env.step(a)
        del hist[0]
        hist.append(s)
        state_ = np.array(hist).flatten()
        seq.append(s, a, r, t)
        state = state_
        s = s_
        if t:
            break
    return seq


memory = PriorityQueue()
memory_size = 50
for i in trange(2000):
    seq = play()
    memory.push(seq.score, seq)
    if len(memory) > memory_size:
        memory.pop()
with open('memory.pkl', 'wb') as f:
    pickle.dump(memory, f)
