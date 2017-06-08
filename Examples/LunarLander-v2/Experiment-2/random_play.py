from sequence import Sequence
from myqueue import PriorityQueue
import gym, pickle
from tqdm import trange

env_name = 'LunarLander-v2'
env = gym.make(env_name)


def play():
    seq = Sequence()
    s = env.reset()
    while True:
        a = env.action_space.sample()
        s_, r, t, info = env.step(a)
        seq.append(s, a, r, t)
        s = s_
        if t:
            break
    return seq


memory = PriorityQueue()
memory_size = 100
for i in trange(20000):
    seq = play()
    memory.push(seq.score, seq)
    if len(memory) > memory_size:
        memory.pop()
with open('memory.pkl', 'wb') as f:
    pickle.dump(memory, f)
