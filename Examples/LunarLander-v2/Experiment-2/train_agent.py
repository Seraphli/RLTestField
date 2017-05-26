import pickle, random, gym
from agent import Agent
from tqdm import trange

gym.undo_logger_setup()

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

datum = train_data[0]
s, a, r, t, s_ = datum
env_name = 'LunarLander-v2'
env = gym.make(env_name)
agent = Agent(s.shape[0], env.action_space.n)
env.close()
losses = []
train_eps = 100000
for _ in trange(train_eps):
    batch = random.sample(train_data, 32)
    losses.append(agent.train(batch))
with open('loss.csv', 'w') as f:
    for _ in trange(train_eps):
        f.write('%d, %f\n' % (_, losses[_]))
# for _ in trange(2):
#     batch = random.sample(train_data, 32)
#     agent.train(batch)
