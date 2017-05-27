import pickle, random, gym
from agent import Agent
from tqdm import trange
from utility.utility import init_logger, get_path

gym.undo_logger_setup()

with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

datum = train_data[0]
s, a, r, t, s_ = datum
env_name = 'LunarLander-v2'
logger = init_logger(env_name)
env = gym.make(env_name)
agent = Agent(s.shape[0], env.action_space.n, logger)
agent.load_session(get_path('tmp/' + env_name))
env.close()
losses = []
train_eps = 100000
for _ in trange(train_eps):
    batch = random.sample(train_data, 32)
    losses.append(agent.train(batch))
with open('loss.csv', 'w') as f:
    for _ in trange(train_eps):
        f.write('%d, %f\n' % (_, losses[_]))
agent.save_session(get_path('tmp/' + env_name))
