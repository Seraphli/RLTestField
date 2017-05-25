import tensorflow as tf, os, numpy as np
from exp_replay import Replay


class Agent(object):
    def __init__(self, state_size, action_size):
        self.state_size, self.action_size = state_size, action_size
        self.model()
        self.replay = Replay()
        self.step = 0

    def model(self):
        self.q_sa = np.zeros([16, 4])

    def save(self):
        import pickle
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.q_sa, f)

    def load(self):
        import pickle
        with open('model.pkl', 'rb') as f:
            self.q_sa = pickle.load(f)

    def q_s(self, obs):
        q = self.q_sa[obs, :]
        return q

    def take_action(self, obs, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        q = self.q_sa[obs, :]
        return np.argmax(q)

    def eval(self, obs):
        # return self.take_action(obs, 0.001)
        return self.take_action(obs, 0)

    def train(self, obs):
        samples = self.replay.batch(32)
        loss = 0
        for i in range(len(samples)):
            s, a, r, t, s_ = samples[i]
            if t:
                q_target = r
            else:
                q_target = r + 0.99 * np.max(self.q_s(s_))
            loss += (q_target - self.q_sa[s, a])
            self.q_sa[s, a] = self.q_sa[s, a] + min(max(1.0 + (0.01 - 1.0) / 10000 * self.step, 0.01), 1.0) * (
                q_target - self.q_sa[s, a])
        self.step += 1
        # if self.step % 1000 == 0:
        #     print('\n%f' % loss)
        # return self.take_action(obs, min(max(1.0 + (0.01 - 1.0) / 10000 * self.step, 0.01), 1.0))
        return self.take_action(obs, 0)
