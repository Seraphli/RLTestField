import gym
from tqdm import tqdm
from agent import Agent


class Game(object):
    def __init__(self):
        gym.undo_logger_setup()
        self.env = gym.make("CartPole-v0")
        state = self.env.reset()
        self.agent = Agent(state.shape[0], self.env.action_space.n)

    def clear_stat(self):
        self.n_games = 0
        self.min_score = .0
        self.max_score = -50000
        self.total_score = .0
        self.total_step = 0

    def game(self, phase, step):
        self.clear_stat()
        self._step = step
        with tqdm(total=step, desc=phase) as _tqdm:
            if phase == "Random":
                score = 0
                s = self.env.reset()
                while self.total_step < step:
                    a = self.env.action_space.sample()
                    s_, r, t, info = self.env.step(a)
                    self.agent.replay.append((s, a, r, t, s_))
                    s = s_
                    score += r
                    self.total_step += 1
                    _tqdm.update()
                    if t:
                        self.min_score = min(self.min_score, score)
                        self.max_score = max(self.max_score, score)
                        self.total_score += score
                        self.n_games += 1
                        score = 0
                        self.env.reset()
            if phase == "Train":
                score = 0
                s = self.env.reset()
                while self.total_step < step:
                    a = self.agent.train(s)
                    s_, r, t, info = self.env.step(a)
                    self.agent.replay.append((s, a, r, t, s_))
                    s = s_
                    score += r
                    self.total_step += 1
                    _tqdm.update()
                    if t:
                        self.min_score = min(self.min_score, score)
                        self.max_score = max(self.max_score, score)
                        self.total_score += score
                        self.n_games += 1
                        score = 0
                        self.env.reset()
            if phase == "Test":
                score = 0
                s = self.env.reset()
                while self.total_step < step:
                    a = self.agent.eval(s)
                    s_, r, t, info = self.env.step(a)
                    self.agent.replay.append((s, a, r, t, s_))
                    s = s_
                    score += r
                    self.total_step += 1
                    _tqdm.update()
                    if t:
                        self.min_score = min(self.min_score, score)
                        self.max_score = max(self.max_score, score)
                        self.total_score += score
                        self.n_games += 1
                        score = 0
                        self.env.reset()
        self.average_score = self.total_score / self.n_games

        return self.n_games, self.total_step, self.average_score, self.min_score, self.max_score

    def run(self):
        n_games, total_step, average_score, min_score, max_score \
            = self.game("Random", 50000)
        print('PHASE: %s, N: %d, STEP: %d, AVG: %f, MIN: %d, MAX: %d' %
              ("Random", n_games, total_step, average_score, min_score, max_score))
        for _ in range(20):
            n_games, total_step, average_score, min_score, max_score \
                = self.game("Train", 10000)
            print('PHASE: %s, N: %d, STEP: %d, AVG: %f, MIN: %d, MAX: %d' %
                  ("Train", n_games, total_step, average_score, min_score, max_score))
            n_games, total_step, average_score, min_score, max_score \
                = self.game("Test", 2000)
            print('PHASE: %s, N: %d, STEP: %d, AVG: %f, MIN: %d, MAX: %d' %
                  ("Test", n_games, total_step, average_score, min_score, max_score))
        self.agent.save()


if __name__ == '__main__':
    g = Game()
    g.run()
