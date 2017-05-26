class Sequence(object):
    def __init__(self):
        self.buffer = []
        self.score = 0

    def append(self, s, a, r, t):
        self.buffer.append((s, a, r, t))
        self.score += r

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def __setitem__(self, key, value):
        self.buffer[key] = value
