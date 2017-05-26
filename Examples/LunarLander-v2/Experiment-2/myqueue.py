import heapq


class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, priority, x):
        heapq.heappush(self.queue, (priority, x))

    def pop(self):
        _, x = heapq.heappop(self.queue)
        return x

    def empty(self):
        return not self.queue

    def __len__(self):
        return len(self.queue)

    def __iter__(self):
        return iter(self.queue)
