import numpy as np
from sumTree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.eps = 1e-5
        self.max_priority = 1.0

    def push(self, transition):
        p = self.max_priority
        self.tree.add(p, transition)

    def sample(self, batch_size, beta):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = np.random.uniform(i * segment, (i + 1) * segment)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()

        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            p = (np.abs(error) + self.eps) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return self.tree.size