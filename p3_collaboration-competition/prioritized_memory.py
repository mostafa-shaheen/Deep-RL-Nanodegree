import random
import numpy as np
from SumTree import SumTree
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.4
    beta = 0.001   #  reach using full importance sampling after 1000 episodes

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            if type(data) == tuple:
                batch.append(data)
                idxs.append(idx)
                priorities.append(p)
                
        if len(batch) !=n:
            batch.append(batch[-1])
            idxs.append(idxs[-1])
            priorities.append(priorities[-1])
            

        mini_batch = np.array(batch).transpose()
        states = torch.from_numpy(np.vstack(mini_batch[0])).float().to(device)
        actions = torch.from_numpy(np.vstack(mini_batch[1])).float().to(device)
        rewards = torch.from_numpy(np.vstack(mini_batch[2])).float().to(device)
        next_states = torch.from_numpy(np.vstack(mini_batch[3])).float().to(device)
        dones = mini_batch[4]
        dones = torch.from_numpy(np.vstack(dones.astype(int))).float().to(device)            
            
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
                       
        sample = (states, actions, rewards, next_states, dones)

        return sample, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        
