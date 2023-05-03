import random
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, WeightedRandomSampler

class RolloutStorage(object):
    def __init__(self, config):
        self.obs = torch.zeros([config.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.next_obs = torch.zeros([config.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.rewards = torch.zeros([config.max_buff,  1])
        self.actions = torch.zeros([config.max_buff, 1])
        self.actions = self.actions.long()
        self.masks = torch.ones([config.max_buff,  1])
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.num_steps = config.max_buff
        self.step = 0
        self.current_size = 0

    def add(self, obs, actions, rewards, next_obs, masks):
        self.obs[self.step].copy_(torch.tensor(obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.next_obs[self.step].copy_(torch.tensor(next_obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.actions[self.step].copy_(torch.tensor(actions, dtype=torch.float))
        self.rewards[self.step].copy_(torch.tensor(rewards, dtype=torch.float))
        self.masks[self.step].copy_(torch.tensor(masks, dtype=torch.float))
        self.step = (self.step + 1) % self.num_steps
        self.current_size = min(self.current_size + 1, self.num_steps)

    def sample(self, mini_batch_size=None):
        indices = np.random.randint(0, self.current_size, mini_batch_size)
        obs_batch = self.obs[indices]
        obs_next_batch = self.next_obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        masks_batch = self.masks[indices]
        return obs_batch, obs_next_batch, actions_batch, rewards_batch, masks_batch

# @zhuangzh
class SumTree(object):
    ''' a binary tree data structure where the parent's value is the sum of its children '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        print(parent)
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, step, pri):
        idx = step + self.capacity - 1
        self.update(idx, pri)

    def update(self, idx, pri):
        change = pri - self.tree[idx]
        self.tree[idx] = pri
        self._propagate(idx, change)

    def sample(self, s):
        idx = self._retrieve(0, s)
        return idx - self.capacity + 1


class PriorityRolloutStorage(RolloutStorage):
    def __init__(self, config):
        self.max_buff = 2 ** (int(np.log2(config.max_buff)) + 1)
        print((int(np.log2(config.max_buff)) + 1))
        self.obs = torch.zeros([self.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.next_obs = torch.zeros([self.max_buff,  *config.state_shape], dtype=torch.uint8)
        self.rewards = torch.zeros([self.max_buff,  1])
        self.actions = torch.zeros([self.max_buff, 1])
        self.actions = self.actions.long()
        self.masks = torch.ones([self.max_buff,  1])
        self.num_steps = self.max_buff
        self.step = 0
        self.current_size = 0
        self.tree = SumTree(self.max_buff)
        self.pri_max = 0.1
        self.alpha = 0.6
        self.epsilon = 0.01

    def add(self, obs, actions, rewards, next_obs, masks):
        self.obs[self.step].copy_(torch.tensor(obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.next_obs[self.step].copy_(torch.tensor(next_obs[None,:], dtype=torch.uint8).squeeze(0).squeeze(0))
        self.actions[self.step].copy_(torch.tensor(actions, dtype=torch.float))
        self.rewards[self.step].copy_(torch.tensor(rewards, dtype=torch.float))
        self.masks[self.step].copy_(torch.tensor(masks, dtype=torch.float))
        pri = (np.abs(self.pri_max) + self.epsilon) ** self.alpha
        self.tree.add(self.step, pri)
        self.step = (self.step + 1) % self.num_steps
        self.current_size = min(self.current_size + 1, self.num_steps)


    def sample(self, batch_size):
        indices = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx = self.tree.sample(s)
            indices.append(idx)

        obs_batch = self.obs[indices]
        obs_next_batch = self.next_obs[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        masks_batch = self.masks[indices]
        return indices, obs_batch, obs_next_batch, actions_batch, rewards_batch, masks_batch

    def update(self, idxs, td_errs):
        self.pri_max = max(self.pri_max, max(np.abs(td_errs)))
        for i, idx in enumerate(idxs):
            pri = (np.abs(td_errs[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, pri)
