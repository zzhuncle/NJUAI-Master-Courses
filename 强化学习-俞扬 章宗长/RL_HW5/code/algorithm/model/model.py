import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from algorithm.network.dynamics import Stochastic_Dynamics
from algorithm.network.reward import Reward
from algorithm.utils.normalizer import Normalizer


class Ensemble_Guassian_Dynamics:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim

        self.lr = args.lr
        self.T = args.horizon
        self.num_model = args.num_ensemble
        self.batch_size = args.batch_size_mb
        self.state_clip = args.state_clip
        self.device = args.device
        if self.device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.norm = args.norm_input_mb
        self.num_epoch = args.num_epoch
        if self.norm:
            self.normalizer = Normalizer(self.state_dim, self.action_dim).to(self.device)
        else:
            self.normalizer = None

        self.ensemble_dynamics = [Stochastic_Dynamics(self.state_dim, self.action_dim, self.hidden_dim, self.normalizer, self.state_clip).to(self.device) for _ in range(self.num_model)]
        self.ensemble_rewards = [Reward(self.state_dim, self.action_dim, self.hidden_dim).to(self.device) for _ in range(self.num_model)]

        self.optim_dynamics = [torch.optim.Adam(self.ensemble_dynamics[i].parameters(), lr=self.lr) for i in range(self.num_model)]
        self.optim_rewards = [torch.optim.Adam(self.ensemble_rewards[i].parameters(), lr=self.lr) for i in range(self.num_model)]

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def inference(self, state, action):
        pass

    def update_normalizer(self, memory, offline=True):
        state, action, next_state, reward, done = None, None, None, None, None
        if offline:
            state, action, next_state, reward, done = memory.sample(total=True, offline=True)
        else:
            state, action, next_state = memory.get_recent()
        self.normalizer.state_normalizer.update(state)
        self.normalizer.action_normalizer.update(action)
        self.normalizer.diff_normalizer.update(next_state - state)
        if offline:
            del state, action, next_state, reward, done

    def forward(self, state, action, id):
        next_state, _ = self.ensemble_dynamics[id](state, action, eval=True)
        reward = self.ensemble_rewards[id](state, action)

        return next_state, reward

    def train(self, memory):
        if self.norm:
            self.update_normalizer(memory)
        batch_size = memory.size
        ensemble_dynamics_loss = [[] for _ in range(self.num_model)]
        ensemble_rewards_loss = [[] for _ in range(self.num_model)]
        state, action, next_state, reward, done = memory.sample(total=True)
        arr = np.arange(memory.size)

        for id in range(self.num_model):
            for i in range(self.num_epoch):
                np.random.shuffle(arr)
                total_reward_loss = 0
                total_dynamics_loss = 0
                for j in range(batch_size // self.batch_size):
                    batch_index = arr[self.batch_size * j : self.batch_size * (j+1)]
                    batch_index = torch.LongTensor(batch_index)

                    batch_state = torch.FloatTensor(state[batch_index]).to(self.device)
                    batch_action = torch.FloatTensor(action[batch_index]).to(self.device)
                    batch_next_state = torch.FloatTensor(next_state[batch_index]).to(self.device)
                    batch_reward = torch.FloatTensor(reward[batch_index]).to(self.device)
                    batch_done = torch.FloatTensor(done[batch_index]).to(self.device)

                    pre_next_state, pre_reward = self.forward(batch_state, batch_action, id)

                    dynamics_loss = torch.mean(torch.pow(batch_next_state - pre_next_state, 2) * (1. - batch_done))
                    reward_loss = torch.mean(torch.pow(batch_reward - pre_reward, 2) * (1. - batch_done))

                    self.optim_dynamics[id].zero_grad()
                    dynamics_loss.backward()
                    self.optim_dynamics[id].step()

                    self.optim_rewards[id].zero_grad()
                    reward_loss.backward()
                    self.optim_rewards[id].step()

                    total_dynamics_loss += dynamics_loss.item()
                    total_reward_loss += reward_loss.item()

                ensemble_dynamics_loss[id].append(total_dynamics_loss / (batch_size // self.batch_size))
                ensemble_rewards_loss[id].append(total_reward_loss / (batch_size // self.batch_size))

        return ensemble_dynamics_loss, ensemble_rewards_loss

    def train_(self, memory):
        if self.norm:
            self.update_normalizer(memory)
        batch_size = memory.size
        ensemble_dynamics_loss = [[] for _ in range(self.num_model)]
        ensemble_rewards_loss = [[] for _ in range(self.num_model)]
        for id in range(self.num_model):
            for i in range(self.num_epoch):
                total_reward_loss = 0
                total_dynamics_loss = 0
                for j in range(batch_size // self.batch_size + 1):
                    state, action, next_state, reward, done = memory.sample(self.batch_size)
                    pre_next_state, pre_reward = self.forward(state, action, id)

                    dynamics_loss = F.mse_loss(next_state, pre_next_state)
                    reward_loss = F.mse_loss(reward, pre_reward)

                    self.optim_dynamics[id].zero_grad()
                    dynamics_loss.backward()
                    self.optim_dynamics[id].step()

                    self.optim_rewards[id].zero_grad()
                    reward_loss.backward()
                    self.optim_rewards[id].step()

                    total_dynamics_loss += dynamics_loss.item()
                    total_reward_loss += reward_loss.item()

                ensemble_dynamics_loss[id].append(total_dynamics_loss)
                ensemble_rewards_loss[id].append(total_reward_loss)

        return ensemble_dynamics_loss, ensemble_rewards_loss

    def load_model(self, path):
        state_dict = torch.load(path)

        dynamics_state_dict = state_dict['ensemble_dynamics']
        rewards_state_dict = state_dict['ensemble_rewards']
        dynamics_optim_state_dict = state_dict['dynamics_optim']
        rewards_optim_state_dict = state_dict['rewards_optim']

        for id in range(self.num_model):
            self.ensemble_dynamics[id].load_state_dict(dynamics_state_dict[id])
            self.ensemble_rewards[id].load_state_dict(rewards_state_dict[id])
            self.optim_dynamics[id].load_state_dict(dynamics_optim_state_dict[id])
            self.optim_rewards[id].load_state_dict(rewards_optim_state_dict[id])

    def save_model(self, path):
        dynamics_state_dict = [self.ensemble_dynamics[id].state_dict() for id in range(self.num_model)]
        rewards_state_dict = [self.ensemble_rewards[id].state_dict() for id in range(self.num_model)]
        dynamics_optim_state_dict = [self.optim_dynamics[id].state_dict() for id in range(self.num_model)]
        rewards_optim_state_dict = [self.optim_rewards[id].state_dict() for id in range(self.num_model)]

        state_dict = {'ensemble_dynamics': dynamics_state_dict,
                      'ensemble_rewards': rewards_state_dict,
                      'dynamics_optim': dynamics_optim_state_dict,
                      'rewards_optim': rewards_optim_state_dict}

        torch.save(state_dict, path)