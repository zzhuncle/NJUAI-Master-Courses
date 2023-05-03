import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from algorithm.network.actor import Deterministic_Actor
from algorithm.network.critic import Twin_Qnetwork


class TD3:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed

        self.state_dim = self.args.state_dim
        self.action_dim = self.args.action_dim
        self.hidden_dim = self.args.hidden_dim
        self.action_clip = self.args.action_clip
        self.grad_norm_clip = self.args.grad_norm_clip

        self.gamma = args.gamma
        self.tau = args.tau
        self.lr = args.lr
        self.batch_size = args.batch_size_mf
        self.device = args.device

        if self.device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.policy_noise = args.policy_noise
        self.act_noise = args.act_noise
        self.noise_clip = args.noise_clip
        self.policy_freq = args.policy_freq

        self.actor_eval = Deterministic_Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_clip).to(self.device)
        self.critic_eval = Twin_Qnetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.actor_target = copy.deepcopy(self.actor_eval).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_eval).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.total_it = 0

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def inference(self, state, deterministic=False):
        input_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        output_action = self.actor_eval(input_state)
        action = output_action.data.cpu().numpy()[0]
        if not deterministic:
            action += np.random.normal(0, self.action_clip * self.act_noise, size=self.action_dim)
            #action += np.clip((np.random.randn(self.action_dim) * self.act_noise), -self.noise_clip, self.noise_clip)
            action = np.clip(action, -self.action_clip, self.action_clip)

        return action

    def update_actor(self, state, bc_action, lamb=0.1, beta=0.1, ada=True):
        action = self.actor_eval(state)
        q1, q2 = self.critic_eval(state, action)
        q = torch.min(q1, q2)
        if ada:
            actor_loss = -q.mean() * 2.5 / q1.mean().detach() + F.mse_loss(bc_action, action)
        else:
            actor_loss = -q.mean() * lamb + F.mse_loss(bc_action, action) * beta

        self.actor_optim.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), self.grad_norm_clip)
        self.actor_optim.step()

        return actor_loss.item()

    def compute_target_q(self, state, action, next_state, reward, done):
        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.action_clip, self.action_clip)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1. - done) * self.gamma * target_q

        return target_q

    def update_critic(self, state, action, next_state, reward, done):
        # with torch.no_grad():
        #     noise = (
        #             torch.randn_like(action) * self.policy_noise
        #     ).clamp(-self.noise_clip, self.noise_clip)
        #
        #     next_action = (
        #             self.actor_target(next_state) + noise
        #     ).clamp(-self.action_clip, self.action_clip)
        #
        #     target_q1, target_q2 = self.critic_target(next_state, next_action)
        #     target_q = torch.min(target_q1, target_q2)
        #     target_q = reward + (1. - done) * self.gamma * target_q
        target_q = self.compute_target_q(state, action, next_state, reward, done)

        eval_q1, eval_q2 = self.critic_eval(state, action)
        critic_loss = F.mse_loss(eval_q1, target_q) + F.mse_loss(eval_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_eval.parameters(), self.grad_norm_clip)
        self.critic_optim.step()

        return critic_loss.item()

    def soft_update(self):
        for param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, memory, lamb=None, beta=None, ada=True):
        self.total_it += 1
        state, action, next_state, reward, done = memory.sample(self.batch_size)

        #state = torch.tensor(batch_data['state'], dtype=torch.float32).to(self.device)
        #action = torch.tensor(batch_data['action'], dtype=torch.float32).to(self.device)
        #reward = torch.tensor(batch_data['reward'], dtype=torch.float32).to(self.device)
        #next_state = torch.tensor(batch_data['next_state'], dtype=torch.float32).to(self.device)
        #done = torch.tensor(batch_data['done'], dtype=torch.float32).to(self.device)

        aloss, closs = None, None
        closs = self.update_critic(state, action, next_state, reward, done)
        if self.total_it % self.policy_freq == 0:
            aloss = self.update_actor(state, action, lamb=lamb, beta=beta, ada=ada)
            self.soft_update()

        return closs, aloss, None

    def save_model(self, path):
        state_dict = {'actor_eval': self.actor_eval.state_dict(),
                      'actor_target': self.actor_target.state_dict(),
                      'critic_eval': self.critic_eval.state_dict(),
                      'critic_target': self.critic_target.state_dict(),
                      'actor_optim': self.actor_optim.state_dict(),
                      'critic_optim': self.critic_optim.state_dict()}

        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.actor_eval.load_state_dict(state_dict['actor_eval'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic_eval.load_state_dict(state_dict['critic_eval'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optim.load_state_dict(state_dict['actor_optim'])
        self.critic_optim.load_state_dict(state_dict['critic_optim'])