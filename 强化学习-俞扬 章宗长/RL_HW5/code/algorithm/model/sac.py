import copy
import torch
import torch.nn.functional as F
import numpy as np

from algorithm.network.actor import Stochastic_Actor
from algorithm.network.critic import Twin_Qnetwork


class SAC:

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
        self.update_interval = args.update_interval
        self.target_entropy = - self.action_dim
        self.batch_size = args.batch_size_mf
        self.device = args.device

        if self.device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.actor_eval = Stochastic_Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_clip).to(self.device)
        self.critic_eval = Twin_Qnetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        #self.actor_target = copy.deepcopy(self.actor_eval).to(self.device)
        self.critic_target = copy.deepcopy(self.critic_eval).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.temp_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.total_it = 0

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def inference(self, state, deterministic=False):
        input_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        output_action, _, output_mean = self.actor_eval(input_state, get_mean=deterministic)
        action = output_action.data.cpu().numpy()[0]
        if deterministic:
            mean = output_mean.data.cpu().numpy()[0]
            return mean
        return action

    def soft_update(self):
        for param, target_param in zip(self.critic_eval.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        #for param, target_param in zip(self.actor_eval.parameters(), self.actor_target.parameters()):
        #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_critic(self, state, action, next_state, reward, done):
        with torch.no_grad():
            next_action, next_logprobs, _ = self.actor_eval(next_state, get_logprob=True)
            q_t1, q_t2 = self.critic_target(next_state, next_action)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward + (1.0 - done) * self.gamma * (q_target - self.alpha * next_logprobs)
        q_1, q_2 = self.critic_eval(state, action)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)

        q_loss_step = loss_1 + loss_2
        self.critic_optim.zero_grad()
        q_loss_step.backward()
        self.critic_optim.step()

        return q_loss_step.item()

    # def update_critic(self, state, action, next_state, reward, done):
    #     with torch.no_grad():
    #         next_action, next_logprob, _ = self.actor_eval(state, get_logprob=True)
    #         target_q1, target_q2 = self.critic_target(next_state, next_action)
    #         target_q = torch.min(target_q1, target_q2)
    #         target_q = reward + (1. - done) * self.gamma * (target_q - self.alpha * next_logprob)
    #
    #     eval_q1, eval_q2 = self.critic_eval(state, action)
    #     critic_loss = F.mse_loss(eval_q1, target_q) + F.mse_loss(eval_q2, target_q)
    #
    #     self.critic_optim.zero_grad()
    #     critic_loss.backward()
    #     #torch.nn.utils.clip_grad_norm_(self.critic_eval.parameters(), self.grad_norm_clip)
    #     self.critic_optim.step()
    #
    #     return critic_loss.item()

    def update_actor(self, state):
        action, logprobs, _ = self.actor_eval(state, get_logprob=True)
        q_b1, q_b2 = self.critic_eval(state, action)
        qval_batch = torch.min(q_b1, q_b2)
        actor_loss = (self.alpha * logprobs - qval_batch).mean()
        temp_loss = -self.log_alpha * (logprobs.detach() + self.target_entropy).mean()

        for p in self.critic_eval.parameters():
            p.requires_grad = False
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        for p in self.critic_eval.parameters():
            p.requires_grad = True

        # self.actor_optim.zero_grad()
        # actor_loss.backward()
        # #torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), self.grad_norm_clip)
        # self.actor_optim.step()
        #
        # self.temp_optim.zero_grad()
        # temp_loss.backward()
        # self.temp_optim.step()

        self.alpha = self.log_alpha.exp()

        return actor_loss.item(), temp_loss.item()

    def train(self, memory):
        self.total_it += 1
        state, action, next_state, reward, done = memory.sample(self.batch_size)

        aloss, tloss, closs = None, None, None
        closs = self.update_critic(state, action, next_state, reward, done)
        aloss, tloss = self.update_actor(state)
        if self.total_it % self.update_interval == 0:
            self.soft_update()

        return closs, aloss, tloss

    def save_model(self, path):
        state_dict = {'actor_eval': self.actor_eval.state_dict(),
                      'logalpha': self.log_alpha,
                      #'actor_target': self.actor_target.state_dict(),
                      'critic_eval': self.critic_eval.state_dict(),
                      'critic_target': self.critic_target.state_dict(),
                      'actor_optim': self.actor_optim.state_dict(),
                      'critic_optim': self.critic_optim.state_dict()}

        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.actor_eval.load_state_dict(state_dict['actor_eval'])
        self.log_alpha = state_dict['logalpha']
        #self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic_eval.load_state_dict(state_dict['critic_eval'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optim.load_state_dict(state_dict['actor_optim'])
        self.critic_optim.load_state_dict(state_dict['critic_optim'])