import copy
import torch
import torch.nn.functional as F
import numpy as np

from algorithm.network.actor import Stochastic_Actor
from algorithm.network.critic import Twin_Qnetwork


class CQL:

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
        self.update_interval = args.update_interval # 
        self.target_entropy = - self.action_dim # 目标熵的大小
        self.batch_size = args.batch_size_mf # 
        self.device = args.device

        if self.device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.actor_eval = Stochastic_Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_clip).to(self.device)
        self.critic_eval = Twin_Qnetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        self.critic_target = copy.deepcopy(self.critic_eval).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic_eval.parameters(), lr=self.lr)
        self.temp_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.total_it = 0 # 训练次数
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # @zhuangzh
        self.beta = args.beta  # CQL损失函数中的系数
        self.num_random = args.num_random  # CQL中的动作采样数


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

        # @zhuangzh
        # 以上与SAC相同,以下Q网络更新是CQL的额外部分
        batch_size = state.shape[0]
        random_unif_action = torch.rand([batch_size * self.num_random, action.shape[-1]], dtype=torch.float).uniform_(-1, 1).to(self.device)
        random_unif_log_pi = np.log(0.5 ** next_action.shape[-1])
        tmp_state = state.unsqueeze(1).repeat(1, self.num_random, 1).view(-1, state.shape[-1])
        tmp_next_state = next_state.unsqueeze(1).repeat(1, self.num_random, 1).view(-1, next_state.shape[-1])
        random_curr_action, random_curr_log_pi, _ = self.actor_eval(tmp_state, get_logprob=True)
        random_next_action, random_next_log_pi, _ = self.actor_eval(tmp_next_state, get_logprob=True)
        
        def util(x, num = self.num_random):
            x1, x2 = x
            return x1.view(-1, num, 1), x2.view(-1, num, 1)
        
        q1_unif, q2_unif = util(self.critic_eval(tmp_state, random_unif_action))
        q1_curr, q2_curr = util(self.critic_eval(tmp_state, random_curr_action))
        q1_next, q2_next = util(self.critic_eval(tmp_state, random_next_action))
        q1_cat = torch.cat([q1_unif - random_unif_log_pi,
            q1_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
            q1_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
        ], dim=1)
        q2_cat = torch.cat([q2_unif - random_unif_log_pi,
            q2_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
            q2_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
        ], dim=1)

        loss_1 += self.beta * (torch.logsumexp(q1_cat, dim=1).mean() - q_1.mean())
        loss_2 += self.beta * (torch.logsumexp(q2_cat, dim=1).mean() - q_2.mean())
        # @zhuangzh

        q_loss_step = loss_1 + loss_2
        self.critic_optim.zero_grad()
        q_loss_step.backward()
        self.critic_optim.step()

        return q_loss_step.item()


    def update_actor(self, state):
        action, logprobs, _ = self.actor_eval(state, get_logprob=True)
        q_b1, q_b2 = self.critic_eval(state, action)
        qval_batch = torch.min(q_b1, q_b2)
        actor_loss = (self.alpha * logprobs - qval_batch).mean()


        for p in self.critic_eval.parameters():
            p.requires_grad = False
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 更新alpha值
        temp_loss = -self.log_alpha * (logprobs.detach() + self.target_entropy).mean() 
        self.temp_optim.zero_grad()
        temp_loss.backward()
        self.temp_optim.step()
        
        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        return actor_loss.item(), temp_loss.item()

    def train(self, memory):
        self.total_it += 1
        state, action, next_state, reward, done = memory.sample(self.batch_size)

        aloss, tloss, closs = None, None, None
        closs = self.update_critic(state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device), done.to(self.device))
        aloss, tloss = self.update_actor(state.to(self.device))
        if self.total_it % self.update_interval == 0:
            self.soft_update()

        return closs, aloss, tloss

    def save_model(self, path):
        state_dict = {'actor_eval': self.actor_eval.state_dict(),
                      'logalpha': self.log_alpha,
                      'critic_eval': self.critic_eval.state_dict(),
                      'critic_target': self.critic_target.state_dict(),
                      'actor_optim': self.actor_optim.state_dict(),
                      'critic_optim': self.critic_optim.state_dict()}

        torch.save(state_dict, path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.actor_eval.load_state_dict(state_dict['actor_eval'])
        self.log_alpha = state_dict['logalpha']
        self.critic_eval.load_state_dict(state_dict['critic_eval'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optim.load_state_dict(state_dict['actor_optim'])
        self.critic_optim.load_state_dict(state_dict['critic_optim'])