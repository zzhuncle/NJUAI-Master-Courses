import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

from algorithm.utils.utils import TanhTransform


# class Stochastic_Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256, logstd_clip=[-20, 2], action_clip=1.0):
#         super(Stochastic_Actor, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.logstd_clip_l = -20 #logstd_clip[0]
#         self.logstd_clip_u = 2 #logstd_clip[1]
#         self.action_clip = action_clip
#
#         self.l1 = nn.Linear(state_dim, hidden_dim)
#         self.l2 = nn.Linear(hidden_dim, hidden_dim)
#         self.l3 = nn.Linear(hidden_dim, hidden_dim)
#         self.l4 = nn.Linear(hidden_dim, action_dim + action_dim)
#
#     def forward(self, state, get_logprob=False, get_mean=False):
#         hidden1 = F.relu(self.l1(state))
#         hidden2 = F.relu(self.l2(hidden1))
#         hidden3 = F.relu(self.l3(hidden2))
#
#         action_and_prob = self.l4(hidden3)
#         mu, logstd = action_and_prob.chunk(2, dim=1)
#         logstd = torch.clamp(logstd, self.logstd_clip_l, self.logstd_clip_u)
#         std = logstd.exp()
#         dist = Normal(mu, std)
#
#         transforms = [TanhTransform(cache_size=1)]
#         dist = TransformedDistribution(dist, transforms)
#         action = dist.rsample()
#
#         mean, log_prob = None, None
#         if get_logprob:
#             log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
#         if get_mean:
#             mean = torch.tanh(mu)
#
#         return action, log_prob, mean
class SVG_Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, logstd_clip=[-20, 2], action_clip=1.0):
        super(SVG_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.logstd_clip_l = -20  # logstd_clip[0]
        self.logstd_clip_u = 2  # logstd_clip[1]
        self.action_clip = action_clip

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, action_dim + action_dim)

    def forward(self, state, real_action=None, eval=False):
        hidden1 = F.relu(self.l1(state))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        mu_logstd= self.l4(hidden3)
        #mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        mean = torch.tanh(mu)

        if real_action is not None:
            eps = (real_action - mean) / std
            action = mean + std * eps.detach()
        elif eval:
            action = mean
        else:
            dist = Normal(mu, std)
            transforms = [TanhTransform(cache_size=1)]
            dist = TransformedDistribution(dist, transforms)
            action = dist.rsample()

        return action

class Stochastic_Actor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, logstd_clip=[-20, 2], action_clip=1.0):
        super(Stochastic_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.logstd_clip_l = -20  # logstd_clip[0]
        self.logstd_clip_u = 2  # logstd_clip[1]
        self.action_clip = action_clip

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, action_dim + action_dim)

    def forward(self, state, get_logprob=False, get_mean=False):
        hidden1 = F.relu(self.l1(state))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        mu_logstd= self.l4(hidden3)
        #mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)

        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()

        mean, logprob = None, None
        if get_logprob:
            logprob = dist.log_prob(action).sum(dim=1, keepdim=True)
        if get_mean:
            mean = torch.tanh(mu)
        return action, logprob, mean


class Deterministic_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_clip=1.0):
        super(Deterministic_Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_clip = action_clip

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        hidden1 = F.relu(self.l1(state))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        action = self.action_clip * torch.tanh(self.l4(hidden3))

        return action