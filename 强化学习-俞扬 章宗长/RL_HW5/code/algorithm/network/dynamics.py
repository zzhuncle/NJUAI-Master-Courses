import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

class Deterministic_Dynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, normalizer=None, state_clip=100):
        super(Deterministic_Dynamics, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, state_dim)

        if normalizer is not None:
            self.normalize = True
            self.state_clip = state_clip
            self.state_normalizer = normalizer.state_normalizer
            self.action_normalizer = normalizer.action_normalizer
            self.diff_normalizer = normalizer.diff_normalizer
        else:
            self.normalize = False
            self.state_normalizer = nn.Identity()
            self.action_normalizer = nn.Identity()
            self.diff_normalizer = nn.Identity()

    def forward(self, state, action, res=True, eval=False):
        state_norm = self.state_normalizer(state)

        sa = torch.cat([state_norm, action], dim=1)
        hidden1 = F.relu(self.l1(sa))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        delta_state = self.l4(hidden3)

        if self.normalize:
            delta_state = self.diff_normalizer(delta_state, inverse=True)
            next_state = delta_state + state
            if eval:
                next_state = self.state_normalizer(self.state_normalizer(next_state).clamp(-self.state_clip, self.state_clip), inverse=True)
        else:
            next_state = delta_state + state
        return next_state


class SVG_Dynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, normalizer=None, state_clip=100):
        super(SVG_Dynamics, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.logstd_clip_l = -20  # logstd_clip[0]
        self.logstd_clip_u = 2  # logstd_clip[1]

        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, state_dim * 2)

        if normalizer is not None:
            self.normalize = True
            self.state_clip = state_clip
            self.state_normalizer = normalizer.state_normalizer
            self.action_normalizer = normalizer.action_normalizer
            self.diff_normalizer = normalizer.diff_normalizer
        else:
            self.normalize = False
            self.state_normalizer = nn.Identity()
            self.action_normalizer = nn.Identity()
            self.diff_normalizer = nn.Identity()

    def forward(self, state, action, real_next_state=None, eval=False):
        state_norm = self.state_normalizer(state)

        sa = torch.cat([state_norm, action], dim=1)
        hidden1 = F.relu(self.l1(sa))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        delta_state_and_prob = self.l4(hidden3)
        mu, logstd = delta_state_and_prob.chunk(2, dim=1)
        logstd = torch.clamp(logstd, self.logstd_clip_l, self.logstd_clip_u)
        std = logstd.exp()

        if real_next_state is None:
            dist = Normal(mu, std)
            delta_state = dist.rsample()
        else:
            diff_state_norm = self.diff_normalizer(real_next_state - state)
            eps = (diff_state_norm - mu) / std
            delta_state = mu + std * eps.detach()

        if self.normalize:
            delta_state = self.diff_normalizer(delta_state, inverse=True)
            next_state = delta_state + state
            if eval:
                next_state = self.state_normalizer(self.state_normalizer(next_state).clamp(-self.state_clip, self.state_clip), inverse=True)
        else:
            next_state = delta_state + state
        return next_state


class Stochastic_Dynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, normalizer=None, state_clip=100):
        super(Stochastic_Dynamics, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.logstd_clip_l = -20  # logstd_clip[0]
        self.logstd_clip_u = 2  # logstd_clip[1]

        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, state_dim * 2)

        if normalizer is not None:
            self.normalize = True
            self.state_clip = state_clip
            self.state_normalizer = normalizer.state_normalizer
            self.action_normalizer = normalizer.action_normalizer
            self.diff_normalizer = normalizer.diff_normalizer
        else:
            self.normalize = False
            self.state_normalizer = nn.Identity()
            self.action_normalizer = nn.Identity()
            self.diff_normalizer = nn.Identity()

    def forward(self, state, action, res=True, eval=False, get_logprob=False, get_mean=False):
        state_norm = self.state_normalizer(state)

        sa = torch.cat([state_norm, action], dim=1)
        hidden1 = F.relu(self.l1(sa))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        delta_state_and_prob = self.l4(hidden3)
        mu, logstd = delta_state_and_prob.chunk(2, dim=1)
        logstd = torch.clamp(logstd, self.logstd_clip_l, self.logstd_clip_u)
        std = logstd.exp()
        dist = Normal(mu, std)
        delta_state = dist.rsample()

        log_prob = None
        if get_logprob:
            log_prob = dist.log_prob(delta_state).sum(dim=1, keepdim=True)
        if get_mean:
            delta_state = mu

        if self.normalize:
            delta_state = self.diff_normalizer(delta_state, inverse=True)
            next_state = delta_state + state
            if eval:
                next_state = self.state_normalizer(self.state_normalizer(next_state).clamp(-self.state_clip, self.state_clip), inverse=True)
        else:
            next_state = delta_state + state
        return next_state, log_prob