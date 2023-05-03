import torch
import torch.nn as nn
import torch.nn.functional as F


class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Reward, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        hidden1 = F.relu(self.l1(sa))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        reward = self.l4(hidden3)

        return reward