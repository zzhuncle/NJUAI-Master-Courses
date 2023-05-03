import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Qnetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        hidden1 = F.relu(self.l1(sa))
        hidden2 = F.relu(self.l2(hidden1))
        hidden3 = F.relu(self.l3(hidden2))
        qvalue = self.l4(hidden3)

        return qvalue


class Twin_Qnetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Twin_Qnetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.q1 = Qnetwork(state_dim, action_dim, hidden_dim)
        self.q2 = Qnetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        qvalue1, qvalue2 = self.q1(state, action), self.q2(state, action)

        return qvalue1, qvalue2