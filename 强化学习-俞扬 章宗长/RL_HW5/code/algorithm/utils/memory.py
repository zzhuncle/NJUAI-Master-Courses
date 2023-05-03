import numpy as np
import torch


class TrajectoryBuffer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_size = args.batch_size_svg + 5000

        if self.device == 'gpu':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.total_sample = 0
        self.max_len = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add_data(self, state, action, next_state, reward, done):
        self.state[self.size] = state
        self.action[self.size] = action
        self.next_state[self.size] = next_state
        self.reward[self.size] = reward
        self.done[self.size] = float(done)

        self.size += 1

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.size
            return (
                torch.FloatTensor(self.state[:batch_size]).to(self.device),
                torch.FloatTensor(self.action[:batch_size]).to(self.device),
                torch.FloatTensor(self.next_state[:batch_size]).to(self.device),
                torch.FloatTensor(self.reward[:batch_size]).to(self.device),
                torch.FloatTensor(self.done[:batch_size]).to(self.device)
            )

        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def clear(self):
        self.total_sample = 0
        self.max_len = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.max_size = args.buffer_size
        self.device = args.device
        self.batch_size = args.batch_size_mf
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

        if self.device == 'gpu':
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recent_buffer = {'state': [],
                              'action': [],
                              'next_state': []}

    def set_buffer(self, state, action, next_state, reward, done):
        if self.size == 0:
            self.state = state
            self.action = action
            self.next_state = next_state
            self.reward = reward
            self.done = done

            self.size = state.shape[0]
            self.ptr = state.shape[0]
            self.max_size = state.shape[0]
        else:
            self.state = np.concatenate([self.state, state], axis=0)
            self.action = np.concatenate([self.action, action], axis=0)
            self.next_state = np.concatenate([self.next_state, next_state], axis=0)
            self.reward = np.concatenate([self.reward, reward], axis=0)
            self.done = np.concatenate([self.done, done], axis=0)

            self.size += state.shape[0]
            self.ptr += state.shape[0]
            self.max_size += state.shape[0]

    def clear_buffer(self):
        self.batch_size = self.args.batch_size_mf
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, self.state_dim))
        self.action = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.recent_buffer['state'].append(state)
        self.recent_buffer['action'].append(action)
        self.recent_buffer['next_state'].append(next_state)

    def sample(self, batch_size=None, total=False, offline=False):
        if batch_size is None and not total:
            batch_size = self.batch_size
        elif total and not offline:
            batch_size = self.size
            return (
                self.state[:batch_size],
                self.action[:batch_size],
                self.next_state[:batch_size],
                self.reward[:batch_size],
                self.done[:batch_size]
            )
        elif total and offline:
            batch_size = self.size
            return (
                torch.FloatTensor(self.state[:batch_size]).to(self.device),
                torch.FloatTensor(self.action[:batch_size]).to(self.device),
                torch.FloatTensor(self.next_state[:batch_size]).to(self.device),
                torch.FloatTensor(self.reward[:batch_size]).to(self.device),
                torch.FloatTensor(self.done[:batch_size]).to(self.device)
            )
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def get_recent(self):
        state = self.recent_buffer['state']
        action = self.recent_buffer['action']
        next_state = self.recent_buffer['next_state']
        self.recent_buffer = {'state': [],
                              'action': [],
                              'next_state': []}

        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(next_state).to(self.device)
        )

    def load_buffer(self, path):
        state_dict = np.load(path+'buffer.npy')
        self.state = state_dict['state']
        self.action = state_dict['action']
        self.next_state = state_dict['next_state']
        self.reward = state_dict['reward']
        self.done = state_dict['done']


    def save_buffer(self, path):
        state_dict = {
            'state': self.state,
            'action': self.action,
            'next_state': self.next_state,
            'reward': self.reward,
            'done': self.done
        }
        np.save(path+'buffer.npy', state_dict)