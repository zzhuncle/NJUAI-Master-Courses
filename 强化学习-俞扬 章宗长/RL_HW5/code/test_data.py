#import d4rl

# 数据集的读取
import gym
import numpy as np
import torch
from algorithm_offline.model.td3bc import TD3 as TD3BC
from algorithm_offline.utils.memory import ReplayBuffer
from algorithm_offline.utils.params import get_args


task_name = "{}-{}-v0"
env_names = ['hopper'] #['halfcheetah', 'hopper', 'walker2d']
levels = ['random', 'medium', 'expert']


def save_data(task_name, env_name, level):
    path = './dataset_mujoco/{}_{}_data.npy'.format(env_name, level)
    env = gym.make(task_name.format(env_name, level))
    dataset = env.get_dataset()

    states = dataset["observations"][:]
    actions = dataset["actions"][:]
    next_states = np.concatenate([dataset["observations"][1:], np.zeros_like(states[0])[np.newaxis, :]], axis=0)
    rewards = dataset["rewards"][:, np.newaxis]
    terminals = dataset["terminals"][:, np.newaxis] + 0.

    state_dict = {'state': states,
                  'action': actions,
                  'next_state': next_states,
                  'reward': rewards,
                  'terminal': terminals}

    np.save(path, state_dict)


def load_data(path):
    data = np.load(path, allow_pickle=True).item()
    states = data['state']
    actions = data['action']
    next_states = data['next_state']
    rewards = data['reward']
    terminals = data['terminal']

    dataset = {'state': states,
               'action': actions,
               'next_state': next_states,
               'reward': rewards,
               'terminal': terminals}

    return dataset


args = get_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.state_dim = 11
args.action_dim = 3

for env_name in env_names:
    for level in levels:
        dataset = load_data('./dataset_mujoco/{}_{}_data.npy'.format(env_name, level))
        states, actions, next_states, rewards, terminals = dataset['state'], dataset['action'], dataset['next_state'], dataset['reward'], dataset['terminal']

        replay_buffer = ReplayBuffer(args)
        replay_buffer.set_buffer(states, actions, next_states, rewards, terminals)
        policy = TD3BC(args)

        for i in range(1000000):
            closs, aloss, _ = policy.train(replay_buffer)

