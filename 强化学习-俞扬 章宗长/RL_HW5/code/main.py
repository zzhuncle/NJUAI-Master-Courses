# 加载并评估训练好的模型

import gym
import numpy as np
import torch
from algorithm.agent.cql import CQL
from algorithm.model.td3bc import TD3 as TD3BC
from algorithm.utils.memory import ReplayBuffer
from algorithm.utils.params import get_args
from matplotlib import pyplot as plt

env_names = ['hopper']
levels = ['random', 'medium', 'expert']

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

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        while True:
            action = policy.inference(state, True)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

args = get_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.state_dim = 11
args.action_dim = 3
args.num_random = 40
args.beta = 10.0

env = gym.make('Hopper-v2')
for env_name in env_names:
    for level in levels:
        # policy = TD3BC(args)
        policy = CQL(args)
        policy.load_model('../model_{}.pt'.format(level))
        reward = evaluate(env, policy)
        print('{}: {}'.format(level, reward))
