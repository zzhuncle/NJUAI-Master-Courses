# 训练、保存、评估模型

import gym
import numpy as np
import torch
from algorithm.agent.cql import CQL
from algorithm.model.sac import SAC as SAC
from algorithm.model.td3bc import TD3 as TD3BC
from algorithm.utils.memory import ReplayBuffer
from algorithm.utils.params import get_args
from matplotlib import pyplot as plt

task_name = "{}-{}-v0"
env_names = ['hopper'] #['halfcheetah', 'hopper', 'walker2d']
levels = ['random', 'medium', 'expert']
run_dir = 'run'

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
args.num_random = 20
args.beta = 5.0


num_epoches = 1000000
log_epoches = 10000

import os
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

env = gym.make('Hopper-v2')
for env_name in env_names:
    for level in levels:
        if level == 'expert':
            args.num_random = 40
            args.beta = 10.0
            
        return_list = [] # zhuangzh
        best_reward = -1
        
        dataset = load_data('./dataset_mujoco/{}_{}_data.npy'.format(env_name, level))
        states, actions, next_states, rewards, terminals = dataset['state'], dataset['action'], dataset['next_state'], dataset['reward'], dataset['terminal']

        replay_buffer = ReplayBuffer(args)
        replay_buffer.set_buffer(states, actions, next_states, rewards, terminals)
        # policy = TD3BC(args)
        policy = CQL(args)

        for i in range(num_epoches):
            closs, aloss, _ = policy.train(replay_buffer)
            
            if i % log_epoches == 0:
                reward = evaluate(env, policy)
                return_list.append(reward)
                if reward >= best_reward:
                    best_reward = reward
                    policy.save_model('{}/model_{}.pt'.format(run_dir, level))
                
        epochs_list = list(range(len(return_list)))
        plt.plot(epochs_list, return_list)
        plt.xlabel('Epochs')
        plt.ylabel('Returns')
        plt.title('CQL on {} {}'.format(env_name, level))
        plt.savefig('{}/{}.png'.format(run_dir, level))
        np.save('{}/{}.npy'.format(run_dir, level), return_list)
        # plt.close()
