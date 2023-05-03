from arguments import get_args
from algo import *
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from env import Make_Env
from gym_minigrid.wrappers import *
import scipy.misc
import sys

t = str(time.time())

def plot(record, info):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    import os
    os.makedirs(t + '-{}'.format(info), exist_ok=True)
    fig.savefig(t + '-{}/performance.png'.format(info))
    plt.close()


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {'steps': [0],
              'max': [0],
              'mean': [0],
              'min': [0]}

    # environment initial
    envs = Make_Env(env_mode=2)
    action_shape = envs.action_shape
    observation_shape = envs.state_shape
    print(action_shape, observation_shape)


    epsilon = 0.2
    alpha = 1
    gamma = 0.9999
    n = 100 # 采样的轨迹条数
    m = 0 # 转移训练的频率
    start_planning = 0 # 开始使用model based 提高样本利用率
    h = 1 # 一条轨迹执行的长度

    # agent initial
    # you should finish your agent with QAgent
    # e.g. agent = myQAgent()
    agent = QLearningAgent(alpha, gamma)  # @zhuangzh
    dynamics_model = DynaModel(8, 8, policy=agent)
    # dynamics_model = NetworkModel(8, 8, policy=agent)

    # start to train your agent
    for i in range(num_updates * 10):
        # an example of interacting with the environment
        obs = envs.reset()
        obs = obs.astype(int)
        for step in range(args.num_steps):
            # Sample actions with epsilon greedy policy
            if np.random.rand() < epsilon:
                action = envs.action_sample()
            else:
                action = agent.select_action(obs)

            # interact with the environment
            obs_next, reward, done, info = envs.step(action)
            obs_next = obs_next.astype(int)
            # add your Q-learning algorithm
            agent.update(obs, action, reward, obs_next) # @zhuangzh
            ''' 改进2 新增部分
            agent.qtable[str(obs)][action] = np.clip(agent.qtable[str(obs)][action], -100, 100)
            '''
            dynamics_model.store_transition(obs, action, reward, obs_next)
            obs = obs_next

            if done:
                obs = envs.reset()

        if i > start_planning:
            for _ in range(n):
                s, idx = dynamics_model.sample_state()
                # buf_tuple = dynamics_model.buffer[idx]
                for _ in range(h):
                    if np.random.rand() < epsilon:
                        a = envs.action_sample()
                    else:
                        a = agent.select_action(s)
                    s_ = dynamics_model.predict(s, a)
                    r = envs.R(s, a, s_)
                    done = envs.D(s, a, s_)
                    # add your Q-learning algorithm
                    agent.update(s, a, r, s_)
                    s = s_
                    if done:
                        break

        for _ in range(m):
            dynamics_model.train_transition(32)

        if (i + 1) % (args.log_interval) == 0:
            total_num_steps = (i + 1) * args.num_steps
            obs = envs.reset()
            obs = obs.astype(int)
            reward_episode_set = []
            reward_episode = 0.
            for step in range(args.test_steps):
                action = agent.select_action(obs)
                obs_next, reward, done, info = envs.step(action)
                reward_episode += reward
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0.
                    obs = envs.reset()

            end = time.time()
            print("TIME {} Updates {}, num timesteps {}, FPS {} \n avrage/min/max reward {:.1f}/{:.1f}/{:.1f}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    i, total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            plot(record, args.info)
            plt.close('all')  # 避免内存泄漏

            if record['mean'][-1] >= 92 and record['max'][-1] - record['min'][-1] <= 15:
                sys.exit(0)




if __name__ == "__main__":
    main()
