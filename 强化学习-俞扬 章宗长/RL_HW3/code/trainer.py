import math

import numpy as np
from config import Config
from torch.utils.tensorboard import SummaryWriter
from core.util import get_output_folder
import time
import imageio
from PIL import Image
class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = SummaryWriter(self.outputdir)
        print(self.outputdir)

    def train(self, pre_fr=0):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False
        start = time.time()
        state = self.env.reset()
        for fr in range(pre_fr + 1, self.config.frames + 1):
            if fr % self.config.gif_interval >= 1 and fr % self.config.gif_interval<=200:
                if fr % self.config.gif_interval == 1:
                    frames = []
                img = state[0, 0:3].transpose(1,2,0).astype('uint8')
                frames.append(Image.fromarray(img).convert('RGB'))
                if fr % self.config.gif_interval == 200:
                    imageio.mimsave(self.outputdir + '/record.gif', frames, 'GIF', duration=0.1) # @zhuangzh

            epsilon = self.epsilon_by_frame(fr)
            action = self.agent.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = 0
            if fr > self.config.init_buff and fr % self.config.learning_interval==0:
                loss = self.agent.learning(fr)
                losses.append(loss)
                self.board_logger.add_scalar('Loss per frame', loss, fr)


            if fr % self.config.print_interval == 0:
                print(
                    "TIME {}  num timesteps {}, FPS {} \n Loss {:.3f}, avrage reward {:.1f}"
                        .format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                                fr,
                                int(fr / (time.time() - start)),
                                loss, np.mean(all_rewards[-10:])))

            if fr % self.config.log_interval == 0:
                self.board_logger.add_scalar('Reward per episode', all_rewards[-1], ep_num)

            if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                self.agent.save_checkpoint(fr, self.outputdir)

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.add_scalar('Best 100-episodes average reward', avg_reward, ep_num)

                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                    is_win = True
                    self.agent.save_model(self.outputdir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                    if self.config.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
