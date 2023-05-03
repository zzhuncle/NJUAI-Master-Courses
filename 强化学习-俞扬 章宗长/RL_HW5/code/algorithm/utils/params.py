import argparse


def get_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--policy', type=str, default='TD3')
    parser.add_argument('--start_episode', type=int, default=0, help='start')
    parser.add_argument('--log_path', type=str, default='./td3.log')

    parser.add_argument('--state_dim', type=int, default=17)
    parser.add_argument('--action_dim', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--action_clip', type=float, default=1.0)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--grad_norm_clip', type=float, default=10)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--act_noise', type=float, default=0.1)
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--num_update', type=int, default=40)
    parser.add_argument('--actor_update', type=int, default=1)
    parser.add_argument('--critic_update', type=int, default=2)
    parser.add_argument('--batch_size_mf', type=int, default=256)

    parser.add_argument('--batch_size_mb', type=int, default=256)
    parser.add_argument('--num_ensemble', type=int, default=2)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--norm_input_mb', type=bool, default=True)
    parser.add_argument('--state_clip', type=float, default=100)

    parser.add_argument('--rollout_length', type=int, default=5)
    parser.add_argument('--batch_size_br', type=int, default=256)
    parser.add_argument('--batch_size_svg', type=int, default=32)
    parser.add_argument('--epoch_svg_mb', type=int, default=5)
    parser.add_argument('--epoch_svg_mf', type=int, default=500)
    parser.add_argument('--total_samples_mb', type=int, default=100000)

    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--eval_episode', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=5e3)
    parser.add_argument('--total_steps', type=int, default=2e6)
    parser.add_argument('--collect_steps', type=int, default=10000)
    parser.add_argument('--start_steps_mf', type=int, default=1000)
    parser.add_argument('--start_steps_mb', type=int, default=0)
    parser.add_argument('--model_steps', type=int, default=1000)
    parser.add_argument('--policy_steps', type=int, default=1)

    parser.add_argument('--update_interval', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=10000)
    parser.add_argument('--device', type=str, default='gpu')

    args = parser.parse_args()
    return args