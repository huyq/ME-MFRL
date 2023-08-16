"""Self Play
"""

import argparse
import os
import random
import time

import tensorflow.compat.v1 as tf
import numpy as np
from examples.spread_model.algo import spawn_ai
from examples.spread_model.algo import tools
from examples.spread_model.senario_spread import play
from examples.spread_model.algo.tools import Color

from env.mpe.make_env import make_env

os.environ["WANDB_START_METHOD"] = "thread"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

tf.disable_v2_behavior()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v
    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break
    return eps

def test_env(env):
    print('action dim:', env.action_space[0].shape[0], env.action_space[1].shape[0])
    print('obs dim:', env.observation_space[0].shape[0], env.observation_space[1].shape[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ppo', 'me_mfppo', 'quantile_mfppo', 'mappo', 'sac'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--agent_density', type=float, default=0.04, help='set the density of agents')
    parser.add_argument('--save_every', type=int, default=50, help='decide the self-play update interval')
    parser.add_argument('--checkpoint_dir', type=str, help='required when use bi-network')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=25, help='set the max steps')
    parser.add_argument('--seed', type=int, default=0, help='setup random seed')
    parser.add_argument('--order', type=int, default=4, help='moment order')
    parser.add_argument('--name', type=str, help='name of WandB file', required=False)

    args = parser.parse_args()

    # Initialize the environment

    setup_seed(args.seed)
    
    env = make_env('exp_spread')
    
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp/{}_{}'.format(args.algo, args.seed))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}_{}'.format(args.algo, args.seed))
    if 'mf' in args.algo:
        log_dir = os.path.join(BASE_DIR,f'data/tmp/{args.algo}_{args.order}_{args.seed}')
        model_dir = os.path.join(BASE_DIR, f'data/models/{args.algo}_{args.order}_{args.seed}')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    start_from = 0

    sess = tf.Session(config=tf_config)
    model = spawn_ai(args.algo, sess, env, None, args.algo + 'main', args.max_steps)

    sess.run(tf.global_variables_initializer())



    for k in range(start_from, start_from + args.n_round):
        mean_rewards = play(env=env, n_round=k, map_size=args.map_size, max_steps=args.max_steps, handles=None,
                    model=model, print_every=50, eps=0, render=False, train=True)
        
        info = {'agent': {'mean_reward': mean_rewards}}
        print('\n[INFO] {}'.format(info))

        if (k + 1) % args.save_every == 0:
            print(Color.INFO.format('[INFO] Saving model ...'))
            model.save(model_dir + '-main', k)

    
