"""Self Play
"""

import argparse
import os
import random
import time

import tensorflow.compat.v1 as tf
import numpy as np
from examples.tag_model.algo import spawn_ai
from examples.tag_model.algo import tools
from examples.tag_model.senario_tag import play

from env.mpe.make_env import make_env

os.environ["WANDB_START_METHOD"] = "thread"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

tf.disable_v2_behavior()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_seed(seed=42):
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
    parser.add_argument('--algo', type=str, choices={'ppo', 'me_mfppo', 'grid_mfppo', 'mappo', 'sac'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--agent_density', type=float, default=0.04, help='set the density of agents')
    parser.add_argument('--save_every', type=int, default=50, help='decide the self-play update interval')
    parser.add_argument('--checkpoint_dir', type=str, help='required when use bi-network')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=1000, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=100, help='set the max steps')
    parser.add_argument('--seed', type=int, default=0, help='setup random seed')
    parser.add_argument('--order', type=int, default=4, help='moment order')
    parser.add_argument('--name', type=str, help='name of WandB file', required=False)

    args = parser.parse_args()

    setup_seed(seed=args.seed)
    # Initialize the environment
    
    env = make_env('exp_tag')
    
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp/{}_{}'.format(args.algo, args.seed))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}_{}'.format(args.algo, args.seed))

    if 'mf' in args.algo:
        log_dir = os.path.join(BASE_DIR,f'data/tmp/{args.algo}_{args.order}_{args.seed}')
        model_dir = os.path.join(BASE_DIR, f'data/models/{args.algo}_{args.order}_{args.seed}')

    start_from = 0

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, None, args.algo + 'predator', args.max_steps, args.order),
              spawn_ai(args.algo, sess, env, None, args.algo + 'prey', args.max_steps, args.order)]

    sess.run(tf.global_variables_initializer())
    runner = tools.Runner(sess, env, None, args.map_size, args.max_steps, models, play, 
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True)


    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
    
