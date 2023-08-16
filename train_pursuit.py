"""Self Play
"""

import argparse
import os
import random
import time

import tensorflow.compat.v1 as tf
import numpy as np
import wandb
import magent
from examples.pursuit_model.algo import spawn_ai
from examples.pursuit_model.algo import tools
from examples.pursuit_model.pursuit import play

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'attention_mfq', 'causal_mfq', 'ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--agent_density', type=float, default=0.04, help='set the density of agents')
    parser.add_argument('--save_every', type=int, default=10, help='decide the self-play update interval')
    parser.add_argument('--checkpoint_dir', type=str, help='required when use bi-network')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=1000, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--seed', type=int, default=0, help='setup random seed')
    parser.add_argument('--name', type=str, default='pursuit', help='name of WandB file', required=False)

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('pursuit', map_size=args.map_size)
    # env.set_render_dir(os.path.join(BASE_DIR, 'examples/pursuit_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp/{}'.format(args.name))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.name))

    if args.algo in ['mfq', 'mfac', 'attention_mfq', 'me_mfq']:
        use_mf = True
    else:
        use_mf = False

    start_from = 0

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-predator', args.max_steps),
              spawn_ai(args.algo, sess, env, handles[1], args.algo + '-prey', args.max_steps)]

    sess.run(tf.global_variables_initializer())
    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True)


    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
