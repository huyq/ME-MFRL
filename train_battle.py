"""Self Play
"""

import argparse
import os
import random
import time

import tensorflow.compat.v1 as tf
import numpy as np
# import wandb
import magent
from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.senario_battle import play

os.environ["WANDB_START_METHOD"] = "thread"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

tf.disable_v2_behavior()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int = 42) -> None:
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_random_seed(seed)
  tf.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")

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
    parser.add_argument('--algo', type=str, choices={'attention_mfq', 'ac', 'mfac', 'mfq', 'il', 'me_mfq','me_mfq_leg'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--save_every', type=int, default=50, help='decide the self-play update interval')
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--order', type=int, default=4, help='moment order')

    args = parser.parse_args()

    # set_seed(args.seed)
    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    # env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    log_dir = os.path.join(BASE_DIR,'data/tmp/{}_{}'.format(args.algo, args.seed))
    model_dir = os.path.join(BASE_DIR, 'data/models/{}_{}'.format(args.algo, args.seed))

    if args.algo == 'me_mfq':
        log_dir = os.path.join(BASE_DIR,f'data/tmp/{args.algo}_{args.order}_{args.seed}')
        model_dir = os.path.join(BASE_DIR, f'data/models/{args.algo}_{args.order}_{args.seed}')

    if args.algo in ['mfq', 'mfac', 'attention_mfq','me_mfq']:
        use_mf = True
    else:
        use_mf = False

    start_from = 0

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps, args.order),
              spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent', args.max_steps, args.order)]

    sess.run(tf.global_variables_initializer())
    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, seed=args.seed, train=True)


    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
