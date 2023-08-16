"""Battle
"""

import argparse
import os
# import tensorflow as tf
import numpy as np
import magent

from examples.pursuit_model.algo import spawn_ai
from examples.pursuit_model.algo import tools
from examples.pursuit_model.pursuit import play

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, choices={'attention_mfq', 'causal_mfq','ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--prey', type=str, choices={'attention_mfq', 'causal_mfq', 'ac', 'mfac', 'mfq', 'il'}, help='indicate the opponent model')
    parser.add_argument('--pred_dir', type=str, help='the path of the algorithm')
    parser.add_argument('--prey_dir', type=str, help='the path of the opponent model')
    parser.add_argument('--n_round', type=int, default=50, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')
    parser.add_argument('--idx', nargs='*', required=True)

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('pursuit', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    pred_model_dir = os.path.join(BASE_DIR, 'data/models/{}-0'.format(args.pred_dir))
    prey_model_dir = os.path.join(BASE_DIR, 'data/models/{}-1'.format(args.prey_dir))
    log_dir = os.path.join(BASE_DIR,'data/tmp/{}'.format(args.pred_dir))

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.pred, sess, env, handles[0], args.pred + '-predator', args.max_steps),
              spawn_ai(args.prey, sess, env, handles[1], args.prey + '-prey', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(pred_model_dir, step=args.idx[0])
    models[1].load(prey_model_dir, step=args.idx[1])

    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, play, render_every=0, log_dir=log_dir)
    win_cnt = {'predator_ave': 0, 'prey_ave': 0}

    for k in range(0, args.n_round):
        runner.run(0.0, k, win_cnt=win_cnt)

    print('\n[*] >>> Reward: Predator[{0}] {1} Prey[{2}] {3}'.format(args.pred, win_cnt['predator_ave'] / args.n_round, args.prey, win_cnt['prey_ave'] / args.n_round))
