import os
import numpy as np
import tensorflow.compat.v1 as tf
from scipy import signal

from . import tools

EPS = 1e-8
LOG_STD_MAX = -1
LOG_STD_MIN = -10

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class SAC:
    def __init__(self, sess, name, handle, env, sub_len, memory_size=2**10, batch_size=1024, update_every=1):
        self.sess = sess

        self.name = name
        self.state_dim = env.observation_space[0].shape[0] if 'predator' in name else env.observation_space[-1].shape[0]
        self.action_dim = env.action_space[0].shape[0] if 'predator' in name else env.action_space[-1].shape[0]
        self.tau = 0.995
        self.gamma = 0.99
        self.alpha = 0.2
        self.action_scale = 1
        self.hidden_sizes = (256,64)

        self.batch_size = batch_size
        self.update_every = update_every
        # self.lr = 2e-4
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.actor_global_step = tf.Variable(0, trainable=False)
        self.critic_global_step = tf.Variable(0, trainable=False)

        # init training buffers
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, memory_size)

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network()
        
        self.sess.run(tf.global_variables_initializer())
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]
    

    def _flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    

    def flush_buffer(self, **kwargs):
        for i in kwargs['ids']:
            obs = kwargs['state'][i]
            act = kwargs['acts'][i]
            rew = kwargs['rewards'][i]
            next_obs = kwargs['next_state'][i]
            done = kwargs['dones'][i]
            self.replay_buffer.store(obs, act, rew, next_obs, done)
    
    def act(self, **kwargs):
        feed_dict = {self.s_ph:kwargs['state'],
                     self.a_ph:np.zeros((len(kwargs['state']),self.action_dim))}  
        # act_op = self.mu if deterministic else self.pi
        _a = self.sess.run([self.pi], feed_dict)[0]
        a = np.clip(_a, -1, 1)
        return a, None, None
    
    
    def actor_critic(self, x, a, hidden_sizes):
        # pi
        with tf.variable_scope('pi'):
            x_pi = mlp(x, list(hidden_sizes), activation=tf.nn.relu, output_activation=tf.nn.relu)
            mu = tf.layers.dense(x_pi, self.action_dim, tf.nn.tanh)
            log_std = tf.layers.dense(x_pi, self.action_dim, activation=None)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_pi = gaussian_likelihood(pi, mu, log_std)
            logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
            mu = tf.tanh(mu)
            pi = tf.tanh(pi)

        mu *= self.action_scale
        pi *= self.action_scale
        
        # vfs
        vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], tf.nn.relu, None), axis=1)
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([x,a], axis=-1))
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([x,a], axis=-1))

        return mu, pi, logp_pi, q1, q2

    def _create_network(self):
        self.s_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.s2_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.a_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.r_ph = tf.placeholder(tf.float32, (None,))
        self.d_ph = tf.placeholder(tf.float32, (None,))
    

        # self.all_phs = [self.s_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        with tf.variable_scope('main'):
            mu, pi, logp_pi, q1, q2 = self.actor_critic(self.s_ph, self.a_ph, self.hidden_sizes)
            self.eval_name = tf.get_variable_scope().name
            self.e_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)
            self.mu = mu
            self.pi = pi

        with tf.variable_scope('main', reuse=True):
            _, _, _, q1_pi, q2_pi = self.actor_critic(self.s_ph, pi, self.hidden_sizes)
            _, pi_next, logp_pi_next, _, _ = self.actor_critic(self.s2_ph, self.a_ph, self.hidden_sizes)

        with tf.variable_scope('target'):     
            _, _, _, q1_targ, q2_targ  = self.actor_critic(self.s2_ph, pi_next, self.hidden_sizes)
            self.target_name = tf.get_variable_scope().name
            self.t_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
        q_backup = tf.stop_gradient(self.r_ph + self.gamma*(1-self.d_ph)*(min_q_targ - self.alpha * logp_pi_next))

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(self.alpha * logp_pi - min_q_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
        value_loss = q1_loss + q2_loss

        actor_lr_decated = tf.train.linear_cosine_decay(self.actor_lr, self.actor_global_step, 1000)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr_decated)
        train_pi_op = pi_optimizer.minimize(pi_loss, self.actor_global_step, var_list=self.get_vars('main/pi'))

        critic_lr_decated = tf.train.linear_cosine_decay(self.critic_lr, self.critic_global_step, 1000)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr_decated)
        value_params = self.get_vars('main/q')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, self.critic_global_step, var_list=value_params)
        
        with tf.control_dependencies([train_value_op]):
            target_update = [tf.assign(self.t_variables[i], self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                    for i in range(len(self.t_variables))]
            
        
        self.step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, 
                train_pi_op, train_value_op, target_update]


    

    def _train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, obs_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            feed_dict = {self.s_ph: obs,
                         self.s2_ph: obs_next,
                         self.a_ph: actions,
                         self.r_ph: rewards,
                         self.d_ph: dones,
                        }
            
            outs = self.sess.run(self.step_ops, feed_dict)

            if i % 50 == 0:
                print('[*] LossPi:', np.round(outs[0], 6), '/ LossQ1:', np.round(outs[1], 6), '/ LossQ2:', np.round(outs[2], 6),
                  '/ Q1Vals:', np.round(outs[3], 6), '/ Q2Vals:', np.round(outs[4], 6), '/ LogPi:', np.round(outs[5], 6)) 
        
    

    def train(self):
        for _ in range(self.update_every):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            feed_dict = {self.s_ph: batch['obs1'],
                         self.s2_ph: batch['obs2'],
                         self.a_ph: batch['acts'],
                         self.r_ph: batch['rews'],
                         self.d_ph: batch['done'],
                        }

            outs = self.sess.run(self.step_ops, feed_dict)

            print('[*] LossPi:', np.round(outs[0], 6), '/ LossQ1:', np.round(outs[1], 6), '/ LossQ2:', np.round(outs[2], 6),
                  '/ Q1Vals:', np.round(outs[3], 6), '/ Q2Vals:', np.round(outs[4], 6), '/ LogPi:', np.round(outs[5], 6)) 


    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "ac_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "ac_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))