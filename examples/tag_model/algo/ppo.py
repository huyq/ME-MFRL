import os
import numpy as np
import tensorflow.compat.v1 as tf
from scipy import signal

from . import tools

EPS = 1e-8
LOG_STD = -1.7

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPO:
    def __init__(self, sess, name, handle, env, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4):
        self.sess = sess

        self.name = name
        self.state_dim = env.observation_space[0].shape[0] if 'predator' in name else env.observation_space[-1].shape[0]
        self.action_dim = env.action_space[0].shape[0] if 'predator' in name else env.action_space[-1].shape[0]
        self.gamma = gamma
        self.lam = 0.99
        self.target_kl = 0.01
        self.clip_ratio = 0.2

        self.batch_size = batch_size
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.actor_update_steps = 2
        self.critic_update_steps = 2
        self.actor_global_step = tf.Variable(0, trainable=False)
        self.critic_global_step = tf.Variable(0, trainable=False)


        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.replay_buffer = tools.EpisodesBuffer()

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network()
        
        self.sess.run(tf.global_variables_initializer())
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    
    def act(self, **kwargs):
        inputs = {self.s_ph:kwargs['state']}  
        pi, v, logp_pi = self.sess.run([self.pi, self.v, self.logp_pi], inputs)
        a = np.clip(pi, -1, 1)
        return a, v, logp_pi
    
    def act_test(self, **kwargs): 
        inputs = {self.s_ph:kwargs['state']} 
        a = self.sess.run(self.mu, inputs)
        a = np.clip(a, -1, 1)
        return a, None, None
    
    def get_v(self, **kwargs):
        inputs = {self.s_ph:kwargs['state']}  
        return self.sess.run(self.v, inputs)[0, 0]

    def _create_network(self):
        self.s_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.a_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.adv_ph = tf.placeholder(tf.float32, (None,))
        self.ret_ph = tf.placeholder(tf.float32, (None,))
        self.logp_old_ph = tf.placeholder(tf.float32, (None,))

        self.all_phs = [self.s_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]

        # actor critic
        x_pi = tf.layers.dense(self.s_ph, 256, tf.nn.relu)
        x_pi = tf.layers.dense(x_pi, 64, tf.nn.relu)
        mu = tf.layers.dense(x_pi, self.action_dim, tf.nn.tanh)
        log_std = tf.get_variable(name='log_std', initializer=LOG_STD*np.ones(self.action_dim, dtype=np.float32))
        std = tf.exp(log_std)
        self.pi = mu + tf.random_normal(tf.shape(mu)) * std
        self.logp = gaussian_likelihood(self.a_ph, mu, log_std)
        self.logp_pi = gaussian_likelihood(self.pi, mu, log_std)
        self.mu = mu
    
        x_v = tf.layers.dense(self.s_ph, 256, tf.nn.relu)
        self.v = tf.layers.dense(x_v, 1)

        # loss
        ratio = tf.exp(self.logp - self.logp_old_ph)
        min_adv = tf.where(self.adv_ph>0, (1+self.clip_ratio)*self.adv_ph, (1-self.clip_ratio)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        # info
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)      # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)                  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1+self.clip_ratio), ratio < (1-self.clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # train_op
        # self.train_pi = tf.train.AdamOptimizer(learning_rate=self.actor_lr).minimize(self.pi_loss)
        # self.train_v = tf.train.AdamOptimizer(learning_rate=self.critic_lr).minimize(self.v_loss)
        actor_lr_decated = tf.train.linear_cosine_decay(self.actor_lr, self.actor_global_step, 1000)
        critic_lr_decated = tf.train.linear_cosine_decay(self.critic_lr, self.critic_global_step, 1000)
        self.train_pi = tf.train.AdamOptimizer(learning_rate=actor_lr_decated).minimize(self.pi_loss, self.actor_global_step)
        self.train_v = tf.train.AdamOptimizer(learning_rate=critic_lr_decated).minimize(self.v_loss, self.critic_global_step)



    def train(self):
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer()

        # calc buffer size
        n = 0
        for episode in batch_data:
            n += len(episode.rewards)

        state_buf = np.empty((n,) + (self.state_dim,), dtype=np.float32)
        act_buf = np.empty((n,) + (self.action_dim,), dtype=np.float32)
        adv_buf = np.empty(n,dtype=np.float32)
        ret_buf = np.empty(n,dtype=np.float32)
        logp_buf = np.empty(n,dtype=np.float32)

        ptr = 0
        # collect episodes from multiple separate buffers to a continuous buffer
        for episode in batch_data:
            state = episode.states
            action = episode.actions
            reward = episode.rewards
            done = np.asarray(episode.dones,dtype=np.bool)
            value = episode.values
            logp_ = episode.logps

            T = len(reward)

            bootstrap_value = self.sess.run(self.v, feed_dict={self.s_ph: [state[-1]],})[0]
            value = np.append(value, bootstrap_value)
            discount = (~done).astype(np.float32)*self.gamma
            delta = reward + discount*value[1:] - value[:-1]
            advantage = discount_cumsum(delta, self.gamma * self.lam)
            return_ = advantage + value[:-1]

            state_buf[ptr:ptr+T] = state[:-1]
            act_buf[ptr:ptr+T] = action
            adv_buf[ptr:ptr+T] = advantage
            ret_buf[ptr:ptr+T] = return_
            logp_buf[ptr:ptr+T] = logp_

            ptr += T

        assert n == ptr

        # train
        inputs = {
            self.s_ph: state_buf,
            self.a_ph: act_buf,
            self.adv_ph: adv_buf, 
            self.ret_ph: ret_buf, 
            self.logp_old_ph: logp_buf,
        }
        pg_loss, vf_loss, ent_loss = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)
        
        for i in range(self.actor_update_steps):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
        
        for _ in range(self.critic_update_steps):
            self.sess.run(self.train_v, feed_dict=inputs)
        

        print('[*] PG_LOSS:', np.round(pg_loss, 6), '/ VF_LOSS:', np.round(vf_loss, 6), '/ ENT_LOSS:', np.round(ent_loss))

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


class MFPPO:
    def __init__(self, sess, name, handle, env, value_coef=0.1, ent_coef=0.08, gamma=0.95, batch_size=64, learning_rate=1e-4, order=4):
        self.sess = sess

        self.name = name
        self.state_dim = env.observation_space[0].shape[0] if 'predator' in name else env.observation_space[-1].shape[0]
        self.action_dim = env.action_space[0].shape[0] if 'predator' in name else env.action_space[-1].shape[0]
        self.order = order
        self.moment_dim = order**self.action_dim if 'grid' in name else order*self.action_dim
        self.gamma = gamma
        self.lam = 0.99
        self.target_kl = 0.01
        self.clip_ratio = 0.2

        self.batch_size = batch_size
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.actor_update_steps = 2
        self.critic_update_steps = 2
        self.actor_global_step = tf.Variable(0, trainable=False)
        self.critic_global_step = tf.Variable(0, trainable=False)


        self.value_coef = value_coef  # coefficient of value in the total loss
        self.ent_coef = ent_coef  # coefficient of entropy in the total loss

        # init training buffers
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True, use_g=True)

        with tf.variable_scope(name):
            self.name_scope = tf.get_variable_scope().name
            self._create_network()
    
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)
    
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)
    
    def act(self, **kwargs):
        inputs = {self.s_ph:kwargs['state'], self.m_ph:kwargs['meanaction'], self.g_ph:kwargs['g']} 
        pi, v, logp_pi = self.sess.run([self.pi, self.v, self.logp_pi], inputs)
        a = np.clip(pi, -1, 1)
        return a, v, logp_pi
     
    def act_test(self, **kwargs): 
        inputs = {self.s_ph:kwargs['state'], self.m_ph:kwargs['meanaction']} 
        a = self.sess.run(self.mu, inputs)
        a = np.clip(a, -1, 1)
        return a, None, None
        
    
    def get_v(self, **kwargs):
        inputs = {self.s_ph:kwargs['state'], self.m_ph:kwargs['meanaction']}  
        return self.sess.run(self.v, inputs)[0, 0]

    def _create_network(self):
        print(self.state_dim, self.moment_dim, self.action_dim)
        self.s_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.m_ph = tf.placeholder(tf.float32, [None, self.moment_dim])
        self.a_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.g_ph = tf.placeholder(tf.float32, [None, self.moment_dim])
        self.adv_ph = tf.placeholder(tf.float32, (None,))
        self.ret_ph = tf.placeholder(tf.float32, (None,))
        self.logp_old_ph = tf.placeholder(tf.float32, (None,))

        self.all_phs = [self.s_ph, self.a_ph, self.adv_ph, self.ret_ph, self.logp_old_ph]


        # actor 

        # pi(a|s,mu)
        # x_pi = tf.layers.dense(self.s_ph, 256, tf.nn.relu)
        # x_moment_pi = tf.layers.dense(self.m_ph, 64, tf.nn.relu)
        # x_concat_pi = tf.concat([x_pi, x_moment_pi], axis=1)
        # mu = tf.layers.dense(x_concat_pi, self.action_dim, tf.nn.tanh)


        # pi(a|s)
        x_pi = tf.layers.dense(self.s_ph, 256, tf.nn.relu)
        x_pi = tf.layers.dense(x_pi, 64, tf.nn.relu)
        mu = tf.layers.dense(x_pi, self.action_dim, tf.nn.tanh)

        log_std = tf.get_variable(name='log_std', initializer=LOG_STD*np.ones(self.action_dim, dtype=np.float32))
        std = tf.exp(log_std)
        self.pi = mu + tf.random_normal(tf.shape(mu)) * std
        self.logp = gaussian_likelihood(self.a_ph, mu, log_std)
        self.logp_pi = gaussian_likelihood(self.pi, mu, log_std)
        self.mu = mu

        # critic
        x_v = tf.layers.dense(self.s_ph, 256, tf.nn.relu)
        x_moment = tf.layers.dense(self.m_ph, 64, tf.nn.relu)
        x_concat = tf.concat([x_v, x_moment], axis=1)
        h = tf.layers.dense(x_concat, 1)
        delta_h = tf.gradients(h, self.m_ph)[0]
        G = tf.keras.backend.batch_dot(self.g_ph, delta_h)
        self.v = h+G

        # loss
        ratio = tf.exp(self.logp - self.logp_old_ph)
        min_adv = tf.where(self.adv_ph>0, (1+self.clip_ratio)*self.adv_ph, (1-self.clip_ratio)*self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        # info
        self.approx_kl = tf.reduce_mean(self.logp_old_ph - self.logp)      # a sample estimate for KL-divergence, easy to compute
        self.approx_ent = tf.reduce_mean(-self.logp)                  # a sample estimate for entropy, also easy to compute
        clipped = tf.logical_or(ratio > (1+self.clip_ratio), ratio < (1-self.clip_ratio))
        self.clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # train_op
        # self.train_pi = tf.train.AdamOptimizer(learning_rate=self.actor_lr).minimize(self.pi_loss)
        # self.train_v = tf.train.AdamOptimizer(learning_rate=self.critic_lr).minimize(self.v_loss)
        actor_lr_decated = tf.train.linear_cosine_decay(self.actor_lr, self.actor_global_step, 1000)
        critic_lr_decated = tf.train.linear_cosine_decay(self.critic_lr, self.critic_global_step, 1000)
        self.train_pi = tf.train.AdamOptimizer(learning_rate=actor_lr_decated).minimize(self.pi_loss, self.actor_global_step)
        self.train_v = tf.train.AdamOptimizer(learning_rate=critic_lr_decated).minimize(self.v_loss, self.critic_global_step)



    def train(self):
        batch_data = self.replay_buffer.episodes()
        self.replay_buffer = tools.EpisodesBuffer(use_mean=True, use_g=True)

        # calc buffer size
        n = 0
        for episode in batch_data:
            n += len(episode.rewards)

        state_buf = np.empty((n,) + (self.state_dim,), dtype=np.float32)
        act_buf = np.empty((n,) + (self.action_dim,), dtype=np.float32)
        adv_buf = np.empty(n,dtype=np.float32)
        ret_buf = np.empty(n,dtype=np.float32)
        logp_buf = np.empty(n,dtype=np.float32)
        meanaction_buf = np.empty((n,) + (self.moment_dim,), dtype=np.float32)
        g_buf = np.empty((n,) + (self.moment_dim,), dtype=np.float32)

        ptr = 0
        # collect episodes from multiple separate buffers to a continuous buffer
        for episode in batch_data:
            state = episode.states
            action = episode.actions
            reward = episode.rewards
            done = np.asarray(episode.dones,dtype=np.bool)
            value = episode.values
            logp_ = episode.logps
            meanaction = episode.meanactions
            g = episode.g

            T = len(reward)
            bootstrap_value = self.sess.run(self.v, feed_dict={self.s_ph: [state[-1]], self.m_ph: [meanaction[-1]], self.g_ph:[g[-1]]})[0]
            value = np.append(value, bootstrap_value)
            discount = (~done).astype(np.float32)*self.gamma
            delta = reward + discount*value[1:] - value[:-1]
            advantage = discount_cumsum(delta, self.gamma * self.lam)
            return_ = advantage + value[:-1]

            state_buf[ptr:ptr+T] = state[:-1]
            act_buf[ptr:ptr+T] = action
            adv_buf[ptr:ptr+T] = advantage
            ret_buf[ptr:ptr+T] = return_
            logp_buf[ptr:ptr+T] = logp_
            meanaction_buf[ptr:ptr+T] = meanaction[:-1]
            g_buf[ptr:ptr+T] = g[:-1]

            ptr += T

        assert n == ptr

        # train
        inputs = {
            self.s_ph: state_buf,
            self.m_ph: meanaction_buf,
            self.a_ph: act_buf,
            self.adv_ph: adv_buf, 
            self.ret_ph: ret_buf, 
            self.logp_old_ph: logp_buf,
            self.g_ph: g_buf,
        }
        pg_loss, vf_loss, ent_loss = self.sess.run([self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)
        
        for i in range(self.actor_update_steps):
            _, kl = self.sess.run([self.train_pi, self.approx_kl], feed_dict=inputs)
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
        
        for _ in range(self.critic_update_steps):
            self.sess.run(self.train_v, feed_dict=inputs)
        
        # train_steps = self.sess.run([self.actor_global_step,self.critic_global_step])
        # print('golbal steps:', train_steps)
        print('[*] PG_LOSS:', np.round(pg_loss, 6), '/ VF_LOSS:', np.round(vf_loss, 6), '/ ENT_LOSS:', np.round(ent_loss))

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