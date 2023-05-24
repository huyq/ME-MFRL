import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import os

tf.disable_v2_behavior()


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'


class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError


class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value


class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.logps = []
        self.meanactions = []
        self.terminal = False

    def append(self, state, action, reward, done, value, logp, meanaction=None):
        self.states.append(state)
        if action is not None:
            self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)
        if done is not None:
            self.dones.append(done)
        if value is not None:
            self.values.append(value)
        if logp is not None:    
            self.logps.append(logp)
        if meanaction is not None:
            self.meanactions.append(meanaction)
        if done:
            self.terminal = True


class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        state = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        dones = kwargs['dones']
        values = kwargs['values']
        logps = kwargs['logps']
        ids = kwargs['ids']

        if self.use_mean:
            meanaction = kwargs['meanaction']
        
        for i in ids:
            entry = self.buffer.get(i)
            if entry is None:
                entry = EpisodesBufferEntry()
                self.buffer[i] = entry
            
            if self.use_mean:
                entry.append(state[i], acts[i], rewards[i], dones[i], values[i], logps[i], meanaction=meanaction[i])
            else:
                entry.append(state[i], acts[i], rewards[i], dones[i], values[i], logps[i])


    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()


class AgentMemory(object):
    def __init__(self, obs_dim, act_dim, max_len, use_mean=False):
        self.state = MetaBuffer((obs_dim,), max_len)
        self.actions = MetaBuffer((act_dim,), max_len)
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean

        if self.use_mean:
            self.meanaction = MetaBuffer((act_dim,), max_len)

    def append(self, state, act, reward, done, prob=None):
        self.state.append(np.array([state]))
        self.actions.append(np.array([act]))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([done], dtype=np.bool))

        if self.use_mean:
            self.meanaction.append(np.array([prob]))

    def pull(self):
        res = {
            'state': self.state.pull(),
            'acts': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'dones': self.terminals.pull(),
            'meanaction': None if not self.use_mean else self.meanaction.pull()
        }

        return res


class MemoryGroup(object):
    def __init__(self, obs_dim, act_dim, max_len, batch_size, sub_len, use_mean=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_dim = act_dim

        self.state = MetaBuffer((obs_dim,), max_len)
        self.actions = MetaBuffer((act_dim,), max_len)
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.meanaction = MetaBuffer((act_dim,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.state.append(kwargs['state'])
        self.actions.append(kwargs['acts'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['dones'])

        if self.use_mean:
            self.meanaction.append(kwargs['meanaction'])

        mask = np.where(kwargs['dones'] == True, False, True)
        mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.obs_dim, self.act_dim, self.sub_len, use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(state=kwargs['state'][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], done=kwargs['dones'][i], prob=kwargs['prob'][i])
            else:
                self.agent[_id].append(state=kwargs['state'][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], done=kwargs['dones'][i])

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            self._new_add += len(tmp['state'])
            self._flush(**tmp)
        self.agent = dict()  # clear

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries

        obs = self.state.sample(idx)
        obs_next = self.state.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
            return obs, actions, act_prob, obs_next, act_next_prob, rewards, dones, masks
        else:
            return obs, obs_next, dones, rewards, actions, masks

    def get_batch_num(self):
        print('\n[INFO] Length of buffer and new add:', len(self.state), self._new_add)
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.state)


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name, n_group=1):
        self.name_set = set()
        self.gra = tf.Graph()
        self.n_group = n_group

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with self.gra.as_default():
            self.sess = tf.Session(graph=self.gra, config=sess_config)
            self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
            self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        with self.gra.as_default():
            for name in name_list:
                if name in self.name_set:
                    raise Exception("You cannot define different operations with same name: `{}`".format(name))
                self.name_set.add(name)
                setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                     for i in range(self.n_group)])
                setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                             for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)

        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i], feed_dict={
                        getattr(self, key)[i]: value[i]}), global_step=step)
            else:
                self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0], feed_dict={
                        getattr(self, key)[0]: value}), global_step=step)


class Runner(object):
    def __init__(self, sess, env, handles, map_size, max_steps, models,
                play_handle, render_every=None, save_every=None, tau=None, log_name=None, log_dir=None, model_dir=None, train=False, use_moment=True):
        """Initialize runner

        Parameters
        ----------
        sess: tf.Session
            session
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train

        if self.train:
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            summary_items = ['ave_agent_reward', 'mean_reward', 'kill', "Sum_Reward", "Kill_Sum"]
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            assert isinstance(sess, tf.Session)
            self.sess = sess

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def run(self, variant_eps, iteration, mean_reward=None):
        # pass
        info = {'agent': {'mean_reward': 0.}}

        render = (iteration + 1) % self.render_every if self.render_every > 0 else False
        mean_rewards = self.play(env=self.env, n_round=iteration, map_size=self.map_size, max_steps=self.max_steps, handles=self.handles,
                    models=self.models, print_every=50, eps=variant_eps, render=render, train=self.train)


        info['mean_reward'] = mean_rewards

        # Change keys for logging both main and opponent
        log_info = dict()
        for key, value in info.items():
            log_info.update({key + 'tot_rew': value['mean_reward']})

        if self.train:
            print('\n[INFO] {}'.format(info))
            if self.save_every and (iteration + 1) % self.save_every == 0:
                print(Color.INFO.format('[INFO] Saving model ...'))
                self.models[0].save(self.model_dir + '-predator', iteration)
                self.models[1].save(self.model_dir + '-prey', iteration)
        else:
            mean_reward['predator'].append(info['predator']['mean_reward'])
            mean_reward['prey'].append(info['prey']['mean_reward'])
            print('\n[INFO] {0}'.format(info))


