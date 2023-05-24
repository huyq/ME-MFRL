import numpy as np 

class RandomActor:
    def __init__(self, env, handle):
        self.env = env
        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]

    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        agent_num = len(kwargs['state'][0])
        actions = np.random.randint(self.num_actions, size=agent_num)

        return actions, None

    def load(self, dir, step):
        pass
