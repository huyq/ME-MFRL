from . import ppo
from . import q_learning
from . import sac


PPO = ppo.PPO
ME_MFPPO = ppo.ME_MFPPO
GRID_MFPPO = ppo.GRID_MFPPO
MAPPO = ppo.MAPPO
SAC = sac.SAC


def spawn_ai(algo_name, sess, env, handle, human_name, max_steps, order=4):
    if algo_name == 'ppo':
        model = PPO(sess, human_name, handle, env)
    elif algo_name == 'me_mfppo':
        model = ME_MFPPO(sess, human_name, handle, env, order=order)
    elif algo_name == 'grid_mfppo':
        model = GRID_MFPPO(sess, human_name, handle, env, order=order)
    elif algo_name == 'mappo':
        model = MAPPO(sess, human_name, handle, env, order=order)
    elif algo_name == 'sac':
        model = SAC(sess, human_name, handle, env, max_steps, memory_size=80000)
    return model
