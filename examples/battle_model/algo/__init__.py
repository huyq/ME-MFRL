from . import ac
from . import q_learning
from . import random_actor

AC = ac.ActorCritic
MFAC = ac.MFAC
IL = q_learning.DQN
MFQ = q_learning.MFQ
AttMFQ = q_learning.AttentionMFQ
MEMFQ = q_learning.MEMFQ
MEMFQ_LEG = q_learning.MEMFQ_LEG
RandomActor = random_actor.RandomActor


def spawn_ai(algo_name, sess, env, handle, human_name, max_steps, moment_order=4):
    if algo_name == 'mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'attention_mfq':
        model = AttMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env)
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'me_mfq':
        model = MEMFQ(sess, human_name, handle, env, max_steps, memory_size=80000,moment_order=moment_order)
    elif algo_name == 'me_mfq_leg':
        model = MEMFQ_LEG(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'random':
        model = RandomActor(env, handle)
    return model
