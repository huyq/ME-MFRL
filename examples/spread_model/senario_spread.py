"""
Pursuit: predators get reward when they attack prey.
"""

import argparse
import logging
import time
import logging as log
import numpy as np


from env.mpe.make_env import make_env

logging.basicConfig(level=logging.ERROR)

num_pred = 30
num_prey = 10

# order = 4
# moment_dim = order * 2
# bin_dim = order ** 2

def _calc_bin_density(a, order=4, action_max=1.0, action_min=-1.0, eps=1e-4):
    bin_density = np.zeros(shape=(order, order), dtype=np.float)
    action_tmp = (a - action_min) / (action_max - action_min) * (order - eps-4)
    action_tmp = np.floor(action_tmp).astype(np.int)

    for i in range(len(action_tmp)):
        bin_density[action_tmp[i][0], action_tmp[i][1]] += 1

    bin_density = bin_density.flatten()
    bin_density = bin_density / np.sum(bin_density)
    return bin_density


def _calc_moment(a, order=4):
    moments = []
    for i in range(1, order+1):
        moment_1 = np.mean(np.power(a[:,0], i))
        moment_1 = np.sign(moment_1) * np.power(np.abs(moment_1), 1/i)
        moment_2 = np.mean(np.power(a[:,1], i))
        moment_2 = np.sign(moment_2) * np.power(np.abs(moment_2), 1/i)
        moments.append(moment_1)
        moments.append(moment_2) 
    moment = np.array(moments)
    return moment.reshape(1,-1)


def play(env, n_round, map_size, max_steps, handles, model, print_every=10, record=False, render=False, eps=None, train=False):
    
    step_ct = 0
    done = False
    rewards = 0
    action_dim = env.action_space[0].shape[0]
    num_agents = len(env.agents)

    obs = env.reset()
    acts = np.zeros((num_agents,action_dim), dtype=np.float32)
    values = np.zeros((num_agents,), dtype=np.float32)
    logprobs = np.zeros((num_agents,), dtype=np.float32)

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, num_agents))
    mean_rewards = []
    total_rewards = []

    former_meanaction = []

    if 'mf' in model.name:
        former_meanaction = np.zeros((1, model.moment_dim))
    elif 'ma' in model.name:
        former_meanaction = np.zeros((1, num_agents*action_dim))

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        if 'mf' in model.name or 'ma' in model.name:
            former_meanaction = np.tile(former_meanaction, (num_agents, 1))
        acts, values, logprobs = model.act(state=obs, meanaction=former_meanaction)

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        obs, rewards, dones, _ = env.step(stack_act)
        done = all(dones)

        buffer = {
            'state': old_obs, 
            'acts': acts, 
            'rewards': rewards, 
            'dones': dones,
            'values': values, 
            'logps': logprobs,
            'ids': range(num_agents), 
        }
        if 'mf' in model.name:
            buffer['meanaction'] = former_meanaction
        if 'ma' in model.name:
            buffer['meanaction'] = former_meanaction
        if 'sac' in model.name:
            buffer['next_state'] = obs

  

        model.flush_buffer(**buffer)

        #############################
        # Calculate mean field #
        #############################
        if 'quantile' in model.name:
            former_meanaction = _calc_bin_density(acts, model.order)
        elif 'mf' in model.name:
            former_meanaction = _calc_moment(acts, model.order)
        elif 'ma' in model.name:
            former_meanaction = stack_act


        sum_reward = sum(rewards)
        total_rewards.append(sum_reward)

        if render:
            env.render()

        info = {}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    if 'ppo' in model.name:
        buffer = {
            'state': obs, 
            'acts': [None for i in range(num_agents)], 
            'rewards': [None for i in range(num_agents)], 
            'dones': [None for i in range(num_agents)],
            'values': [None for i in range(num_agents)], 
            'logps': [None for i in range(num_agents)],
            'ids': range(num_agents), 
        }
        if 'mf' in model.name:
            buffer['meanaction'] = np.tile(former_meanaction, (num_agents, 1))
        
        if 'ma' in model.name:
            buffer['meanaction'] = np.tile(former_meanaction, (num_agents, 1))
        
    
        model.flush_buffer(**buffer)
    
    model.train()

    mean_rewards = sum(total_rewards)/num_agents

    return mean_rewards


def test(env, n_round, map_size, max_steps, handles, model, print_every=10, record=False, render=False, eps=None, train=False):
    step_ct = 0
    done = False
    rewards = 0
    action_dim = env.action_space[0].shape[0]
    num_agents = len(env.agents)

    obs = env.reset()
    acts = np.zeros((num_agents,action_dim), dtype=np.float32)
    values = np.zeros((num_agents,), dtype=np.float32)
    logprobs = np.zeros((num_agents,), dtype=np.float32)


    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, num_agents))
    mean_rewards = []
    total_rewards = []

    former_meanaction = []
    if 'mf' in model.name:
        former_meanaction = np.zeros((1, model.moment_dim))
    elif 'ma' in model.name:
        former_meanaction = np.zeros((1, num_agents*action_dim))


    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        former_meanaction = np.tile(former_meanaction, (num_agents, 1))
        acts, values, logprobs = model.act(state=obs, meanaction=former_meanaction)


        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        obs, rewards, dones, _ = env.step(stack_act)
        done = all(dones)

        #############################
        # Calculate mean field #
        #############################
        if 'quantile' in model.name:
            former_meanaction = _calc_bin_density(acts, model.order)
        elif 'mf' in model.name:
            former_meanaction = _calc_moment(acts, model.order)
        elif 'ma' in model.name:
            former_meanaction = stack_act
                


        sum_reward = sum(rewards)
        total_rewards.append(sum_reward)
        
        info = {}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))


    mean_rewards = sum(total_rewards)/num_agents

    return mean_rewards
