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

def _cal_ca(a, order=4):
    c_a = []
    for i in range(a.shape[1]):
        mean_action = np.mean(a[:,i])
        deviation = mean_action - a[:,i]
        m = np.array([j*a[:,i]**(j-1) for j in range(1,order+1)])
        c = m@deviation.T 
        c_a.append(c)

    c_a = np.array(c_a).T
    return c_a.reshape(1,-1)



def play(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, eps=None, train=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    max_nums = [num_pred, num_prey]  # num_pred predators, 40 prey


    action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]

    all_obs = env.reset()
    obs = [all_obs[:num_pred], all_obs[num_pred:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],action_dim[i]), dtype=np.int32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_meanaction = [[] for _ in range(n_group)]
    former_g = [[] for _ in range(n_group)]
    for i in range(n_group):
        if 'me' in models[i].name:
            former_g[i] = np.zeros((1, models[i].moment_dim))
        if 'mf' in models[i].name:
            former_meanaction[i] = np.zeros((1, models[i].moment_dim))
        elif 'ma' in models[i].name:
            former_meanaction[i] = np.zeros((1, max_nums[i]*models[i].action_dim))

    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        for i in range(n_group):
            if 'mf' in models[i].name or 'ma' in models[i].name:
                former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
            if 'me' in models[i].name:
                former_g[i] = np.tile(former_g[i], (max_nums[i], 1))
            acts[i], values[i], logprobs[i] = models[i].act(state=obs[i], meanaction=former_meanaction[i], g=former_g[i])

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, _ = env.step(stack_act)
        obs = [all_obs[:num_pred], all_obs[num_pred:]]
        rewards = [all_rewards[:num_pred], all_rewards[num_pred:]]
        done = all(all_done)

        predator_buffer = {
            'state': old_obs[0], 
            'acts': acts[0], 
            'rewards': rewards[0], 
            'dones': all_done[:num_pred],
            'values': values[0], 
            'logps': logprobs[0],
            'ids': range(max_nums[0]), 
        }
        if 'me' in models[0].name:
            predator_buffer['g'] = former_g[0]
        if 'mf' in models[0].name or 'ma' in models[0].name:
            predator_buffer['meanaction'] = former_meanaction[0]
        if 'sac' in models[0].name:
            predator_buffer['next_state'] = obs[0]

        prey_buffer = {
            'state': old_obs[1], 
            'acts': acts[1], 
            'rewards': rewards[1], 
            'dones': all_done[num_pred:],
            'values': values[1], 
            'logps': logprobs[1],
            'ids': range(max_nums[1]), 
        }
        if 'me' in models[1].name:
            prey_buffer['g'] = former_g[1]
        if 'mf' in models[1].name or 'ma' in models[1].name:
            prey_buffer['meanaction'] = former_meanaction[1]
        if 'sac' in models[1].name:
            prey_buffer['next_state'] = obs[1]

        models[0].flush_buffer(**predator_buffer)
        models[1].flush_buffer(**prey_buffer)
        
        #############################
        # Calculate mean field #
        #############################
        for i in range(n_group):
            if 'me' in models[i].name:
                former_g[i] = _cal_ca(acts[i])
                former_meanaction[i] = _calc_moment(acts[i])
            if 'grid' in models[i].name:
                former_meanaction[i] = _calc_bin_density(acts[i])      
            if 'ma' in models[i].name:
                former_meanaction = stack_act


        for i in range(n_group):
            sum_reward = sum(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        info = {"kill": sum(total_rewards[0])/10}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    if 'ppo' in models[0].name:
        predator_buffer = {
            'state': obs[0], 
            'acts': [None for i in range(max_nums[0])], 
            'rewards': [None for i in range(max_nums[0])], 
            'dones': [None for i in range(max_nums[0])],
            'values': [None for i in range(max_nums[0])], 
            'logps': [None for i in range(max_nums[0])],
            'ids': range(max_nums[0]), 
        }
        if 'me' in models[0].name:
            predator_buffer['g'] = np.tile(former_g[0], (max_nums[0], 1))
        if 'mf' in models[0].name or 'ma' in models[0].name:
            predator_buffer['meanaction'] = np.tile(former_meanaction[0], (max_nums[0], 1))
    
        models[0].flush_buffer(**predator_buffer)
    
    if 'ppo' in models[1].name:
        prey_buffer = {
            'state': obs[1], 
            'acts': [None for i in range(max_nums[1])], 
            'rewards': [None for i in range(max_nums[1])], 
            'dones': [None for i in range(max_nums[1])],
            'values': [None for i in range(max_nums[1])], 
            'logps': [None for i in range(max_nums[1])],
            'ids': range(max_nums[1]), 
        }
        if 'me' in models[1].name:
            prey_buffer['g'] = np.tile(former_g[1], (max_nums[1], 1))
        if 'mf' in models[1].name or 'ma' in models[1].name:
            prey_buffer['meanaction'] = np.tile(former_meanaction[1], (max_nums[1], 1))

        models[1].flush_buffer(**prey_buffer)


    models[0].train()
    models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(total_rewards[i])/max_nums[i]

    return mean_rewards


def test(env, n_round, map_size, max_steps, handles, models, print_every=10, record=False, render=False, eps=None, train=False):
    env.reset()

    step_ct = 0
    done = False
    n_group = 2

    rewards = [None for _ in range(n_group)]
    max_nums = [num_pred, num_prey]

    action_dim = [env.action_space[0].shape[0], env.action_space[-1].shape[0]]

    all_obs = env.reset()
    obs = [all_obs[:num_pred], all_obs[num_pred:]]  # gym-style: return first observation
    acts = [np.zeros((max_nums[i],action_dim[i]), dtype=np.int32) for i in range(n_group)]
    values = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    logprobs = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, max_nums[0]))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_meanaction = [[] for _ in range(n_group)]
    for i in range(n_group):
        if 'mf' in models[i].name:
            former_meanaction[i] = np.zeros((1, models[i].moment_dim))


    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        #################
        # Choose action #
        #################
        # print('\n===============obs len: ', len(obs))
        for i in range(n_group):
            if 'mf' in models[i].name:
                former_meanaction[i] = np.tile(former_meanaction[i], (max_nums[i], 1))
            acts[i], values[i], logprobs[i] = models[i].act(state=obs[i], meanaction=former_meanaction[i])
        ## random predator
        # acts[0] = np.random.rand(num_pred,2)*2-1  

        old_obs = obs
        stack_act = np.concatenate(acts, axis=0)
        all_obs, all_rewards, all_done, _ = env.step(stack_act)
        obs = [all_obs[:num_pred], all_obs[num_pred:]]
        rewards = [all_rewards[:num_pred], all_rewards[num_pred:]]
        done = all(all_done)

        #############################
        # Calculate mean field #
        #############################
        for i in range(n_group):
            if 'grid' in models[i].name:
                former_meanaction[i] = _calc_bin_density(acts[i], order=models[i].order)
            elif 'mf' in models[i].name:
                former_meanaction[i] = _calc_moment(acts[i], order=models[i].order)
                

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            total_rewards[i].append(sum_reward)
        
        info = {"kill": sum(total_rewards[0])/10}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(total_rewards[i])/max_nums[i]

    return mean_rewards
