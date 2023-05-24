import copy
import random
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()




def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    obs = [None for _ in range(n_group)]
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    ids = [None for _ in range(n_group)]
    alive_idx = [None for _ in range(n_group)]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]
    ########################
    # Actor start sampling #
    ########################
    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            obs[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])
            alive_idx[i] = np.array(list(map(lambda x: x % max_nums[i], ids[i])), dtype=int)

        #################
        # Choose action #
        #################
        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(alive_idx[i]), 1))
            acts[i][alive_idx[i]], _ = models[i].act(state=obs[i], prob=former_act_prob[i], eps=eps, train=True)

        for i in range(n_group):
            env.set_action(handles[i], acts[i][alive_idx[i]])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': obs[0], 'acts': acts[0][alive_idx[0]], 'rewards': rewards[0], 
            'alives': alives[0], 'ids': ids[0], 
        }
  
        buffer['prob'] = former_act_prob[0]      

        #############################
        # Calculate former_act_prob #
        #############################
        for i in range(n_group):
            if 'me' in models[i].name:
                former_act_prob[i] = np.sum(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])),axis=0, keepdims=True)
            else:
                former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])), axis=0, keepdims=True)
            
        if train:
            models[0].flush_buffer(**buffer)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards



def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    # generate_rand_map(env, map_size, density, handles)
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    obs = [None for _ in range(n_group)]
    acts = [np.zeros((max_nums[i],), dtype=np.int32) for i in range(n_group)]
    ids = [None for _ in range(n_group)]
    alive_idx = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            obs[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])
            alive_idx[i] = np.array(list(map(lambda x: x % max_nums[i], ids[i])), dtype=int)

        #################
        # Choose action #
        #################
        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(alive_idx[i]), 1))
            acts[i][alive_idx[i]], _ = models[i].act(state=obs[i], prob=former_act_prob[i], eps=eps, train=False)

        for i in range(n_group):
            env.set_action(handles[i], acts[i][alive_idx[i]])

        # simulate one step
        done = env.step()

        #############################
        # Calculate former_act_prob #
        #############################
        for i in range(n_group):
            if 'me' in models[i].name:
                former_act_prob[i] = np.sum(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])),axis=0, keepdims=True)
            else:
                former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i][alive_idx[i]])), axis=0, keepdims=True)
            
        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        # if step_ct % print_every == 0:
            # print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
