import math
from matplotlib import pyplot as plt
import numpy as np
import os
import random
from agents.base import Agent
from collections import deque, defaultdict
from heapq import heapify, heappush, heappop

def Prioritized_Sweeping(
        MDP,
        agent,
        init_mode=0,
        init_theta=10,
        alpha_k=100,
        sigma=10.0,
        n_episodes=int(1e6),
        n=10,
):

    q = agent.get_initial_action_values(mode=init_mode, terminal_states=True)
    n_steps = {(s, a): 1 for s in MDP.S for a in MDP.A_list}
    model = {(s, a): defaultdict(lambda: 1) for s in MDP.S for a in MDP.A_list}
    p_queue = []
    p_queue_set = set()

    prevs = MDP.get_all_prev_()

    for i in range(n_episodes): #n_episodes
        n_steps_episode = 0
        theta = (init_theta+10) / (i+10)
        #print(theta, p_queue_set)

        state = MDP.get_initial_state()
        while state not in MDP.terminal_states and n_steps_episode < MDP.max_steps:
            action = agent.get_action(state=state, q=q, sigma=sigma)
            next_state = MDP.get_next_state(state=state, action=action)
            reward = MDP.get_reward(next_state=next_state)

            model[(state, action)][(reward, next_state)] += 1

            #print(s, a, r)

            max_next_q = max([q[(next_state, next_action)] for next_action in MDP.A_list])
            p = abs(reward + MDP.gamma * max_next_q - q[(state, action)])

            if p > theta and (state, action) not in p_queue_set:
                heappush(p_queue, (-p, (state, action)))
                p_queue_set.add((state, action))

            for j in range(n):
                if not p_queue:
                    break
                p_neg, (s, a) = heappop(p_queue)
                p_queue_set.remove((s, a))
                if not model[(s, a)]:
                    continue
                r, ns = random.choices(list(model[(s, a)].keys()), weights=list(model[(s, a)].values()))[0]

                alpha = 0.1 * (alpha_k / (n_steps[(s, a)] + alpha_k))
                max_next_q_ = max([q[(ns, na)] for na in MDP.A_list])
                q[(s, a)] = q[(s, a)] + alpha * (r + MDP.gamma * max_next_q_ - q[(s, a)])


                for prev_s, prev_a, prev_r in prevs[s]:
                    #print(prev_s, prev_a, prev_r, s)
                    prev_max_next_q = max(q[(s, a)] for a in MDP.A_list)
                    p = abs(prev_r + MDP.gamma * prev_max_next_q - q[(prev_s, prev_a)])
                    if p > theta and (prev_s, prev_a) not in p_queue_set:
                        heappush(p_queue, (-p, (prev_s, prev_a)))
                        p_queue_set.add((prev_s, prev_a))

            state = next_state
            n_steps_episode += 1

        if (i+1) % 100 == 0:
            v = {s: max(q[s, a] for a in MDP.A_list) for s in MDP.S}
            MDP.print_values(v)





if __name__ == '__main__':
    from MDPs.gridworld import get_Gridworld_MDP
    #from MDPs.mountain_car import get_Mountain_Car_MDP

    w_hidden_sizes = [50] * 2
    theta_hidden_sizes = [50] * 3
    w_lr = 1e-5
    theta_lr = 1e-5
    input_mode = 'default'
    n = 10 ** 5

    MDP = get_Gridworld_MDP()

    agent = Agent(MDP, mode='softmax')

    MDP.calc_optimal_v_pi(agent)

    Prioritized_Sweeping(MDP, agent)
