import numpy as np
import random
import math

from .MDP_base import MDP_base

class Mountain_Car_MDP(MDP_base):
    def __init__(self, S, A, p, R, d0, gamma, terminal_states, n_bins):
        """

        :param S: (set) states (excluding terminal states)
        :param A: (tuple: actions, tuple=directions) actions and the corresponding directions
        :param p: (tuple: directions/indications, tuple: probs) dynamics
        :param R: (dict: key=state, value=reward)
        :param d0: (tuple: initial_states, tuple: probs) (initial states, probs)
        :param gamma: (int) discount rate
        :param terminal_states: (set) terminal states
        """
        MDP_base.__init__(self, S, A, p, R, d0, gamma, terminal_states)
        self.n_bins = n_bins

    def get_next_state(self, state, action):
        i, j = state
        x = -1.2 + 1.7 * i / (self.n_bins - 1)
        v = -0.07 + 0.14 * j / (self.n_bins - 1)

        vv = v + 0.001 * self.A[action] - 0.0025 * math.cos(3 * x)
        xx = x + vv


        if xx > 0.5:
            ii, jj = -1, -1
        elif xx < -1.2:
            ii, jj = 0, self.n_bins // 2
        else:
            ii = round((xx + 1.2) / 1.7 * (self.n_bins - 1))
            if ii == self.n_bins:
                jj = 0
            else:
                vv = min(0.07, max(-0.07, vv))
                jj = round((vv + 0.07) / 0.14 * (self.n_bins-1))

        #print(state, action, (x, v), (xx, vv), (ii, jj))
        return (ii, jj)

    def get_all_next_states_rewards(self, state, action):
        pass

    def get_reward(self, state=None, action=None, next_state=None):
        return self.R

    def estimate_q_from_v(self, state, action, values):
        next_state = self.get_next_state(state, action)
        if next_state in self.terminal_states:
            return self.R
        return self.R + values[next_state]

    def print_values(self, values, text=''):
        #print('average values:', sum(values.values()) / len(values.values()))
        return

    def print_pi(self, pi, text=''):
        return

def get_Mountain_Car_MDP(n_bins):
    """
    Get MDPs for the default environment
    :return: tuple of states(S), actions(A), dynamics(p), reward by entering state(R), initial state(d0), discount rate(gamma), terminal states
    """
    if n_bins & 1 == 0:
        n_bins += 1

    # States (excluding terminal state)
    S = set([(i, j) for i in range(n_bins) for j in range(n_bins)])

    # Terminal State:
    terminal_states = {(-1, -1)}

    # Actions and directions: (dict)
    A = {'reverse': -1, 'neutral': 0, 'forward': 1}

    # Dynamics: (dict)
    p = dict()

    # Rewards:
    R = -1

    # Discount:
    gamma = 1.0

    # Initial State: (dict)
    i1, i2 = int((n_bins - 1) / 1.7 * 0.6) + 1, int((n_bins - 1) / 1.7 * 0.8) + 1
    j = n_bins // 2
    d0 = {(i, j): 1 for i in range(i1, i2)}


    ######################################################################
    R = -1
    #gamma = 0.9
    #######################################################################

    return Mountain_Car_MDP(S, A, p, R, d0, gamma, terminal_states, n_bins)