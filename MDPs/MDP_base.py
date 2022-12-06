import numpy as np
import random
import math

from collections import defaultdict
from algorithms.Value_Iteration import value_iteration

class MDP_base():
    def __init__(self, S, A, p, R, d0, gamma, terminal_states, max_steps=float('inf')):
        """

        :param S: (set) states (excluding terminal states)
        :param A: (tuple: actions, tuple=directions) actions and the corresponding directions
        :param p: (tuple: directions/indications, tuple: probs) dynamics
        :param R: (dict: key=state, value=reward)
        :param d0: (tuple: initial_states, tuple: probs) (initial states, probs)
        :param gamma: (int) discount rate
        :param terminal_states: (set) terminal states
        """

        def dict_to_tuples(D):
            keys = tuple(D.keys())
            values = tuple(D[key] for key in keys)
            return (keys, values)

        self.S = S
        self.A = A
        self.A_list = sorted(list(A.keys()), key=lambda x: A[x])
        self.p_tuples = dict_to_tuples(p)
        self.R = R
        self.d0_tuples = dict_to_tuples(d0)
        self.gamma = gamma
        self.terminal_states = terminal_states
        self.max_steps = max_steps

        self.optimal_values = None
        self.optimal_pi = None

    def get_initial_state(self):
        return random.choices(population=self.d0_tuples[0], weights=self.d0_tuples[1])[0]

    def get_next_state(self, state, action):
        raise NotImplementedError()

    def get_reward(self, state=None, action=None, next_state=None):
        raise NotImplementedError()

    def estimate_q_from_v(self, state, action, values):
        raise NotImplementedError()

    def calc_optimal_v_pi(self, agent):
        v = value_iteration(self, agent, 1e-6)[0]
        pi = agent.get_greedy_pi(v=v)

        self.optimal_values, self.optimal_pi = v, pi
        return v, pi

    def print_values(self, values, text=''):
        raise NotImplementedError()

    def print_pi(self, pi, text=''):
        raise NotImplementedError()

    def get_all_next_states_rewards(self, state, action):
        raise NotImplementedError()

    def get_all_prev_(self):
        res = defaultdict(list)
        for s in self.S:
            for a in self.A_list:
                ret = self.get_all_next_states_rewards(s, a)
                for r, ns in ret:
                    res[ns].append((s, a, r))
        return res



