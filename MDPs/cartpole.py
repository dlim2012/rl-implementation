import math
import numpy as np
from .MDP_base import MDP_base

g = -9.8
m_c = 1
m = 0.1
lp = 0.5
mu_c = 0.0005
mu_p = 0.000002
F = 10

class CartPole(MDP_base):
    def __init__(self, S, A, p, R, d0, gamma, terminal_states, n_bins):
        super().__init__(S, A, p, R, d0, gamma, terminal_states)
        self.n_bins = n_bins

    def get_next_state(self, state, action):
        action = self.A[action]
        i, j, k, l = state
        x = -2.4 + i*(4.8)/(self.n_bins-1)
        theta = -math.radians(-12) + k * (2 * math.radians(12)) / (self.n_bins-1)
        x_v = -5 + j * (10)/(self.n_bins-1)
        theta_v = -math.radians(20) + l * (2 * math.radians(20)) / (self.n_bins-1)

        num_inner = (-action - m * lp * theta_v ** 2 * np.sin(theta) + mu_c * np.sign(x_v)) / (m_c + m)
        num_sub = (mu_p * theta_v) / (m*lp)
        num = g * np.sin(theta) + np.cos(theta) * num_inner - num_sub

        deno_sub = m * (np.cos(theta) ** 2) / (m_c + m)
        deno = lp * (4/3 - deno_sub)

        theta_acc = num / deno
        x_acc = (action + m*lp * (theta_v **2 * np.sin(theta) - theta_acc * np.cos(theta)) - mu_c * np.sign(x_v)) / (m_c + m)

        x_v_new = x_v + x_acc
        x_new = x + x_v_new

        theta_v_new = theta_v + theta_acc
        theta_new = theta + theta_v_new

        ii = round((x_new + 2.4) / 4.8 * (self.n_bins - 1))
        jj = round((x_v_new + 5) / 10 * (self.n_bins - 1))
        kk = round((theta_new + math.radians(12)) / 2 * math.radians(12) * (self.n_bins - 1))
        ll = round((theta_v_new + math.radians(20)) / 2 * math.radians(20) * (self.n_bins - 1))

        ii = max(0, ii)
        ii = min(ii, self.n_bins-1)
        jj = max(0, jj)
        jj = min(jj, self.n_bins-1)
        kk = max(0, kk)
        kk = min(kk, self.n_bins-1)
        ll = max(0, ll)
        ll = min(ll, self.n_bins-1)

        return (ii, jj, kk, ll)

    def estimate_q_from_v(self, state, action, values):
        next_state = self.get_next_state(state, action)
        if next_state in self.terminal_states:
            return self.R
        return self.R + values[next_state]

    def get_reward(self, state=None, action=None, next_state=None):
        return self.R



def get_CartPole_MDP(n_bins):
    S = set([(i, j, k, l) for i in range(n_bins) for j in range(n_bins) for k in range(n_bins) for l in range(n_bins)])
    A = {'left': -1, 'right': 1}
    p = dict()
    R = 1
    gamma = 1.0

    x, xv, t, tv = tuple(np.random.uniform(-0.5, 0.5, (4)))
    
    ii = round((x + 2.4) / 4.8 * (n_bins - 1))
    jj = round((xv + 5) / 10 * (n_bins - 1))
    kk = round((t + math.radians(12)) / 2 * math.radians(12) * (n_bins - 1))
    ll = round((tv + math.radians(20)) / 2 * math.radians(20) * (n_bins - 1))

    terminal_states = set([(i, j, k, l) for i in [0, n_bins-1] for k in [0, n_bins-1] for j in range(n_bins) for l in range(n_bins)])
    d0 = {(ii, jj, kk, ll): 1}

    return CartPole(S, A, p, R, d0, gamma, terminal_states, n_bins)  