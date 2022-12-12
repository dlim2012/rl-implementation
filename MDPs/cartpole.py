import math
import numpy as np
from .MDP_base import MDP_base

g = -9.8
m_c = 1.0
m = 0.1
lp = 0.5
mu_c = 0.0005
mu_p = 0.000002
F = 10
t_step = 0.02

class CartPole(MDP_base):
    def __init__(self, S, A, p, R, d0, gamma, terminal_states, n_bins):
        super().__init__(S, A, p, R, d0, gamma, terminal_states)
        self.n_bins = n_bins
        self.x_bins = np.linspace(-2.4, 2.4, n_bins)
        self.xv_bins = np.linspace(-5, 5, n_bins)
        self.theta_bins = np.linspace(math.radians(-12), math.radians(12), n_bins)
        self.theta_v_bins = np.linspace(math.radians(-20), math.radians(20), n_bins)

    def get_next_state(self, state, action):
        action = self.A[action]
        i, j, k, l = state
        x = self.x_bins[i]
        theta = self.theta_bins[k]
        x_v = self.xv_bins[j]
        theta_v = self.theta_v_bins[l]

        num_inner = (-action*F - m * lp * theta_v ** 2 * np.sin(theta) + mu_c * np.sign(x_v)) / (m_c + m)
        num_sub = (mu_p * theta_v) / (m*lp)
        num = g * np.sin(theta) + np.cos(theta) * num_inner - num_sub

        deno_sub = m * (np.cos(theta) ** 2) / (m_c + m)
        deno = lp * (4/3 - deno_sub)

        theta_acc = num / deno
        x_acc = (action*F + m*lp * (theta_v **2 * np.sin(theta) - theta_acc * np.cos(theta)) - mu_c * np.sign(x_v)) / (m_c + m)

        x_v_new = x_v + x_acc * t_step
        x_new = x + x_v_new * t_step

        theta_v_new = theta_v + theta_acc * t_step
        theta_new = theta + theta_v_new * t_step

        ii = np.digitize(x_new, self.x_bins, right=True)
        jj = np.digitize(x_v_new, self.xv_bins, right=True)
        kk = np.digitize(theta_new, self.theta_bins, right=True)
        ll = np.digitize(theta_v_new, self.theta_v_bins, right=True)

        (ii, jj, kk, ll) = map(lambda x: min(self.n_bins-1, x), (ii, jj, kk, ll))

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

    x, xv, t, tv = tuple(np.random.uniform(-0.05, 0.05, (4)))
    
    bins = np.linspace(-2.4, 2.4, n_bins)
    ii = np.digitize(x, bins, right=True)
    bins = np.linspace(-5, 5, n_bins)
    jj = np.digitize(xv, bins, right=True)
    bins = np.linspace(math.radians(-12), math.radians(12), n_bins)
    kk = np.digitize(t, bins, right=True)
    bins = np.linspace(math.radians(-20), math.radians(20), n_bins)
    ll = np.digitize(tv, bins, right=True)

    terminal_states = set([(i, j, k, l) for i in [0, n_bins-1] for k in [0, n_bins-1] for j in range(n_bins) for l in range(n_bins)])
    d0 = {(ii, jj, kk, ll): 1}

    return CartPole(S, A, p, R, d0, gamma, terminal_states, n_bins)  