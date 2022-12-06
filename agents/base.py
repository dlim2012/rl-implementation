import random
import numpy as np
import math

class Agent:
    def __init__(self, MDP, mode=None):
        self.S = MDP.S
        self.A = MDP.S
        self.A_list = MDP.A_list
        self.terminal_states = MDP.terminal_states
        self.MDP = MDP
        self.mode = mode


    def get_initial_pi(self):
        # pi: (dict: key=state, value=[probs for each action in self.A])
        return {s: [1/len(self.A_list) for a in self.A_list] for s in self.S}

    def get_initial_state_values(self, terminal_states=True):
        if terminal_states:
            return {s: 0.0 for s in self.S.union(self.terminal_states)}
        return {s: 0.0 for s in self.S}

    def get_initial_action_values(self, mode='zeros', terminal_states=True):
        if mode == 'random':
            q = {(s, a): random.random() for s in self.S for a in self.A_list}
        elif type(mode) == tuple:
            q = {(s, a): random.uniform(mode[0], mode[1]) for s in self.S for a in self.A_list }
        elif type(mode) in (int, float):
            q = {(s, a): mode for s in self.S for a in self.A_list}
        else:
            raise ValueError('invalid mode: %s' % mode)
        if terminal_states:
            for s in self.terminal_states:
                for a in self.A_list:
                    q[(s, a)] = 0.0
        return q

    def get_action_weights_from_q(self, action_values, eps=0.0, sigma=1.0, return_probs=False):
        if self.mode == 'epsilon-greedy':
            max_action_value = max(action_values)
            max_actions = []
            for i, action_value in enumerate(action_values):
                if action_value > max_action_value - 1e-10:
                    max_actions.append(i)
            probs = [eps / len(self.A_list) for action in self.A_list]
            for action in max_actions:
                probs[action] += (1 - eps) / len(max_actions)
            return probs
        elif self.mode == 'softmax':
            max_action_value = max(action_values)
            weights = [math.exp(sigma * (action_value - max_action_value) + 700) for action_value in action_values]
            if return_probs:
                denom = sum(weights)
                return [weight / denom for weight in weights]
            return weights
        else:
            raise ValueError("invalid mode: %s" % self.mode)

    def get_action(self, state, pi=None, q=None, eps=0.0, sigma=1.0):
        if pi is not None:
            if type(pi[state]) == str:
                return pi[state]
            else:
                weights = pi[state]
        else:
            action_values = [q[(state, action)] for action in self.A_list]
            weights = self.get_action_weights_from_q(action_values=action_values, eps=eps, sigma=sigma)

        return random.choices(population=self.A_list, weights=weights)[0]

    def get_greedy_pi(self, q=None, v=None):
        pi = dict()
        for state in self.S:
            if q != None:
                action_values = [q[(state, action)] for action in self.A_list]
            else:
                action_values = [self.MDP.estimate_q_from_v(state, action, v) for action in self.A_list]
            pi[state] = self.A_list[np.argmax(action_values)]
        return pi

    def get_stochastic_pi_from_q(self, q, eps=0.0, sigma=1.0):
        pi = dict()
        for state in self.S:
            action_values = [q[(state, action)] for action in self.A_list]
            probs = self.get_action_weights_from_q(action_values=action_values, eps=eps, sigma=sigma, return_probs=True)
            for i, action in enumerate(self.A_list):
                pi[(state, action)] = probs[i]
        return pi