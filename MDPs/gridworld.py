
import random

from .MDP_base import MDP_base

optimal_values = [
    [4.0187, 4.5548, 5.1575, 5.8336, 6.4553],
    [4.3716, 5.0324, 5.8013, 6.6473, 7.3907],
    [3.8672, 4.3900, 0.0000, 7.5769, 8.4637],
    [3.4182, 3.8319, 0.0000, 8.5738, 9.6946],
    [2.9977, 2.9309, 6.0733, 9.6946, 0.0000]
]

class GridWorld(MDP_base):
    def __init__(self, S, A, p, R, d0, gamma, terminal_states):
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

        self.dx, self.dy = (-1, 0, 1, 0), (0, 1, 0, -1)


        # Arrows to print out deterministic policy
        self.arrows = {'AU': u'\u2191', 'AD': u'\u2193', 'AL': u'\u2190', 'AR': u'\u2192', '': '', 'G': 'G'}

    def get_reward(self, state=None, action=None, next_state=None):
        return self.R[next_state]

    def get_next_state(self, state, action):
        x, y = state
        veer = random.choices(population=self.p_tuples[0], weights=self.p_tuples[1])[0]

        # return current state if veer == -1
        if veer == -1:
            return state

        direction = (self.A[action] + veer) % 4
        next_state = (x + self.dx[direction], y + self.dy[direction])
        if next_state in self.S or next_state in self.terminal_states:
            return next_state

        # return current state if the agent can't move to the direction
        return state

    def get_all_next_states_rewards(self, state, action):
        x, y = state

        next_states = []
        for veer in self.p_tuples[0]:

            # return current state if veer == -1
            if veer == -1:
                next_states.append((self.R[state], state))
                continue

            direction = (self.A[action] + veer) % 4
            next_state = (x + self.dx[direction], y + self.dy[direction])
            if next_state in self.S or next_state in self.terminal_states:
                next_states.append((self.R[next_state], next_state))
                continue

            # return current state if the agent can't move to the direction
            next_states.append((self.R[state], state))
        return next_states


    def estimate_q_from_v(self, state, action, values):
        x, y = state
        action_value = 0
        for veer, prob in zip(self.p_tuples[0], self.p_tuples[1]):
            if veer == -1:
                 action_value += prob * (self.R[state] + self.gamma * values[state])
            else:
                d = (self.A[action] + veer) % 4
                next_state = (x + self.dx[d], y + self.dy[d])
                if next_state in self.S:
                    action_value += prob * (self.R[next_state] + self.gamma * values[next_state])
                elif next_state in self.terminal_states:
                    action_value += prob * (self.R[next_state])
                else:
                    action_value += prob * (self.R[state] + self.gamma * values[state])

        return action_value


    def print_values(self, values, text=''):
        """
        print values in a grid-like shape
        :param values: value function
        :param text: text to print
        :return: None
        """
        print('Value Function', text)
        for x in range(5):
            for y in range(5):
                if (x, y) in values:
                    print('{:10.4f}'.format(values[(x, y)]),end=' ')
                else:
                    print('{:10.4f}'.format(0),end=' ')
            print()

    def print_pi(self, pi, text=''):
        """
        print out a policy in a grid-like shape
        :param pi: policy
        :param text: text
        :return: None
        """
        print('Policy', text)
        for x in range(5):
            for y in range(5):
                if (x, y) in pi:
                    print('{:5s}'.format(self.arrows[pi[(x, y)]]), end=' ')
                elif (x, y) in self.terminal_states:
                    print('{:5s}'.format('G'), end=' ')
                else:
                    print('{:5s}'.format(''), end=' ')

            print()

def get_Gridworld_MDP():
    """
    Get MDPs for the default environment
    :return: tuple of states(S), actions(A), dynamics(p), reward by entering state(R), initial state(d0), discount rate(gamma), terminal states
    """
    # States (excluding terminal state)
    S = set([(x, y) for y in range(5) for x in range(5) if (x, y) not in [(2, 2), (3, 2), (4, 4)]])

    # Terminal State:
    terminal_states = {(4, 4)}

    # Actions and directions: (dict)
    A = {'AU': 0, 'AR': 1, 'AD': 2, 'AL': 3}

    # Dynamics: (dict)
    p = {0: 0.8, 3: 0.05, 1: 0.05, -1: 0.1}

    # Rewards:
    R = {s: 0 for s in S.union(terminal_states)}
    R[(4, 2)] = -10.0
    R[(4, 4)] = 10.0

    # Discount:
    gamma = 0.9

    # Initial State: (dict)
    d0 = {s: 1/len(S) for s in S}
    ##################################################################
    #d0 = {(4, 1): 1.0 for i in range(5)}
    #R[(0, 0)] = 10
    #R[(0, 2)] = -10
    #R[(0, 4)] = -5
    #R[(4, 0)] = -5
    #gamma = 0.5
    #R[(4, 2)] = -20
    #R[(4, 4)] = 1
    #terminal_states = {(0, 0), (0, 4), (4, 4), (4, 0)}
    ##################################################################

    S = S - terminal_states
    return GridWorld(S, A, p, R, d0, gamma, terminal_states)