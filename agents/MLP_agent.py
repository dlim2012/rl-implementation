from .base import Agent
import torch
import torchvision
import math
import random
import numpy as np
from torch.autograd import Variable
from tools.torch_MLP import MLP

class MLP_Agent(Agent):
    def __init__(
            self,
            MDP,
            state_size,
            w_hidden_sizes,
            w_lr,
            theta_hidden_sizes,
            theta_lr,
            mode='softmax',
            input_mode='default'
    ):
        Agent.__init__(self, MDP)
        self.state_size = state_size

        self.w_hidden_sizes = w_hidden_sizes
        self.w_lr = w_lr
        self.theta_hidden_sizes = theta_hidden_sizes
        self.theta_lr = theta_lr

        self.input_mode = input_mode
        self.mode = mode

        #print('w_hidden_sizes', w_hidden_sizes)
        #print('w_lr', w_lr)
        #print('theta_hidden_sizes', theta_hidden_sizes)
        #print('theta_lr', theta_lr)
        #print('input_mode', input_mode)
        in_size = len(self.state_to_input(torch.zeros(state_size)))

        self.w_MLP = MLP(
            in_channels=in_size,
            hidden_channels=w_hidden_sizes+[1],
            activation_layer=torch.nn.LeakyReLU,
        )

        self.theta_MLP = MLP(
            in_channels=in_size,
            hidden_channels=theta_hidden_sizes+[len(self.A_list)],
            activation_layer=torch.nn.LeakyReLU,
        )


        self.action_to_index = {a: i for i, a in enumerate(self.A_list)}

        self.softmax1 = torch.nn.Softmax()
        self.w_optimizer = torch.optim.SGD(self.w_MLP.parameters(), lr=w_lr)
        self.theta_optimizer = torch.optim.SGD(self.theta_MLP.parameters(), lr=theta_lr)

    def get_initial_pi(self):
        # pi: (dict: key=state, value=[probs for each action in self.A])
        return {s: torch.tensor([1/len(self.A_list) for a in self.A_list]) for s in self.S}

    def get_initial_state_values(self, terminal_states=True):
        if terminal_states:
            return {s: torch.tensor(0.0) for s in self.S.union(self.terminal_states)}
        return {s: torch.tensor(0.0) for s in self.S}

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

    def state_to_input(self, state):
        if self.input_mode == 'default':
            ret = state
        elif self.input_mode == 'quadratic':
            ret = []
            for i, x in enumerate(state):
                ret.append(x)
                for j, y in enumerate(state):
                    if i > j:
                        continue
                    ret.append(x * y)
        elif self.input_mode == 'trigonometric':
            ret = []
            for x in state:
                ret.append(x)
                ret.append(math.cos(x))
                ret.append(math.sin(x))
        elif self.input_mode == 'mountain_car':
            x, v = state
            ret = [x, math.cos(x), math.sin(x), v]
        else:
            raise ValueError('invalid input_mode: %s' % self.input_mode)

        return Variable(torch.tensor(ret, dtype=torch.float32))

    def get_action_weights_from_q(self, action_values=None, eps=0.0, sigma=1.0, return_probs=True):
        # q: action values for current state

        if self.mode == 'softmax':
            if return_probs:
                if action_values != None:
                    probs = torch.softmax(sigma * action_values, dim=0)
                    return probs

        """
        action_values = [q[(state, action)] for action in self.A_list]
        if mode == 'epsilon-greedy':
            max_action_value = max(action_values)
            max_actions = []
            for i, action_value in enumerate(action_values):
                if action_value > max_action_value - 1e-10:
                    max_actions.append(i)
            probs = [eps / len(self.A_list) for action in self.A_list]
            for action in max_actions:
                probs[action] += (1 - eps) / len(max_actions)
            return probs
        elif mode == 'softmax':
            max_action_value = max(action_values)
            weights = [math.exp(sigma * (action_value - max_action_value) + 700) for action_value in action_values]
            if return_probs:
                denom = sum(weights)
                return [weight / denom for weight in weights]
            return weights
        else:
            raise ValueError("invalid mode: %s" % mode)
        """

    def get_log_prob_for_action(self, state, action, eps=0.0, sigma=1.0):
        action_values = self.theta_MLP.forward(self.state_to_input(state))
        probs = self.get_action_weights_from_q(action_values=action_values, eps=eps, sigma=sigma, return_probs=True)
        #return probs[self.action_to_index[action]]
        return torch.log(probs[self.action_to_index[action]])

    def get_action(self, state, pi=None, q=None, mode='epsilon-greedy', eps=0.0, sigma=1.0):
        if pi is not None:
            if type(pi[state]) == str:
                return pi[state], None
            else:
                probs = Variable(torch.tensor(pi[state], dtype=torch.float32))
        else:
            if q is None:
                action_values = self.theta_MLP.forward(self.state_to_input(state))
            else:
                action_values = [q[(state, action)] for action in self.A_list]
            probs = self.get_action_weights_from_q(action_values=action_values, eps=eps, sigma=sigma, return_probs=True)

        #print(action_values, probs)
        action_idx = np.random.choice(len(self.A_list), p=probs.detach().numpy())
        return self.A_list[action_idx], torch.log(probs[action_idx])

    def calc_values(self, state):
        return self.w_MLP.forward(self.state_to_input(state))

    def train_w(self, loss_w):
        self.w_optimizer.zero_grad()
        loss_w.backward()
        self.w_optimizer.step()

    def train_theta(self, loss_theta):
        self.theta_optimizer.zero_grad()
        loss_theta.backward()
        self.theta_optimizer.step()

    def get_greedy_pi(self, q=None, v=None):
        pi = dict()
        for state in self.S:
            if q != None:
                action_values = [q[(state, action)] for action in self.A_list]
            else:
                action_values = [self.MDP.estimate_q_from_v(state, action, v) for action in self.A_list]
            pi[state] = self.A_list[np.argmax(action_values)]
        return pi

    def get_stochastic_pi_from_q(self, q, mode='epsilon-greedy', eps=0.0, sigma=1.0):
        pi = dict()
        for state in self.S:
            probs = self.get_action_weights_from_q(state=state, q=q, mode=mode, eps=eps, sigma=sigma, return_probs=True)
            for i, action in enumerate(self.A_list):
                pi[(state, action)] = probs[i]
        return pi