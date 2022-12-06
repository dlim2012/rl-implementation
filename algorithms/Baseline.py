
from torch.autograd import Variable
import torch
import numpy as np

def Baseline_Algorithm(
        MDP,
        agent,
        n=10**4,
        min_sigma = 1.,
        max_sigma= 2.,
        print_output=0,
):

    n_episodes = 0
    n_steps_list = []
    returns = []
    for i in range(n):
        n_episodes += 1

        sigma = min_sigma + (max_sigma - min_sigma) * (i+1) / n
        states, actions, rewards = [], [], []

        n_steps = 1
        s = MDP.get_initial_state()
        while s not in MDP.terminal_states and n_steps <= 1000:
            a, log_prob = agent.get_action(s, sigma=sigma)
            ns = MDP.get_next_state(state=s, action=a)
            r = MDP.get_reward(state=s, action=a, next_state=ns)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = ns
            n_steps += 1

        G = Variable(torch.tensor(0.0, dtype=torch.float32))
        n_steps = len(actions)

        for reverse_t, (s, a, r) in enumerate(zip(states[::-1], actions[::-1], rewards[::-1])):
            G = G * MDP.gamma + r
            value = agent.calc_values(s)

            delta = (G - value).clone().detach()

            loss_w = -1 * delta * value
            agent.train_w(loss_w)

            log_prob = agent.get_log_prob_for_action(s, a)
            loss_theta = -1 * delta * log_prob * (MDP.gamma ** (n_steps - reverse_t - 1))
            agent.train_theta(loss_theta)

        n_steps_list.append(n_steps)
        returns.append(G.tolist())

        if n_episodes == 10:
            if (np.mean(returns)) < -999.9:
                print(returns)
                return []

        if n_episodes % 100 == 0:
            if np.mean(returns[-100:]) < -999.9:
                print(returns)
                return []

        if print_output > 0 and (n_episodes == 10 or n_episodes % 100 == 0):

            v = dict()
            for s in MDP.S:
                v[s] = agent.calc_values(s).tolist()[0]
            diff = [abs(v[key]-MDP.optimal_values[key]) for key in v.keys()]
            s = MDP.get_initial_state()
            print(
                ['w',
                agent.w_hidden_sizes,
                '%.2e' % agent.w_lr],
                ['th',
                agent.theta_hidden_sizes,
                '%.2e' % agent.theta_lr],
                [n_episodes],
                sum(n_steps_list) / len(n_steps_list),
                s,
                agent.calc_values(s),
                'max-norm diff: %.6f' % max(diff),
                'avg diff: %.6f' % (sum(diff) / len(diff))
            )
            n_steps_list = []
            if print_output == 1:
                v = dict()
                for s in MDP.S:
                    v[s] = agent.calc_values(s).tolist()[0]

                max_norm = max([abs(v[key]-MDP.optimal_values[key]) for key in v.keys()])
                MDP.print_values(v, 'n_episodes: %d, Max-Norm: %.4f' % (n_episodes, max_norm))
                if v[(0, 1)] != v[(0, 1)]:
                    break
            else:
                max_norm = float('nan')


            # print(qq)

            #MDP.print_values(MDP.optimal_values)

            if print_output == 1:
                q = dict()
                for s in MDP.S:
                    action_values = agent.theta_MLP.forward(agent.state_to_input(s))
                    # print(q)
                    probs = torch.nn.functional.softmax(action_values, dim=0).tolist()
                    for i, a in enumerate(MDP.A_list):
                        q[(s, a)] = probs[i]

                if 0:
                    for s in MDP.S:
                        for a in MDP.A_list:
                            key = (s, a)
                            print((key, q[key]), end=' ')
                        print()
                pi = agent.get_greedy_pi(q)
                MDP.print_pi(pi, 'n_episodes: %d, Max-Norm: %.4f' % (n_episodes, max_norm))
            #print('sigma:', sigma, '%.6f, %.6f' % (q[((4, 1), 'AR')], q[((3, 4), 'AD')]))
            #print(log_prob)

            import random
            s = random.choice(list(MDP.S))
            action_values = agent.theta_MLP.forward(agent.state_to_input(s))
            probs = agent.get_action_weights_from_q(action_values=action_values, eps=0, sigma=sigma, return_probs=True)
            #print(s, action_values, probs)

    return returns


if __name__ == '__main__':
    from MDPs.gridworld import get_Gridworld_MDP
    #from MDPs.mountain_car import get_Mountain_Car_MDP
    from agents.MLP_agent import MLP_Agent

    w_hidden_sizes = [50] * 2
    theta_hidden_sizes = [50] * 3
    w_lr = 1e-5
    theta_lr = 1e-5
    input_mode = 'default'
    n = 10 ** 5

    MDP = get_Gridworld_MDP()
    agent = MLP_Agent(
        MDP=MDP,
        state_size=2,
        w_hidden_sizes=w_hidden_sizes,
        w_lr=2e-4,
        theta_hidden_sizes=theta_hidden_sizes,
        theta_lr=1e-4,
        input_mode=input_mode
    )

    MDP.calc_optimal_v_pi(agent)
    returns = Baseline_Algorithm(
        MDP,
        agent,
        min_sigma=1.0,
        max_sigma=2.0,
        n=n,
        print_output=True
    )


# HP search for mountain car 2, 3 layers converged best