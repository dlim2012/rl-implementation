
from MDPs.gridworld import get_Gridworld_MDP
from agents.MLP_agent import MLP_Agent
from torch.autograd import Variable

def Artor_Critic_Algorithm(
        MDP,
        agent,
        mode='softmax',
        sigma=100.0,
        n=10**5,
        lambda_w = 0.9,
        lambda_theta = 0.9
):

    MDP = get_Gridworld_MDP()
    agent = MLP_Agent(MDP, 2, [20, 20], [20, 20], 2e-4, 1e-4)
    MDP.calc_optimal_v_pi(agent)

    n_episodes = 0
    for i in range(n):
    #for i in range(n):
        n_episodes += 1
        s = MDP.get_initial_state()

        sig = 1 + (sigma-1) * (i+1) / n
        #sig = 10
        I = 1
        z_theta = Variable(torch.tensor(0.0))
        z_w = Variable(torch.tensor(0.0))
        states = []
        log_probs = []

        while s not in MDP.terminal_states:
            a, log_prob = agent.get_action(s, mode=mode, sigma=sig)

            ns = MDP.get_next_state(s, a)
            r = MDP.get_reward(s, a, ns)

            value = MDP.optimal_values[s] #agent.calc_values(s)
            if ns in MDP.terminal_states:
                delta = r - value
            else:
                #delta = r + MDP.gamma * agent.calc_values(ns) - value
                delta = r + MDP.gamma * MDP.optimal_values[ns] - value

            #z_w = MDP.gamma * lambda_w * z_w.clone() + value
            #z_theta = MDP.gamma * lambda_theta * z_theta.clone() + I * log_prob

            #_, __ = agent.train(delta, value, I*log_prob)
            _, __ = agent.train(delta, value, I*log_prob)

            states.append(s)
            log_probs.append(log_prob)

            s = ns
            I = MDP.gamma * I
        #print(_, __)

        if n_episodes % 100 == 0:
            v = dict()
            for s in MDP.S:
                v[s] = agent.calc_values(s).tolist()[0]
            MDP.print_values(v, 'n_episodes: %d' % n_episodes)

            q = dict()
            for s in MDP.S:
                action_values = agent.theta_MLP.forward(torch.tensor(s, dtype=torch.float32))
                probs = agent.get_action_weights_from_q(s, action_values, mode='softmax', return_probs=True).tolist()
                for i, a in enumerate(MDP.A_list):
                    q[(s, a)] = probs[i]
            pi = agent.get_greedy_pi(q)

            #for key in sorted(qq.keys()):
            #    print((key, qq[key]), end=' ')
            MDP.print_values(MDP.optimal_values)
            MDP.print_pi(pi)
            print(sig)

if __name__ == '__main__':
    import torch

    torch.autograd.set_detect_anomaly(True)

    Artor_Critic_Algorithm(None, None)