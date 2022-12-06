
from MDPs.gridworld import get_Gridworld_MDP
from agents.MLP_agent import MLP_Agent
from torch.autograd import Variable

def Monte_Carlo_Policy_Gradients(
        MDP,
        agent,
        mode='softmax',
        sigma=100.0,
        n=10**5
):

    MDP = get_Gridworld_MDP()
    #MDP.R[(4, 4)] = 0
    #MDP.R[(0, 2)] = 10
    #MDP.terminal_states = set([(0, 2)])
    agent = MLP_Agent(
        MDP=MDP,
        state_size=2,
        w_hidden_sizes=[50],
        theta_hidden_sizes=[50],
        w_lr=2e-4,
        theta_lr=1e-5)
    MDP.calc_optimal_v_pi(agent)

    n_episodes = 0
    for i in range(10000):
    #for i in range(n):
        n_episodes += 1
        s = MDP.get_initial_state()

        #sig = sigma * (i+1) / n
        sig = 2
        G = 0
        discounted_rewards = []
        states = []
        actions = []
        rewards = []
        log_probs = []
        while s not in MDP.terminal_states:
            a, log_prob = agent.get_action(s, mode=mode, sigma=sig)

            ns = MDP.get_next_state(s, a)
            r = MDP.get_reward(s, a, ns)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            log_probs.append(log_prob)

            s = ns

        G = Variable(torch.Tensor([0.]))
        #G = 0
        for s, a, r, log_prob in zip(states[::-1], actions[::-1], rewards[::-1], log_probs[::-1]):
            G = Variable(torch.tensor([r])) + MDP.gamma * G
            q = agent.theta_MLP.forward(torch.tensor(s, dtype=torch.float32))
            log_prob = agent.get_action_weights_from_q(s, q=q, mode=mode, sigma=sig)[agent.A_list.index(a)]
            #G = MDP.gamma * G + reward
            loss = log_prob * -1 * G
            #og_prob.backward()
            #loss = (-1.0) * G * log_prob
            agent.theta_optimizer.zero_grad()
            loss.backward()
            agent.theta_optimizer.step()


        if n_episodes % 10 == 0:
            qq = dict()
            for s in MDP.S:
                q = agent.theta_MLP.forward(torch.tensor(s, dtype=torch.float32))
                # print(q)
                probs = torch.nn.functional.softmax(q, dim=0).tolist()
                for i, a in enumerate(MDP.A_list):
                    qq[(s, a)] = probs[i]
            # print(qq)
            for s in MDP.S:
                for a in MDP.A_list:
                    key=(s, a)
                    print((key, qq[key]), end=' ')
                print()
            #MDP.print_values(MDP.optimal_values)
            pi = agent.get_greedy_pi(qq)
            MDP.print_pi(pi)
            print(loss)


if __name__ == '__main__':
    import torch

    torch.autograd.set_detect_anomaly(True)

    Monte_Carlo_Policy_Gradients(None, None)