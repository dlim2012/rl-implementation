
from MDPs.gridworld import get_Gridworld_MDP
from agents.MLP_agent import MLP_Agent
from algorithms.Baseline import Baseline_Algorithm

if __name__ == '__main__':

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
        print_output=1
    )