
from MDPs.gridworld import get_Gridworld_MDP
from agents.MLP_agent import MLP_Agent
from algorithms.Baseline import Baseline_Algorithm
from tools.savefig import savefig_mean_std
import time
from multiprocessing import Pool, cpu_count
import numpy as np

def func(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr, input_mode, n):
    while True:
        print(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr)
        start = time.time()

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

        lc1, lc2, returns = Baseline_Algorithm(
            MDP,
            agent,
            min_sigma=1.0,
            max_sigma=2.0,
            n=n,
            print_output=0
        )

        if len(returns) == n:
            print(time.time() - start)
            return lc1, lc2, returns

if __name__ == '__main__':
    n_runs = 20

    w_hidden_sizes = [50] * 2
    theta_hidden_sizes = [50] * 3
    w_lr = 1e-5
    theta_lr = 1e-5
    input_mode = 'default'
    n = 10 ** 5

    parameters = [(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr, input_mode, n) for _ in range(n_runs)]

    ##################################################################
    n_runs = 2
    n = 10 ** 3
    ##################################################################

    with Pool(processes=min(20, n_runs)) as pool:
        res = pool.starmap(func, parameters)

    save_dir = 'results'
    save_name = 'baseline_mountain_car'

    savefig_mean_std(
        [item[0] for item in res],
        eval_interval=1,
        save_dir=save_dir,
        save_name=save_name,
        xlabel='Number of actions',
        ylabel='Number of episodes'
    )

    savefig_mean_std(
        [np.array(item[1]) for item in res],
        eval_interval=1,
        save_dir=save_dir,
        save_name=save_name,
        xlabel='Number of episodes',
        ylabel='Mean squared error'
    )

    savefig_mean_std(
        [-np.array(item[2]) for item in res],
        eval_interval=1,
        save_dir=save_dir,
        save_name=save_name,
        xlabel='Number of episodes',
        ylabel='Rewards'
    )
