
from MDPs.cartpole import get_CartPole_MDP
from agents.MLP_agent import MLP_Agent
from algorithms.Baseline import Baseline_Algorithm
import time
from tools.tools import time_hr_min_sec
from tools.savefig import savefig_mean_std
import numpy as np
from multiprocessing import Pool, cpu_count
import torch

def func(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr, input_mode, n, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr)
    while True:
        start = time.time()

        MDP = get_CartPole_MDP(n_bins=5)
        agent = MLP_Agent(
            MDP=MDP,
            state_size=2,
            w_hidden_sizes=w_hidden_sizes,
            w_lr=w_lr,
            theta_hidden_sizes=theta_hidden_sizes,
            theta_lr=theta_lr,
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

        print('%.2f' % (time.time() - start), end=' ')
        if len(returns) == n:
            print(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr)
            return lc1, lc2, returns

if __name__ == '__main__':
    n_runs = 20

    w_hidden_sizes = [50] * 2
    theta_hidden_sizes = [50] * 3
    w_lr = 1e-7
    theta_lr = 3e-9
    input_mode = 'default'
    n = 10 ** 5

    ##################################################################
    #n_runs = 4
    #n = 10 ** 4
    ##################################################################

    parameters = [(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr, input_mode, n, _) for _ in range(n_runs)]
    res = func(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr, input_mode, n, 1)

    #with Pool(processes=min(20, n_runs)) as pool:
    #    res = pool.starmap(func, parameters)

    save_dir = 'results'
    save_name = '1211_baseline_cartpole'

    # savefig_mean_std(
    #     [item[0] for item in res],
    #     eval_interval=1,
    #     save_dir=save_dir,
    #     save_name=save_name+"_lc1",
    #     xlabel='Number of actions',
    #     ylabel='Number of episodes'
    # )

    # savefig_mean_std(
    #     [np.array(item[1]) for item in res],
    #     eval_interval=1,
    #     save_dir=save_dir,
    #     save_name=save_name+'_lc2',
    #     xlabel='Number of episodes',
    #     ylabel='Mean squared error'
    # )

    # savefig_mean_std(
    #     [-np.array(item[2]) for item in res],
    #     eval_interval=1,
    #     save_dir=save_dir,
    #     save_name=save_name+'_lc3',
    #     xlabel='Number of episodes',
    #     ylabel='number of steps'
    # )
