
import matplotlib.pyplot as plt
import numpy as np
import os
from algorithms.Baseline import Baseline_Algorithm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from MDPs.gridworld import get_Gridworld_MDP
from MDPs.mountain_car import get_Mountain_Car_MDP
from agents.MLP_agent import MLP_Agent
import time
from tools.tools import time_hr_min_sec
from tools.savefig import savefig_mean_std

def func(MDP, agent, algorithm, n=int(1e4), save_dir='test'):
    MDP.calc_optimal_v_pi(agent)
    returns_list = []
    lc1_list = []
    lc2_list = []

    while len(returns_list) < 5:
        start = time.time()
        lc1, lc2, returns = algorithm(
            MDP,
            agent,
            min_sigma=1.0,
            max_sigma=2.0,
            n=n,
            print_output=0
        )
        end = time.time()

        if len(returns) > 1:
            print('time: %d hr %d min %d sec' % time_hr_min_sec(end-start))
            lc1_list.append(lc1)
            lc2_list.append(lc2)
            returns_list.append(returns)

    lc1_save_dir = os.path.join(save_dir, 'lc1')
    lc2_save_dir = os.path.join(save_dir, 'lc2')
    lc3_save_dir = os.path.join(save_dir, 'lc3')

    lc1_length = min([len(lc1) for lc1 in lc1_list])
    for i in range(len(lc1_list)):
        lc1_list[i] = lc1_list[i][:lc1_length]

    os.makedirs(lc1_save_dir, exist_ok=True)
    os.makedirs(lc2_save_dir, exist_ok=True)
    os.makedirs(lc3_save_dir, exist_ok=True)

    save_name = '%.4f_' % np.mean(returns[-100:]) + str(agent.w_hidden_sizes) + '_%.2e_' % agent.w_lr + \
                str(agent.theta_hidden_sizes) + '_%.2e_' % agent.theta_lr + '_%s' % agent.input_mode + '.png'

    savefig_mean_std(lc1_list, 1, lc1_save_dir, save_name)
    savefig_mean_std(lc2_list, 10, lc2_save_dir, save_name)
    savefig_mean_std(returns_list, 1, lc3_save_dir, save_name, ylim=(-10, 10))

    return returns

def main():
    n = 10000
    save_dir = 'hp_search/1210_gw1'
    print('save_dir:', save_dir)


    pool = ProcessPoolExecutor(max_workers=10)

    for input_mode in ['default']:
        for w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr in \
                [
                    ([20] * 2, [20] * 2, 3e-5, 3e-4),
                    ([20] * 2, [20] * 2, 1e-5, 3e-4),
                    ([50] * 2, [50] * 2, 1e-5, 3e-4),
                    ([50] * 2, [50] * 2, 1e-4, 1e-4),
                    ([50] * 2, [50] * 2, 3e-4, 3e-4),
                    ([50] * 2, [50] * 2, 3e-5, 1e-4),
                    ([50] * 2, [50] * 2, 1e-4, 3e-4),
                    ([20] * 2, [20] * 3, 1e-3, 1e-3),
                    ([20] * 2, [20] * 2, 1e-3, 3e-4),
                    ([50] * 2, [50] * 2, 3e-5, 3e-5),
                ]:
                    MDP = get_Gridworld_MDP()
                    agent = MLP_Agent(
                        MDP=MDP,
                        state_size=2,
                        w_hidden_sizes=w_hidden_sizes,
                        w_lr=w_lr,
                        theta_hidden_sizes=theta_hidden_sizes,
                        theta_lr=theta_lr,
                        input_mode=input_mode
                    )
                    #func(MDP, agent, Baseline_Algorithm, n, save_dir); #return
                    pool.submit(func, MDP, agent, Baseline_Algorithm, n, save_dir)

    """
    for input_mode in ['default']:
        for k in reversed([20, 50]):
            for w_hidden_sizes in reversed([[k] * 2]):
                for theta_hidden_sizes in reversed([[k] * 3]):
                    for w_lr in reversed([1e-3, 3e-4, 1e-4, 3e-5, 1e-5]):
                        for theta_lr in reversed([1e-3, 3e-4, 1e-4, 3e-5, 1e-5]):
                            MDP = get_Gridworld_MDP()
                            agent = MLP_Agent(
                                MDP=MDP,
                                state_size=2,
                                w_hidden_sizes=w_hidden_sizes,
                                w_lr=w_lr,
                                theta_hidden_sizes=theta_hidden_sizes,
                                theta_lr=theta_lr,
                                input_mode=input_mode
                            )
                            #func(MDP, agent, Baseline_Algorithm, n, save_dir); #return
                            pool.submit(func, MDP, agent, Baseline_Algorithm, n, save_dir)
    """



if __name__ == '__main__':
    main()