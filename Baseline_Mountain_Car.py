
from MDPs.mountain_car import get_Mountain_Car_MDP
from agents.MLP_agent import MLP_Agent
from algorithms.Baseline import Baseline_Algorithm
import time
from tools.tools import time_hr_min_sec

if __name__ == '__main__':

    for i in range(1000):
        w_hidden_sizes = [50] * 2
        theta_hidden_sizes = [50] * 3
        w_lr = 1e-8
        theta_lr = 1e-8
        input_mode = 'default'
        n = 10 ** 5
        print(w_hidden_sizes, theta_hidden_sizes, w_lr, theta_lr)

        MDP = get_Mountain_Car_MDP(n_bins=100)
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

        start = time.time()
        returns = Baseline_Algorithm(
            MDP,
            agent,
            min_sigma=1.0,
            max_sigma=2.0,
            n=n,
            print_output=2
        )
        end = time.time()

        if len(returns) > 1:
            print('time: %d hr %d min %d sec' % time_hr_min_sec(end-start))
            break
        else:
            print('Retrying...')