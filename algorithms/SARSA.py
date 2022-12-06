import math
import random

def SARSA_Algorithm(
        MDP,
        agent,
        init_mode='optimistic', # 'zeros', 'random', 'optimistic'
        alpha_a=0.1,
        alpha_k=10,
        alpha_decay=1,
        epsilon=0.05,
        epsilon_decay = 0,
        sigma=10.0,
        n=int(1e6),
        eval_interval=0,
        learning_curve_b_steps=10000,
        learning_curve_c_type='MSE',
        max_steps_in_episode=float('inf'),
        print_output=False,
        seed=0
    ):
    # directory to save hyperparameter search plots
    #os.makedirs(save_dir, exist_ok=True)
    random.seed(seed)

    q = agent.get_initial_action_values(mode=init_mode, terminal_states=True)
    n_steps = {(s, a): 1 for s in MDP.S for a in MDP.A_list}
    diffs = {(s, a): float('inf') for s in MDP.S for a in MDP.A_list}

    eps = epsilon
    n_episodes = 0
    learning_curve_b, learning_curve_c = [], []
    for i in range(n):
        n_episodes += 1
        n_steps_in_episode = 0

        if epsilon_decay:
            eps = epsilon * math.exp(-n_episodes / epsilon_decay)
        sig = sigma * n_episodes / n

        s = MDP.get_initial_state()
        a = agent.get_action(state=s, q=q, eps=eps, sigma=sig)
        while s not in MDP.terminal_states and n_steps_in_episode < max_steps_in_episode:
            ns = MDP.get_next_state(state=s, action=a)
            r = MDP.get_reward(state=s, action=a, next_state=ns)
            na = agent.get_action(state=s, q=q, eps=eps, sigma=sig)

            # update values
            alpha = alpha_a * (alpha_k / (n_steps[(s, a)] + alpha_k)) ** alpha_decay
            #alpha = 1 / n_steps[state]
            #alpha = 0.01


            action_value = q[(s, a)]
            q[(s, a)] = q[(s, a)] + alpha * (r + MDP.gamma * q[(ns, na)] - q[(s, a)])
            diffs[s] = abs(q[(s, a)] - action_value)
            #print(s, a, r, ns, na, action_value, q[(s, a)], q[(ns, na)])
            #print(('sarsa', s, MDP.A[a]))#, end=' ')

            n_steps_in_episode += 1
            n_steps[(s, a)] += 1
            if len(learning_curve_b) < learning_curve_b_steps:
                learning_curve_b.append(n_episodes)
            s, a = ns, na

        if n_episodes % eval_interval == 0:
            if learning_curve_c_type == 'MSE':
                pi = agent.get_stochastic_pi_from_q(q, eps=eps, sigma=sigma)
                v = {s: sum([q[(s, a)] * pi[(s, a)] for a in MDP.A]) for s in MDP.S}
                error = sum([(v[s] - MDP.optimal_values[s]) ** 2 for s in MDP.S]) / len(MDP.S)
                learning_curve_c.append(error)
                if print_output:
                    print('n_episodes: %d, alpha: %.2e, eps: %.2e, error: %.6e' % (n_episodes, alpha, eps, error))
                    MDP.print_values(MDP.optimal_values)
                    MDP.print_values(v)
            elif learning_curve_c_type == 'n_steps':
                learning_curve_c.append(n_steps_in_episode)

        if print_output and learning_curve_c_type == 'n_steps' and n_episodes % 100 == 0:
            if MDP.explore_mode == 'epsilon-greedy':
                print('n_episodes: %d, alpha: %.2e, eps: %.2e, return: %.2f' % (
                    n_episodes, alpha, eps, sum(learning_curve_c[-100:]) / 100))
            else:
                print('n_episodes: %d, alpha: %.2e, sigma: %.2e, steps: %.2f' % (
                    n_episodes, alpha, sigma, sum(learning_curve_c[-100:]) / 100))

    return q, learning_curve_b, learning_curve_c