import random

def TD_Algorithm(MDP, agent, pi, delta=1e-8, seed=0):
    random.seed(seed)

    v = agent.get_initial_state_values(terminal_states=True)

    n_steps = {s: 1 for s in MDP.S}
    diffs = {s: float('inf') for s in MDP.S}

    n_episodes = 0
    while True:
        s = MDP.get_initial_state()
        while s not in MDP.terminal_states:
            a = agent.get_action(s, pi)
            next_state = MDP.get_next_state(s, a)
            r = MDP.get_reward(next_state=next_state)

            # update values
            k = 25
            alpha = 0.1 * k / (n_steps[s] + k)
            """
            1, 1: not converging well (2.08e8 steps, 0.244)
            1e1, 1: 2,767,010,204 steps
            1e2, 1: 2,851,900,000 ongoing 0.0041~0.0042 fluctuating
            1e3, 1: fluctuate (5.75e8 steps, fluctuating ~0.05)
            1e4, 1: fluctuate (5.13e8 steps, fluctuating ~0.09)
            1e2, 1.5: not converging (3.46e8 steps, nearly stopped at 0.0173)
            1e3, 1.5:  2,164,900,000 ongoing 0.002
            1e4, 1.5:  2,265,100,000 ongoing 0.0004
            1e5, 1.5: took 1,480,066,935 steps
            """
            #alpha = 0.01

            value = v[s]
            v[s] = v[s] + alpha * (r + MDP.gamma * v[next_state] - v[s])
            diffs[s] = abs(v[s] - value)
            #print(state, action, next_state, reward, value, values[state])

            n_steps[s] += 1
            s = next_state


        n_episodes += 1
        #if n_episodes % 100000 == 0:
        #    print([n_episodes], max(diffs.values()), max([abs(v[state] - MDP.optimal_values[state]) for state in MDP.S]))
        #    MDP.print_values(v)

        #if max([abs(v[state] - MDP.optimal_values[state]) for state in MDP.S]) < epsilon:
        #    break

        # break if max norm of diffs < delta
        if max(diffs.values()) < delta:
            break

    return v, n_episodes

