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

            value = v[s]
            v[s] = v[s] + alpha * (r + MDP.gamma * v[next_state] - v[s])
            diffs[s] = abs(v[s] - value)

            n_steps[s] += 1
            s = next_state

        n_episodes += 1

        # break if max norm of diffs < delta
        if max(diffs.values()) < delta:
            break

    return v, n_episodes

