
def value_iteration(MDP, agent, epsilon=0.0001):
    # initial value
    values = agent.get_initial_state_values()

    n_iter = 0
    while True:
        delta = 0
        new_values = dict()
        for state in MDP.S:
            new_values[state] = max(MDP.estimate_q_from_v(state, action, values) for action in MDP.A.keys())
            delta = max(delta, abs(new_values[state] - values[state]))
        values = new_values

        n_iter += 1
        if delta < epsilon:
            break

    return values, n_iter


