import numpy as np
import itertools
from environment import Team, State, ExhaustiveAgent, Environment


def init():
    init_state = State(
        teams=[
            Team(
                healths=np.array([1000., 1000.]),
                dps=np.array([10., 10.]),
                timers=np.array([0., 0.]),
                change_time=np.array([1., 1.])
            ),
            Team(
                healths=np.array([1000., 1000.]),
                dps=np.array([10., 10.]),
                timers=np.array([0., 0.]),
                change_time=np.array([1., 1.])
            ),
        ]
    )
    agent = ExhaustiveAgent()
    env = Environment()
    return eval_state(init_state, env, agent)


def eval_state(state:State, env:Environment, agent:ExhaustiveAgent) -> dict:
    summary = {
        'actions': {},
        'state': state,
    }
    # Return state with no actions if end state
    if state.is_terminated:
        return summary

    # Get new action iterators from each team
    action_iterators = [agent.next_targets(state.teams, t) for t in range(len(state.teams))]
    # Iterate through all action combinations
    for action in itertools.product(*action_iterators):
        # Take step in environment
        next_state = env.step(state, action)
        # Evaluate resulting state
        next_summary = eval_state(next_state, env, agent)
        # Add action-state pair to summary
        summary['actions'][tuple(map(tuple, action))] = next_summary

    return summary


if __name__ == '__main__':
    result = init()
    pass
