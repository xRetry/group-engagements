import numpy as np
import itertools
from environment import Team, State, ExhaustiveAgent, Environment
from typing import Dict, List, Callable, Iterable


class ExhaustiveSimulation:
    agent: ExhaustiveAgent
    env: Environment
    states: Dict[State, dict]
    init_state: State
    _n_teams: int

    def __init__(self, teams: List[Team]):
        # Set up environment
        self.env = Environment()
        # Set up agent (one for all teams)
        self.agent = ExhaustiveAgent()
        # Set up initial state
        self.init_state = State(
            teams=tuple(teams)
        )
        # Set up states memory
        self.states = {}
        # Use shorthand for amount of teams
        self._n_teams = len(teams)

    def run(self) -> None:
        # Initialization for recursive function calls
        self._exec_state(self.init_state)

    def eval(self, eval_function: Callable[[np.ndarray], Iterable]) -> (List[List[tuple]], List[State], np.ndarray):
        # Find the trajectory based on the eval function
        action_trace, reward = self._eval_state(self.init_state, eval_function)
        # Collect the corresponding states
        state_trace = self._sample_trajectory(action_trace)
        return action_trace, state_trace, reward

    def _exec_state(self, state:State) -> None:
        # Check if state already visited -> do nothing
        memory_check = self.states.get(state)
        if memory_check is not None:
            return None
        # Set up empty state summary
        state_summary = {
            'actions': [],
            'states': [],
        }
        # Return empty state summary if end state
        if state.is_terminated:
            self.states[state] = state_summary
            return None

        # Get action list from each team
        actions_per_team = [np.array(list(self.agent.next_targets(state.teams, t))) for t in range(len(state.teams))]
        # Initialize the state matrix (numpy object matrix for easier indexing)
        state_matrix = np.zeros([len(t) for t in actions_per_team], dtype=object)

        # Iterate through all action (index) combinations
        for idxs in itertools.product(*[range(len(a)) for a in actions_per_team]):
            # Get corresponding actions from indices
            action = [actions_per_team[i][idxs[i]] for i in range(len(idxs))]
            # Take step in environment
            next_state = self.env.step(state, action)
            # Evaluate resulting state
            self._exec_state(next_state)
            # Add state to matrix
            state_matrix[idxs] = next_state

        # Convert all actions to tuples and store in state summary
        state_summary['actions'] = [list(map(tuple, a)) for a in actions_per_team]
        # Store state matrix in state summary
        state_summary['states'] = state_matrix
        # Add state summary to memory
        self.states[state] = state_summary

    def _eval_state(self, state, eval_function: Callable[[np.ndarray], Iterable]) -> (list, np.ndarray):
        # Check if terminal state is reached -> return empty action list and rewards
        if state.is_terminated:
            return [], state.rewards
        # Extract next actions and states
        next_states = self.states[state]['states']
        next_actions = self.states[state]['actions']
        # Set up reward and (previous) actions matrix
        reward_matrix = np.zeros(list(next_states.shape))
        action_matrix = np.zeros_like(reward_matrix, dtype=object)
        reward_matrix = np.repeat(np.expand_dims(reward_matrix, -1), self._n_teams, axis=-1)
        # Filling up reward matrix and corresponding (previous) action matrix
        for idxs in itertools.product(*[range(len(a)) for a in next_actions]):
            # Evaluate next state given current action
            action_trace, reward = self._eval_state(next_states[idxs], eval_function)
            # Add resulting reward to reward matrix
            reward_matrix[idxs] = reward
            # Add previous action list to action matrix
            action_matrix[idxs] = action_trace
        # Select reward matrix indices based on provided eval function
        idx_sel = tuple(eval_function(reward_matrix))
        # Select actions based on selected indices
        actions_sel = [next_actions[idx_sel[i]][0] for i in range(len(idx_sel))]
        # Add actions to action list
        actions_list = [actions_sel] + action_matrix[idx_sel]
        return actions_list, reward_matrix[idx_sel]

    def _sample_trajectory(self, action_list: List[List[tuple]]) -> List[State]:
        # Set up state list with init state
        state_list = [self.init_state]
        # Iterate through all action-pairs in provided action list
        for a in action_list:
            # Get last entry in action list
            state = self.states[state_list[-1]]
            # Find indices of current action-pair for the state matrix
            idx = []
            for t in range(len(a)):
                actions_available = state['actions'][t]
                for i in range(len(actions_available)):
                    if a[t] == actions_available[i]:
                        idx.append(i)
                        break
            # Add current state to state list
            state_list.append(state['states'][tuple(idx)])
        return state_list


if __name__ == '__main__':
    pass
