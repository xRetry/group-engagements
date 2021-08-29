import numpy as np
import itertools
import copy
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Team:
    healths: tuple
    dps: tuple
    timers: tuple
    change_time: tuple
    targets: tuple = None


@dataclass(frozen=True)
class State:
    teams: Tuple[Team]
    time: float = 0.
    is_terminated:bool = False
    rewards: tuple = None


class ExhaustiveAgent:
    float_cutoff: float

    def __init__(self):
        self.float_cutoff = 1e-5

    def next_targets(self, teams:Tuple[Team], idx_own:int) -> iter:
        targets_old = teams[idx_own].targets
        healths_own = np.array(teams[idx_own].healths)
        healths_enemy = []
        for i in range(len(teams)):
            if i != idx_own:
                healths_enemy.append(teams[i].healths)
        healths_enemy = np.concatenate(healths_enemy)

        # All alive enemies as possible next targets
        targets_available = self._idx_alive(healths_enemy)
        # Select all alive players to change target
        idx_needs_change = self._idx_alive(healths_own)

        if targets_old is not None:
            target_dead = np.isin(targets_old, targets_available) == False
            # Filter player selection further depending on target is dead
            idx_needs_change = idx_needs_change[np.isin(idx_needs_change, np.where(target_dead)[0])]
        else:
            # Initializing targets (will be overwritten)
            targets_old = np.zeros_like(healths_own, dtype=np.int32)

        if len(idx_needs_change) == 0:
            targets_iterator = iter([targets_old])
        else:
            combinations_iterator = itertools.product(*[targets_available for i in range(len(idx_needs_change))])
            targets_iterator = map(self._merge, itertools.cycle([targets_old]), itertools.cycle([idx_needs_change]), combinations_iterator)
        return targets_iterator

    def _idx_alive(self, healths:np.ndarray) -> np.ndarray:
        return np.where(healths > self.float_cutoff)[0]

    @staticmethod
    def _merge(targets_old, idx_new, targets_new):
        targets = np.array(targets_old)
        targets[idx_new] = targets_new
        return targets


class Environment:
    def __init__(self):
        self.float_cutoff = 1e-5

    def step(self, state:State, targets_idx:List[np.ndarray]) -> State:
        healths, dps_per_target, timers = self._aggregate_teams(state.teams, targets_idx)
        # Convert targets by index to one-hot matrix
        targets_new = self._targets_to_matrix(healths, targets_idx)
        # Calculate the combined DPS per target
        dps = self._dps_per_target(dps_per_target, targets_new)
        # Get the time duration to next kill
        dt_min = self._get_dt_min(healths, dps, timers)
        # Compute new health values
        healths_new = self._update_healths(healths, dps, dt_min)

        # Build teams with new values
        teams_new = []
        healths_final = []
        idx = 0
        for t in range(len(state.teams)):
            n_players_team = len(targets_idx[t])
            healths_team = healths_new[idx:idx+n_players_team]
            healths_final.append(healths_team)
            teams_new.append(
                Team(
                    healths=tuple(healths_team),
                    dps=tuple(state.teams[t].dps),
                    targets=tuple(targets_idx[t]),
                    timers=tuple(timers[idx:idx+n_players_team]),
                    change_time=tuple(state.teams[t].change_time)
                )
            )
            idx += n_players_team

        rewards, is_terminated = self._check_termination(healths_final)

        state_new = State(
            time=state.time + dt_min,
            teams=tuple(teams_new),
            is_terminated=is_terminated,
            rewards=tuple(rewards)
        )

        return state_new

    @staticmethod
    def _aggregate_teams(teams: Tuple[Team], targets_idx: List[np.ndarray]):
        healths = []
        dps_per_target = []
        timers = []
        for t in range(len(teams)):
            healths.append(teams[t].healths)
            dps_per_target.append(teams[t].dps)

            is_changing = np.equal(targets_idx[t], teams[t].targets)
            timers_team = np.array(teams[t].timers)
            timers_team[is_changing] = np.array(teams[t].change_time)[is_changing]
            timers.append(timers_team)
        return np.concatenate(healths), np.concatenate(dps_per_target), np.concatenate(timers)

    def _targets_to_matrix(self, healths: np.ndarray, targets:List[np.ndarray]) -> np.ndarray:
        # Get player counts
        n_players = [len(t) for t in targets]
        # Determine index offsets
        idx_offsets = np.cumsum([0]+n_players)
        # Initialize target matrix
        matrix = np.zeros((sum(n_players), sum(n_players)))
        # Fill matrix with targets
        for t in range(len(targets)):
            targets_team = copy.copy(targets[t])
            # Adjust target indices according to offsets
            targets_team[targets_team >= idx_offsets[t]] += n_players[t]
            # Set target in matrix
            matrix[np.arange(idx_offsets[t], idx_offsets[t + 1]), targets_team] = 1
        # Deactivate dead players
        matrix[healths <= self.float_cutoff,:] = 0
        return matrix

    @staticmethod
    def _dps_per_target(dps_per_player: np.ndarray, targets: np.ndarray):
        # Compute combined DPS per target
        return np.dot(targets.T, dps_per_player)

    @staticmethod
    def _get_dt_min(healths: np.ndarray, dps: np.ndarray, timers: np.ndarray) -> float:
        # Determine all next kill times
        dt = np.divide(
            healths,
            dps,
            out=np.ones_like(dps) * np.inf,
            where=(dps > 0)
        ) + timers
        # Select lowest next time step
        dt_min = np.min(dt)
        return dt_min

    @staticmethod
    def _update_healths(healths: np.ndarray, dps_per_target:np.ndarray, dt:float) -> np.ndarray:
        return healths - dps_per_target * dt

    def _check_termination(self, healths: List[np.ndarray]) -> (np.ndarray, bool):
        # Count and sort alive players per team
        n_alive = [(h > self.float_cutoff).sum() for h in healths]
        n_alive_sorted = np.sort(n_alive)
        # Initialize rewards as zero
        rewards = np.zeros(len(healths))

        is_teminated = False
        # Check if everyone is dead in the team with second most alive players -> terminate
        if n_alive_sorted[-2] == 0:
            is_teminated = True
            # Check if someone is alive in the team with the most alive players -> rewards != 0
            if n_alive_sorted[-1] > 0:
                idx_winner = np.argmax(n_alive)
                is_alive = healths[idx_winner] > self.float_cutoff
                health_left = healths[idx_winner][is_alive].sum()
                # Losing teams get the negative remaining health as reward (to ensure E[r]=0 if random policy)
                rewards -= health_left
                # Winner gets the positive remaining health as rewards
                rewards[idx_winner] = health_left
        return rewards, is_teminated


if __name__ == '__main__':
    pass
