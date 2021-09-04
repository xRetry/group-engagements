import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
from environment import State
from typing import Optional, Dict, List


def plot_team(state:State, team_idx:int, targets:Optional[List[Dict[tuple, float]]]=None, ax=None):
    # Check if targets are provided, else grab from state
    if targets is None:
        targets_team = {state.teams[team_idx].targets, 1.}
    else:
        targets_team = targets[team_idx]
    # Collect all health values
    healths = [state.teams[t].healths for t in range(len(state.teams))]
    # Collect the amount of players per team
    n_players = np.array([len(h) for h in healths])
    # Create mask for current team
    is_team = np.concatenate([np.ones(n_players[t])*t for t in range(len(n_players))]) == team_idx
    # Concatenate health values of all teams together
    healths = np.concatenate(healths)
    # Create mask for alive players
    is_alive = healths > 1e-5
    # Define team colors for plotting
    base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:pink', 'tab:brown']
    # Create a color array according to team affiliation
    colors = np.concatenate([np.repeat(base_colors[t], n_players[t]) for t in range(len(n_players))])
    # Check if plot is part of another figure
    is_figure = True if ax is None else False
    if is_figure:
        fig, ax = plt.subplots(1, 1)
    # Initialize array with y-coordinates
    y = np.zeros_like(is_team, dtype=np.float32)
    # Create x-coordinates according to team affiliation
    x = np.invert(is_team).astype(int)

    # PLAYER POINTS AND DESCRIPTIONS
    # Loop through current team and current enemies
    for b in [True, False]:
        # Overwrite y-coordinates
        y[is_team == b] = np.arange((is_team == b).sum()) * 1 / ((is_team == b).sum() - 1)
        # Create combined mask for team selection and alive/dead
        mask_alive = (is_team == b) & is_alive
        mask_dead = (is_team == b) & np.invert(is_alive)
        # Plot points
        ax.scatter(x[mask_alive], y[mask_alive], marker='o', c=colors[mask_alive], s=100)
        ax.scatter(x[mask_dead], y[mask_dead], marker='x', c=colors[mask_dead], s=100)
        # Define health text alignment
        alignment = 'right' if b else 'left'
        # Define health text x-offset
        x_offset = -0.05 if b else 0.05
        # Plot healths next to points
        for i, h in enumerate(healths[is_team == b]):
            ax.text(x[is_team == b][i] + x_offset, y[is_team == b][i] - 0.015, str(round(h)), horizontalalignment=alignment)

    # TARGET ARROWS
    # Define colormap for arrows
    cmap = plt.cm.get_cmap('jet')
    # Set up list for legend handles and labels
    handles = []
    labels = []
    # Iterate through all target probabilities
    for target, prob in targets_team.items():
        # Skip if team targets are non-existent
        if target is None:
            continue
        # Convert targets to numpy array
        target = np.array(target)
        # Get arrow color from colormap according to probability
        color = cmap(prob)
        # Plot arrow
        ax.quiver(
            x[is_team & is_alive]+0.1,
            y[is_team & is_alive],
            np.ones_like(x[is_team & is_alive])-0.1,
            y[np.invert(is_team)][target[np.where(is_alive[is_team])]] - y[is_team & is_alive],
            color=color,
            scale=1.15,
            scale_units='xy',
            angles='xy',
            label=prob
        )
        # Add label and dummy handle to lists
        labels.append(prob)
        handles.append(plt.arrow(0, 0, 0, 0, color=color))

    # Define axis appearance
    ax.set_xlim([-0.2, 1.2])
    ax.set_xticks([])
    ax.set_yticks([])

    # LEGEND
    # Add legend if more targets exist
    if len(handles) > 1:
        # Convert to numpy array for indexing
        handles = np.array(handles)
        labels = np.array(labels)
        # Get indices of sorted target probabilities
        idx_sorted = np.argsort(labels)
        # Add legend with sorted entries
        ax.legend(handles[idx_sorted], labels[idx_sorted], loc='upper center', handler_map={mpatches.FancyArrow: HandlerPatch(patch_func=make_arrow_patch),})
    # Show plot if not part of other figure
    if is_figure:
        plt.show()


def make_arrow_patch(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    # Create arrow patch for legend from dummy array
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height)
    return p


def plot_state(state:State, targets:Optional[List[Dict[tuple, float]]]=None, axs=None):
    # Get targets from state if not provided
    if targets is None:
        targets = [{t.targets:1.} for t in state.teams]
    # Get the amount of teams
    n_teams = len(state.teams)
    # Create figure if not part of another one
    if axs is None:
        fig, axs = plt.subplots(1, n_teams, figsize=(5*n_teams, 5))
    # Iterate through all teams
    for t in range(n_teams):
        # Plot state from the perspective of current team
        plot_team(state, t, targets=targets, ax=axs[t])
    # Show plot if figure
    if axs is None:
        plt.tight_layout()
        plt.show()


def plot_engagement(states:List[State], actions:List[list]):
    # Get amount of teams and states
    n_teams = len(states[0].teams)
    n_states = len(states)
    # Add empty action at end of action list
    actions.append(None)
    # Create figure
    fig, axs = plt.subplots(n_states, n_teams)
    # Iterate through all state
    for s in range(n_states):
        # Get current actions
        actions_state = actions[s]
        # Reformat current actions if actions exist
        if actions_state is not None:
            actions_state = [{a_team: 1.} for a_team in actions_state if not isinstance(a_team, dict)]
        # Add state plots to figure
        plot_state(states[s], actions_state, axs=axs[s, :])
    # Show figure
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
