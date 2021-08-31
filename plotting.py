import numpy as np
import matplotlib.pyplot as plt
from environment import State


def plot_team(state:State, team_idx:int, ax=None):
    targets_team = np.array(state.teams[team_idx].targets) if state.teams[team_idx].targets is not None else None

    healths = [state.teams[t].healths for t in range(len(state.teams))]
    n_players = np.array([len(h) for h in healths])
    is_team = np.concatenate([np.ones(n_players[t])*t for t in range(len(n_players))]) == team_idx
    healths = np.concatenate(healths)
    is_alive = healths > 1e-5

    base_colors = ['b', 'r', 'g']
    colors = np.concatenate([np.repeat(base_colors[t], n_players[t]) for t in range(len(n_players))])

    is_figure = True if ax is None else False
    if is_figure:
        fig, ax = plt.subplots(1, 1)

    y = np.zeros_like(is_team, dtype=np.float32)
    x = np.invert(is_team).astype(int)

    for b in [True, False]:
        y[is_team == b] = np.arange((is_team == b).sum()) * 1 / ((is_team == b).sum() - 1)

        mask_alive = (is_team == b) & is_alive
        mask_dead = (is_team == b) & np.invert(is_alive)

        ax.scatter(x[mask_alive], y[mask_alive], marker='o', c=colors[mask_alive], s=100)
        ax.scatter(x[mask_dead], y[mask_dead], marker='x', c=colors[mask_dead], s=100)

        alignment = 'right' if b else 'left'
        x_offset = -0.05 if b else 0.05
        for i, h in enumerate(healths[is_team == b]):
            ax.text(x[is_team == b][i] + x_offset, y[is_team == b][i] - 0.0155, str(round(h)), horizontalalignment=alignment)

    # Arrows
    if targets_team is not None:
        ax.quiver(
            x[is_team & is_alive]+0.1,
            y[is_team & is_alive],
            np.ones_like(x[is_team & is_alive])-0.1,
            y[np.invert(is_team)][targets_team[np.where(is_team & is_alive)]] - y[is_team & is_alive],
            scale=1.15,
            scale_units='xy',
            angles='xy',
        )

    ax.set_xlim([-0.2, 1.2])
    ax.set_xticks([])
    ax.set_yticks([])
    if is_figure:
        plt.show()


def plot_state(state:State):
    n_teams = len(state.teams)
    fig, axs = plt.subplots(1, n_teams, figsize=(5*n_teams, 5))
    for t in range(n_teams):
        plot_team(state, t, axs[t])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
