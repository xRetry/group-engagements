import numpy as np
from typing import List


def pure_nash(rewards_row, rewards_col) -> List[tuple]:
    # Find maximum reward values for row and column
    max_row = rewards_row.max(axis=0)[:, None]
    max_col = rewards_col.max(axis=1)[:, None]
    # Find all occurrences of maximum reward values in specific row/column
    idx_row = list(map(tuple, np.argwhere(rewards_row == max_row.T)))
    idx_col = list(map(tuple, np.argwhere(rewards_col == max_col)))
    # Select coordinates which occur for both players
    pure = [t for t in idx_row if t in idx_col]
    return pure


def mixed_nash(rewards: np.ndarray) -> np.ndarray:  # TODO: check behavior for non-symmetric matrices
    # Add ones to bottom row of design matrix
    A = np.row_stack((rewards, np.ones(rewards.shape[1])))
    # Set up y array -> [0, ..., 0, 1]
    b = np.zeros(rewards.shape[0]+1)
    b[-1] = 1
    # Add right column to design matrix
    A = np.column_stack((A, b-1))
    # Solve system of linear equations
    x = np.dot(np.linalg.pinv(A), b)
    return x


def eval_mixed(mixed_row: np.ndarray, mixed_col: np.ndarray, rewards_row: np.ndarray, rewards_col: np.ndarray) -> (np.ndarray, np.ndarray):
    # Calculate the joint probabilities for each action pair
    p = np.dot(mixed_row[:-1, None], mixed_col[:-1, None].T)
    # Multiply joint probabilities with respective rewards
    rewards_row_adj = p * rewards_row
    rewards_col_adj = p * rewards_col
    return rewards_row_adj, rewards_col_adj


if __name__ == '__main__':
    pass
