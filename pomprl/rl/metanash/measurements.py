from open_spiel.python.egt import alpharank
from open_spiel.python.egt import alpharank_visualizer
from open_spiel.python.egt import utils, heuristic_payoff_table

from open_spiel.python.algorithms.projected_replicator_dynamics import projected_replicator_dynamics

import numpy as np


def get_alpha_rank_pi(payoff_table):

    # matrix must be symmetric
    assert len(np.shape(payoff_table)) == 2
    assert np.shape(payoff_table)[0] == np.shape(payoff_table)[1]

    payoff_tables = (payoff_table, payoff_table.T)
    payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_tables[0]),
                     heuristic_payoff_table.from_matrix_game(payoff_tables[1].T)]

    # Check if the game is symmetric (i.e., players have identical strategy sets
    # and payoff tables) and return only a single-player’s payoff table if so.
    # This ensures Alpha-Rank automatically computes rankings based on the
    # single-population dynamics.
    is_symmetric, payoff_tables = utils.is_symmetric_matrix_game(payoff_tables)

    assert is_symmetric

    # Compute Alpha-Rank
    (rhos, rho_m, pi, num_profiles, num_strats_per_population) = alpharank.compute(
        payoff_tables, alpha=1e2)

    for i in range(len(pi)):
        if np.isclose(pi[i], 0.0):
            pi[i] = 0.0

    return pi

def get_projected_replicator_dynamics_pi(payoff_table):
    # matrix must be symmetric
    assert len(np.shape(payoff_table)) == 2
    assert np.shape(payoff_table)[0] == np.shape(payoff_table)[1]

    payoff_tables = [payoff_table, payoff_table.T]
    # payoff_tables = [heuristic_payoff_table.from_matrix_game(payoff_tables[0]),
    #                  heuristic_payoff_table.from_matrix_game(payoff_tables[1].T)]

    # Check if the game is symmetric (i.e., players have identical strategy sets
    # and payoff tables) and return only a single-player’s payoff table if so.
    # This ensures Alpha-Rank automatically computes rankings based on the
    # single-population dynamics.
    is_symmetric, _ = utils.is_symmetric_matrix_game(payoff_tables)

    assert is_symmetric
    assert len(payoff_tables) == 2

    pi = projected_replicator_dynamics(payoff_tensors=payoff_tables,
                                  prd_initial_strategies=None,
                                  prd_iterations=int(1e6),
                                  prd_dt=1e-3,
                                  prd_gamma=0.0,
                                  average_over_last_n_strategies=None)


    return pi





def get_effective_population_diversity(payoff_table, pi):
    # https://arxiv.org/pdf/1901.08106.pdf

    # matrix must be symmetric
    assert len(np.shape(payoff_table)) == 2
    assert np.shape(payoff_table)[0] == np.shape(payoff_table)[1]

    # only positive values
    rectified_payoff_table = np.where(payoff_table < 0, 0, payoff_table)

    effective_diversity = np.dot(np.dot(pi.T, rectified_payoff_table), pi)
    effective_diversity = float(np.squeeze(effective_diversity))

    assert effective_diversity >= 0

    return effective_diversity


def get_relative_population_performance(payoff_table, pi1, pi2):
    # measures pi1 relative to pi2

    # https://arxiv.org/pdf/1901.08106.pdf

    # matrix must be symmetric
    assert len(np.shape(payoff_table)) == 2
    assert np.shape(payoff_table)[0] == np.shape(payoff_table)[1]

    relative_performance = np.dot(np.dot(pi1.T, payoff_table), pi2)
    relative_performance = float(np.squeeze(relative_performance))

    return relative_performance



if __name__ == '__main__':

    payoff_table = [[0,1, 1],
                    [0,0, 0],
                    [-1,-1, 0]]

    payoff_table = np.asarray(payoff_table)

    print(get_alpha_rank_pi(payoff_table))


    print(get_projected_replicator_dynamics_pi(payoff_table))