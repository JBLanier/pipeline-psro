import numpy as np
import logging
from mprl.utility_services.worker.base_interface import BaseClientManagerInterface
from mprl.utility_services.lock_server.lock_client_interface import LockServerInterface

logger = logging.getLogger(__name__)

def _get_br_to_strat(strat, payoffs):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    return br


def fictitious_play(iters, payoffs):
    dim = len(payoffs)
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = _get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps


def get_rectified_selection_probs_for_policy_key(payoff_table, policy_key, fp_iters):

    policy_index = payoff_table.get_policy_spec_for_key(policy_key).get_payoff_matrix_index()
    payoff_matrix = payoff_table.get_payoff_matrix()

    as_policy_against_opponents_row = payoff_matrix[policy_index]
    opponent_indexes_to_include = []
    for i, payoff_against_opponent in enumerate(as_policy_against_opponents_row):
        if payoff_against_opponent >= 0.0:
            opponent_indexes_to_include.append(i)

    original_selection_probs = get_fp_metanash_for_payoff_table(payoff_table=payoff_table,
                                                        fp_iters=fp_iters)
    print(f"original selection probs: {original_selection_probs}")
    out_selection_probs = np.zeros_like(original_selection_probs)
    for include_index in opponent_indexes_to_include:
        out_selection_probs[include_index] = original_selection_probs[include_index]
    out_selection_probs = out_selection_probs / sum(out_selection_probs)
    assert np.isclose(sum(out_selection_probs), 1.0)

    return out_selection_probs


def get_fp_metanash_for_payoff_table(payoff_table,
                                 fp_iters,
                                 accepted_opponent_policy_class_names=None,
                                 accepted_opponent_model_config_keys=None,
                                 add_payoff_matrix_noise_std_dev=None,
                                 mix_with_uniform_dist_coeff=None,
                                 only_first_n_policies=None,
                                 leave_out_indexes=None,
                                 p_or_lower_rounds_to_zero=1e-4):

    filtered_payoff_matrix = payoff_table.get_filtered_payoff_matrix(
        accepted_policy_class_names=accepted_opponent_policy_class_names,
        accepted_model_config_keys=accepted_opponent_model_config_keys,
        add_payoff_matrix_noise_std_dev=add_payoff_matrix_noise_std_dev,
        only_first_n_policies=only_first_n_policies,
        leave_out_indexes=leave_out_indexes
    )

    logger.debug(f"Performing {fp_iters} iters of fictitious play on payoff table")
    averages, exps = fictitious_play(iters=fp_iters, payoffs=filtered_payoff_matrix)
    logger.debug(f"Finished fictitious play on payoff table")

    selection_probs = np.copy(averages[-1])

    if mix_with_uniform_dist_coeff is not None and mix_with_uniform_dist_coeff > 0:
        uniform_dist = np.ones_like(selection_probs) / len(selection_probs)
        selection_probs = mix_with_uniform_dist_coeff * uniform_dist + (
                1.0 - mix_with_uniform_dist_coeff) * selection_probs

    assert np.isclose(sum(selection_probs), 1.0)

    for i, prob in enumerate(selection_probs):
        if prob <= p_or_lower_rounds_to_zero:
            selection_probs[i] = 0.0
    selection_probs = selection_probs / sum(selection_probs)
    assert np.isclose(sum(selection_probs), 1.0)

    return selection_probs


def get_fp_metanash_for_latest_payoff_table(
        manager_interface: BaseClientManagerInterface,
        fp_iters,
        accepted_opponent_policy_class_names=None,
        accepted_opponent_model_config_keys=None,
        add_payoff_matrix_noise_std_dev=None,
        mix_with_uniform_dist_coeff=None,
        only_first_n_policies=None,
        leave_out_indexes=None,
        p_or_lower_rounds_to_zero=1e-4):
    payoff_table, payoff_table_key = manager_interface.get_latest_payoff_table(infinite_retry_on_error=True)
    if payoff_table is None:
        logger.debug("Manager payoff table is empty right now.")
        return None, None, None

    selection_probs = get_fp_metanash_for_payoff_table(payoff_table=payoff_table,
                                                       fp_iters=fp_iters,
                                                       accepted_opponent_policy_class_names=accepted_opponent_policy_class_names,
                                                       accepted_opponent_model_config_keys=accepted_opponent_model_config_keys,
                                                       add_payoff_matrix_noise_std_dev=add_payoff_matrix_noise_std_dev,
                                                       mix_with_uniform_dist_coeff=mix_with_uniform_dist_coeff,
                                                       only_first_n_policies=only_first_n_policies,
                                                       leave_out_indexes=leave_out_indexes,
                                                       p_or_lower_rounds_to_zero=p_or_lower_rounds_to_zero)

    return selection_probs, payoff_table, payoff_table_key


def get_unreserved_policy_key_with_priorities(
        lock_server_interface: LockServerInterface,
        policy_keys,
        policy_priorities):

    keys_in_order_of_highest_priority_first = [key for _, key in sorted(zip(policy_priorities, policy_keys), key=lambda pair: pair[0], reverse=True)]
    reserved_key_or_none = lock_server_interface.try_to_reserve_item_from_list(
        possible_item_names_in_order_of_highest_priority_first=keys_in_order_of_highest_priority_first)
    return reserved_key_or_none
