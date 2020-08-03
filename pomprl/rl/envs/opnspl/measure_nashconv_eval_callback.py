
from ray.rllib.policy import Policy
from pomprl.rl.common.stratego_model import StrategoModel
from pomprl.rl.envs.opnspl.poker_multiagent_env import PARTIAL_OBSERVATION, VALID_ACTIONS_MASK, VALID_ACTIONS_SHAPES, KUHN_POKER, LEDUC_POKER
from open_spiel.python.policy import Policy as OSPolicy, PolicyFromCallable, TabularPolicy
from open_spiel.python.algorithms.exploitability import nash_conv, exploitability
from pomprl.rl.common.stratego_preprocessor import StrategoDictFlatteningPreprocessor

import pyspiel
import numpy as np
from open_spiel.python.rl_environment import Environment, TimeStep


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def tabular_policy_from_weighted_policies(game, policy_iterable, weights):
    """Converts multiple Policy instances into an weighted averaged TabularPolicy.

    Args:
      game: The game for which we want a TabularPolicy.
      policy_iterable: An iterable that returns Openspiel policies
      weights: probabilities of selecting actions from each policy

    Returns:
      A averaged TabularPolicy over the policy_iterable.
    """

    empty_tabular_policy = TabularPolicy(game)

    assert np.isclose(1.0, sum(weights))

    # initially zero out all policy probs
    for state_index, state in enumerate(empty_tabular_policy.states):
        infostate_policy = [
            0
            for _ in range(game.num_distinct_actions())
        ]
        empty_tabular_policy.action_probability_array[
        state_index, :] = infostate_policy

    # add weighted probs from each policy we're averaging over
    for policy, weight, in zip(policy_iterable, weights):
        for state_index, state in enumerate(empty_tabular_policy.states):
            old_action_probabilities = empty_tabular_policy.action_probabilities(state)
            add_action_probabilities = policy.action_probabilities(state)
            infostate_policy = [
                old_action_probabilities.get(action, 0.) + add_action_probabilities.get(action, 0.) * weight
                for action in range(game.num_distinct_actions())
            ]
            empty_tabular_policy.action_probability_array[
            state_index, :] = infostate_policy

    # check that all action probs pers state add up to one in the newly created policy
    for state_index, state in enumerate(empty_tabular_policy.states):
        action_probabilities = empty_tabular_policy.action_probabilities(state)
        infostate_policy = [
            action_probabilities.get(action, 0.)
            for action in range(game.num_distinct_actions())
        ]

        assert np.isclose(1.0, sum(infostate_policy)), "INFOSTATE POLICY: {}".format(infostate_policy)

    return empty_tabular_policy


def openspiel_policy_from_nonlstm_rllib_policy(openspiel_game, poker_game_version, rllib_policy):

    preprocessor = StrategoDictFlatteningPreprocessor(obs_space=rllib_policy.observation_space.original_space)

    def policy_callable(state: pyspiel.State):

        valid_actions = state.legal_actions_mask()
        legal_actions_list = state.legal_actions()

        # assert np.array_equal(valid_actions, np.ones_like(valid_actions)) # should be always true at least for Kuhn

        info_state_vector = state.information_state_as_normalized_vector()
        obs = preprocessor.transform({
            PARTIAL_OBSERVATION: np.asarray(info_state_vector, dtype=np.float32),
            VALID_ACTIONS_MASK: np.asarray(valid_actions, dtype=np.float32)})

        _, _, action_info = rllib_policy.compute_single_action(obs=obs, state=[])

        action_probs = None
        for key in ['policy_targets', 'action_probs']:
            if key in action_info:
                action_probs = action_info[key]
        if action_probs is None:
            action_logits = action_info['behaviour_logits']
            action_probs = softmax(action_logits)

        legal_action_probs = []
        for idx in range(len(valid_actions)):
            if valid_actions[idx] == 1.0:
                legal_action_probs.append(action_probs[idx])

        return {action_name: action_prob for action_name, action_prob in zip(legal_actions_list, legal_action_probs)}

    return PolicyFromCallable(game=openspiel_game, callable_policy=policy_callable)


def measure_nash_conv_nonlstm(rllib_policy, poker_game_version, policy_mixture_dict=None, set_policy_weights_fn=None):
    if poker_game_version in [KUHN_POKER, LEDUC_POKER]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    if policy_mixture_dict is None:
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                      poker_game_version=poker_game_version,
                                                                      rllib_policy=rllib_policy)
    else:
        if set_policy_weights_fn is None:
            raise ValueError("If policy_mixture_dict is passed a value, a set_policy_weights_fn must be passed as well.")

        def policy_iterable():
            for weights_key in policy_mixture_dict.keys():
                set_policy_weights_fn(weights_key)

                single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                              poker_game_version=poker_game_version,
                                                                              rllib_policy=rllib_policy)
                yield single_openspiel_policy

        openspiel_policy = tabular_policy_from_weighted_policies(game=openspiel_game,
                                                                 policy_iterable=policy_iterable(),
                                                                 weights=policy_mixture_dict.values())

    nash_conv_result = exploitability(game=openspiel_game, policy=openspiel_policy)
    return nash_conv_result


def get_measure_nash_conv_nonlstm_eval_callback(eval_name, poker_game_version, measure_policy_ids):

    def measure_nash_conv_nonlstm_eval_callback(trainer, eval_metrics):
        eval_workers, eval_config = trainer.extra_eval_worker_sets_and_configs[eval_name]

        for measure_policy_id in measure_policy_ids:
            rllib_policy = eval_workers.local_worker().policy_map[measure_policy_id]

            nash_conv_result = measure_nash_conv_nonlstm(rllib_policy=rllib_policy,
                                                         poker_game_version=poker_game_version)

            eval_metrics[measure_policy_id + '_ground_truth_nashconv'] = nash_conv_result
            print("NASH CONV {}:".format(measure_policy_id), nash_conv_result)

    return measure_nash_conv_nonlstm_eval_callback


def get_measure_nash_conv_nonlstm_policy_mixture_eval_callback(eval_name, poker_game_version, measure_policy_ids):

    def measure_nash_conv_nonlstm_eval_callback(trainer, eval_metrics):
        eval_workers, eval_config = trainer.extra_eval_worker_sets_and_configs[eval_name]

        for measure_policy_id in measure_policy_ids:
            rllib_policy = eval_workers.local_worker().policy_map[measure_policy_id]

            nash_conv_result = measure_nash_conv_nonlstm(rllib_policy=rllib_policy,
                                                         poker_game_version=poker_game_version)

            eval_metrics[measure_policy_id + '_ground_truth_nashconv'] = nash_conv_result
            print("NASH CONV {}:".format(measure_policy_id), nash_conv_result)

    return measure_nash_conv_nonlstm_eval_callback


class NFSPPolicies(OSPolicy):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        # legal_actions = state.legal_actions(cur_player)
        #
        # self._obs["current_player"] = cur_player
        # self._obs["info_state"][cur_player] = (
        #     state.information_state_as_normalized_vector(cur_player))
        # self._obs["legal_actions"][cur_player] = legal_actions
        #
        # info_state = TimeStep(
        #     observations=self._obs, rewards=None, discounts=None, step_type=None)
        #
        # p = self._policies[cur_player].step(info_state, is_evaluation=True).probs

        prob_dict = self._policies[cur_player].action_probabilities(state, player_id)
        # prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def nfsp_measure_nash_conv_nonlstm(rllib_p0_and_p1_policies, poker_game_version):
    if poker_game_version in [KUHN_POKER, LEDUC_POKER]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)
    openspiel_env = Environment(poker_game_version, open_spiel_env_config)

    openspiel_policies = []

    for rllib_policy in rllib_p0_and_p1_policies:

        if not isinstance(rllib_policy, OSPolicy):
            openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                          poker_game_version=poker_game_version,
                                                                          rllib_policy=rllib_policy)
        else:
            openspiel_policy = rllib_policy

        openspiel_policies.append(openspiel_policy)

    nfsp_os_policy = NFSPPolicies(env=openspiel_env, nfsp_policies=openspiel_policies)

    nash_conv_result = exploitability(game=openspiel_game, policy=nfsp_os_policy)
    return nash_conv_result