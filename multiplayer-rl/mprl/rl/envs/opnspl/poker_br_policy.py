from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import numpy as np
import logging
import collections

from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy
from ray.rllib.models.catalog import ModelCatalog
from mprl.rl.common.util import numpy_unpack_obs
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV
from mprl.utility_services.cloud_storage import maybe_download_object

from ray.tune.registry import ENV_CREATOR, _global_registry

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from mprl.rl.envs.opnspl.measure_exploitability_eval_callback import openspiel_policy_from_nonlstm_rllib_policy, tabular_policy_from_weighted_policies
from mprl.rl.envs.opnspl.util import policy_to_dict_but_we_can_actually_use_it
import pyspiel

# Used to return tuple actions as a list of batches per tuple element
TupleActions = namedtuple("TupleActions", ["batches"])

POLICY_TARGETS = "policy_targets"

OBSERVATION = 'observation'
VALID_ACTIONS_MASK = 'valid_actions_mask'

logger = logging.getLogger(__name__)

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class PokerOracleBestResponsePolicy(Policy):
    @override(Policy)
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space=observation_space, action_space=action_space, config=config)

        if config["custom_preprocessor"]:
            self.preprocessor = ModelCatalog.get_preprocessor_for_space(
                observation_space=self.observation_space.original_space,
                options={"custom_preprocessor": config["custom_preprocessor"]})
        else:
            raise ValueError("Custom preprocessor for PokerCFRPolicy needs to be specified on its passed config.")

        env_id = config['env']
        assert env_id == POKER_ENV

        env_config = config['env_config']
        self.game_version = env_config['version']

        self.game = pyspiel.load_game(self.game_version)

        self.policy_dict = None

        # information state has player as onehot in both kuhn and leduc
        # from c++ code:
        #  // Mark who I am.
        #   (*values)[player] = 1;

        # exit()
        # print("LOADED CFR POLICY EXPLOITABILITY:", exploitability.exploitability(game, self.tabular_policy))

    def compute_best_response(self, policy_to_exploit, br_only_as_player_id=None, policy_mixture_dict=None, set_policy_weights_fn=None):
        if policy_mixture_dict is None:
            openspiel_policy_to_exploit = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=self.game,
                                                                          poker_game_version=self.game_version,
                                                                          rllib_policy=policy_to_exploit)
        else:
            if set_policy_weights_fn is None:
                raise ValueError(
                    "If policy_mixture_dict is passed a value, a set_policy_weights_fn must be passed as well.")

            def policy_iterable():
                for weights_key in policy_mixture_dict.keys():
                    print("setting policy weights")
                    set_policy_weights_fn(weights_key)
                    print("making openspeil policy")
                    single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=self.game,
                                                                                         poker_game_version=self.game_version,
                                                                                         rllib_policy=policy_to_exploit)
                    print("made policy")
                    yield single_openspiel_policy

            openspiel_policy_to_exploit = tabular_policy_from_weighted_policies(game=self.game,
                                                                     policy_iterable=policy_iterable(),
                                                                     weights=policy_mixture_dict.values())
            print("made policy average")

        br_player_ids = [br_only_as_player_id] if br_only_as_player_id is not None else [0, 1]

        br = {player_id: BestResponsePolicy(game=self.game, player_id=player_id,
                                                  policy=openspiel_policy_to_exploit) for player_id in br_player_ids}

        policy_dict = {}
        for player_id, br_policy in br.items():
            policy_dict.update(policy_to_dict_but_we_can_actually_use_it(
                player_policy=br_policy, game=self.game, player_id=player_id))

        return policy_dict

    def set_policy_dict(self, policy_dict):
        self.policy_dict = policy_dict

    def _get_action_probs_for_infoset(self, infoset):
        action_probs = np.zeros(shape=(self.action_space.n,), dtype=np.float32)
        policy_lookup_val = self.policy_dict[str(np.asarray(infoset, dtype=np.float32).tolist())]
        for action, prob in policy_lookup_val:
            action_probs[action] = prob

        return action_probs

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions for the current policy.

        Arguments:
            obs_batch (np.ndarray): batch of observations
            state_batches (list): list of RNN state input batches, if any
            prev_action_batch (np.ndarray): batch of previous action values
            prev_reward_batch (np.ndarray): batch of previous rewards
            info_batch (info): batch of info objects
            episodes (list): MultiAgentEpisode for each obs in obs_batch.
                This provides access to all of the internal episode state,
                which may be useful for model-based or multiagent algorithms.
            kwargs: forward compatibility placeholder

        Returns:
            actions (np.ndarray): batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (list): list of RNN state output batches, if any, with
                shape like [STATE_SIZE, BATCH_SIZE].
            info (dict): dictionary of extra feature batches, if any, with
                shape like {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """

        # print("\n\n\n\n\n\ngame type provides info state: ", self.tabular_policy.game_type.provides_information_state)
        # print("game type provides observation: ", self.tabular_policy.game_type.provides_observation)

        obs_batch = numpy_unpack_obs(obs=np.asarray(obs_batch), space=self.observation_space.original_space,
                                     preprocessor=self.preprocessor)

        info_states = obs_batch["partial_observation"]
        valid_actions = obs_batch['valid_actions_mask']
        actions = []
        policy_probs = []

        for info_state, valid_mask in zip(info_states, valid_actions):
            if self.policy_dict is None:
                action_probs = valid_mask.copy() / sum(valid_mask)
            else:
                action_probs = self._get_action_probs_for_infoset(info_state)

            action = np.random.choice(range(self.action_space.n), p=action_probs)
            assert valid_mask[action] == 1.0
            actions.append(action)
            policy_probs.append(action_probs)

        return actions, [], {POLICY_TARGETS: np.asarray(policy_probs)}

    def compute_gradients(self, postprocessed_batch):
        """Computes gradients against a batch of experiences.

        Either this or learn_on_batch() must be implemented by subclasses.

        Returns:
            grads (list): List of gradient output values
            info (dict): Extra policy-specific values
        """
        pass

    def apply_gradients(self, gradients):
        """Applies previously computed gradients.

        Either this or learn_on_batch() must be implemented by subclasses.
        """
        pass

    def get_weights(self):
        """Returns model weights.

        Returns:
            weights (obj): Serializable copy or view of model weights
        """
        return None

    def set_weights(self, weights):
        """Sets model weights.

        Arguments:
            weights (obj): Serializable copy or view of model weights
        """
        pass

    def get_initial_state(self):
        """Returns initial RNN state for the current policy."""
        return []

    def get_state(self):
        """Saves all local state.

        Returns:
            state (obj): Serialized local state.
        """
        return self.get_weights()

    def set_state(self, state):
        """Restores all local state.

        Arguments:
            state (obj): Serialized local state.
        """
        self.set_weights(state)

    def on_global_var_update(self, global_vars):
        """Called on an update to global vars.

        Arguments:
            global_vars (dict): Global variables broadcast from the driver.
        """
        pass

    def export_model(self, export_dir):
        """Export Policy to local directory for serving.

        Arguments:
            export_dir (str): Local writable directory.
        """
        raise NotImplementedError

    def export_checkpoint(self, export_dir):
        """Export Policy checkpoint to local directory.

        Argument:
            export_dir (str): Local writable directory.
        """
        raise NotImplementedError


