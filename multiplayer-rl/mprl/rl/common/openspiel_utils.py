from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray

from collections import namedtuple
import numpy as np
import random

from ray.rllib.agents.trainer import Trainer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from ray.tune.logger import pretty_print

from mprl.rl.envs.opnspl.poker_br_policy import PokerOracleBestResponsePolicy
from mprl.rl.common.stratego_model import STRATEGO_MODEL
from mprl.rl.common.stratego_preprocessor import STRATEGO_PREPROCESSOR
from mprl.rl.ppo.ppo_custom_eval_trainer import PPOCustomEvalTrainer
from mprl.rl.ppo.ppo_stratego_model_policy import PPOStrategoModelTFPolicy
from mprl.rl.common.util import numpy_unpack_obs
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV
from mprl.rl.envs.opnspl.measure_exploitability_eval_callback import openspiel_policy_from_nonlstm_rllib_policy
from mprl.rl.envs.opnspl.util import policy_to_dict_but_we_can_actually_use_it
from mprl.rl.envs.opnspl.poker_multiagent_env import PokerMultiAgentEnv

from open_spiel.python.policy import tabular_policy_from_policy
from open_spiel.python import policy
import pyspiel

tf = try_import_tf()

RL_BR_POLICY = "rl_br_policy"
ORACLE_BR_POLICY = "oracle_br_policy"
EXPLOIT_POLICY = "exploit_policy"

# Used to return tuple actions as a list of batches per tuple element
TupleActions = namedtuple("TupleActions", ["batches"])

POLICY_TARGETS = "policy_targets"

OBSERVATION = 'observation'
VALID_ACTIONS_MASK = 'valid_actions_mask'


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PokerOpenSpeilPolicy(Policy):
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
        self.policy_dict = None

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


def get_openspeil_format_rl_br_policy(game_name, br_player_id, policy_to_exploit, policy_to_exploit_player_id):
    ray.init(local_mode=True, ignore_reinit_error=True)

    poker_game_version = game_name
    observation_mode = "partially_observable"

    poker_env_config = {
        'version': poker_game_version,
        'fixed_players': True
    }

    make_env_fn = lambda env_config: PokerMultiAgentEnv(env_config)
    temp_env = make_env_fn(poker_env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    model_config = {
        # === Options for custom models ===
        # Name of a custom preprocessor to use
        "custom_preprocessor": STRATEGO_PREPROCESSOR,
        # Name of a custom model to use
        "custom_model": STRATEGO_MODEL,

        "custom_options": {
            "mask_invalid_actions": True,
            "observation_mode": observation_mode,
            "q_fn": False
        },
    }

    def train_policy_mapping_fn(agent_id):
        if agent_id == br_player_id:

            # this is just to quickly check that we're matching the Oracle BR by having it
            # also play some games and report win stats too
            # TODO: you can remove this if-statement if you dont care about verifying against the oracle BR
            if random.random() < 0.1:
                return ORACLE_BR_POLICY

            return RL_BR_POLICY

        elif agent_id == policy_to_exploit_player_id:
            return EXPLOIT_POLICY
        else:
            raise ValueError(f"The env requested a policy for a player ID of {agent_id} "
                             f"but the BR policy has a player ID of {br_player_id} "
                             f"and the exploit policy has player ID of {policy_to_exploit_player_id}")

    trainer_config = {
        "log_level": "INFO",
        "num_workers": 0,  # 0 means a single worker instance in the same process as the optimizer
        "memory_per_worker": 1419430400,
        "num_gpus": 0,  # (GPUs for training) not using gpus for anything by default
        "num_gpus_per_worker": 0,  # (GPUs per experience gathering worker process, can be a fraction)
        "num_envs_per_worker": 1,
        "env": POKER_ENV,
        "env_config": poker_env_config,

        "multiagent": {
            "policies": {
                RL_BR_POLICY: (PPOStrategoModelTFPolicy, obs_space, act_space, {
                    # the config dicts in these "policies" override any non-policy-specific params
                    'model': model_config,
                    "lr": 0.001,
                }),

                # TODO: you can remove the ORACLE BR policy here if you dont want to verify against it
                # (there are two other TODO's in this file with Oracle BR stuff you can remove)
                ORACLE_BR_POLICY: (PokerOracleBestResponsePolicy, obs_space, act_space, {
                    'custom_preprocessor': STRATEGO_PREPROCESSOR,
                }),

                EXPLOIT_POLICY: (PokerOpenSpeilPolicy, obs_space, act_space, {
                    'custom_preprocessor': STRATEGO_PREPROCESSOR,
                }),
            },
            "policy_mapping_fn": train_policy_mapping_fn,
            "policies_to_train": [RL_BR_POLICY],
        },

        "metrics_smoothing_episodes": 1000,  # all reported RLLib metrics are averaged over this size episode window

        "gamma": 1.0,  # discount
        "num_sgd_iter": 10,  # train over train batch this many times each train() call
        "sgd_minibatch_size": 128,  #break train batch in to this size minibatches
        "train_batch_size": 500,
        "sample_batch_size": 10,  # each worker returns chunks of this size (PPO continues gathering exp until train_batch_size is gathered in total among all policies)
        "simple_optimizer": True,  # non-simple optimizer does multi-gpu/preloading fancy stuff
        "model": {
            "conv_filters": [],
            "fcnet_hiddens": [40, 40, 40],  # poker network size here
        },
    }

    trainer_class = PPOCustomEvalTrainer
    trainer: Trainer = trainer_class(config=trainer_config)

    # For technical reasons (can't pickle certain things),
    # I have to set the policy probs for openspiel-based exploit policy here
    def set_openspeil_exploit_policy_probs(worker):
        game = pyspiel.load_game(game_name)
        worker.policy_map[EXPLOIT_POLICY].set_policy_dict(
            policy_to_dict_but_we_can_actually_use_it(player_policy=policy_to_exploit,
                                                      game=game,
                                                      player_id=policy_to_exploit_player_id))

    trainer.workers.foreach_worker(set_openspeil_exploit_policy_probs)

    ###################
    # For technical reasons (can't pickle certain things),
    # I have to set the policy probs for openspiel-based BR policy here
    # TODO: you can remove this chunk of logic if you remove the other two bits of Oracle BR code in earlier lines
    local_br_policy = trainer.workers.local_worker().policy_map[ORACLE_BR_POLICY]
    local_exploit_rllib_policy = trainer.workers.local_worker().policy_map[EXPLOIT_POLICY]
    br_policy_probs_dict = local_br_policy.compute_best_response(policy_to_exploit=local_exploit_rllib_policy,
                                                                 br_only_as_player_id=br_player_id)
    def set_openspeil_oracle_br_policy_probs(worker):
        worker.policy_map[ORACLE_BR_POLICY].set_policy_dict(br_policy_probs_dict)
    trainer.workers.foreach_worker(set_openspeil_oracle_br_policy_probs)
    ####################

    iterations = 100
    for it in range(1, iterations + 1):
        result = trainer.train()
        print(f"Iteration {it} out of {iterations}")
        print(pretty_print(result))

    game = pyspiel.load_game(game_name)
    open_spiel_policy_from_callable = openspiel_policy_from_nonlstm_rllib_policy(
        openspiel_game=game, poker_game_version=poker_game_version,
        rllib_policy=trainer.workers.local_worker().policy_map[RL_BR_POLICY])

    return tabular_policy_from_policy(game=game, policy=open_spiel_policy_from_callable)


if __name__ == '__main__':
    game_name = "kuhn_poker"
    game = pyspiel.load_game(game_name)
    tabular_policy = policy.TabularPolicy(game)

    # rl_br_policy should have the same interface as a openspeil br policy from something like
    # open_spiel.python.algorithms.best_response.BestResponsePolicy

    rl_br_policy = get_openspeil_format_rl_br_policy(game_name=game_name,
                                                     br_player_id=0,
                                                     policy_to_exploit_player_id=1,
                                                     policy_to_exploit=tabular_policy)
