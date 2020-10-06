from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil

import numpy as np
import ray
from mprl.utility_services.payoff_table import PolicySpec
from mprl.utility_services.worker import LearnerManagerInterface
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR

from mprl.utility_services.cloud_storage import connect_storage_client, maybe_download_object, get_tune_sync_to_cloud_fn, \
    upload_file, DEFAULT_LOCAL_SAVE_PATH
from mprl.rl.sac.sac import SACTrainer
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from socket import gethostname
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_latest_payoff_table
from mprl.utils import datetime_str, ensure_dir, with_updates
import os
import time
from multiprocessing import Pool, cpu_count, Lock, get_logger
from mprl.utils import pretty_print
from mprl.utility_services.worker import ConsoleManagerInterface, FalseConfirmationError
from mprl.scripts.poker_parallel_algos.evaluators.evaluator_utils import make_get_policy_fn, eval_policy_matchup
from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.rl.envs.opnspl.measure_nashconv_eval_callback import measure_nash_conv_nonlstm
import itertools
from mprl.utility_services.cloud_storage import maybe_download_object, connect_storage_client
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.rl.ppo.ppo_stratego_model_policy import PPOStrategoModelTFPolicy
from mprl.rl.common.stratego_preprocessor import STRATEGO_PREPROCESSOR, StrategoDictFlatteningPreprocessor
from ray.rllib.agents.trainer import with_common_config, with_base_config
from ray.rllib.models.catalog import MODEL_DEFAULTS
import ray
from ray.rllib.utils import try_import_tf
import json
import os
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, \
    PokerMultiAgentEnv
from mprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_latest_payoff_table

tf = try_import_tf()

POKER_GAME_VERSION = os.getenv("POKER_GAME_VERSION", KUHN_POKER)
OBSERVATION_MODE = PARTIALLY_OBSERVABLE

POLICY_CLASS = SACDiscreteTFPolicy
POLICY_CLASS_NAME = SACDiscreteTFPolicy.__name__
MODEL_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY

MANAGER_SEVER_HOST = "localhost"
MANAGER_PORT = os.getenv("MANAGER_PORT", 2727)

POKER_ENV_CONFIG = {
    'version': POKER_GAME_VERSION,
}

logger = logging.getLogger(__name__)


def measure_exploitability_of_metanashes_as_they_become_available():
    logger = get_logger()

    storage_client = connect_storage_client()

    worker_id = f"Exploitability_Tracker_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"

    manager_interface = ConsoleManagerInterface(server_host=MANAGER_SEVER_HOST,
                                                port=MANAGER_PORT,
                                                worker_id=worker_id,
                                                storage_client=storage_client,
                                                minio_bucket_name=BUCKET_NAME,
                                                minio_local_dir=DEFAULT_LOCAL_SAVE_PATH)

    logger.info(f"Started worker \'{worker_id}\'")

    # If you use ray for more than just this single example fn, you'll need to move ray.init to the top of your main()
    ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True, local_mode=True)

    model_config_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                      bucket_name=BUCKET_NAME,
                                                      object_name=MODEL_CONFIG_KEY,
                                                      force_download=False)

    with open(model_config_file_path, 'r') as config_file:
        model_config = json.load(fp=config_file)

    example_env = PokerMultiAgentEnv(env_config=POKER_ENV_CONFIG)

    logger.info("\n\n\n\n\n__________________________________________\n"
                f"LAUNCHED FOR {POKER_GAME_VERSION}\n"
                f"__________________________________________\n\n\n\n\n")

    obs_space = example_env.observation_space
    act_space = example_env.action_space

    preprocessor = StrategoDictFlatteningPreprocessor(obs_space=obs_space)
    graph = tf.Graph()
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}), graph=graph)

    def fetch_logits(policy):
        return {
            "behaviour_logits": policy.model.last_output(),
        }

    _policy_cls = POLICY_CLASS.with_updates(
        extra_action_fetches_fn=fetch_logits
    )

    with graph.as_default():
        with sess.as_default():
            policy = _policy_cls(
                obs_space=preprocessor.observation_space,
                action_space=act_space,
                config=with_common_config({
                    'model': with_base_config(base_config=MODEL_DEFAULTS, extra_config=model_config),
                    'env': POKER_ENV,
                    'env_config': POKER_ENV_CONFIG,
                    'custom_preprocessor': STRATEGO_PREPROCESSOR}))

    def set_policy_weights(weights_key):
        weights_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                     bucket_name=BUCKET_NAME,
                                                     object_name=weights_key,
                                                     force_download=False)
        policy.load_model_weights(weights_file_path)

    print("(Started Successfully)")

    last_payoff_table_key = None
    while True:
        payoff_table, payoff_table_key = manager_interface.get_latest_payoff_table(infinite_retry_on_error=True)
        if payoff_table_key == last_payoff_table_key:
            time.sleep(20)
            continue
        last_payoff_table_key = payoff_table_key

        metanash_probs, _, _ = get_fp_metanash_for_latest_payoff_table(manager_interface=manager_interface,
                                                                       fp_iters=20000,
                                                                       accepted_opponent_policy_class_names=[
                                                                           POLICY_CLASS_NAME],
                                                                       accepted_opponent_model_config_keys=[
                                                                           POKER_ENV_CONFIG],
                                                                       add_payoff_matrix_noise_std_dev=0.000,
                                                                       mix_with_uniform_dist_coeff=None,
                                                                       p_or_lower_rounds_to_zero=0.0)

        if metanash_probs is not None:
            policy_weights_keys = payoff_table.get_ordered_keys_in_payoff_matrix()

            policy_dict = {key: prob for key, prob in zip(policy_weights_keys, metanash_probs)}

            exploitabilitly = measure_nash_conv_nonlstm(rllib_policy=policy,
                                                        poker_game_version=POKER_GAME_VERSION,
                                                        policy_mixture_dict=policy_dict,
                                                        set_policy_weights_fn=set_policy_weights)
            print(f"Exploitability: {exploitabilitly}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    measure_exploitability_of_metanashes_as_they_become_available()
