from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil

import numpy as np
import ray
from population_server.payoff_table import PolicySpec
from population_server.worker import LearnerManagerInterface
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR

from pomprl.rl.common.cloud_storage import connect_minio_client, maybe_download_object, get_tune_sync_to_cloud_fn, \
    upload_file, MINIO_DEFAULT_SAVE_PATH
from pomprl.rl.sac.sac import SACTrainer
from pomprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from socket import gethostname
from pomprl.scripts.population_server.utils.metanash import get_fp_metanash_for_latest_payoff_table
from pomprl.util import datetime_str, ensure_dir, with_updates
from pomprl.rl.envs.stratego.stratego_spatial_multiagent_env import SpatialStrategoMultiAgentEnv, BARRAGE, SHORT_BARRAGE, STANDARD, SHORT_STANDARD, FIVES

import os
import time
from multiprocessing import Pool, cpu_count, Lock, get_logger
from pomprl.util import pretty_print
from population_server.worker.evaluator import EvaluatorManagerInterface, FalseConfirmationError
from pomprl.scripts.poker_parallel_algos.evaluators.evaluator_utils import make_get_policy_fn, eval_policy_matchup
from pomprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from termcolor import colored
import itertools
from pomprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv
from pomprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

try:
    POKER_GAME_VERSION = os.environ["POKER_GAME_VERSION"]
except KeyError:
    raise ValueError("POKER_GAME_VERSION environment variable needs to be set")

OBSERVATION_MODE = PARTIALLY_OBSERVABLE
CHANNEL_MODE = 'extended'

ACCEPTED_OPPONENT_POLICY_CLASSES = [SACDiscreteTFPolicy]
ACCEPTED_OPPONENT_POLICY_CLASS_NAMES = [cls.__name__ for cls in ACCEPTED_OPPONENT_POLICY_CLASSES]
ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS = [POKER_ARCH1_MODEL_CONFIG_KEY]

MANAGER_SERVER_HOST = os.getenv("MANAGER_SERVER_HOST", 'localhost')
try:
    MANAGER_PORT = os.environ["MANAGER_PORT"]
    print(colored(f"MANAGER_PORT is {MANAGER_PORT}", "yellow"))
except KeyError:
    raise ValueError("MANAGER_PORT environment variable needs to be set")

EVAL_POOL_PROCESSES = int(os.environ['NUM_EVAL_WORKERS'])

WAIT_SECONDS_BEFORE_TRYING_AGAIN_IF_NO_MATCHUPS = 2

if POKER_GAME_VERSION in [LEDUC_POKER, KUHN_POKER]:
    POKER_ENV_CONFIG = {
        'version': POKER_GAME_VERSION,
        'env_class': PokerMultiAgentEnv
    }
    ENV_CLASS = PokerMultiAgentEnv
elif POKER_GAME_VERSION == 'barrage':
    POKER_ENV_CONFIG = {
        'version': SHORT_BARRAGE,
        'observation_mode': OBSERVATION_MODE,
        'channel_mode': CHANNEL_MODE,
        'repeat_games_from_other_side': True,
        'random_player_assignment': False,
        'human_inits': True,
        'penalize_ties': False,
        'env_class': SpatialStrategoMultiAgentEnv
    }
    ENV_CLASS = SpatialStrategoMultiAgentEnv
elif POKER_GAME_VERSION == 'fives':
    POKER_ENV_CONFIG = {
        'version': FIVES,
        'observation_mode': OBSERVATION_MODE,
        'channel_mode': CHANNEL_MODE,
        'repeat_games_from_other_side': True,
        'random_player_assignment': False,
        'human_inits': False,
        'penalize_ties': False,
        'env_class': SpatialStrategoMultiAgentEnv
    }
    ENV_CLASS = SpatialStrategoMultiAgentEnv
else:
    raise NotImplementedError(f"Not setup for game {POKER_GAME_VERSION}")

logger = logging.getLogger(__name__)


def perform_eval_matchups_as_they_are_available(i):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    # if os.getenv("EVALUATOR_USE_GPU") == 'true':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(i % len(''.join(i for i in os.environ['CUDA_VISIBLE_DEVICES'] if i.isdigit())))

    minio_client = connect_minio_client(endpoint=MINIO_ENDPOINT,
                                        access_key=MINIO_ACCESS_KEY,
                                        secret_key=MINIO_SECRET_KEY)

    worker_id = f"evaluator_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"

    manager_interface = EvaluatorManagerInterface(server_host=MANAGER_SERVER_HOST,
                                                  port=MANAGER_PORT,
                                                  worker_id=worker_id,
                                                  minio_client=minio_client,
                                                  minio_bucket_name=BUCKET_NAME,
                                                  minio_local_dir=MINIO_DEFAULT_SAVE_PATH)

    logger.info(f"Started worker \'{worker_id}\'")

    env = ENV_CLASS(env_config=POKER_ENV_CONFIG)

    while True:
        matchup = manager_interface.get_eval_matchup(infinite_retry_on_error=True)
        if matchup is None:
            # no matchups available right now, wait a bit and try again
            time.sleep(WAIT_SECONDS_BEFORE_TRYING_AGAIN_IF_NO_MATCHUPS)
            continue

        logger.info(f"[{worker_id}] Evaluating Matchup:\n{pretty_print(matchup)}")

        as_policy: PolicySpec = matchup['as_policy']
        against_policy: PolicySpec = matchup['against_policy']
        num_games_to_play = matchup['num_games']

        get_as_policy_fn = make_get_policy_fn(model_weights_object_key=as_policy.key,
                                              model_config_object_key=as_policy.config_key,
                                              policy_name=as_policy.key,
                                              policy_class_name=as_policy.class_name,
                                              minio_client=minio_client,
                                              minio_bucket_name=BUCKET_NAME,
                                              download_lock=download_lock,
                                              manual_config=None,
                                              process_id=i)

        get_against_policy_fn = make_get_policy_fn(model_weights_object_key=against_policy.key,
                                                   model_config_object_key=against_policy.config_key,
                                                   policy_name=against_policy.key,
                                                   policy_class_name=against_policy.class_name,
                                                   minio_client=minio_client,
                                                   minio_bucket_name=BUCKET_NAME,
                                                   download_lock=download_lock,
                                                   manual_config=None,
                                                   process_id=i)

        as_policy_payoff, tie_percentage = eval_policy_matchup(
            get_policy_fn_a=get_as_policy_fn,
            get_policy_fn_b=get_against_policy_fn,
            env=env,
            stratego_env_config=POKER_ENV_CONFIG,
            games_per_matchup=num_games_to_play)

        logger.info(f"\n\nFinal Result for {as_policy.key}\nvs\n{against_policy.key}\n{as_policy_payoff}\n\n")

        try:
            manager_interface.submit_eval_matchup_result(as_policy_key=as_policy.key,
                                                         against_policy_key=against_policy.key,
                                                         as_policy_avg_payoff=as_policy_payoff,
                                                         games_played=num_games_to_play,
                                                         infinite_retry_on_error=True)
        except FalseConfirmationError as err:
            logger.warning(f"[{worker_id}] Got False confirmation from manager:\n{err}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    logger.info("\n\n\n\n\n__________________________________________\n"
                f"LAUNCHED FOR {POKER_GAME_VERSION}\n"
                f"__________________________________________\n\n\n\n\n")

    def worker_process_init(lock):
        global download_lock
        download_lock = lock

    download_lock = Lock()
    eval_pool = Pool(processes=EVAL_POOL_PROCESSES, maxtasksperchild=1,
                     initializer=worker_process_init, initargs=(download_lock,))
    logger.info(f"Starting {EVAL_POOL_PROCESSES} eval workers")
    try:
        eval_pool.map(perform_eval_matchups_as_they_are_available, range(EVAL_POOL_PROCESSES))
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: stopping")
        eval_pool.terminate()
