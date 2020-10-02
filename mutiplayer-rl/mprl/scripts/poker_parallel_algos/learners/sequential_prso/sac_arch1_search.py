from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import time

import dill
import numpy as np
import ray
from mprl.utility_services.payoff_table import PolicySpec
from mprl.utility_services.worker import LearnerManagerInterface, ConsoleManagerInterface
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR

from mprl.utility_services.cloud_storage import connect_storage_client, maybe_download_object, get_tune_sync_to_cloud_fn, \
    upload_file
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv
from mprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL
import string
import random
from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.rl.sac.sac import SACTrainer
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_latest_payoff_table
from mprl.utils import datetime_str, ensure_dir, with_updates
from socket import gethostname
from mprl import TUNE_SAVE_DIR

# log level for our code, not for Ray/rllib
logger = logging.getLogger(__name__)

tf = try_import_tf()

TRAIN_POLICY = "train_policy"
STATIC_POLICY = "static_policy"

POKER_GAME_VERSION = os.getenv("POKER_GAME_VERSION", KUHN_POKER)

OBSERVATION_MODE = PARTIALLY_OBSERVABLE

TRAINER_CLASS = SACTrainer
TRAIN_POLICY_CLASS = SACDiscreteTFPolicy
TRAIN_POLICY_MODEL_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY
SUBMISSION_POLICY_TAGS = [f"hostname {gethostname()}"]

STATIC_POLICY_CLASS = SACDiscreteTFPolicy
STATIC_POLICY_MODEL_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY

ACCEPTED_OPPONENT_POLICY_CLASSES = [SACDiscreteTFPolicy]
ACCEPTED_OPPONENT_POLICY_CLASS_NAMES = [cls.__name__ for cls in ACCEPTED_OPPONENT_POLICY_CLASSES]
ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS = [POKER_ARCH1_MODEL_CONFIG_KEY]

MANAGER_SEVER_HOST = os.getenv("MANAGER_SERVER_HOST", "localhost")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", 2727))

METANASH_FICTITIOUS_PLAY_ITERS = 20000
PSRO_EXPLORATION_COEFF = 0.0

SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = 0.01
SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS = int(20000)
SUBMISSION_THRESHOLD_STEPS_START = int(20000)

SUBMISSION_MIN_TIMESTEPS = int(20000)
SUBMISSION_MAX_TIMESTEPS = None
CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "")


if __name__ == "__main__":
    while True:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("\n\n\n\n\n__________________________________________\n"
                    f"LAUNCHED FOR {POKER_GAME_VERSION}\n"
                    f"__________________________________________\n\n\n\n\n")


        storage_client = connect_storage_client()

        ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True)
        logger.info("Ray Web UI at {}".format(ray.get_webui_url()))

        base_experiment_name = f"{CLOUD_PREFIX}learner_{POKER_GAME_VERSION}_sac_arch1_hparam_search_multexp"
        full_experiment_name = f"{base_experiment_name}_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"
        experiment_save_dir = os.path.join(DEFAULT_RESULTS_DIR, full_experiment_name)


        def init_static_policy_distribution_after_trainer_init_callback(trainer):
            trainer.storage_client = connect_storage_client()

            logger.info("Initializing trainer manager interface")
            trainer.manager_interface = LearnerManagerInterface(server_host=MANAGER_SEVER_HOST,
                                                                port=MANAGER_PORT,
                                                                worker_id=full_experiment_name,
                                                                storage_client=trainer.storage_client,
                                                                minio_bucket_name=BUCKET_NAME)

            selection_probs, payoff_table, payoff_table_key = get_fp_metanash_for_latest_payoff_table(
                manager_interface=trainer.manager_interface,
                fp_iters=METANASH_FICTITIOUS_PLAY_ITERS,
                accepted_opponent_policy_class_names=ACCEPTED_OPPONENT_POLICY_CLASS_NAMES,
                accepted_opponent_model_config_keys=ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS,
                add_payoff_matrix_noise_std_dev=0.0,
                mix_with_uniform_dist_coeff=PSRO_EXPLORATION_COEFF
            )

            if selection_probs is None:
                assert payoff_table is None
                assert payoff_table_key is None
                print("Payoff table is empty so using random weights for static policy.")
            else:
                print(f"Payoff table loaded from {payoff_table_key}")
                print(f"Policy selection probs: {selection_probs}")

            payoff_table_dill_str = dill.dumps(payoff_table)
            def worker_set_static_policy_distribution(worker):
                worker.policy_map[STATIC_POLICY].static_policy_selection_probs = selection_probs
                worker.policy_map[STATIC_POLICY].payoff_table = dill.loads(payoff_table_dill_str)
                worker.policy_map[STATIC_POLICY].current_policy_key = None

            trainer.workers.foreach_worker(worker_set_static_policy_distribution)


        # def init_train_policy_weights_from_static_policy_distribution_after_trainer_init_callback(trainer):
        #     local_static_policy = trainer.workers.local_worker().policy_map[STATIC_POLICY]
        #     local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]
        #     if local_static_policy.static_policy_selection_probs is None:
        #         print("Payoff table is empty so using random weights for train policy init.")
        #         local_train_policy.init_tag = "init from random"
        #         return
        #
        #     selected_policy_index = np.random.choice(
        #         a=list(range(len(local_static_policy.static_policy_selection_probs))),
        #         p=local_static_policy.static_policy_selection_probs
        #     )
        #     selected_policy_spec: PolicySpec = local_static_policy.payoff_table.get_policy_for_index(selected_policy_index)
        #     local_train_policy.init_tag = f"full init from {selected_policy_spec.key}"
        #
        #     # may not necessarily be true in all scripts
        #     assert selected_policy_spec.class_name == TRAIN_POLICY_CLASS.__name__
        #     assert selected_policy_spec.config_key == TRAIN_POLICY_MODEL_CONFIG_KEY
        #     storage_client = connect_storage_client(endpoint=MINIO_ENDPOINT,
        #                                         access_key=MINIO_ACCESS_KEY,
        #                                         secret_key=MINIO_SECRET_KEY)
        #     weights_local_path, _ = maybe_download_object(storage_client=storage_client,
        #                                                   bucket_name=BUCKET_NAME,
        #                                                   object_name=selected_policy_spec.key,
        #                                                   force_download=False)
        #
        #     def worker_set_train_policy_weights(worker):
        #         train_policy = worker.policy_map[TRAIN_POLICY]
        #         train_policy.load_model_weights(load_file_path=weights_local_path,
        #                                         add_scope_prefix=TRAIN_POLICY)
        #
        #     trainer.workers.foreach_worker(worker_set_train_policy_weights)


        def sample_new_static_policy_weights_for_each_worker_on_train_result_callback(params):
            trainer = params['trainer']

            def worker_sample_new_static_policy(worker):
                static_policy = worker.policy_map[STATIC_POLICY]

                if static_policy.static_policy_selection_probs is None:
                    return

                selected_policy_index = np.random.choice(
                    a=list(range(len(static_policy.static_policy_selection_probs))),
                    p=static_policy.static_policy_selection_probs
                )
                selected_policy_spec: PolicySpec = static_policy.payoff_table.get_policy_for_index(selected_policy_index)
                assert selected_policy_spec.class_name in ACCEPTED_OPPONENT_POLICY_CLASS_NAMES
                assert selected_policy_spec.config_key in ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS

                if static_policy.current_policy_key != selected_policy_spec.key:
                    # print(f"sampled policy {selected_policy_spec.key} (loading weights)")
                    storage_client = connect_storage_client()
                    weights_local_path, _ = maybe_download_object(storage_client=storage_client,
                                                                  bucket_name=BUCKET_NAME,
                                                                  object_name=selected_policy_spec.key,
                                                                  force_download=False)
                    static_policy.load_model_weights(load_file_path=weights_local_path,
                                                     add_scope_prefix=STATIC_POLICY)
                    static_policy.current_policy_key = selected_policy_spec.key
                # else:
                #     print(f"sampled policy {selected_policy_spec.key} (weights already loaded)")

            trainer.workers.foreach_worker(worker_sample_new_static_policy)


        def stop_and_submit_if_not_improving_on_train_result_callback(params):
            trainer = params['trainer']
            result = params['result']
            result['stop_signal'] = False

            should_submit = False
            submit_reason = None

            if not hasattr(trainer, 'previous_threshold_check_reward'):
                trainer.previous_threshold_check_reward = -100.0
                trainer.next_threshold_check_timesteps = SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS + SUBMISSION_THRESHOLD_STEPS_START
                print(f"fist threshold check at {trainer.next_threshold_check_timesteps} timesteps")

            if result['timesteps_total'] >= SUBMISSION_THRESHOLD_STEPS_START and \
                    SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS is not None and \
                    SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD is not None:

                if result['timesteps_total'] >= trainer.next_threshold_check_timesteps:
                    trainer.next_threshold_check_timesteps = max(
                        trainer.next_threshold_check_timesteps + SUBMISSION_IMPROVEMENT_THRESHOLD_PER_TIMESTEPS,
                        result['timesteps_total'] + 1)

                    target_reward = trainer.previous_threshold_check_reward + SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
                    result['target_reward'] = target_reward
                    measured_reward = result['policy_reward_mean'][TRAIN_POLICY]
                    print(
                        f"{result['timesteps_total']} timesteps: {TRAIN_POLICY} reward: {measured_reward}, target reward: {target_reward}")

                    if measured_reward < target_reward and \
                            (SUBMISSION_MIN_TIMESTEPS is None or result['timesteps_total'] >= SUBMISSION_MIN_TIMESTEPS):
                        should_submit = True
                        submit_reason = f"plateaued at {measured_reward} reward"
                        print(
                            f"{result['timesteps_total']} timesteps: {TRAIN_POLICY} didn\'t reach target reward. Submitting policy.")
                    else:
                        print(f"next threshold check at {trainer.next_threshold_check_timesteps} timesteps")

                    trainer.previous_threshold_check_reward = measured_reward

            if SUBMISSION_MAX_TIMESTEPS is not None and result['timesteps_total'] >= SUBMISSION_MAX_TIMESTEPS:
                should_submit = True
                submit_reason = f"hit max timesteps of {SUBMISSION_MAX_TIMESTEPS}"
                print(f"Trainer hit max timesteps. Submitting policy.")

            if should_submit:
                assert submit_reason is not None
                result['stop_signal'] = True
                # local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]
                #
                # tags = [*SUBMISSION_POLICY_TAGS, submit_reason]
                # if hasattr(local_train_policy, "init_tag"):
                #     tags += local_train_policy.init_tag
                #
                # checkpoints_dir = os.path.join(experiment_save_dir, "policy_submissions")
                # checkpoint_name = f"{datetime_str()}_iter_{result['training_iteration']}.dill"
                # checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_name)
                # local_train_policy.save_model_weights(save_file_path=checkpoint_save_path,
                #                                       remove_scope_prefix=TRAIN_POLICY)
                # policy_key = os.path.join(base_experiment_name, full_experiment_name,
                #                           "policy_submissions", checkpoint_name)
                # storage_client = connect_storage_client(endpoint=MINIO_ENDPOINT,
                #                                     access_key=MINIO_ACCESS_KEY,
                #                                     secret_key=MINIO_SECRET_KEY)
                # upload_file(storage_client=storage_client,
                #             bucket_name=BUCKET_NAME,
                #             object_key=policy_key,
                #             local_source_path=checkpoint_save_path)
                # trainer.manager_interface.submit_new_policy_for_population(
                #     policy_weights_key=policy_key,
                #     policy_config_key=TRAIN_POLICY_MODEL_CONFIG_KEY,
                #     policy_class_name=TRAIN_POLICY_CLASS.__name__,
                #     policy_tags=tags
                # )

        def all_on_train_result_callbacks(params):
            stop_and_submit_if_not_improving_on_train_result_callback(params=params)
            sample_new_static_policy_weights_for_each_worker_on_train_result_callback(params=params)

        train_model_config_local_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                      bucket_name=BUCKET_NAME,
                                                                      object_name=TRAIN_POLICY_MODEL_CONFIG_KEY)
        with open(train_model_config_local_file_path, 'r') as config_file:
            train_model_config = json.load(fp=config_file)

        static_model_config_local_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                       bucket_name=BUCKET_NAME,
                                                                       object_name=STATIC_POLICY_MODEL_CONFIG_KEY)
        with open(static_model_config_local_file_path, 'r') as config_file:
            static_model_config = json.load(fp=config_file)


        def train_policy_mapping_fn(agent_id):
            if agent_id == 1:
                return TRAIN_POLICY
            elif agent_id == 0 or agent_id == -1:
                return STATIC_POLICY
            else:
                raise ValueError("train_policy_mapping_fn: wasn't expecting an agent_id other than 1 or -1")


        poker_env_config = {
            'version': POKER_GAME_VERSION,
        }

        temp_env = ENV_CLASS(poker_env_config)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space

        total_gpus_for_trial = 2
        num_gpus_for_opt = 1
        num_workers = (total_gpus_for_trial - num_gpus_for_opt) * 10

        base_trainer_config = {
            "log_level": "DEBUG",
            "metrics_smoothing_episodes": 10000,
            "memory_per_worker": 1019430400,
            "num_envs_per_worker": 1,
            "num_workers": 5,
            "num_gpus_per_worker": 0.0,
            "env": POKER_ENV,
            "env_config": tune.grid_search([
                with_updates(base_dict=poker_env_config, updates_dict={
                    'num_envs': 1,
                }),
                with_updates(base_dict=poker_env_config, updates_dict={
                    'num_envs': 16,
                }),
                with_updates(base_dict=poker_env_config, updates_dict={
                    'num_envs': 32,
                }),
                with_updates(base_dict=poker_env_config, updates_dict={
                    'num_envs': 64,
                }),
            ]),
            "multiagent": {
                "policies": {
                    TRAIN_POLICY: (TRAIN_POLICY_CLASS, obs_space, act_space, {
                        'model': train_model_config,
                    }),
                    STATIC_POLICY: (STATIC_POLICY_CLASS, obs_space, act_space, {
                        'model': static_model_config,
                    }),
                },
                "policy_mapping_fn": train_policy_mapping_fn,
                "policies_to_train": [TRAIN_POLICY],
            },

            "callbacks_after_trainer_init": [
                init_static_policy_distribution_after_trainer_init_callback,
                # init_train_policy_weights_from_static_policy_distribution_after_trainer_init_callback
            ],
            "callbacks": {
                "on_train_result": all_on_train_result_callbacks
            },
            "buffer_size": tune.choice([int(1e4), int(1e5), int(1e6)]),
            "learning_starts": tune.choice([0.0, 1000, 10000]),
            "tau": tune.choice([1e-1, 1e-2, 1e-3, 1e-4]),
            "gamma": 1.0,
            "train_batch_size": tune.choice([256, 512, 1024, 2048]),
            "optimization": tune.choice([
                {"actor_learning_rate": 1e-1, "critic_learning_rate": 1e-1, "entropy_learning_rate": 1e-1},
                {"actor_learning_rate": 1e-2, "critic_learning_rate": 1e-2, "entropy_learning_rate": 1e-2},
                {"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3, "entropy_learning_rate": 1e-3},
                {"actor_learning_rate": 1e-4, "critic_learning_rate": 1e-4, "entropy_learning_rate": 1e-4},

            ]),
            "max_entropy_target_proportion": 0.0,
            "batch_mode": 'truncate_episodes',
            "num_gpus": num_gpus_for_opt,
            "sample_batch_size": tune.choice([1, 10, 20, 50, 100]),
            #
            "timesteps_per_iteration": 1,
            "min_iter_time_s": 0,
        }

        # save running script to file
        current_code_file_path = os.path.abspath(__file__)
        copy_code_to_path = os.path.join(experiment_save_dir, "launch_script.py")
        ensure_dir(copy_code_to_path)
        shutil.copy2(src=current_code_file_path, dst=copy_code_to_path, follow_symlinks=True)


        def random_string(stringLength=8):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(stringLength))

        def trial_name_creator(trial):
            config = trial.config
            return f"{random_string(2)}_sac_bufsz_{config['buffer_size']}_lrnstrts_{config['learning_starts']}_tau_{config['tau']}_" \
                   f"trnbtchsz_{config['train_batch_size']}_lr_{config['optimization']['actor_learning_rate']}_" \
                   f"smplbtchsz_{config['sample_batch_size']}_nmwrkrs_{config['num_workers']}_" \
                   f"nmenvs_{config['env_config']['num_envs']}"

        analysis = tune.run(
            local_dir=TUNE_SAVE_DIR,
            name=full_experiment_name,
            upload_dir=base_experiment_name,
            sync_to_cloud=get_tune_sync_to_cloud_fn(storage_client=storage_client, bucket_name=BUCKET_NAME),
            checkpoint_at_end=False,
            keep_checkpoints_num=0,
            checkpoint_freq=0,
            num_samples=2,
            max_failures=0,
            reuse_actors=False,
            trial_name_creator=trial_name_creator,
            export_formats=[],
            stop={"stop_signal": True},
            run_or_experiment=TRAINER_CLASS,
            config=base_trainer_config)

        print("Experiment Done!")
