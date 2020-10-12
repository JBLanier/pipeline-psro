from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import shutil
import time
from socket import gethostname

from termcolor import colored
import dill
import numpy as np
import ray
from mprl.utility_services.payoff_table import PolicySpec
from mprl.utility_services.worker import LearnerManagerInterface, ConsoleManagerInterface
from mprl.utility_services.lock_server.lock_client_interface import LockServerInterface
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.experiment import DEFAULT_RESULTS_DIR
from mprl.utility_services.cloud_storage import connect_storage_client, maybe_download_object, upload_file, get_tune_sync_to_cloud_fn, BUCKET_NAME
from mprl.rl.envs.opnspl.poker_multiagent_env import PARTIALLY_OBSERVABLE, \
    PokerMultiAgentEnv
from mprl.rl.sac.sac import SACTrainer
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.scripts.poker_parallel_algos.utils.configs import POKER_TRAINER_BASE_CONFIG, POKER_ENV_CONFIG, \
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS, POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD, \
    POKER_SUBMISSION_MAX_STEPS, POKER_SUBMISSION_MIN_STEPS, POKER_SUBMISSION_THRESHOLD_STEPS_START, \
    POKER_METANASH_FICTITIOUS_PLAY_ITERS, POKER_GAME_VERSION, POKER_PSRO_EXPLORATION_COEFF, \
    POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION, POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS, \
    POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS, POKER_PAYOFF_MATRIX_NOISE_STD_DEV, ENV_CLASS, \
    SELECTED_CONFIG_KEY, POKER_PIPELINE_INIT_FROM_POP
from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_latest_payoff_table, get_fp_metanash_for_payoff_table, get_unreserved_policy_key_with_priorities
from mprl.utils import datetime_str, ensure_dir, with_updates
from mprl import TUNE_SAVE_DIR

# log level for our code, not for Ray/rllib
logger = logging.getLogger(__name__)

tf = try_import_tf()

TRAIN_POLICY = "train_policy"
STATIC_POLICY = "static_policy"

OBSERVATION_MODE = PARTIALLY_OBSERVABLE

TRAINER_CLASS = SACTrainer
TRAIN_POLICY_CLASS = SACDiscreteTFPolicy
TRAIN_POLICY_MODEL_CONFIG_KEY = SELECTED_CONFIG_KEY
SUBMISSION_POLICY_TAGS = [f"hostname {gethostname()}"]

STATIC_POLICY_CLASS = SACDiscreteTFPolicy
STATIC_POLICY_MODEL_CONFIG_KEY = SELECTED_CONFIG_KEY

ACCEPTED_OPPONENT_POLICY_CLASSES = [SACDiscreteTFPolicy]
ACCEPTED_OPPONENT_POLICY_CLASS_NAMES = [cls.__name__ for cls in ACCEPTED_OPPONENT_POLICY_CLASSES]
ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS = [SELECTED_CONFIG_KEY]

MANAGER_SEVER_HOST = os.getenv("MANAGER_SERVER_HOST", "localhost")

MANAGER_PORT = int(os.getenv("MANAGER_PORT"))
if not MANAGER_PORT:
    raise ValueError("Environment variable MANAGER_PORT needs to be set.")

LOCK_SERVER_HOST = os.getenv("LOCK_SERVER_HOST", 'localhost')
LOCK_SERVER_PORT = int(os.getenv("LOCK_SERVER_PORT"))
if not LOCK_SERVER_PORT:
    raise ValueError("Environment variable LOCK_SERVER_PORT needs to be set.")

PSRO_EXPLORATION_COEFF = POKER_PSRO_EXPLORATION_COEFF
METANASH_FICTITIOUS_PLAY_ITERS = POKER_METANASH_FICTITIOUS_PLAY_ITERS
SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS = POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS
SUBMISSION_THRESHOLD_STEPS_START = POKER_SUBMISSION_THRESHOLD_STEPS_START
SUBMISSION_MIN_STEPS = POKER_SUBMISSION_MIN_STEPS
SUBMISSION_MAX_STEPS = POKER_SUBMISSION_MAX_STEPS

CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "")

if __name__ == "__main__":
    new_learner_wait_for_key_in_payoff_table = None
    while True:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("\n\n\n\n\n__________________________________________\n"
                    f"LAUNCHED FOR {POKER_GAME_VERSION}\n"
                    f"__________________________________________\n\n\n\n\n")

        storage_client = connect_storage_client()

        size_checker = ConsoleManagerInterface(server_host=MANAGER_SEVER_HOST,
                                               port=MANAGER_PORT,
                                               worker_id=f"size_checker_{gethostname()}_pid_{os.getpid()}",
                                               storage_client=storage_client,
                                               minio_bucket_name=BUCKET_NAME)

        while True:
            if new_learner_wait_for_key_in_payoff_table is not None:
                if not size_checker.is_policy_key_in_current_payoff_matrix(policy_key=new_learner_wait_for_key_in_payoff_table):
                    logger.info(
                        f"waiting for payoff matrix to include the policy key {new_learner_wait_for_key_in_payoff_table} ")
                    time.sleep(5)
                else:
                    break
            else:
                break

        ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True)
        logger.info("Ray Web UI at {}".format(ray.get_webui_url()))

        base_experiment_name = f"{CLOUD_PREFIX}learner_{POKER_GAME_VERSION}_sac_arch1_metanash_subset"
        full_experiment_name = f"{base_experiment_name}_{gethostname()}_pid_{os.getpid()}_{datetime_str()}"
        experiment_save_dir = os.path.join(DEFAULT_RESULTS_DIR, full_experiment_name)

        checkpoints_dir = os.path.join(experiment_save_dir, "policy_submissions")
        checkpoint_name = f"policy_submission_{datetime_str()}.dill"
        checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_name)
        submit_checkpoint_policy_key = os.path.join(base_experiment_name, full_experiment_name,
                                                    "policy_submissions", checkpoint_name)
        new_learner_wait_for_key_in_payoff_table = submit_checkpoint_policy_key


        def init_static_policy_distribution_after_trainer_init_callback(trainer):
            trainer.storage_client = connect_storage_client()

            logger.info("Initializing trainer manager interface")
            trainer.manager_interface = LearnerManagerInterface(server_host=MANAGER_SEVER_HOST,
                                                                port=MANAGER_PORT,
                                                                worker_id=full_experiment_name,
                                                                storage_client=trainer.storage_client,
                                                                minio_bucket_name=BUCKET_NAME)

            logger.info("Initializing trainer lock server interface")
            trainer.lock_server_interface = LockServerInterface(server_host=LOCK_SERVER_HOST,
                                                                port=LOCK_SERVER_PORT,
                                                                worker_id=full_experiment_name)

            orig_selection_probs, payoff_table, payoff_table_key = get_fp_metanash_for_latest_payoff_table(
                manager_interface=trainer.manager_interface,
                fp_iters=METANASH_FICTITIOUS_PLAY_ITERS,
                accepted_opponent_policy_class_names=ACCEPTED_OPPONENT_POLICY_CLASS_NAMES,
                accepted_opponent_model_config_keys=ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS,
                add_payoff_matrix_noise_std_dev=0.0,
                mix_with_uniform_dist_coeff=PSRO_EXPLORATION_COEFF
            )

            if orig_selection_probs is None:
                assert payoff_table is None
                assert payoff_table_key is None
                selection_probs = None
                print("Payoff table is empty so using random weights for static policy.")
            else:
                print(f"Payoff table loaded from {payoff_table_key}")
                print(f"Original Selection Probs: {orig_selection_probs}")

                policy_key_to_leave_out = get_unreserved_policy_key_with_priorities(
                    lock_server_interface=trainer.lock_server_interface,
                    policy_keys=payoff_table.get_ordered_keys_in_payoff_matrix(),
                    policy_priorities=orig_selection_probs
                )

                if policy_key_to_leave_out is None:
                    selection_probs = orig_selection_probs
                    print("No policy keys available to reserve so using unaltered selection probs")
                else:
                    chosen_policy_selection_prob = orig_selection_probs[
                        payoff_table.get_policy_spec_for_key(policy_key_to_leave_out).get_payoff_matrix_index()]
                    print(f"\n\nLeaving out {policy_key_to_leave_out}\n"
                          f"(Had selection prob of ({chosen_policy_selection_prob})\n\n")

                    selection_probs = get_fp_metanash_for_payoff_table(
                        payoff_table=payoff_table,
                        fp_iters=METANASH_FICTITIOUS_PLAY_ITERS,
                        accepted_opponent_policy_class_names=ACCEPTED_OPPONENT_POLICY_CLASS_NAMES,
                        accepted_opponent_model_config_keys=ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS,
                        add_payoff_matrix_noise_std_dev=0.0,
                        leave_out_indexes=[payoff_table.get_policy_spec_for_key(policy_key_to_leave_out).get_payoff_matrix_index()],
                        mix_with_uniform_dist_coeff=PSRO_EXPLORATION_COEFF
                    )
                    print(f"Subset Selection Probs: {selection_probs}")

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

        # def sample_new_static_policy_weights_for_each_worker_on_train_result_callback(params):
        #     trainer = params['trainer']
        #
        #     def worker_sample_new_static_policy(worker):
        #         static_policy = worker.policy_map[STATIC_POLICY]
        #
        #         if static_policy.static_policy_selection_probs is None:
        #             return
        #
        #         selected_policy_index = np.random.choice(
        #             a=list(range(len(static_policy.static_policy_selection_probs))),
        #             p=static_policy.static_policy_selection_probs
        #         )
        #         selected_policy_spec: PolicySpec = static_policy.payoff_table.get_policy_for_index(selected_policy_index)
        #         assert selected_policy_spec.class_name in ACCEPTED_OPPONENT_POLICY_CLASS_NAMES
        #         assert selected_policy_spec.config_key in ACCEPTED_OPPONENT_MODEL_CONFIG_KEYS
        #
        #         if static_policy.current_policy_key != selected_policy_spec.key:
        #             # print(f"sampled policy {selected_policy_spec.key} (loading weights)")
        #             storage_client = connect_storage_client(endpoint=MINIO_ENDPOINT,
        #                                                 access_key=MINIO_ACCESS_KEY,
        #                                                 secret_key=MINIO_SECRET_KEY)
        #             weights_local_path, _ = maybe_download_object(storage_client=storage_client,
        #                                                           bucket_name=BUCKET_NAME,
        #                                                           object_name=selected_policy_spec.key,
        #                                                           force_download=False)
        #             static_policy.load_model_weights(load_file_path=weights_local_path,
        #                                              add_scope_prefix=STATIC_POLICY)
        #             static_policy.current_policy_key = selected_policy_spec.key
        #         # else:
        #         #     print(f"sampled policy {selected_policy_spec.key} (weights already loaded)")
        #
        #     trainer.workers.foreach_worker(worker_sample_new_static_policy)

        def sample_new_static_policy_weights_for_each_worker_on_episode_start(params):
            policies = params['policy']

            static_policy = policies[STATIC_POLICY]

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


        def stop_and_submit_if_not_improving_on_train_result_callback(params):
            trainer = params['trainer']
            result = params['result']
            result['stop_signal'] = False

            should_submit = False
            submit_reason = None

            if not hasattr(trainer, 'previous_threshold_check_reward'):
                trainer.previous_threshold_check_reward = -100.0
                trainer.next_threshold_check_steps = SUBMISSION_THRESHOLD_STEPS_START + result['timesteps_total']

                print(colored(f"first threshold check at {trainer.next_threshold_check_steps} steps", "white"))

            if SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS is not None and \
                SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD is not None:

                if result['timesteps_total'] >= trainer.next_threshold_check_steps:
                    trainer.next_threshold_check_steps = max(
                        trainer.next_threshold_check_steps + SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS,
                        result['timesteps_total'] + 1)

                    target_reward = trainer.previous_threshold_check_reward + SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD
                    result['target_reward'] = target_reward
                    measured_reward = result['policy_reward_mean'][TRAIN_POLICY]
                    print(colored(f"{result['timesteps_total']} steps: {TRAIN_POLICY} reward: {measured_reward}, target reward: {target_reward}", "white"))

                    if measured_reward < target_reward and \
                            (SUBMISSION_MIN_STEPS is None or result['timesteps_total'] >= SUBMISSION_MIN_STEPS):
                        should_submit = True
                        submit_reason = f"plateaued at {measured_reward} reward"
                        print(
                            colored(f"{result['timesteps_total']} steps: {TRAIN_POLICY} didn\'t reach target reward. Submitting policy.", "white"))
                    else:
                        print(colored(f"next threshold check at {trainer.next_threshold_check_steps} steps", "white"))

                    trainer.previous_threshold_check_reward = measured_reward

            if SUBMISSION_MAX_STEPS is not None and result['timesteps_total'] >= SUBMISSION_MAX_STEPS:
                should_submit = True
                submit_reason = f"hit max steps of {SUBMISSION_MAX_STEPS}"
                print(colored(f"Trainer hit max steps. Submitting policy.", "white"))

            if should_submit:
                assert submit_reason is not None
                result['stop_signal'] = True
                local_train_policy = trainer.workers.local_worker().policy_map[TRAIN_POLICY]

                tags = [*SUBMISSION_POLICY_TAGS,
                        submit_reason,
                        f"timesteps: {result['timesteps_total']}",
                        f"episodes: {result['episodes_total']}",
                        f"iter: {result['training_iteration']}"]
                if hasattr(local_train_policy, "init_tag"):
                    tags += local_train_policy.init_tag

                # checkpoints_dir = os.path.join(experiment_save_dir, "policy_submissions")
                # checkpoint_name = f"policy_{trainer.claimed_policy_num}_{datetime_str()}_iter_{result['training_iteration']}.dill"
                # checkpoint_save_path = os.path.join(checkpoints_dir, checkpoint_name)
                local_train_policy.save_model_weights(save_file_path=checkpoint_save_path,
                                                      remove_scope_prefix=TRAIN_POLICY)
                # submit_checkpoint_policy_key = os.path.join(base_experiment_name, full_experiment_name,
                #                           "policy_submissions", checkpoint_name)
                storage_client = connect_storage_client()
                upload_file(storage_client=storage_client,
                            bucket_name=BUCKET_NAME,
                            object_key=submit_checkpoint_policy_key,
                            local_source_path=checkpoint_save_path)
                trainer.manager_interface.submit_new_policy_for_population(
                    policy_weights_key=submit_checkpoint_policy_key,
                    policy_config_key=TRAIN_POLICY_MODEL_CONFIG_KEY,
                    policy_class_name=TRAIN_POLICY_CLASS.__name__,
                    policy_tags=tags
                )


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


        temp_env = ENV_CLASS(POKER_ENV_CONFIG)
        obs_space = temp_env.observation_space
        act_space = temp_env.action_space

        trainer_config = with_updates(base_dict=POKER_TRAINER_BASE_CONFIG, updates_dict={
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
            ],
            "callbacks": {
                "on_train_result": stop_and_submit_if_not_improving_on_train_result_callback,
                'on_episode_start': sample_new_static_policy_weights_for_each_worker_on_episode_start,
            },
        })

        # save running script to file
        current_code_file_path = os.path.abspath(__file__)
        copy_code_to_path = os.path.join(experiment_save_dir, "launch_script.py")
        ensure_dir(copy_code_to_path)
        shutil.copy2(src=current_code_file_path, dst=copy_code_to_path, follow_symlinks=True)


        def trial_name_creator(trial):
            config = trial.config
            return "sac_learner"


        analysis = tune.run(
            local_dir=TUNE_SAVE_DIR,
            name=full_experiment_name,
            upload_dir=base_experiment_name,
            sync_to_cloud=get_tune_sync_to_cloud_fn(storage_client=storage_client, bucket_name=BUCKET_NAME),
            checkpoint_at_end=False,
            keep_checkpoints_num=0,
            checkpoint_freq=0,
            num_samples=1,
            max_failures=0,
            reuse_actors=False,
            trial_name_creator=trial_name_creator,
            export_formats=[],
            stop={"stop_signal": True},
            run_or_experiment=TRAINER_CLASS,
            config=trainer_config)

        print("Experiment Done!")
