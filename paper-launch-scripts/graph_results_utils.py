
import ray
from ray.rllib.agents.trainer import with_common_config, with_base_config
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.utils import try_import_tf

import json
import os
import pandas as pd
import multiprocessing
from itertools import repeat

from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from mprl.rl.envs.opnspl.measure_exploitability_eval_callback import measure_exploitability_nonlstm
from mprl.utility_services.cloud_storage import maybe_download_object, connect_storage_client, BUCKET_NAME
from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
from mprl.rl.common.stratego_preprocessor import STRATEGO_PREPROCESSOR, StrategoDictFlatteningPreprocessor
from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv
from mprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL
from mprl.scripts.poker_parallel_algos.utils.metanash import get_fp_metanash_for_payoff_table
from mprl.utility_services.payoff_table import PayoffTable

tf = try_import_tf()

POLICY_CLASS = SACDiscreteTFPolicy
POLICY_CLASS_NAME = SACDiscreteTFPolicy.__name__
MODEL_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY


def get_stats_for_single_payoff_table(payoff_table_key, experiment_name, poker_game_version, model_config_key):
    POKER_ENV_CONFIG = {
        'version': poker_game_version,
    }

    storage_client = connect_storage_client()

    # If you use ray for more than just this single example fn, you'll need to move ray.init to the top of your main()
    ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True, local_mode=True)

    model_config_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                      bucket_name=BUCKET_NAME,
                                                      object_name=model_config_key,
                                                      force_download=False)

    with open(model_config_file_path, 'r') as config_file:
        model_config = json.load(fp=config_file)

    example_env = PokerMultiAgentEnv(env_config=POKER_ENV_CONFIG)

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

    payoff_table_local_path, _ = maybe_download_object(storage_client=storage_client,
                                                           bucket_name=BUCKET_NAME,
                                                           object_name=payoff_table_key,
                                                           force_download=False)

    payoff_table = PayoffTable.from_dill_file(dill_file_path=payoff_table_local_path)
    stats_out = {
        'payoff_table_key': [],
        'experiment_name': [],
        'num_policies': [],
        'exploitability': [],
        'total_steps': [],
        'total_episodes': [],
    }

    exploitability_per_generation = []
    total_steps_per_generation = []
    total_episodes_per_generation = []
    num_policies_per_generation = []

    for i, n_policies in enumerate(range(1,payoff_table.size() + 1)):
        metanash_probs = get_fp_metanash_for_payoff_table(payoff_table=payoff_table,
                                                                 fp_iters=40000,
                                                                 accepted_opponent_policy_class_names=[POLICY_CLASS_NAME],
                                                                 accepted_opponent_model_config_keys=[POKER_ENV_CONFIG],
                                                                 add_payoff_matrix_noise_std_dev=0.000,
                                                                 mix_with_uniform_dist_coeff=None,
                                                                 only_first_n_policies=n_policies,
                                                                 p_or_lower_rounds_to_zero=0.0)

        policy_weights_keys = payoff_table.get_ordered_keys_in_payoff_matrix()

        policy_dict = {key: prob for key, prob in zip(policy_weights_keys, metanash_probs)}

        exploitability_this_gen = measure_exploitability_nonlstm(rllib_policy=policy,
                                  poker_game_version=poker_game_version,
                                  policy_mixture_dict=policy_dict,
                                  set_policy_weights_fn=set_policy_weights)

        print(f"{experiment_name}: {n_policies} policies, {exploitability_this_gen} exploitability")

        policy_added_this_gen = payoff_table.get_policy_for_index(i)
        latest_policy_tags = policy_added_this_gen.tags
        steps_prefix = "timesteps: "
        latest_policy_steps = int([tag for tag in latest_policy_tags if steps_prefix in tag][0][len(steps_prefix):])
        episodes_prefix = "episodes: "
        latest_policy_episodes = int([tag for tag in latest_policy_tags if episodes_prefix in tag][0][len(episodes_prefix):])

        if i > 0:
            total_steps_this_generation = latest_policy_steps + total_steps_per_generation[i-1]
            total_episodes_this_generation = latest_policy_episodes + total_episodes_per_generation[i-1]
        else:
            total_steps_this_generation = latest_policy_steps
            total_episodes_this_generation = latest_policy_episodes

        exploitability_per_generation.append(exploitability_this_gen)
        total_steps_per_generation.append(total_steps_this_generation)
        total_episodes_per_generation.append(total_episodes_this_generation)
        num_policies_per_generation.append(n_policies)

        num_new_entries = len(exploitability_per_generation)
        stats_out['payoff_table_key'] = stats_out['payoff_table_key'] + [payoff_table_key] * num_new_entries
        stats_out['experiment_name'] = stats_out['experiment_name'] + [experiment_name] * num_new_entries
        stats_out['num_policies'] = stats_out['num_policies'] + num_policies_per_generation
        stats_out['exploitability'] = stats_out['exploitability'] + exploitability_per_generation
        stats_out['total_steps'] = stats_out['total_steps'] + total_steps_per_generation
        stats_out['total_episodes'] = stats_out['total_episodes'] + total_episodes_per_generation
    return stats_out


def get_exploitability_stats_over_time_for_payoff_tables_all_same_poker_version(
        payoff_table_keys, exp_names, poker_game_version, model_config_key):

    num_processes = max(multiprocessing.cpu_count()//2, 1)
    with multiprocessing.get_context("spawn").Pool(processes=num_processes) as pool:
        results = pool.starmap(func=get_stats_for_single_payoff_table,
                               iterable=zip(payoff_table_keys, exp_names, repeat(poker_game_version), repeat(model_config_key)))

    pool.close()
    pool.join()

    combined_stats = {}
    for result in results:
        for key, val in result.items():
            if key not in combined_stats:
                combined_stats[key] = val
            else:
                combined_stats[key] = [*combined_stats[key], *val]

    return pd.DataFrame(combined_stats).drop_duplicates()
