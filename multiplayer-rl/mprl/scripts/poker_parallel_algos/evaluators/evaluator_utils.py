from mprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv
from mprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL

from progress.bar import Bar
import numpy as np
import dill
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from multiprocessing import Lock
import logging
import time
logger = logging.getLogger(__name__)

# def run_dill_encoded(payload):
#     fun, args = dill.loads(payload)
#     return fun(*args)
#
# def apply_async(pool, fun, args):
#     payload = dill.dumps((fun, args))
#     return pool.apply_async(run_dill_encoded, (payload,))


def eval_policy_matchup(get_policy_fn_a, get_policy_fn_b, env, stratego_env_config, games_per_matchup):
    resample_policy_fn_a = False
    if isinstance(get_policy_fn_a, tuple):
        get_policy_fn_a, resample_policy_fn_a = get_policy_fn_a

    policy_a_name, policy_a_get_action_index = get_policy_fn_a(stratego_env_config)

    resample_policy_fn_b = False
    if isinstance(get_policy_fn_b, tuple):
        get_policy_fn_b, resample_policy_fn_b = get_policy_fn_b

    policy_b_name, policy_b_get_action_index = get_policy_fn_b(stratego_env_config)
    policy_funcs = [policy_a_get_action_index, policy_b_get_action_index]

    policy_a_state = None
    policy_b_state = None
    policy_states = [policy_a_state, policy_b_state]

    def policy_index(agent_id):
        if agent_id == 1:
            return 0
        else:
            return 1

    policy_a_total_payoff = 0
    ties = 0

    max_reward = None
    min_reward = None

    time_since_last_output = time.time()

    # with Bar('Evaluating {} vs {}'.format(policy_a_name, policy_b_name), max=games_per_matchup) as bar:
    for game in range(games_per_matchup):


        if game % 10 == 0:
            now = time.time()
            logger.debug(f"{policy_a_name} vs {policy_b_name}: {game}/{games_per_matchup} games played, {now - time_since_last_output} seconds")
            time_since_last_output = now

        if resample_policy_fn_a:
            policy_a_get_action_index(None, None, resample=True)

        if resample_policy_fn_b:
            policy_b_get_action_index(None, None, resample=True)

        obs = env.reset()
        dones = {}
        infos = {}
        game_length = 0

        player_a_total_game_reward = 0.0
        while True:
            if "__all__" in dones:
                if dones["__all__"]:
                    break
            game_length += 1
            assert len(obs) == 1
            acting_agent_id, acting_agent_observation = list(obs.items())[0]
            acting_policy_fn = policy_funcs[policy_index(acting_agent_id)]
            acting_policy_state = policy_states[policy_index(acting_agent_id)]

            action_index, new_policy_state = acting_policy_fn(acting_agent_observation, acting_policy_state)
            policy_states[policy_index(acting_agent_id)] = new_policy_state

            obs, rewards, dones, infos = env.step(action_dict={acting_agent_id: action_index})
            player_a_total_game_reward += rewards.get(1, 0.0)

        player_a_won = infos[1]['game_result'] == 'won'
        tied = infos[1]['game_result'] == 'tied'

        policy_a_total_payoff += player_a_total_game_reward
        if player_a_total_game_reward > 0:
            assert player_a_won, f"player_a_total_game_reward: {player_a_total_game_reward}"

        if tied:
            ties += 1

        if max_reward is None or player_a_total_game_reward > max_reward:
            max_reward = player_a_total_game_reward
        if min_reward is None or player_a_total_game_reward < min_reward:
            min_reward = player_a_total_game_reward

        # print(f"game length: {game_length}")

        # bar.next()

    policy_a_avg_payoff = policy_a_total_payoff / games_per_matchup
    tie_percentage = ties / games_per_matchup

    logger.info(f"max reward for player a: {max_reward}, min reward {min_reward}")
    logger.info(f"avg payoff: {policy_a_avg_payoff}")
    return policy_a_avg_payoff, tie_percentage


def make_get_policy_fn(model_weights_object_key, model_config_object_key, policy_name, policy_class_name,
                       storage_client, minio_bucket_name, download_lock=None, manual_config=None,
                       population_policy_keys_to_selection_probs=None):

    if download_lock is None:
        download_lock = Lock()

    def get_policy_fn(stratego_env_config):

        from mprl.utility_services.cloud_storage import maybe_download_object
        from mprl.rl.sac.sac_policy import SACDiscreteTFPolicy
        from mprl.rl.ppo.ppo_stratego_model_policy import PPOStrategoModelTFPolicy
        from mprl.rl.common.stratego_preprocessor import STRATEGO_PREPROCESSOR, StrategoDictFlatteningPreprocessor
        from ray.rllib.agents.trainer import with_common_config, with_base_config
        from ray.rllib.models.catalog import MODEL_DEFAULTS
        from mprl.rl.common.sac_spatial_stratego_model import SAC_SPATIAL_STRATEGO_MODEL
        import ray
        from ray.rllib.utils import try_import_tf
        import json
        import os
        tf = try_import_tf()

        from tensorflow.python.client import device_lib

        def get_available_gpus():
            local_device_protos = device_lib.list_local_devices()
            return [x.name for x in local_device_protos if x.device_type == 'GPU']


        # If you use ray for more than just this single example fn, you'll need to move ray.init to the top of your main()
        ray.init(address=os.getenv('RAY_HEAD_NODE'), ignore_reinit_error=True, local_mode=True)

        if policy_class_name == 'PPOStrategoModelTFPolicy':
            _policy_class = PPOStrategoModelTFPolicy
        elif policy_class_name == 'SACDiscreteTFPolicy':
            _policy_class = SACDiscreteTFPolicy
        else:
            raise NotImplementedError(f"Eval for policy class \'{policy_class_name}\' not implemented.")

        if model_config_object_key:
            with download_lock:
                model_config_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                  bucket_name=minio_bucket_name,
                                                                  object_name=model_config_object_key,
                                                                  force_download=False)

                with open(model_config_file_path, 'r') as config_file:
                    model_config = json.load(fp=config_file)
        else:
            model_config = manual_config

        example_env = stratego_env_config['env_class'](env_config=stratego_env_config)
        obs_space = example_env.observation_space
        act_space = example_env.action_space

        preprocessor = StrategoDictFlatteningPreprocessor(obs_space=obs_space)


        graph = tf.Graph()

        if os.getenv("EVALUATOR_USE_GPU") == 'true':
            gpu = 1
        else:
            gpu = 0

        config = tf.ConfigProto(device_count={'GPU': gpu})
        if gpu:
            config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        with graph.as_default():
            with sess.as_default():
                policy = _policy_class(
                    obs_space=preprocessor.observation_space,
                    action_space=act_space,
                    config=with_common_config({
                        'model': with_base_config(base_config=MODEL_DEFAULTS, extra_config=model_config),
                        'env': POKER_ENV,
                        'env_config': stratego_env_config,
                        'custom_preprocessor': STRATEGO_PREPROCESSOR,
                    }))

                if model_weights_object_key:
                    with download_lock:
                        weights_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                     bucket_name=minio_bucket_name,
                                                                     object_name=model_weights_object_key,
                                                                     force_download=False)
                        policy.load_model_weights(weights_file_path)
                    policy.current_model_weights_key = weights_file_path
                else:
                    policy.current_model_weights_key = None

        def policy_fn(observation, policy_state=None):
            if policy_state is None:
                policy_state = policy.get_initial_state()

            current_player_perspective_action_index, policy_state, _ = policy.compute_single_action(
                obs=preprocessor.transform(observation),
                state=policy_state)

            return current_player_perspective_action_index, policy_state

        if population_policy_keys_to_selection_probs is not None:

            def sample_new_policy_weights_from_population():
                new_policy_key = np.random.choice(a=list(population_policy_keys_to_selection_probs.keys()),
                                                  p=list(population_policy_keys_to_selection_probs.values()))
                if new_policy_key != policy.current_model_weights_key:
                    with download_lock:
                        weights_file_path, _ = maybe_download_object(storage_client=storage_client,
                                                                     bucket_name=minio_bucket_name,
                                                                     object_name=new_policy_key,
                                                                     force_download=False)
                        policy.load_model_weights(weights_file_path)
                        logger.debug(f"Sampling new population weights from {new_policy_key}")
                    policy.current_model_weights_key = new_policy_key

            return policy_name, policy_fn, sample_new_policy_weights_from_population

        # policy name must be unique
        return policy_name, policy_fn

    return get_policy_fn