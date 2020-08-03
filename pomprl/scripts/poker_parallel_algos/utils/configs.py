import os
from pomprl.rl.envs.opnspl.poker_multiagent_env import POKER_ENV, KUHN_POKER, LEDUC_POKER, PARTIALLY_OBSERVABLE, PokerMultiAgentEnv
from pomprl.util import with_updates
from pomprl.rl.common.stratego_preprocessor import STRATEGO_PREPROCESSOR
from pomprl.rl.common.sac_stratego_model import SAC_STRATEGO_MODEL
from socket import gethostname
from pomprl.rl.envs.stratego.stratego_spatial_multiagent_env import SpatialStrategoMultiAgentEnv, BARRAGE, SHORT_BARRAGE, FIVES
from pomprl.rl.envs.stratego.stratego_spatial_parallel_env import SPATIAL_STRATEGO_PARALLEL_ENV
from pomprl.scripts.poker_parallel_algos.utils.policy_config_keys import POKER_ARCH1_MODEL_CONFIG_KEY
from pomprl.scripts.population_server.utils.policy_config_keys import SAC_ARCH1P_MODEL_CONFIG_KEY, SAC_ARCH1P_FIVES_MODEL_CONFIG_KEY
from pomprl.rl.common.sac_spatial_stratego_model import SAC_SPATIAL_STRATEGO_MODEL

POKER_GAME_VERSION = os.getenv("POKER_GAME_VERSION")
if not POKER_GAME_VERSION:
    raise ValueError("Environment variable POKER_GAME_VERSION needs to be set.")


if POKER_GAME_VERSION == LEDUC_POKER:
    POKER_ENV_CONFIG = {
        'version': POKER_GAME_VERSION,
    }
    SELECTED_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY
    ENV_CLASS = PokerMultiAgentEnv
    POKER_TRAINER_BASE_CONFIG = {
        "log_level": "DEBUG",
        "metrics_smoothing_episodes": 10000,
        "memory_per_worker": 1019430400,
        "num_envs_per_worker": 1,
        "num_workers": 2,
        "num_gpus_per_worker": 0.0,
        "env": POKER_ENV,
        "env_config": with_updates(base_dict=POKER_ENV_CONFIG, updates_dict={
            'num_envs': 1,
        }),

        "buffer_size": int(20000),
        "learning_starts": 10000,
        "tau": 0.01,
        "gamma": 1.0,
        "train_batch_size": 1024,
        "optimization": {
            "actor_learning_rate": 0.01,
            "critic_learning_rate": 0.01,
            "entropy_learning_rate": 0.01,
        },
        "max_entropy_target_proportion": 0.0,
        "batch_mode": 'complete_episodes',
        "num_gpus": 0,
        "sample_batch_size": 20,
        #
        "timesteps_per_iteration": 1,
        "min_iter_time_s": 0,
    }

    POKER_METANASH_FICTITIOUS_PLAY_ITERS = 20000
    POKER_PSRO_EXPLORATION_COEFF = 0.1
    POKER_PAYOFF_MATRIX_NOISE_STD_DEV = 0.0
    POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION = 0.2
    POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS = True
    POKER_PIPELINE_INIT_FROM_POP = False
    POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS = int(15000)

    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = 0.01

    avg_episode_length = 4.5

    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS = int(2500 * avg_episode_length)
    POKER_SUBMISSION_THRESHOLD_STEPS_START = int(7500 * avg_episode_length)
    POKER_SUBMISSION_MIN_STEPS = int(10000 * avg_episode_length)
    POKER_SUBMISSION_MAX_STEPS = None


elif POKER_GAME_VERSION == KUHN_POKER:
    POKER_ENV_CONFIG = {
        'version': POKER_GAME_VERSION,
    }
    SELECTED_CONFIG_KEY = POKER_ARCH1_MODEL_CONFIG_KEY
    ENV_CLASS = PokerMultiAgentEnv
    POKER_TRAINER_BASE_CONFIG = {
        "log_level": "DEBUG",
        "metrics_smoothing_episodes": 10000,
        "memory_per_worker": 1019430400,
        "num_envs_per_worker": 1,
        "num_workers": 2,
        "num_gpus_per_worker": 0.0,
        "env": POKER_ENV,
        "env_config": with_updates(base_dict=POKER_ENV_CONFIG, updates_dict={
            'num_envs': 1,
        }),

        "buffer_size": int(20000),
        "learning_starts": 10000,
        "tau": 0.01,
        "gamma": 1.0,
        "train_batch_size": 1024,
        "optimization": {
            "actor_learning_rate": 0.01,
            "critic_learning_rate": 0.01,
            "entropy_learning_rate": 0.01,
        },
        "max_entropy_target_proportion": 0.0,
        "batch_mode": 'complete_episodes',
        "num_gpus": 0,
        "sample_batch_size": 20,
        #
        "timesteps_per_iteration": 1,
        "min_iter_time_s": 0,
    }

    POKER_METANASH_FICTITIOUS_PLAY_ITERS = 20000
    POKER_PSRO_EXPLORATION_COEFF = 0.1
    POKER_PAYOFF_MATRIX_NOISE_STD_DEV = 0.0

    POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION = 0.2
    POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS = True
    POKER_PIPELINE_INIT_FROM_POP = False

    POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS = int(10000)

    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = 0.01
    avg_episode_length = 2.5
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS = int(2500 * avg_episode_length)
    POKER_SUBMISSION_THRESHOLD_STEPS_START = int(7500 * avg_episode_length)
    POKER_SUBMISSION_MIN_STEPS = int(10000 * avg_episode_length)
    POKER_SUBMISSION_MAX_STEPS = None

elif POKER_GAME_VERSION == BARRAGE:
    IS_ON_ARCUS = "arcus" in gethostname().lower()

    POKER_ENV_CONFIG = {
        'version': SHORT_BARRAGE,
        'observation_mode': PARTIALLY_OBSERVABLE,
        'channel_mode': 'extended',
        'repeat_games_from_other_side': False,
        'random_player_assignment': True,
        'human_inits': True,
        'penalize_ties': False
    }
    SELECTED_CONFIG_KEY = SAC_ARCH1P_MODEL_CONFIG_KEY
    ENV_CLASS = SpatialStrategoMultiAgentEnv

    num_workers = 5

    POKER_TRAINER_BASE_CONFIG = {
        "log_level": "DEBUG",
        "metrics_smoothing_episodes": 10000,
        "memory_per_worker": 7019430400 if IS_ON_ARCUS else 4019430400,
        "num_envs_per_worker": 1,
        "num_workers": num_workers,
        "num_gpus_per_worker": 1 / num_workers,
        "env": SPATIAL_STRATEGO_PARALLEL_ENV,
        "env_config": with_updates(base_dict=POKER_ENV_CONFIG, updates_dict={
            'num_envs': 64,
        }),

        "buffer_size": int(1e6),
        "tau": 5e-3,
        "gamma": 0.99,
        "train_batch_size": 2048,
        "optimization": {
            "actor_learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "entropy_learning_rate": 1e-3,
        },
        "max_entropy_target_proportion": 0.3,
        "batch_mode": 'truncate_episodes',
        "num_gpus": 1,
        "sample_batch_size": 100,
        #
        "timesteps_per_iteration": 1,
        "min_iter_time_s": 0,
    }

    POKER_METANASH_FICTITIOUS_PLAY_ITERS = 20000
    POKER_PSRO_EXPLORATION_COEFF = 0.1
    POKER_PAYOFF_MATRIX_NOISE_STD_DEV = 0.01
    POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION = 0.3
    POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS = True
    POKER_PIPELINE_INIT_FROM_POP = True

    POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS = int(1.5e6)
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = 0.03
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS = int(3e6) # original was int(15e6)
    POKER_SUBMISSION_THRESHOLD_STEPS_START = int(1e6)
    POKER_SUBMISSION_MIN_STEPS = int(3e6)
    POKER_SUBMISSION_MAX_STEPS = None

elif POKER_GAME_VERSION == FIVES:
    POKER_ENV_CONFIG = {
        'version': FIVES,
        'observation_mode': PARTIALLY_OBSERVABLE,
        'channel_mode': 'extended',
        'repeat_games_from_other_side': False,
        'random_player_assignment': True,
        'human_inits': False,
        'penalize_ties': False
    }
    SELECTED_CONFIG_KEY = SAC_ARCH1P_FIVES_MODEL_CONFIG_KEY
    ENV_CLASS = SpatialStrategoMultiAgentEnv

    num_workers = 3

    POKER_TRAINER_BASE_CONFIG = {
        "log_level": "DEBUG",
        "metrics_smoothing_episodes": 10000,
        "memory_per_worker": 4019430400,
        "num_envs_per_worker": 1,
        "num_workers": num_workers,
        "num_gpus_per_worker": 0,
        "env": SPATIAL_STRATEGO_PARALLEL_ENV,
        "env_config": with_updates(base_dict=POKER_ENV_CONFIG, updates_dict={
            'num_envs': 1,
        }),

        "buffer_size": int(20000),
        "learning_starts": 10000,
        "tau": 0.01,
        "gamma": 0.99,
        "train_batch_size": 1024,
        "optimization": {
            "actor_learning_rate": 3e-3,
            "critic_learning_rate": 3e-3,
            "entropy_learning_rate": 3e-3,
        },
        "max_entropy_target_proportion": 0.2,
        "batch_mode": 'complete_episodes',
        "num_gpus": 0,
        "sample_batch_size": 100,
        #
        "timesteps_per_iteration": 1,
        "min_iter_time_s": 0,
    }

    POKER_METANASH_FICTITIOUS_PLAY_ITERS = 20000
    POKER_PSRO_EXPLORATION_COEFF = 0.1
    POKER_PAYOFF_MATRIX_NOISE_STD_DEV = 0.0
    POKER_PIPELINE_WARMUP_ENTROPY_TARGET_PROPORTION = 0.2
    POKER_PIPELINE_LIVE_PAYOFF_TABLE_CALC_IS_ASYNCHRONOUS = True
    POKER_PIPELINE_INIT_FROM_POP = False

    POKER_PIPELINE_CHECKPOINT_AND_REFRESH_LIVE_TABLE_EVERY_N_STEPS = int(200000)
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_REWARD = 0.03
    POKER_SUBMISSION_IMPROVEMENT_THRESHOLD_PER_STEPS = int(100000)
    POKER_SUBMISSION_THRESHOLD_STEPS_START = int(40000)
    POKER_SUBMISSION_MIN_STEPS = int(200000)
    POKER_SUBMISSION_MAX_STEPS = None

else:
    raise ValueError(f"Unknown poker game version set with env variable POKER_GAME_VERSION :{POKER_GAME_VERSION}")