from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer, check_config_and_setup_param_noise, setup_exploration, update_target_if_needed
from ray.rllib.agents.sac.sac_policy import SACTFPolicy
from ray.rllib.agents.trainer import with_base_config
from mprl.rl.common.custom_evals_trainer_mixin import perform_relevant_custom_evals, CustomEvaluationsTrainerMixin
from mprl.rl.common.weights_utils_trainer_mixin import WeightsUtilsTrainerMixin
from mprl.rl.sac.async.sac_async_replay_optimizer import AsyncReplayOptimizer

OPTIMIZER_SHARED_CONFIGS = [
    "buffer_size", "prioritized_replay", "prioritized_replay_alpha",
    "prioritized_replay_beta", "prioritized_replay_eps", "sample_batch_size",
    "train_batch_size", "learning_starts"
]

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Model ===
    "twin_q": True,
    "use_state_preprocessor": False,
    # "policy": "GaussianLatentSpacePolicy",
    # RLlib model options for the Q function
    # "Q_model": {
    #     "hidden_activation": "relu",
    #     "hidden_layer_sizes": (256, 256),
    # },
    # # RLlib model options for the policy function
    # "policy_model": {
    #     "hidden_activation": "relu",
    #     "hidden_layer_sizes": (256, 256),
    # },

    # === Learning ===
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 5e-3,
    # Target entropy lower bound. This is the inverse of reward scale,
    # and will be optimized automatically.
    "target_entropy": "auto",
    # Disable setting done=True at end of episode.
    "no_done_at_end": False,
    # N-step target updates
    "n_step": 1,

    "max_entropy_target_proportion": 0.2,

    # === Evaluation ===
    # The evaluation stats will be reported under the "evaluation" metric key.
    "evaluation_interval": 0,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 200,
    # Extra configuration that disables exploration.
    "evaluation_config": {
        "exploration_enabled": False,
    },

    # === Exploration ===
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 100,
    "exploration_enabled": True,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(1e6),
    # If True prioritized replay buffer will be used.
    # TODO(hartikainen): Make sure this works or remove the option.
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "beta_annealing_fraction": 0.2,
    "final_prioritized_replay_beta": 0.4,
    "compress_observations": False,

    # === Optimization ===
    "optimization": {
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 1e-4,
        "entropy_learning_rate": 1e-4,
    },
    # If not None, clip gradients during optimization at this value
    "grad_norm_clipping": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 30000,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "sample_batch_size": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 256,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,

    # === Parallelism ===
    # Whether to use a GPU for local optimization.
    "num_gpus": 0,
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,

    # TODO(ekl) these are unused; remove them from sac config
    "per_worker_exploration": False,
    "exploration_fraction": 0.1,
    "schedule_max_timesteps": 100000,
    "exploration_final_eps": 0.02,
})
# __sphinx_doc_end__
# yapf: enable


def make_async_optimizer(workers, config):
    return AsyncReplayOptimizer(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        train_batch_size=config["train_batch_size"],
        sample_batch_size=config["sample_batch_size"],
        **config["optimizer"])


def validate_config(config):
    check_config_and_setup_param_noise(config=config)

    if config["callbacks"]["on_episode_end"]:
        episode_end_callback = config["callbacks"]["on_episode_end"]
    else:
        episode_end_callback = None

    def on_episode_end(params):
        # Add policy win rates metrics
        if episode_end_callback is not None:
            episode_end_callback(params)
        episode = params['episode']

        agent_last_infos = {policy_key: episode.last_info_for(agent_id) for agent_id, policy_key in episode.agent_rewards.keys()}
        for policy_key, info in agent_last_infos.items():
            if info is not None:
                if info['game_result'] == 'won':
                    episode.custom_metrics[policy_key + '_win'] = 1.0
                else:
                    episode.custom_metrics[policy_key + '_win'] = 0.0

    config["callbacks"]["on_episode_end"] = on_episode_end

    for eval_name, eval_config in config['extra_evaluation_configs'].items():
        if 'env_config' not in eval_config or not eval_config['env_config']:
            eval_config['env_config'] = {}
        eval_config['env_config'] = with_base_config(base_config=config["default_extra_eval_env_config"],
                                                     extra_config=eval_config['env_config'])
def pg_custom_eval_trainer_setup_mixins(trainer):
    CustomEvaluationsTrainerMixin.__init__(trainer)


def pg_custom_eval_trainer_before_init(trainer):
    setup_exploration(trainer)
    if any(trainer.config["callbacks_before_trainer_init"]):
        for callback in trainer.config["callbacks_before_trainer_init"]:
            callback(trainer)


def after_optimizer_step(trainer, fetches):
    update_target_if_needed(trainer, fetches)
    perform_relevant_custom_evals(trainer=trainer, fetches=fetches)


def collect_metrics(trainer, selected_workers=None):
    metrics = trainer.optimizer.collect_metrics(
        trainer.config["collect_metrics_timeout"],
        min_history=trainer.config["metrics_smoothing_episodes"],
        selected_workers=selected_workers)

    if hasattr(trainer, "extra_evaluation_metrics") and trainer.extra_evaluation_metrics:
        metrics.update(trainer.extra_evaluation_metrics)

    return metrics

def pg_custom_eval_trainer_after_init(trainer):
    pg_custom_eval_trainer_setup_mixins(trainer)

    if any(trainer.config["callbacks_after_trainer_init"]):
        for callback in trainer.config["callbacks_after_trainer_init"]:
            callback(trainer)


ppo_custom_eval_trainer_added_config_items = {
    "extra_evaluation_configs": {},
    "default_extra_eval_env_config": {},
    "callbacks_after_trainer_init": [],
    "callbacks_before_trainer_init": [],
    "export_policy_weights_ids": [],
    "redo_invalid_games": False,
    "wandb": {},
    "ed": None,
    "policy_catalog": None,
    "eq_iters": None,
    "adaptive_pval_test": False,
    "br_thres": None,
    "eq_thres": None,
    "br_eval_against_policy": None,
    "thres_is_pval": None,
    "adaptive_pval": None
}

PG_CUSTOM_EVAL_TRAINER_DEFAULT_CONFIG = with_base_config(base_config=DEFAULT_CONFIG,
                                                          extra_config=ppo_custom_eval_trainer_added_config_items)

mixins = [CustomEvaluationsTrainerMixin, WeightsUtilsTrainerMixin]

SACAsyncTrainer = GenericOffPolicyTrainer.with_updates(
    name="SACAsyncDiscrete",
    make_policy_optimizer=make_async_optimizer,
    default_config=PG_CUSTOM_EVAL_TRAINER_DEFAULT_CONFIG,
    default_policy=SACTFPolicy,
    validate_config=validate_config,
    before_init=pg_custom_eval_trainer_before_init,
    after_init=pg_custom_eval_trainer_after_init,
    after_optimizer_step=after_optimizer_step,
    collect_metrics_fn=collect_metrics,
    mixins=mixins
)
