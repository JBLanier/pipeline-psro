from mprl.rl.common.custom_evals_trainer_mixin import perform_relevant_custom_evals, CustomEvaluationsTrainerMixin
from mprl.rl.common.weights_utils_trainer_mixin import WeightsUtilsTrainerMixin
from ray.rllib.agents.ppo.ppo import PPOTrainer, update_kl, validate_config, DEFAULT_CONFIG
from ray.rllib.agents.trainer import with_base_config
from ray import tune

from mprl.rl.common.util import get_redo_sample_if_game_result_was_invalid_worker_callback


def ppo_custom_eval_trainer_setup_mixins(trainer):
    CustomEvaluationsTrainerMixin.__init__(trainer)


def ppo_custom_eval_trainer_before_init(trainer):
    if any(trainer.config["callbacks_before_trainer_init"]):
        for callback in trainer.config["callbacks_before_trainer_init"]:
            callback(trainer)


def after_optimizer_step(trainer, fetches):
    update_kl(trainer=trainer, fetches=fetches)
    perform_relevant_custom_evals(trainer=trainer, fetches=fetches)


def collect_metrics(trainer, selected_workers=None):
    metrics = trainer.optimizer.collect_metrics(
        trainer.config["collect_metrics_timeout"],
        min_history=trainer.config["metrics_smoothing_episodes"],
        selected_workers=selected_workers)

    if hasattr(trainer, "extra_evaluation_metrics") and trainer.extra_evaluation_metrics:
        metrics.update(trainer.extra_evaluation_metrics)

    return metrics


def ppo_custom_eval_trainer_validate_config(config):

    if config['sgd_minibatch_size'] == -1:
        config['sgd_minibatch_size'] = config['train_batch_size']

    validate_config(config)

    if config["redo_invalid_games"]:
        assert config['num_envs_per_worker'] == 1, "redo_invalid_games requires num_envs_per_worker to be set to 1"
        assert config['sample_batch_size'] == 1, "redo_invalid_games requires sample_batch_size to be set to 1"
        assert config['batch_mode'] == 'complete_episodes', "redo_invalid_games requires batch_mode to be set to \'complete_episodes\'"

        if config["callbacks"]["on_sample_end"]:
            sample_end_callback = config["callbacks"]["on_sample_end"]
        else:
            sample_end_callback = None

        config["callbacks"]["on_sample_end"] = tune.function(
            get_redo_sample_if_game_result_was_invalid_worker_callback(sample_end_callback))

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

def ppo_custom_eval_trainer_after_init(trainer):
    ppo_custom_eval_trainer_setup_mixins(trainer)

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

PPO_CUSTOM_EVAL_TRAINER_DEFAULT_CONFIG = with_base_config(base_config=DEFAULT_CONFIG,
                                                          extra_config=ppo_custom_eval_trainer_added_config_items)

ppo_custom_eval_trainer_mixins = [CustomEvaluationsTrainerMixin, WeightsUtilsTrainerMixin]

# Add custom evaluation logic to PPOTrainer
PPOCustomEvalTrainer = PPOTrainer.with_updates(
    name="PPOCustomEvalTrainer",
    default_config=PPO_CUSTOM_EVAL_TRAINER_DEFAULT_CONFIG,
    before_init=ppo_custom_eval_trainer_before_init,
    after_init=ppo_custom_eval_trainer_after_init,
    validate_config=ppo_custom_eval_trainer_validate_config,
    after_optimizer_step=after_optimizer_step,
    collect_metrics_fn=collect_metrics,
    mixins=ppo_custom_eval_trainer_mixins
)