import copy

import logging

from ray.rllib import SampleBatch
from ray.rllib.agents import Trainer
from ray.rllib.evaluation import collect_metrics
from ray.rllib.utils.memory import ray_get_and_free
from ray.tune.util import merge_dicts
from mprl.rl.common.shared_pol_rollout_worker_set import SharedPolicyRolloutWorkerSet
from collections import defaultdict
logger = logging.getLogger(__name__)


class CustomEvaluationsTrainerMixin(object):

    _allow_unknown_subkeys = [
        "tf_session_args", "env_config", "model", "optimizer", "multiagent",
        "custom_resources_per_worker", "evaluation_config", "extra_evaluation_configs",
        "default_extra_eval_env_config", "wandb", "ed"
    ]

    # def _make_workers(self, env_creator, policy, config, num_workers,
    #                   local_shared_policy_map=None, local_shared_preprocessors=None, local_shared_tf_sess=None):
    #     return SharedPolicyRolloutWorkerSet(
    #         env_creator,
    #         policy,
    #         config,
    #         num_workers=num_workers,
    #         logdir=self.logdir,
    #         local_shared_policy_map=local_shared_policy_map,
    #         local_shared_preprocessors=local_shared_preprocessors,
    #         local_shared_tf_sess=local_shared_tf_sess)

    def __init__(self):

        self.extra_eval_worker_sets_and_configs = {}
        self.extra_eval_interval_counters = {}
        self.extra_eval_data = {}

        if any(self.config["extra_evaluation_configs"]):
            for eval_name, eval_config in self.config["extra_evaluation_configs"].items():
                # Update env_config with evaluation settings:exi
                extra_config = copy.deepcopy(eval_config)
                extra_config.update({
                    "num_envs_per_worker": 1,
                    "batch_mode": "complete_episodes",
                    "sample_batch_size": 1,
                    "remote_worker_envs": False,
                })

                self.extra_eval_data[eval_name] = {}

                if 'local_eval_workers_share_main_policy_instance' in extra_config and \
                        extra_config['local_eval_workers_share_main_policy_instance']:

                    print("\n\n\n\n\nSharing driver policy between main workers and rollout worker\n\n\n\n\n")

                    policy_map_to_share = self.workers.local_worker().policy_map
                    preprocessors_to_share = self.workers.local_worker().preprocessors
                    tf_sess_to_share = self.workers.local_worker().tf_sess
                else:
                    policy_map_to_share = None
                    preprocessors_to_share = None
                    tf_sess_to_share = None

                raise NotImplementedError("This is super old code")
                evaluation_workers = self._make_workers(
                    self.env_creator,
                    self._policy,
                    merge_dicts(self.config, extra_config),
                    num_workers=extra_config["evaluation_num_workers"],
                    local_shared_policy_map=policy_map_to_share,
                    local_shared_preprocessors=preprocessors_to_share,
                    local_shared_tf_sess=tf_sess_to_share)

                self.extra_eval_worker_sets_and_configs[eval_name] = (evaluation_workers, extra_config)
                self.extra_eval_interval_counters[eval_name] = 0

        # self.self_play_evaluation_metrics = self._self_play_evaluate()

    def _stop(self):
        # perform the same stop functions as in trainer.py
        if hasattr(self, "workers"):
            self.workers.stop()
        if hasattr(self, "optimizer"):
            self.optimizer.stop()

        # also stop evaluation workers
        for worker_set, config in self.extra_eval_worker_sets_and_configs.values():
            worker_set.stop()

    def _perform_custom_evaluation(self, eval_worker_set, eval_name, eval_config):

        if eval_worker_set.remote_workers():

            raise NotImplementedError("remote workers needs to be checked for custom evals")
            # num_workers = len(eval_worker_set.remote_workers())
            # episodes_per_worker = eval_config["evaluation_num_episodes"] // num_workers
            #
            # logger.info("{}: Evaluating {} episodes across {} workers".format(eval_name, episodes_per_worker * num_workers, num_workers))
            #
            # local_worker_save = self.workers.local_worker().save()
            # for w in eval_worker_set.remote_workers():
            #     w.restore.remote(local_worker_save)
            #
            # def gather_samples(worker):
            #     samples = []
            #     for _ in range(episodes_per_worker):
            #         samples.append(worker.sample())
            #     return SampleBatch.concat_samples(samples)
            #
            # batch = SampleBatch.concat_samples(ray_get_and_free(
            #     [w.apply.remote(gather_samples) for w in eval_worker_set.remote_workers()]))
            #
            # metrics = collect_metrics(remote_workers=eval_worker_set.remote_workers())

        else:
            if not ('local_eval_workers_share_main_policy_instance' in eval_config and
                    eval_config["local_eval_workers_share_main_policy_instance"]):
                eval_worker_set.local_worker().restore(
                    self.workers.local_worker().save())

            if hasattr(eval_worker_set, 'eval_num_episodes_override'):
                num_episodes = eval_worker_set.eval_num_episodes_override
            else:
                num_episodes = eval_config["evaluation_num_episodes"]

            print("{}: Evaluating {} episodes on a local worker".format(eval_name, num_episodes))

            extra_metrics_by_policy = {}
            total_games = 0
            last_batch = None
            for _ in range(num_episodes):
                last_batch = SampleBatch.concat_samples(
                    [eval_worker_set.local_worker().sample()]
                )
                for policy_id, s in last_batch.policy_batches.items():
                    if policy_id not in extra_metrics_by_policy:
                        extra_metrics_by_policy[policy_id] = {
                            'win_percentage': 0.0,
                            'loss_percentage': 0.0,
                            'tie_percentage': 0.0,
                            'invalid_games_that_weren\'t_retried': 0.0
                        }

                    for row in s.rows():
                        infos = row['infos']
                        if 'game_result' in infos:

                            total_games += 1

                            if infos['game_result'] == 'won':
                                extra_metrics_by_policy[policy_id]['win_percentage'] += 1
                            elif infos['game_result'] == 'lost':
                                extra_metrics_by_policy[policy_id]['loss_percentage'] += 1
                            elif infos['game_result'] == 'tied':
                                extra_metrics_by_policy[policy_id]['tie_percentage'] += 1
                            else:
                                raise ValueError(
                                    "\'game_result\' returned in the environment\'s info contained an unknown key: {}".format(
                                        infos['game_result']))

                            if 'game_result_was_invalid' in infos and infos['game_result_was_invalid']:
                                extra_metrics_by_policy[policy_id]['invalid_games_that_weren\'t_retried'] += 1

        metrics = collect_metrics(eval_worker_set.local_worker())

        for policy_id, s in last_batch.policy_batches.items():
            extra_metrics_by_policy[policy_id]['win_percentage'] /= total_games
            extra_metrics_by_policy[policy_id]['loss_percentage'] /= total_games
            extra_metrics_by_policy[policy_id]['tie_percentage'] /= total_games

            metrics[policy_id] = extra_metrics_by_policy[policy_id]

        return metrics


def perform_relevant_custom_evals(trainer: Trainer, fetches=None):

    trainer.extra_evaluation_metrics = None

    # Do any extra evaluations specified in the config

    if any(trainer.extra_eval_worker_sets_and_configs):
        trainer.extra_evaluation_metrics = {}

        for eval_name, (eval_worker_set, eval_config) in trainer.extra_eval_worker_sets_and_configs.items():
            trainer.extra_eval_interval_counters[eval_name] += 1

            do_eval = False

            if 'eval_scheduler' in eval_config:
                do_eval, scheduler_post_eval_callback = eval_config['eval_scheduler'](
                    trainer, trainer.extra_eval_data[eval_name])
            else:
                scheduler_post_eval_callback = None

            # Do eval if eval interval exists and has been met
            if 'evaluation_interval' in eval_config and eval_config['evaluation_interval'] is not None and \
                trainer.extra_eval_interval_counters[eval_name] >= eval_config['evaluation_interval']:
                do_eval = True
                trainer.extra_eval_interval_counters[eval_name] = 0

            # forced evals dont reset interval counter
            if "force_evals_to_happen_on_iterations" in eval_config:
                if trainer.iteration in eval_config["force_evals_to_happen_on_iterations"]:
                    do_eval = True

            # Do eval if we are supposed to eval every time fetches includes 'trained_this_step'
            if fetches is not None and 'eval_after_discrete_train_phase' in eval_config and eval_config['eval_after_discrete_train_phase'] and fetches['trained_this_step']:
                do_eval = True

            # Do eval is trainer has a dict attribute 'do_custom_eval' with eval name in it set to True
            if hasattr(trainer, 'do_custom_eval') and eval_name in trainer.do_custom_eval and trainer.do_custom_eval[eval_name]:
                # unset this flag now that we're doing the requested eval
                trainer.do_custom_eval[eval_name] = False
                do_eval = True

            if do_eval:

                if "iteration_specific_num_episodes_overrides" in eval_config:
                    overrides = eval_config["iteration_specific_num_episodes_overrides"]
                    if str(trainer.iteration) in overrides.keys():
                        eval_config = copy.deepcopy(eval_config)
                        eval_config['evaluation_num_episodes'] = overrides[str(trainer.iteration)]

                # before_custom_evaluation_callback may make changes to the trainer
                if 'before_custom_evaluation_callback' in eval_config:
                    callbacks = eval_config['before_custom_evaluation_callback']
                    if not isinstance(callbacks, tuple):
                        callbacks = (callbacks,)

                    for callback in callbacks:
                        callback(trainer)

                eval_metrics = trainer._perform_custom_evaluation(eval_worker_set=eval_worker_set, eval_name=eval_name, eval_config=eval_config)

                # after_custom_evaluation_callback may modify eval metrics or make changes to the trainer
                if 'after_custom_evaluation_callback' in eval_config:
                    callbacks = eval_config['after_custom_evaluation_callback']
                    if not isinstance(callbacks, tuple):
                        callbacks = (callbacks,)

                    for callback in callbacks:
                        callback(trainer, eval_metrics)

                trainer.extra_evaluation_metrics[eval_name] = eval_metrics

                if 'metrics_to_track_at_top_level' in eval_config and len(eval_config['metrics_to_track_at_top_level']) > 0:
                    metrics_to_track = eval_config['metrics_to_track_at_top_level']
                    for top_level_key_alias, existing_subkeys in metrics_to_track.items():
                        value_to_track = eval_metrics
                        for subkey in existing_subkeys:
                            value_to_track = value_to_track[subkey]

                        trainer.extra_evaluation_metrics[top_level_key_alias] = value_to_track

                if scheduler_post_eval_callback is not None:
                    scheduler_post_eval_callback(trainer, eval_metrics)

