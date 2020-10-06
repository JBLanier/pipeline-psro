import ray
from ray.rllib.utils.memory import ray_get_and_free
from mprl.rl.envs.opnspl.measure_nashconv_eval_callback import get_measure_nash_conv_nonlstm_eval_callback, measure_nash_conv_nonlstm
from mprl.rl.common.util import get_random_policy_object_key


def get_alpha_rank_callbacks_and_eval_configs(new_population_member_policy_rllib_key,
                                              new_population_member_policy_mapping_id,
                                              static_league_policy_rllib_key,
                                              static_league_policy_policy_mapping_id,

                                              new_population_member_vs_league_eval_name,
                                              new_population_member_vs_league_eval_interval,
                                              new_population_member_vs_league_eval_policy_mapping_fn,
                                              base_experiment_name,
                                              full_experiment_name,
                                              alpha_rank_games_per_policy_matchup,
                                              get_extra_data_dict_for_new_population_rllib_policy_fn=None,
                                              set_static_policy_key_generator_for_training_fn=None,
                                              game_version=None,
                                              measure_nash_conv=False,
                                              attach_tag_to_catalog_submissions: str = None,
                                              submit_policy_to_empty_catalog_at_start=True):


    """
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    Trainer Config
    
    //////////////////////////////////////////////////////////////////////////////////////////////////
    """

    add_to_trainer_config = {}

    def set_policy_catalog_handle_attr_in_policies(trainer):

        policy_catalog = trainer.config['policy_catalog']

        if not isinstance(policy_catalog, ray.actor.ActorHandle):
            assert callable(policy_catalog)
            policy_catalog = policy_catalog(trainer)
            trainer.config['policy_catalog'] = policy_catalog

        def set_policy_catalog_handle(worker):
            for policy in worker.policy_map.values():
                policy.policy_catalog = policy_catalog

        trainer.workers.foreach_worker(set_policy_catalog_handle)
        if hasattr(trainer, 'extra_eval_worker_sets_and_configs'):
            for workers, _ in trainer.extra_eval_worker_sets_and_configs.values():
                workers.foreach_worker(set_policy_catalog_handle)

    def submit_trainable_policy_checkpoint_to_league_with_payoff_table_results(trainer):
        policy_catalog = trainer.config['policy_catalog']

        trainable_policy = trainer.workers.local_worker().policy_map[new_population_member_policy_rllib_key]

        trainable_vs_league_eval_worker_set, _ = trainer.extra_eval_worker_sets_and_configs[new_population_member_vs_league_eval_name]
        local_eval_trainable_policy = trainable_vs_league_eval_worker_set.local_worker().policy_map[new_population_member_policy_rllib_key]

        if hasattr(local_eval_trainable_policy, 'latest_payoff_table'):
            latest_payoff_table = local_eval_trainable_policy.latest_payoff_table
            latest_payoff_games_count = local_eval_trainable_policy.latest_payoff_games_count
            for v in latest_payoff_games_count.values():
                assert v == alpha_rank_games_per_policy_matchup

            latest_payoff_table = {k: v / alpha_rank_games_per_policy_matchup for k, v in latest_payoff_table.items()}
        else:
            print("Trainable policy has no payoff table attribute yet, Submitting policy to catalog with no game results")
            latest_payoff_table = {}
            latest_payoff_games_count = {}

        local_eval_trainable_policy.latest_payoff_table = None
        local_eval_trainable_policy.latest_payoff_games_count = None

        object_key = get_random_policy_object_key(base_experiment_name=base_experiment_name,
                                                  full_experiment_name=full_experiment_name,
                                                  tag=attach_tag_to_catalog_submissions)

        if get_extra_data_dict_for_new_population_rllib_policy_fn is not None:
            extra_data = get_extra_data_dict_for_new_population_rllib_policy_fn(trainer)
        else:
            extra_data = {}

        if measure_nash_conv:
            nash_conv = measure_nash_conv_nonlstm(rllib_policy=trainable_policy, poker_game_version=game_version)
            extra_data['nash_conv'] = nash_conv

        weights_to_submit = trainable_policy.get_model_weights(remove_scope_prefix=new_population_member_policy_rllib_key)

        ray_get_and_free(policy_catalog.submit_new_policy.remote(
            policy_file_key=object_key,
            policy_weights=weights_to_submit,
            policy_keys_to_payoff_dict=latest_payoff_table,
            steps_trained=trainer.optimizer.num_steps_sampled,
            ranked_games_played=sum(latest_payoff_games_count.values()),
            extra_data=extra_data,
            tag=attach_tag_to_catalog_submissions))
        del weights_to_submit

        def set_last_policy_key(worker):
            setattr(worker.policy_map[new_population_member_policy_rllib_key], 'last_checkpoint_league_key', object_key)

        trainer.workers.foreach_worker(set_last_policy_key)
        if hasattr(trainer, 'extra_eval_worker_sets_and_configs'):
            for workers, _ in trainer.extra_eval_worker_sets_and_configs.values():
                workers.foreach_worker(set_last_policy_key)

    def set_games_to_play_for_new_member_vs_league_eval_in_trainer(trainer):

        print("\n\n\n\n\n\nSET GAMES {}\n\n\n\n\n\n".format(attach_tag_to_catalog_submissions))

        policy_catalog = trainer.config['policy_catalog']
        all_policy_keys = ray_get_and_free(policy_catalog.get_all_keys.remote())

        trainable_vs_league_eval_worker_set, _ = trainer.extra_eval_worker_sets_and_configs[
            new_population_member_vs_league_eval_name]

        def next_matchup_key_generator():
            for policy_key in all_policy_keys:
                for _ in range(alpha_rank_games_per_policy_matchup):
                    yield policy_key

        trainable_vs_league_eval_worker_set.eval_num_episodes_override = len(all_policy_keys) * alpha_rank_games_per_policy_matchup

        local_eval_static_policy = trainable_vs_league_eval_worker_set.local_worker().policy_map[static_league_policy_rllib_key]

        local_eval_trainable_policy = trainable_vs_league_eval_worker_set.local_worker().policy_map[new_population_member_policy_rllib_key]

        local_eval_static_policy.next_alpha_rank_matchup_generator = next_matchup_key_generator()

        def matchup_generator_finished_callback():
            submit_trainable_policy_checkpoint_to_league_with_payoff_table_results(trainer)
            set_games_to_play_for_new_member_vs_league_eval_in_trainer(trainer)

        local_eval_static_policy.matchup_generator_finished_callback = matchup_generator_finished_callback

        local_eval_trainable_policy.latest_payoff_table = {}
        local_eval_trainable_policy.latest_payoff_games_count = {}

    add_to_trainer_config["callbacks_after_trainer_init"] = [
        set_policy_catalog_handle_attr_in_policies,
        # set_games_to_play_for_new_member_vs_league_eval_in_trainer
    ]
    if submit_policy_to_empty_catalog_at_start:
        add_to_trainer_config["callbacks_after_trainer_init"].append(submit_trainable_policy_checkpoint_to_league_with_payoff_table_results)

    add_to_trainer_config['callbacks'] = {}
    if set_static_policy_key_generator_for_training_fn is not None:

        add_to_trainer_config["callbacks_after_trainer_init"].append(set_static_policy_key_generator_for_training_fn)

        def get_static_policy_for_training(params):
            policies = params["policy"]
            static_league_policy = policies[static_league_policy_rllib_key]
            policy_catalog = static_league_policy.policy_catalog

            new_league_key = next(static_league_policy.policy_training_generator)

            if not hasattr(static_league_policy, 'current_league_policy_key') or \
                    new_league_key != static_league_policy.current_league_policy_key:
                new_weights = ray_get_and_free(policy_catalog.get_weights_by_key.remote(new_league_key))
                static_league_policy.set_model_weights(weights=new_weights, add_scope_prefix=static_league_policy_rllib_key)
                static_league_policy.current_league_policy_key = new_league_key
                del new_weights

        add_to_trainer_config['callbacks'] = {
               "on_episode_start": get_static_policy_for_training
        }


    """
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    New Population Member vs League Eval

    //////////////////////////////////////////////////////////////////////////////////////////////////
    """


    def get_static_policy_for_new_member_vs_league_eval(params):
        policies = params["policy"]
        static_league_policy = policies[static_league_policy_rllib_key]
        policy_catalog = static_league_policy.policy_catalog
        assert hasattr(static_league_policy, "next_alpha_rank_matchup_generator")

        try:
            if (hasattr(static_league_policy, "_matchup_generator_needs_skip") and static_league_policy._matchup_generator_needs_skip):
                static_league_policy._matchup_generator_needs_skip = False
                _ = next(static_league_policy.next_alpha_rank_matchup_generator)
            next_matchup_key = next(static_league_policy.next_alpha_rank_matchup_generator)
        except StopIteration:
            print("LATEST GAMES COUNT: {}".format(policies[new_population_member_policy_rllib_key].latest_payoff_games_count))
            static_league_policy.matchup_generator_finished_callback()
            next_matchup_key = next(static_league_policy.next_alpha_rank_matchup_generator)
            static_league_policy._matchup_generator_needs_skip = True

        if not hasattr(static_league_policy, 'current_league_policy_key') or \
                next_matchup_key != static_league_policy.current_league_policy_key:
            new_weights = ray_get_and_free(policy_catalog.get_weights_by_key.remote(next_matchup_key))
            static_league_policy.set_model_weights(weights=new_weights, add_scope_prefix=static_league_policy_rllib_key)
            static_league_policy.current_league_policy_key = next_matchup_key
            del new_weights

    def add_game_result_to_policy_payoff_table_stats(params):
        policies = params["policy"]
        episode = params["episode"]

        static_league_key = policies[static_league_policy_rllib_key].current_league_policy_key
        trainable_policy = policies[new_population_member_policy_rllib_key]

        trainable_policy_return = episode.agent_rewards.get((new_population_member_policy_mapping_id, new_population_member_policy_rllib_key), None)
        assert trainable_policy_return is not None

        trainable_policy.latest_payoff_table[static_league_key] = (
                trainable_policy_return + trainable_policy.latest_payoff_table.get(static_league_key, 0.0))

        trainable_policy.latest_payoff_games_count[static_league_key] = (
                1 + trainable_policy.latest_payoff_games_count.get(static_league_key, 0.0))

    def add_new_policy_alpha_rank_to_eval_metrics(trainer, eval_metrics):
        policy_catalog = trainer.config['policy_catalog']
        trainable_policy = trainer.workers.local_worker().policy_map[new_population_member_policy_rllib_key]
        last_checkpoint_key = trainable_policy.last_checkpoint_league_key
        skill_ranking = ray_get_and_free(policy_catalog.get_skill_ranking.remote(policy_file_key=last_checkpoint_key))
        eval_metrics["new_policy_alpha_rank"] = skill_ranking

    def add_effective_population_diversity_to_eval_metrics(trainer, eval_metrics):
        policy_catalog = trainer.config['policy_catalog']
        diversity = ray_get_and_free(policy_catalog.get_current_effective_population_diversity.remote())
        print("effective meta nash population diversity:", diversity)
        eval_metrics["effective_population_diversity"] = diversity

    after_custom_eval_callbacks = []
    metrics_to_track_at_top_level = {}

    if measure_nash_conv:
        measure_nash_conv_latest_policy_callback = get_measure_nash_conv_nonlstm_eval_callback(
            eval_name=new_population_member_vs_league_eval_name, poker_game_version=game_version,
            measure_policy_ids=[new_population_member_policy_rllib_key])

        def measure_nash_conv_of_meta_nash_callback(trainer, eval_metrics):
            policy_catalog = trainer.config['policy_catalog']
            alpha_rank_dict = ray_get_and_free(policy_catalog.get_alpha_rank_scores.remote())
            print("alpha rank scores: {}".format(list(alpha_rank_dict.values())))
            eval_workers, eval_config = trainer.extra_eval_worker_sets_and_configs[new_population_member_vs_league_eval_name]
            rllib_policy = eval_workers.local_worker().policy_map[static_league_policy_rllib_key]

            def set_weights_fn(policy_key):
                weights = ray_get_and_free(policy_catalog.get_weights_by_key.remote(policy_key))
                rllib_policy.set_model_weights(weights=weights, add_scope_prefix=static_league_policy_rllib_key)
                del weights

            nash_conv_result = measure_nash_conv_nonlstm(rllib_policy=rllib_policy,
                                                         poker_game_version=game_version,
                                                         policy_mixture_dict=alpha_rank_dict,
                                                         set_policy_weights_fn=set_weights_fn)

            eval_metrics['meta_nash_ground_truth_nashconv'] = nash_conv_result
            print("NASH CONV {}:".format('meta nash'), nash_conv_result)

        after_custom_eval_callbacks.append(measure_nash_conv_latest_policy_callback)
        metrics_to_track_at_top_level[new_population_member_policy_rllib_key + '_ground_truth_nashconv'] = (new_population_member_policy_rllib_key + '_ground_truth_nashconv',)

        after_custom_eval_callbacks.append(measure_nash_conv_of_meta_nash_callback)
        metrics_to_track_at_top_level['meta_nash_ground_truth_nashconv'] = ('meta_nash_ground_truth_nashconv',)

    after_custom_eval_callbacks.append(add_new_policy_alpha_rank_to_eval_metrics)
    metrics_to_track_at_top_level['new_policy_alpha_rank'] = ('new_policy_alpha_rank',)

    after_custom_eval_callbacks.append(add_effective_population_diversity_to_eval_metrics)
    metrics_to_track_at_top_level['effective_population_diversity'] = ('effective_population_diversity',)

    if set_static_policy_key_generator_for_training_fn is not None:
        after_custom_eval_callbacks.append(lambda trainer, eval_metrics: set_static_policy_key_generator_for_training_fn(trainer))

    def on_episode_end(params):
        add_game_result_to_policy_payoff_table_stats(params)
        episode = params['episode']
        # print("Eval episode length {}".format(episode.length))

    new_member_vs_league_eval = {
        "multiagent": {
            "policy_mapping_fn": new_population_member_vs_league_eval_policy_mapping_fn,
        },
        "callbacks": {
            "on_episode_start": get_static_policy_for_new_member_vs_league_eval,
            "on_episode_end": on_episode_end
        },
        "evaluation_num_workers": 0,
        "evaluation_num_episodes": 1,
        "evaluation_interval": new_population_member_vs_league_eval_interval,
        "before_custom_evaluation_callback": set_games_to_play_for_new_member_vs_league_eval_in_trainer,
        "after_custom_evaluation_callback": tuple(callback for callback in after_custom_eval_callbacks),
        "metrics_to_track_at_top_level": metrics_to_track_at_top_level
    }

    return add_to_trainer_config, new_member_vs_league_eval