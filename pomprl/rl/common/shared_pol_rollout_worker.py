from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.offline import NoopOutput

class SharedPolicyRolloutWorker(RolloutWorker):

    def __init__(self,
                 env_creator,
                 policy,
                 policy_mapping_fn=None,
                 policies_to_train=None,
                 tf_session_creator=None,
                 batch_steps=100,
                 batch_mode="truncate_episodes",
                 episode_horizon=None,
                 preprocessor_pref="deepmind",
                 sample_async=False,
                 compress_observations=False,
                 num_envs=1,
                 observation_filter="NoFilter",
                 clip_rewards=None,
                 clip_actions=True,
                 env_config=None,
                 model_config=None,
                 policy_config=None,
                 worker_index=0,
                 monitor_path=None,
                 log_dir=None,
                 log_level=None,
                 callbacks=None,
                 input_creator=lambda ioctx: ioctx.default_sampler_input(),
                 input_evaluation=frozenset([]),
                 output_creator=lambda ioctx: NoopOutput(),
                 remote_worker_envs=False,
                 remote_env_batch_wait_ms=0,
                 soft_horizon=False,
                 no_done_at_end=False,
                 seed=None,
                 _fake_sampler=False,
                 local_shared_policy_map=None,
                 local_shared_preprocessors=None,
                 ):
        self._local_shared_policy_map = local_shared_policy_map
        self._local_shared_preprocessors = local_shared_preprocessors
        assert False
        super(SharedPolicyRolloutWorker, self).__init__(
            env_creator=env_creator,
            policy=policy,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
            tf_session_creator=tf_session_creator,
            batch_steps=batch_steps,
            batch_mode=batch_mode,
            episode_horizon=episode_horizon,
            preprocessor_pref=preprocessor_pref,
            sample_async=sample_async,
            compress_observations=compress_observations,
            num_envs=num_envs,
            observation_filter=observation_filter,
            clip_rewards=clip_rewards,
            clip_actions=clip_actions,
            env_config=env_config,
            model_config=model_config,
            policy_config=policy_config,
            worker_index=worker_index,
            monitor_path=monitor_path,
            log_dir=log_dir,
            log_level=log_level,
            callbacks=callbacks,
            input_creator=input_creator,
            input_evaluation=input_evaluation,
            output_creator=output_creator,
            remote_worker_envs=remote_worker_envs,
            remote_env_batch_wait_ms=remote_env_batch_wait_ms,
            soft_horizon=soft_horizon,
            no_done_at_end=no_done_at_end,
            seed=seed,
            _fake_sampler=_fake_sampler,
        )

    def _build_policy_map(self, policy_dict, policy_config):

        if self._local_shared_policy_map is not None:
            policy_map = self._local_shared_policy_map
            preprocessors = self._local_shared_preprocessors
            return policy_map, preprocessors
        else:
            return RolloutWorker._build_policy_map(self, policy_dict, policy_config)
