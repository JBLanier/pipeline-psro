import numpy as np
from mprl.scripts.poker_parallel_algos.evaluators.evaluator_utils import make_get_policy_fn
from mprl.scripts.poker_parallel_algos.utils.policy_config_keys import SAC_ARCH1P_MODEL_CONFIG_KEY
from stratego_env import StrategoMultiAgentEnv


class BarrageAgentPolicyPopulationWrapper:

    POLICY_KEYS_TO_METANASH_SELECTION_PROBS = {
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-12_pid_1947_07.07.03PM_May-27-2020/policy_submissions/policy_0_08.17.21AM_May-28-2020_iter_13998.dill': 4.91823731e-06,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-11_pid_1392_07.07.08PM_May-27-2020/policy_submissions/policy_1_08.53.08AM_May-29-2020_iter_41377.dill': 2.31186581e-06,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-12_pid_1941_07.07.03PM_May-27-2020/policy_submissions/policy_2_06.23.52PM_May-29-2020_iter_44447.dill': 2.91053832e-05,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-11_pid_1456_07.07.08PM_May-27-2020/policy_submissions/policy_3_03.03.44AM_May-30-2020_iter_57092.dill': 2.78225070e-01,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-12_pid_1947_08.18.15AM_May-28-2020/policy_submissions/policy_4_11.51.42AM_May-30-2020_iter_48207.dill': 2.41526654e-02,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-11_pid_1392_08.54.37AM_May-29-2020/policy_submissions/policy_5_07.58.36PM_May-30-2020_iter_38228.dill': 3.36704862e-01,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-12_pid_1941_06.25.57PM_May-29-2020/policy_submissions/policy_6_11.24.38AM_May-31-2020_iter_39770.dill': 2.48275259e-01,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-11_pid_1456_03.05.40AM_May-30-2020/policy_submissions/policy_7_08.01.21PM_May-31-2020_iter_42924.dill': 1.12600139e-01,
        'learner_barrage_sac_arch1_pipeline_psro/learner_barrage_sac_arch1_pipeline_psro_arcus-12_pid_1947_11.53.25AM_May-30-2020/policy_submissions/policy_8_05.21.44AM_Jun-01-2020_iter_39247.dill': 5.66871997e-06
    }

    def __init__(self, stratego_env_config):
        stratego_env_config = stratego_env_config.copy()
        self._util_env = StrategoMultiAgentEnv(env_config=stratego_env_config)
        stratego_env_config['env_class'] = StrategoMultiAgentEnv
        self._stratego_env_config = stratego_env_config

        get_policy_fn = make_get_policy_fn(model_weights_object_key=None,
                                           model_config_object_key=SAC_ARCH1P_MODEL_CONFIG_KEY,
                                           policy_name="Barrage_P2SRO_NeurIPS_2020_Agent",
                                           policy_class_name="SACDiscreteTFPolicy", storage_client=None,
                                           minio_bucket_name=None, download_lock=None, manual_config=None,
                                           population_policy_keys_to_selection_probs=self.POLICY_KEYS_TO_METANASH_SELECTION_PROBS)
        _, self._current_policy_fn, self._sample_new_weights_fn = get_policy_fn(stratego_env_config=self._stratego_env_config)
        self._sample_new_weights_fn()

    def sample_new_policy_from_metanash(self):
        self._sample_new_weights_fn()

    def get_action(self, extended_channels_observation):
        if self._current_policy_fn is None:
            self._sample_new_weights_fn()
        action_as_flattened_spatial_index, _ = self._current_policy_fn(extended_channels_observation)
        action_as_spatial_index = np.unravel_index(action_as_flattened_spatial_index, shape=self._util_env.base_env.spatial_action_size)
        action_as_1d_index = self._util_env.base_env.get_action_1d_index_from_spatial_index(action_as_spatial_index)
        return action_as_1d_index
