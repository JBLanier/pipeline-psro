import numpy as np
import gym
from collections import OrderedDict
import logging

from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import Preprocessor, DictFlatteningPreprocessor, NoPreprocessor
from ray.rllib.models import ModelCatalog

STRATEGO_PREPROCESSOR = "stratego_preprocessor"


logger = logging.getLogger(__name__)


class StrategoDictFlatteningPreprocessor(Preprocessor):
    """Preprocesses each dict value, then flattens it all into a vector.

    RLlib models will unpack the flattened output before _build_layers_v2().
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space, options):
        assert isinstance(self._obs_space, gym.spaces.Dict)
        size = 0
        self.preprocessors = []
        for space_id, space in self._obs_space.spaces.items():

            logger.debug("Creating sub-preprocessor for {}".format(space))

            if space_id == 'internal_state':
                self.dummy_internal_state = np.zeros(shape=space.shape, dtype=np.float32)
            preprocessor = NoPreprocessor(space, self._options)
            self.preprocessors.append(preprocessor)
            size += preprocessor.size
        return (size, )

    @override(Preprocessor)
    def transform(self, observation):

        if 'internal_state' in observation and observation['internal_state'] is None:
            observation['internal_state'] = self.dummy_internal_state
        self.check_shape(observation)
        array = np.zeros(self.shape)
        self.write(observation, array, 0)
        return array

    @override(Preprocessor)
    def write(self, observation, array, offset):
        if not isinstance(observation, OrderedDict):
            observation = OrderedDict(sorted(list(observation.items())))
        assert len(observation) == len(self.preprocessors), \
            (len(observation), len(self.preprocessors))
        for o, p in zip(observation.values(), self.preprocessors):
            p.write(o, array, offset)
            offset += p.size

    @property
    @override(Preprocessor)
    def observation_space(self):
        obs_space = gym.spaces.Box(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            self.shape,
            dtype=np.float32)
        # Stash the unwrapped space so that we can unwrap dict spaces
        # automatically in models
        obs_space.original_space = self._obs_space
        return obs_space


ModelCatalog.register_custom_preprocessor(STRATEGO_PREPROCESSOR, StrategoDictFlatteningPreprocessor)
