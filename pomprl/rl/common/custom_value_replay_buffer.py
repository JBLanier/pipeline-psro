from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
import sys


from ray.rllib.utils.compression import unpack_if_needed, pack_if_needed
from ray.rllib.utils.window_stat import WindowStat


class CustomValueReplayBuffer(object):
    """Holds custom keys/values in each batch.
    The normal rllib Replay Buffer is hard coded as to what it can hold.
    """

    def __init__(self, size, keys_to_types_dict=None, can_pack_list=None):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """

        if keys_to_types_dict is not None:
            self.keys_to_types_dict = keys_to_types_dict
        else:
            # Default Values
            self.keys_to_types_dict = {"obs": np.array,
                                       "actions": np.array,
                                       "rewards": float,
                                       "new_obs": np.array,
                                       "dones": bool}

        for k in self.keys_to_types_dict:
            if self.keys_to_types_dict[k] == np.array:
                self.keys_to_types_dict[k] = lambda x: np.array(x, copy=False)

        self.expected_keys = sorted(self.keys_to_types_dict.keys())

        if can_pack_list is not None:
            self.can_pack_list = set(can_pack_list)
        else:
            self.can_pack_list = {"obs", "new_obs"}

        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(size)
        self._num_added = 0
        self._num_sampled = 0
        self._evicted_hit_stats = WindowStat("evicted_hit", 1000)
        self._est_size_bytes = 0

    def __len__(self):
        return len(self._storage)

    def add(self, **kwargs):
        assert len(kwargs) == len(self.expected_keys) and sorted(kwargs.keys()) == self.expected_keys

        for k in kwargs.keys():
            if k in self.can_pack_list:
                kwargs[k] = pack_if_needed(kwargs[k])

        data = [kwargs[k] for k in self.expected_keys]

        if len(self._storage) < self._maxsize:
            self._storage.append(data)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in data)
        else:
            idx = np.random.randint(0, self._num_added + 1)
            if idx < self._maxsize:
                self._storage[idx] = data

                self._evicted_hit_stats.push(self._hit_count[idx])
                self._hit_count[idx] = 0

        self._num_added += 1

    def _encode_sample(self, idxes):

        batch = {k: [] for k in self.expected_keys}

        for i in idxes:
            data = self._storage[i]
            for data_item, k in zip(data, self.expected_keys):

                if k in self.can_pack_list:
                    data_item = unpack_if_needed(data_item)

                data_item = self.keys_to_types_dict[k](data_item)

                batch[k].append(data_item)

            self._hit_count[i] += 1
        return batch

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        batch in dictionary form
        """
        idxes = [
            random.randint(0,
                           len(self._storage) - 1) for _ in range(batch_size)
        ]
        self._num_sampled += batch_size
        return self._encode_sample(idxes)

    def stats(self, debug=False):
        data = {
            "added_count": self._num_added,
            "sampled_count": self._num_sampled,
            "est_size_bytes": self._est_size_bytes,
            "num_entries": len(self._storage),
        }
        if debug:
            data.update(self._evicted_hit_stats.stats())
        return data

    def clear(self):
        self.__init__(size=self._maxsize,
                      keys_to_types_dict=self.keys_to_types_dict,
                      can_pack_list=self.can_pack_list)

    def get_single_epoch_batch_generator(self, batch_size):

        def gen():
            batch_idx = 0
            while batch_idx < int(len(self) / batch_size):
                sample = self.sample(batch_size)
                batch_idx += 1
                yield sample

        return gen
