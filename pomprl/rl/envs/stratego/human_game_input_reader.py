from ray.rllib.offline.input_reader import InputReader
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

from pomprl.games.stratego.config import STANDARD_STRATEGO_CONFIG

from pomprl.games.stratego.stratego_procedural_env import StrategoProceduralEnv
from pomprl.games.stratego.util import create_game_from_data
from pomprl.rl.envs.stratego.stratego_multiagent_env import PARTIALLY_OBSERVABLE, FULLY_OBSERVABLE, PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS, FULLY_OBSERVABLE_OBS_NUM_LAYERS, \
    _get_fully_observable_max_and_min_vals, _get_partially_observable_max_and_min_vals
from pomprl.rl.common.stratego_preprocessor import StrategoDictFlatteningPreprocessor
import threading
from queue import Queue
import ray
from ray.rllib.utils.memory import ray_get_and_free
import random

import numpy as np


def player_index(player: int):
    # player 1 returns 0
    # player -1 returns 1
    return (player - 1) // -2


def partitions(lst, n):
    return [lst[i::n] for i in range(n)]

class QueueRunner(threading.Thread):

    def __init__(self, queue, sample_generator_fn, shuffle_buffer_size, unproccessed_data):

        assert len(unproccessed_data) > 0

        threading.Thread.__init__(self)
        self.daemon = True
        self.queue: Queue = queue
        self._sample_generator_fn = sample_generator_fn
        self._sample_generator = self._sample_generator_fn(unproccessed_data)
        self._shuffle_buffer = []
        self._shuffle_buffer_size = shuffle_buffer_size
        self._unprocessed_data = unproccessed_data

    def enqueue(self):
        try:
            new_sample = next(self._sample_generator)
        except StopIteration:
            self._sample_generator = self._sample_generator_fn(self._unprocessed_data)
            new_sample = next(self._sample_generator)

        if len(self._shuffle_buffer) < self._shuffle_buffer_size:
            self._shuffle_buffer.append(new_sample)
            idx = random.randint(0, len(self._shuffle_buffer) - 1)
            enqueue_sample = self._shuffle_buffer[idx]
        else:
            idx = random.randint(0, len(self._shuffle_buffer)-1)
            enqueue_sample = self._shuffle_buffer[idx]
            self._shuffle_buffer[idx] = new_sample

        self.queue.put(item=enqueue_sample, block=True, timeout=None)

    def run(self):
        while True:
            self.enqueue()



@ray.remote(num_cpus=1)
class DataProcessor:

    def __init__(self, unprocessed_human_data, batch_size, obs_space, observation_mode, train_policy_id, shuffle_buffer_size):
        self._game_counter = 0
        self._human_data = unprocessed_human_data
        self._batch_size = batch_size
        self._preprocessor = StrategoDictFlatteningPreprocessor(obs_space)
        self._observation_mode = observation_mode
        self._train_policy_id = train_policy_id

        piece_amounts = STANDARD_STRATEGO_CONFIG['piece_amounts']

        if self._observation_mode == PARTIALLY_OBSERVABLE:
            self._obs_highs, self._obs_lows = _get_partially_observable_max_and_min_vals(piece_amounts=piece_amounts)
            self._obs_num_layers = PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS
        elif self._observation_mode == FULLY_OBSERVABLE:
            self._obs_highs, self._obs_lows = _get_fully_observable_max_and_min_vals()
            self._obs_num_layers = FULLY_OBSERVABLE_OBS_NUM_LAYERS
        else:
            raise ValueError('Unknown observation mode passed to {}: {}'.format(self.__name__, self.observation_mode))

            # Used to normalize observations from self.base_env to [-1, 1]
        self._obs_ranges = np.reshape((self._obs_highs - self._obs_lows) / np.float32(2.0),
                                      newshape=(1, 1, self._obs_num_layers))
        self._obs_mids = np.reshape((self._obs_highs + self._obs_lows) / np.float32(2.0),
                                    newshape=(1, 1, self._obs_num_layers))



        def sample_generator(human_data):
            env = StrategoProceduralEnv(rows=STANDARD_STRATEGO_CONFIG['rows'],
                                        columns=STANDARD_STRATEGO_CONFIG['columns'])

            illegal_moves_count = 0

            for player1_string, player2_string, winner, moves in human_data:
                state = create_game_from_data(player1_string=player1_string,
                                              player2_string=player2_string,
                                              game_version_config=STANDARD_STRATEGO_CONFIG)

                player_trajectories = [[], []]

                assert winner == 0 or winner == 1

                player1_reward = 1 if winner == 0 else -1
                player2_reward = 1 if winner == 1 else -1

                player_rewards = [player1_reward, player2_reward]

                counter = 0
                for start, end in moves:

                    start_col = ord(start[0]) - ord('A')
                    start_row = ord(start[1]) - ord('1')

                    end_col = ord(end[0]) - ord('A')
                    end_row = ord(end[1]) - ord('1')

                    # (JB) For some reason the letter 'J' is skipped in column numbering
                    if ord(start[0]) > ord('J'):
                        start_col -= 1
                    if ord(end[0]) > ord('J'):
                        end_col -= 1

                    # print("---")
                    # print(start)
                    # print(end)
                    # print("-")
                    #
                    # print("start col, row", start_col, start_row)
                    # print("end col, row", end_col, end_row)
                    # print("fully observable")
                    # env.print_fully_observable_board_to_console(state)

                    if (counter % 2 == 0):
                        player = 1
                    else:
                        player = -1

                    action_index = env.get_action_1d_index_from_positions(start_row, start_col, end_row, end_col)

                    try:
                        new_state, new_player = env.get_next_state(state=state, player=player,
                                                                   action_index=action_index,
                                                                   allow_piece_oscillation=True)
                    except ValueError as e:
                        print(e)
                        illegal_moves_count += 1
                        print("illegal moves count:", illegal_moves_count)
                        break

                    player_perspective_state = env.get_state_from_player_perspective(state=state, player=player)
                    player_perspective_valid_moves = env.get_valid_moves_as_1d_mask(state=player_perspective_state,
                                                                                    player=1)
                    player_perspective_action_index = env.get_action_1d_index_from_player_perspective(
                        action_index=action_index,
                        player=player)

                    player_perspective_action_probs = np.zeros_like(player_perspective_valid_moves, dtype=np.float32)
                    player_perspective_action_probs[player_perspective_action_index] = 1.0

                    player_perspective_reward = player_rewards[player_index(player)]

                    if self._observation_mode == FULLY_OBSERVABLE:
                        player_perspective_observation = env.get_fully_observable_observation(state=state,
                                                                                              player=player)
                    elif self._observation_mode == PARTIALLY_OBSERVABLE:
                        player_perspective_observation = env.get_partially_observable_observation(state=state,
                                                                                                  player=player)
                    else:
                        assert False

                    player_perspective_observation = self._normalize_observation(player_perspective_observation)

                    # print("obs shape:", player_perspective_observation[:, :, 0])
                    # print("obs shape:", player_perspective_observation[:, :, 0])
                    # print("obs shape:", np.shape(player_perspective_observation))
                    # env.print_fully_observable_board_to_console(player_perspective_state)
                    # print(player_perspective_reward)

                    player_trajectories[player_index(player)].append({
                        # 'state': player_perspective_state,
                        # 'action_index': np.float32(player_perspective_action_index),
                        'policy_targets': np.asarray(player_perspective_action_probs, dtype=np.float32),
                        'value_targets': np.float32(player_perspective_reward),
                        # 'full_observation': player_perspective_full_observation,

                        'obs': self._preprocessor.transform({
                            'observation': np.asarray(player_perspective_observation, dtype=np.float32),
                            'valid_actions_mask': np.asarray(player_perspective_valid_moves, dtype=np.float32),
                        })
                        # 'terminal': False
                    })

                    # obs, rewards, dones, infos = env.step(player_action)
                    # print("STEPPED **********")

                    counter += 1

                    if env.get_game_ended(state=new_state, player=1):
                        # print("DONE WITH GAME")
                        break

                    state = new_state
                for trajectory in player_trajectories:
                    if len(trajectory) > 0:
                        if 'terminal' in trajectory[-1]:
                            trajectory[-1]['terminal'] = True
                        for sample in trajectory:
                            yield sample

        self._sample_queue = Queue(maxsize=10000)
        self._queue_runnner = QueueRunner(queue=self._sample_queue, sample_generator_fn=sample_generator, shuffle_buffer_size=shuffle_buffer_size, unproccessed_data=unprocessed_human_data)
        self._queue_runnner.start()

    def _normalize_observation(self, obs):
        return (obs - self._obs_mids) / self._obs_ranges

    def next(self):
        samples = []

        for _ in range(self._batch_size):
            samples.append(self._sample_queue.get(block=True))

        keys = samples[0].keys()

        batch_dict = {key: [sample[key] for sample in samples] for key in keys}

        return MultiAgentBatch(policy_batches={
            self._train_policy_id: SampleBatch(batch_dict)
        }, count=self._batch_size)


class StrategoHumanGameInputReader(InputReader):

    def __init__(self, unprocessed_human_data, batch_size, obs_space, observation_mode, train_policy_id, shuffle_buffer_size, data_processes):
        self.data_processors = []

        for i, human_data_partitition in zip(range(data_processes), partitions(lst=unprocessed_human_data, n=data_processes)):
            self.data_processors.append(DataProcessor.remote(human_data_partitition, batch_size//data_processes, obs_space, observation_mode, train_policy_id, shuffle_buffer_size))
        print("init done")

    def next(self):
        """Return the next batch of experiences read.

        Returns:
            SampleBatch or MultiAgentBatch read.
        """
        batches = []
        for dp in self.data_processors:
            batches.append(ray_get_and_free(dp.next.remote()))
        batch = MultiAgentBatch.concat_samples(samples=batches)

        return batch
