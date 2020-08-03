import numpy as np
import sys

from ray.rllib.env.base_env import BaseEnv
from pomprl.rl.envs.stratego.stratego_multiagent_env import StrategoMultiAgentEnv

# Need raw numba functions
from pomprl.games.stratego.stratego_game_logic import _get_game_result_is_invalid

# Numba Methods
from pomprl.rl.envs.stratego.stratego_parallel_logic import _fill_observation, _step_multi_envs

from ray.tune.registry import register_env


STRATEGO_PARALLEL_ENV = "stratego_parallel_env"

class StrategoParallelEnv(BaseEnv):
    def __init__(self, env_config: dict = None, num_envs: int = 1):
        super().__init__()

        # Convenient to hold local instances of the environments.
        # Might become a problem if were initializing thousands later on
        self.envs = [StrategoMultiAgentEnv(env_config) for _ in range(num_envs)]
        self.base = self.envs[0]

        # Get the shape of the internal state
        example_state = self.base.random_initial_state_fn()
        self.state_shape = example_state.shape
        self.state_type = example_state.dtype

        self.num_envs = num_envs
        self.state_buffer = None
        self.player_buffer = None

        # Create observation buffers
        # These will store all of the current / last observations for each action
        # These will also allow us to parallelize the observation generation
        self.observation_buffers = {
            key: np.empty((2, num_envs, *space.shape), dtype=np.int64 if key == "valid_actions_mask" else np.float32)
            for key, space in self.base.observation_space.spaces.items()
        }

        self.reward_buffers = np.zeros(num_envs, dtype=np.float32)

        if env_config.get('heuristic_reward_matrix') is not None:
            self.add_heuristic_rewards = True
            self.next_player_rewards = np.zeros_like(self.reward_buffers)
            self.heuristic_rewards_matrix = np.float32(env_config['heuristic_reward_matrix'])
        else:
            self.add_heuristic_rewards = False
            self.heuristic_rewards_matrix = np.zeros(shape=(1, 1), dtype=np.float32)

        self.done_buffers = np.zeros(num_envs, dtype=np.bool)

        # Dummy buffer for missing observation buffers when were not using them
        self.dummy_buffer = np.empty((2, num_envs, 1, 1, 1), dtype=np.float32)
        self.dummy_single_buffer = np.empty((1, 1, 1), dtype=np.float32)

        self.__init_observation_dictionaries()

        self.numba_params = (self.base.base_env.action_size, self.base.base_env._mpapsp,
                             self.base.base_env.rows, self.base.base_env.columns,
                             self.base._p_obs_mids, self.base._p_obs_ranges,
                             self.base._f_obs_mids, self.base._f_obs_ranges,
                             "partial_observation" in self.observation_buffers,
                             "full_observation" in self.observation_buffers,
                             "internal_state" in self.observation_buffers,)

    def __init_observation_dictionaries(self):
        # This function is only necessary since rllib requires to observation dict be
        # env -> player -> observation, which is the worst possible index order....

        # This is the most efficient way to create the two observation dictionaries for each player.
        # We create views into the main observation buffer with the current dictionary structure

        # We also make two different versions of these dictionaries
        # An internal one which we can modify
        # and external ones which are immutable to prevent accidental modification outside of the class
        self._internal_player_one_observation = [
            {1: {obs_name: obs[0, env_idx, :]
                 for obs_name, obs in self.observation_buffers.items()}}
            for env_idx in range(self.num_envs)
        ]

        self._internal_player_two_observation = [
            {-1: {obs_name: obs[1, env_idx, :]
                  for obs_name, obs in self.observation_buffers.items()}}
            for env_idx in range(self.num_envs)
        ]

        self.player_one_observation = [
            {1: {obs_name: obs.view()[0, env_idx, :]
                 for obs_name, obs in self.observation_buffers.items()}}
            for env_idx in range(self.num_envs)
        ]

        self.player_two_observation = [
            {-1: {obs_name: obs.view()[1, env_idx, :]
                  for obs_name, obs in self.observation_buffers.items()}}
            for env_idx in range(self.num_envs)
        ]

        # We re-index the observation dictionary so we can easily access both observation in a single loop.
        self._internal_combined_observations = [
            {1: self._internal_player_one_observation[env_idx][1],
             -1: self._internal_player_two_observation[env_idx][-1]}
            for env_idx in range(self.num_envs)]

        self.combined_observations = [
            {1: self.player_one_observation[env_idx][1],
             -1: self.player_two_observation[env_idx][-1]}
            for env_idx in range(self.num_envs)]

        # Make immutable
        for d1 in self.combined_observations:
            for d2 in d1.values():
                for val in d2.values():
                    val.setflags(write=False)

        # Another re-index where we add another player index so we can return all of the observations easily
        self._internal_observation_dict = {1: self._internal_player_one_observation,
                                           -1: self._internal_player_two_observation}
        self.observation_dict = {1: dict(enumerate(self.player_one_observation)),
                                 -1: dict(enumerate(self.player_two_observation))}

    def initialize_states(self):
        # Create internal buffer to store all parallel environment states
        # We can then just call the stateless numba functions on views of this buffer
        self.state_buffer = np.empty((self.num_envs, *self.state_shape), self.state_type)
        self.player_buffer = np.empty(self.num_envs, np.int64)

        for env_id in range(self.num_envs):
            self.try_reset(env_id)

    def poll(self):
        if self.state_buffer is None:
            for env_idx in range(self.num_envs):
                self.try_reset(env_id=env_idx)
            # observations = {env_idx: {} for env_idx in range(self.num_envs)}
            # rewards = {env_idx: {} for env_idx in range(self.num_envs)}
            # dones = {env_idx: {"__all__": False} for env_idx in range(self.num_envs)}
            # info = {env_idx: {} for env_idx in range(self.num_envs)}
        observations = {}
        rewards = {}
        dones = {}
        info = {}

        for env_idx, (player, obs, reward, done, state) in enumerate(
                zip(self.player_buffer, self.combined_observations, self.reward_buffers, self.done_buffers,
                    self.state_buffer)):

            if not done:
                observations[env_idx] = {player: obs[player]}
                if self.add_heuristic_rewards:
                    rewards[env_idx] = {player: reward + self.next_player_rewards[env_idx]}
                    self.next_player_rewards[env_idx] = -reward
                else:
                    rewards[env_idx] = {player: reward}
                dones[env_idx] = {player: False, "__all__": False}
                info[env_idx] = {player: {}}
            else:
                if self.add_heuristic_rewards:
                    self.next_player_rewards[env_idx] = 0.0
                observations[env_idx] = {player: obs[player], -player: obs[-player]}
                rewards[env_idx] = {player: reward, -player: -reward}  # Zero sum game

                dones[env_idx] = {1: True, -1: True, "__all__": True}
                env_info = {1: {}, -1: {}}
                if _get_game_result_is_invalid(state):
                    env_info[1]['game_result_was_invalid'] = True
                    env_info[-1]['game_result_was_invalid'] = True
                    env_info[1]['game_result'] = 'tied'
                    env_info[-1]['game_result'] = 'tied'
                else:
                    env_info[1]['game_result_was_invalid'] = False
                    env_info[-1]['game_result_was_invalid'] = False

                    player_1_reward = rewards[env_idx][1]
                    if player_1_reward == 1:
                        env_info[1]['game_result'] = 'won'
                        env_info[-1]['game_result'] = 'lost'
                    elif player_1_reward == -1:
                        env_info[1]['game_result'] = 'lost'
                        env_info[-1]['game_result'] = 'won'
                    else:
                        env_info[1]['game_result'] = 'tied'
                        env_info[-1]['game_result'] = 'tied'
                info[env_idx] = env_info

        return observations, rewards, dones, info, {}

    def send_actions(self, action_dict):
        actions = np.zeros(self.num_envs, np.int64)
        for i, ac in action_dict.items():
            actions[i] = ac[self.player_buffer[i]]

        _step_multi_envs(self.player_buffer, actions, self.reward_buffers, self.done_buffers, *self.numba_params,
                         self.state_buffer,
                         self.observation_buffers['valid_actions_mask'],
                         self.observation_buffers.get('partial_observation', self.dummy_buffer),
                         self.observation_buffers.get('full_observation', self.dummy_buffer),
                         self.observation_buffers.get('internal_state', self.dummy_buffer),
                         self.add_heuristic_rewards, self.heuristic_rewards_matrix)

    def try_reset(self, env_id, force_state=None, force_player=None):
        if self.state_buffer is None:
            self.initialize_states()

            # return self.observation_dict[self.envs[env_id].player][env_id]

        env = self.envs[env_id]
        env.reset()

        if force_state is not None:
            assert force_player is not None
            env.player = force_player
            env.state = force_state.copy()

        player = env.player

        self.state_buffer[env_id, :] = env.state
        self.player_buffer[env_id] = player

        observation = self._internal_combined_observations[env_id][player]
        _fill_observation(player, *self.numba_params,
                          self.state_buffer[env_id],
                          observation['valid_actions_mask'],
                          observation.get('partial_observation', self.dummy_single_buffer),
                          observation.get('full_observation', self.dummy_single_buffer),
                          observation.get('internal_state', self.dummy_single_buffer))
        return self.observation_dict[player][env_id]


# if __name__ == '__main__':
#     env = StrategoParallelEnv(num_envs=1024)
#     old_env = BaseEnv.to_base_env(StrategoMultiAgentEnv(), lambda x: StrategoMultiAgentEnv(), num_envs=1024)
#
#     env_time = Timer("env.send_actions(a);"
#                      "env.poll()",
#                      setup="env.initialize_states();"
#                            "obs=env.poll();"
#                            "a = {key: {1: np.where(val[1]['valid_actions_mask'] == 1)[0][0]} for key, val in obs[0].items()}",
#                      globals={"env": env, "np": np})
#
#     old_env_time = Timer("env.send_actions(a);"
#                          "env.poll()",
#                          setup="env=BaseEnv.to_base_env(StrategoMultiAgentEnv(), lambda x: StrategoMultiAgentEnv(), num_envs=32);"
#                                "obs=env.poll();"
#                                "a = {key: {1: np.where(val[1]['valid_actions_mask'] == 1)[0][0]} for key, val in obs[0].items()}",
#                          globals={"np": np, "BaseEnv": BaseEnv, "StrategoMultiAgentEnv": StrategoMultiAgentEnv})


def make_stratego_parallel_env(env_config):
    return StrategoParallelEnv(num_envs=env_config["num_envs"], env_config=env_config)


register_env(STRATEGO_PARALLEL_ENV, make_stratego_parallel_env)


if __name__ == '__main__':

    def objects_are_the_same(new, old):
        if type(old) == dict:
            for k, v in old.items():
                if not k in new:
                    return False, f"key \'{k}\' is in the old dict but not in the new dict"
                vals_are_same, msg = objects_are_the_same(new=new[k], old=old[k])
                if not vals_are_same:
                    return False, f"under the dict key \'{k}\',\n{msg}"
        elif type(old) == list or type(old) == np.ndarray:
            if not np.array_equal(new, old):
                if not np.array_equal(np.shape(new), np.shape(old)):
                    return False, f"arrays arent equal, and new is a different shape,\nnew:\n{new}\n\nold:\n{old}\n"
                elif np.array_equal(new, np.zeros_like(new)):
                    return False, f"arrays arent equal, and new is all zeros,\nnew:\n{new}\n\nold:\n{old}\n"
                else:
                    return False, f"arrays arent equal,\nnew:\n{new}\n\nold:\n{old}\n"
        elif type(old) == str:
            if not old == new:
                return False, f"old string: \'{old}\' isnt equal to new string: \'{new}\'"
        elif type(old) == bool:
            if not old == new:
                return False, f"old boolean: \'{old}\' isnt equal to new boolean: \'{new}\'"
        return True, "objects are equal"


    def assert_env_outputs_are_same(new_env_output, old_env_output):
        old_obs, old_rew, old_dones, old_info, old_off_pol_actions = old_env_output
        new_obs, new_rew, new_dones, new_info, old_off_pol_actions = new_env_output

        obs_are_same, msg = objects_are_the_same(new=new_obs, old=old_obs)
        if not obs_are_same:
            assert False, f"In the observations,\n{msg}"

        rew_are_same, msg = objects_are_the_same(new=new_rew, old=old_rew)
        if not rew_are_same:
            assert False, "In the rewards,\n{msg}"

        dones_are_same, msg = objects_are_the_same(new=new_dones, old=old_dones)
        if not dones_are_same:
            assert False, f"In the dones,\n{msg}"

        info_are_same, msg = objects_are_the_same(new=new_info, old=old_info)
        if not info_are_same:
            assert False, f"In the infos,\n{msg}"


    print("\n\nstarted")

    num_envs = 5

    env = StrategoParallelEnv(num_envs=num_envs)
    old_env = BaseEnv.to_base_env(StrategoMultiAgentEnv(), lambda x: StrategoMultiAgentEnv(), num_envs=num_envs)
    old_env.poll()
    env.poll()
    print("envs have been made")

    for env_idx in range(num_envs):
        print(f"ENV INDEX:{env_idx}")
        old_env.try_reset(env_idx)
        print("a")
        old_state = old_env.envs[env_idx].state
        old_player = old_env.envs[env_idx].player
        print("b")
        env.try_reset(env_idx, force_state=old_state, force_player=old_player)
        print("c")

    print("envs have been reset and synced")

    for env_idx in range(num_envs):
        old_state = old_env.envs[env_idx].state
        assert np.array_equal(old_state, env.state_buffer[env_idx, :])
        assert np.array_equal(old_state, env.envs[env_idx].state)
        assert old_env.envs[env_idx].player == env.envs[env_idx].player

    def policy_2_fn(observation, policy_state=None):
        valid_actions_mask = observation['valid_actions_mask']  # 1 for valid, 0 for invalid
        action_probs = valid_actions_mask / np.sum(valid_actions_mask)
        action_index = np.random.choice(a=list(range(len(valid_actions_mask))), p=action_probs)

        policy_state_out = None
        return action_index, policy_state_out

    print("env assertions done")
    has_ever_reset = False
    while True:
        print("polling old env...")
        old_out = old_env.poll()
        print("polling new env...")
        new_out = env.poll()

        print("polling done")

        print(f"type is {type(old_out)}, length is {len(old_out)}")

        assert_env_outputs_are_same(new_env_output=new_out, old_env_output=old_out)
        old_obs, _, old_dones, _, _ = old_out
        _, _, new_dones, _, _ = new_out

        for env_idx in range(num_envs):
            print(f"ENV INDEX:{env_idx}")
            _, old_rew, old_dones, old_info, old_off_pol_actions = old_out
            if old_dones[env_idx]["__all__"]:
                print("\n\n\nRESET\n\n\n")
                has_ever_reset = True
                old_obs[env_idx] = old_env.try_reset(env_idx)
                print("a")
                old_state = old_env.envs[env_idx].state
                old_player = old_env.envs[env_idx].player

                print("b")
                env.try_reset(env_idx, force_state=old_state, force_player=old_player)
                print("c")

        actions = {}
        for env_idx, e in enumerate(old_env.envs):
            env_old_obs = old_obs[env_idx]

            action, _ = policy_2_fn(env_old_obs[e.player])

            actions[env_idx] = {
                e.player: action
            }

        print(f"has ever reset {has_ever_reset}")
        print(f"actions:{actions}")
        print(f"old dones:{old_dones}")
        print(f"new dones:{new_dones}")

        old_env.send_actions(actions)

        print(env)
        env.send_actions(actions)

    # print("new env:")
    # print(env.poll())
    # print("\n\nold env")
    # print()
    # env_time = Timer("env.send_actions(a);"
    #                  "env.poll()",
    #                  setup="env.initialize_states();"
    #                        "obs=env.poll();"
    #                        "a = {key: {1: np.where(val[1]['valid_actions_mask'] == 1)[0][0]} for key, val in obs[0].items()}",
    #                  globals={"env": env, "np": np})
    #
    # old_env_time = Timer("env.send_actions(a);"
    #                      "env.poll()",
    #                      setup="env=BaseEnv.to_base_env(StrategoMultiAgentEnv(), lambda x: StrategoMultiAgentEnv(), num_envs=32);"
    #                            "obs=env.poll();"
    #                            "a = {key: {1: np.where(val[1]['valid_actions_mask'] == 1)[0][0]} for key, val in obs[0].items()}",
    #                      globals={"np": np, "BaseEnv": BaseEnv, "StrategoMultiAgentEnv": StrategoMultiAgentEnv})

    print("\nDONE.\n")

# if __name__ == '__main__':
#
#     def policy_2_fn(observation, policy_state=None):
#         valid_actions_mask = observation['valid_actions_mask']  # 1 for valid, 0 for invalid
#         action_probs = valid_actions_mask / np.sum(valid_actions_mask)
#         action_index = np.random.choice(a=list(range(len(valid_actions_mask))), p=action_probs)
#
#         policy_state_out = None
#         return action_index, policy_state_out
#
#     env = StrategoMultiAgentEnv()
#
#     obs = env.reset()
#     games = 0
#     while True:
#         print(env.base_env.print_fully_observable_board_to_console(state=env.state))
#         valid_actions = obs[env.player]['valid_actions_mask']
#
#         action, _ = policy_2_fn(observation=obs[env.player])
#         obs, _, dones, _ = env.step(action_dict={env.player: action})
#         if dones['__all__']:
#             obs = env.reset()
#             games += 1
#         print("step")
#         print(f"games: {games}")

