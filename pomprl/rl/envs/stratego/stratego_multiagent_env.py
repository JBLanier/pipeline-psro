from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from gym.spaces import Discrete, Box, Dict
import copy
import numpy as np

from pomprl.games.stratego.stratego_procedural_env import StrategoProceduralEnv
from pomprl.games.stratego.stratego_game_logic import RecentMoves, SP, FULLY_OBSERVABLE_OBS_NUM_LAYERS, \
    PartiallyObservableObsLayers, FullyObservableObsLayers, PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS

from pomprl.games.stratego.config import STANDARD_STRATEGO_CONFIG, BARRAGE_STRATEGO_CONFIG, MEDIUM_STRATEGO_CONFIG, \
    TINY_STRATEGO_CONFIG, MICRO_STRATEGO_CONFIG, OCTA_BARRAGE_STRATEGO_CONFIG, STANDARD_STRATEGO_CONFIG2

from pomprl.games.stratego.util import get_random_initial_state_fn, get_random_human_init_fn
from pomprl.rl.envs.stratego import socket_pickle
import socket

from pomprl.util import with_base_config
# env rllib registered name
STRATEGO_ENV = 'Stratego-v1'

# observation modes
PARTIALLY_OBSERVABLE = 'partially_observable'
FULLY_OBSERVABLE = 'fully_observable'
BOTH_OBSERVATIONS = 'both_observations'

# keys in each observation returned by this environment
PARTIAL_OBSERVATION = 'partial_observation'
FULL_OBSERVATION = 'full_observation'
# OBSERVATION = 'observation'
VALID_ACTIONS_MASK = 'valid_actions_mask'
INTERNAL_STATE = 'internal_state'

# game versions
STANDARD = 'standard'
STANDARD2 = 'standard2'
BARRAGE = 'barrage'
OCTA_BARRAGE = 'octa_barrage'
MEDIUM = 'medium'
TINY = 'tiny'
MICRO = 'micro'

VERSION_CONFIGS = {
    STANDARD: STANDARD_STRATEGO_CONFIG,
    STANDARD2: STANDARD_STRATEGO_CONFIG2,
    BARRAGE: BARRAGE_STRATEGO_CONFIG,
    OCTA_BARRAGE: OCTA_BARRAGE_STRATEGO_CONFIG,
    MEDIUM: MEDIUM_STRATEGO_CONFIG,
    TINY: TINY_STRATEGO_CONFIG,
    MICRO: MICRO_STRATEGO_CONFIG
}

DEFAULT_CONFIG = {
    'version': STANDARD,
    'repeat_games_from_other_side': False, # if True, Each game will be played twice, swapping sides players start on.
    # The environment retains state between resets to do this.^

    'observation_mode': BOTH_OBSERVATIONS,  # options are 'partially_observable' or 'fully_observable'
    'observation_includes_internal_state': False,  # Tree search methods require this to be True

    'vs_bot': False,
    'bot_player_num': 1,
    'bot_relative_path': 'basic_python.py',

    'vs_human': False,  # one of the players is a human using a web gui
    'human_player_num': -1,  # 1 or -1
    'human_web_gui_port': 7000,
    'human_inits': False
}


def _get_fully_observable_max_and_min_vals(piece_amounts):
    max_piece_val = SP.BOMB.value
    min_piece_val = SP.NOPIECE.value

    max_po_piece_val = SP.UNKNOWN.value
    min_po_piece_val = SP.NOPIECE.value

    p1_obs_cap_start = FullyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
    p1_obs_cap_end = FullyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
    p2_obs_cap_start = FullyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
    p2_obs_cap_end = FullyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

    # [0] owned_pieces,
    # [1] enemy_pieces,
    # [2] obstacle_map,
    # [3] self_recent_moves,
    # [4] enemy_recent_moves,
    # [5] owned_po_pieces,
    # [6] enemy_po_pieces,
    # [7:19] owned_captured_pieces
    # [19:31] enemy_captured_pieces

    max_vals = np.asarray([
        max_piece_val,
        max_piece_val,
        2,
        RecentMoves.JUST_CAME_FROM.value,
        RecentMoves.JUST_CAME_FROM.value,
        max_po_piece_val,
        max_po_piece_val,
        *[2 for _ in range(p1_obs_cap_start, p1_obs_cap_end)],
        *[2 for _ in range(p2_obs_cap_start, p2_obs_cap_end)],
        2,
        2],
                  dtype=np.float32)

    for piece_type, piece_amount in piece_amounts.items():
        if piece_amount > 1:
            offset = piece_type.value - 1
            for start_layer in [p1_obs_cap_start, p2_obs_cap_start]:
                max_vals[start_layer + offset] = piece_amount

    min_vals = np.asarray([
        min_piece_val,
        min_piece_val,
        0,
        RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value,
        RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value,
        min_po_piece_val,
        min_po_piece_val,
        *[0 for _ in range(p1_obs_cap_start, p1_obs_cap_end)],
        *[0 for _ in range(p2_obs_cap_start, p2_obs_cap_end)],
        0,
        0],
                  dtype=np.float32)

    return max_vals, min_vals


def _get_partially_observable_max_and_min_vals(piece_amounts):
    max_piece_val = SP.BOMB.value
    min_piece_val = SP.NOPIECE.value

    max_po_piece_val = SP.UNKNOWN.value
    min_po_piece_val = SP.NOPIECE.value

    p1_obs_cap_start = PartiallyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_START.value
    p1_obs_cap_end = PartiallyObservableObsLayers.PLAYER_1_CAPTURED_PIECE_RANGE_END.value
    p2_obs_cap_start = PartiallyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_START.value
    p2_obs_cap_end = PartiallyObservableObsLayers.PLAYER_2_CAPTURED_PIECE_RANGE_END.value

    # [0] owned_pieces,
    # [1] owned_po_pieces,
    # [2] enemy_po_pieces,
    # [3] obstacle_map,
    # [4] self_recent_moves,
    # [5] enemy_recent_moves
    # [6:18] owned_captured_pieces
    # [18:30] enemy_captured_pieces

    max_vals = np.asarray([
                max_piece_val,
                max_po_piece_val,
                max_po_piece_val,
                2,
                RecentMoves.JUST_CAME_FROM.value,
                RecentMoves.JUST_CAME_FROM.value,
                *[2 for _ in range(p1_obs_cap_start, p1_obs_cap_end)],
                *[2 for _ in range(p2_obs_cap_start, p2_obs_cap_end)],
                2,
                2],
                          dtype=np.float32)

    for piece_type, piece_amount in piece_amounts.items():
        if piece_amount > 1:
            offset = piece_type.value - 1
            for start_layer in [p1_obs_cap_start, p2_obs_cap_start]:
                max_vals[start_layer + offset] = piece_amount

    min_vals = np.asarray([
                min_piece_val,
                min_po_piece_val,
                min_po_piece_val,
                0,
                RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value,
                RecentMoves.JUST_ARRIVED_AND_CANT_DOUBLE_BACK.value,
                *[0 for _ in range(p1_obs_cap_start, p1_obs_cap_end)],
                *[0 for _ in range(p2_obs_cap_start, p2_obs_cap_end)],
                0,
                0],
                          dtype=np.float32)

    return max_vals, min_vals


class StrategoMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):

        env_config = with_base_config(base_config=DEFAULT_CONFIG, extra_config=env_config if env_config else {})

        # combine env_config with the config for the specific stratego version selected
        env_config = with_base_config(base_config=VERSION_CONFIGS[env_config['version']], extra_config=env_config)

        rows = env_config['rows']
        columns = env_config['columns']
        piece_amounts = env_config['piece_amounts']

        self.base_env: StrategoProceduralEnv = StrategoProceduralEnv(rows=rows, columns=columns)

        if env_config['human_inits']:

            self.random_initial_state_fn = get_random_human_init_fn(game_version=env_config['version'],
                                                                    game_version_config=env_config,
                                                                    procedural_env=self.base_env)

        else:
            self.random_initial_state_fn = get_random_initial_state_fn(base_env=self.base_env,
                                                                       game_version_config=VERSION_CONFIGS[env_config['version']])

        self.repeat_games_from_other_side = env_config['repeat_games_from_other_side']
        self.episodes_completed = 0
        self.last_initial_state = None
        self.action_space = Discrete(self.base_env.action_size)

        self.observation_mode = env_config['observation_mode']

        self.observation_includes_internal_state = env_config['observation_includes_internal_state']

        # if self.observation_mode == PARTIALLY_OBSERVABLE or self.observation_mode == BOTH_OBSERVATIONS:
        self._p_obs_highs, self._p_obs_lows = _get_partially_observable_max_and_min_vals(piece_amounts=piece_amounts)
        self._p_obs_num_layers = PARTIALLY_OBSERVABLE_OBS_NUM_LAYERS
        # if self.observation_mode == FULLY_OBSERVABLE or self.observation_mode == BOTH_OBSERVATIONS:
        self._f_obs_highs, self._f_obs_lows = _get_fully_observable_max_and_min_vals(piece_amounts=piece_amounts)
        self._f_obs_num_layers = FULLY_OBSERVABLE_OBS_NUM_LAYERS

        assert self.observation_mode in [PARTIALLY_OBSERVABLE, FULLY_OBSERVABLE, BOTH_OBSERVATIONS]

        # Used to normalize observations from self.base_env to [-1, 1]
        self._p_obs_ranges = np.reshape((self._p_obs_highs - self._p_obs_lows) / np.float32(2.0), newshape=(1, 1, self._p_obs_num_layers))
        self._p_obs_mids = np.reshape((self._p_obs_highs + self._p_obs_lows) / np.float32(2.0), newshape=(1, 1, self._p_obs_num_layers))

        self._f_obs_ranges = np.reshape((self._f_obs_highs - self._f_obs_lows) / np.float32(2.0), newshape=(1, 1, self._f_obs_num_layers))
        self._f_obs_mids = np.reshape((self._f_obs_highs + self._f_obs_lows) / np.float32(2.0),newshape=(1, 1, self._f_obs_num_layers))

        self.observation_space = {
            VALID_ACTIONS_MASK: Box(
                low=0,
                high=1,
                shape=(self.base_env.action_size,)
            )}

        if self.observation_mode == PARTIALLY_OBSERVABLE or self.observation_mode == BOTH_OBSERVATIONS:
            self.observation_space[PARTIAL_OBSERVATION] = Box(
                low=-1.0,
                high=1.0,
                shape=(rows, columns, self._p_obs_num_layers))

        if self.observation_mode == FULLY_OBSERVABLE or self.observation_mode == BOTH_OBSERVATIONS:
            self.observation_space[FULL_OBSERVATION] = Box(
                low=-1.0,
                high=1.0,
                shape=(rows, columns, self._f_obs_num_layers))

        if self.observation_includes_internal_state:
            sample_state = self.random_initial_state_fn()
            self.observation_space[INTERNAL_STATE] = Box(
                low=-np.inf,
                high=np.inf,
                shape=sample_state.shape
            )

        self.observation_space = Dict(self.observation_space)

        self.vs_human = env_config['vs_human']
        self.human_web_gui_port = env_config['human_web_gui_port']
        self.human_player_num = env_config['human_player_num']
        if self.vs_human:
            from pomprl.games.stratego.stratego_human_server import StrategoHumanGUIServer
            self.gui_server = StrategoHumanGUIServer(base_env=self.base_env, port=self.human_web_gui_port)

        self.vs_bot = env_config['vs_bot']
        self.bot_player_num = env_config['bot_player_num']
        self.bot_relative_path = env_config['bot_relative_path']
        if self.vs_bot:
            self.bot_controller_socket = None
        if self.vs_bot and self.vs_human:
            assert self.human_player_num != self.bot_player_num

    def _get_current_obs(self, player=None):

        if player is None:
            player = self.player

        player_perspective_state = self.base_env.get_state_from_player_perspective(self.state, player)

        valid_actions_mask = self.base_env.get_valid_moves_as_1d_mask(state=player_perspective_state, player=1)

        return_dict = {VALID_ACTIONS_MASK: valid_actions_mask}

        if self.observation_mode == PARTIALLY_OBSERVABLE or self.observation_mode == BOTH_OBSERVATIONS:
            p_obs = self.base_env.get_partially_observable_observation(
                state=player_perspective_state,
                player=1)

            # normalize observation to be in the range [-1, 1]
            # each obs layer is normalized by different values
            p_obs = self.normalize_p_observation(p_obs)
            if np.isnan(p_obs).any():
                assert False, "There's been a mistake. An observation was generated with a NAN value."

            return_dict[PARTIAL_OBSERVATION] = p_obs

        if self.observation_mode == FULLY_OBSERVABLE or self.observation_mode == BOTH_OBSERVATIONS:

            f_obs = self.base_env.get_fully_observable_observation(
                state=player_perspective_state,
                player=1)

            f_obs = self.normalize_f_observation(f_obs)
            if np.isnan(f_obs).any():
                assert False, "There's been a mistake. An observation was generated with a NAN value."

            return_dict[FULL_OBSERVATION] = f_obs

        if self.observation_includes_internal_state:
            return_dict[INTERNAL_STATE] = player_perspective_state

        return return_dict

    def normalize_f_observation(self, f_obs):

        return (f_obs - self._f_obs_mids) / self._f_obs_ranges

    def denormalize_f_observation(self, f_obs: np.ndarray):
        return (f_obs * self._f_obs_ranges) + self._f_obs_mids

    def normalize_p_observation(self, p_obs):

        return (p_obs - self._p_obs_mids) / self._p_obs_ranges

    def denormalize_p_observation(self, p_obs: np.ndarray):
        return (p_obs * self._p_obs_ranges) + self._p_obs_mids

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """

        if self.repeat_games_from_other_side and self.episodes_completed % 2 == 1:
            assert not self.vs_bot
            initial_state = self.base_env.get_state_from_player_perspective(state=self.last_initial_state, player=-1)
            self.player = -1
        else:
            initial_state = self.random_initial_state_fn()
            self.player = 1

        self.last_initial_state = initial_state
        self.state = initial_state

        self.episodes_completed += 1

        if self.vs_human:
            self.gui_server.reset_game(initial_state=initial_state)


        if self.vs_bot:
            if self.bot_controller_socket is not None:
                # self.bot_controller_socket.shutdown(socket.SHUT_RDWR)
                self.bot_controller_socket.close()

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 9935))
            s.listen(5)
            self.launcher_socket = socket_pickle.get_connection("localhost", 9934)
            socket_pickle.pickle_send(self.launcher_socket, data=("new_game", (initial_state, self.bot_player_num, self.bot_relative_path)))
            self.bot_controller_socket, bot_controller_address = s.accept()
            # print("\nbot controller socket connected")

            s.shutdown(socket.SHUT_RDWR)
            s.close()

        obs = {self.player: self._get_current_obs()}

        while (self.vs_human and self.player == self.human_player_num) or (self.vs_bot and self.player == self.bot_player_num):
            dones = None
            if self.vs_human and self.player == self.human_player_num:
                action_positions = self.gui_server.get_action_by_position(state=self.state, player=self.player)
                action = self.base_env.get_action_1d_index_from_positions(*action_positions)
                obs, _, dones, _ = self.step(action_dict={self.human_player_num: action}, check_for_bot_move=False)

                if self.vs_bot:
                    # print("sending human move to bot...")
                    action_positions_to_bot = self.base_env.get_action_positions_from_player_perspective(self.human_player_num,
                                                                                          *action_positions)
                    socket_pickle.pickle_send(self.bot_controller_socket, data=("new_move", action_positions_to_bot))

            if self.vs_bot and self.player == self.bot_player_num:
                cmd, bot_move = socket_pickle.pickle_recv(self.bot_controller_socket)
                assert cmd == 'bot made move'
                # print(f"Reset(): Got bot move: {bot_move}")
                bot_move = self.base_env.get_action_positions_from_player_perspective(self.bot_player_num, *bot_move)
                assert self.base_env.is_move_valid_by_position(self.state, self.player, *bot_move)
                bot_move_action_index = self.base_env.get_action_1d_index_from_positions(*bot_move)
                obs, _, dones, _ = self.step(action_dict={self.bot_player_num: bot_move_action_index},
                                             check_for_human_move=False)
            if dones is not None and dones["__all__"]:
                assert False, "Game finished on its own"

        return obs


    def step(self, action_dict, check_for_human_move=True, check_for_bot_move=True, allow_piece_oscillation=False):
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """

        # action should only be for current player
        assert self.player in action_dict
        assert self.player * -1 not in action_dict

        action = action_dict[self.player]
        action = self.base_env.get_action_1d_index_from_player_perspective(action_index=action, player=self.player)

        self.state, self.player = self.base_env.get_next_state(self.state, self.player, action,
                                                               allow_piece_oscillation=allow_piece_oscillation)

        # if self.vs_bot:
        #     debug_action_pos = self.base_env.get_action_positions_from_1d_index(action)
        #     print(f"action by player {self.player*-1}: {debug_action_pos}")
        #     self.base_env.print_fully_observable_board_to_console(state=self.state)

        reward = self.base_env.get_game_ended(self.state, self.player)

        if reward == 0:

            if self.vs_human and check_for_human_move and self.player == self.human_player_num:
                action_positions = self.gui_server.get_action_by_position(state=self.state, player=self.player)
                action = self.base_env.get_action_1d_index_from_positions(*action_positions)
                return self.step(action_dict={self.human_player_num: action})

            if self.vs_bot and check_for_bot_move and self.player == self.bot_player_num:
                action_positions_to_bot = self.base_env.get_action_positions_from_1d_index(action_index=action)
                socket_pickle.pickle_send(self.bot_controller_socket, data=("new_move", action_positions_to_bot))

                try:
                    cmd, bot_move = socket_pickle.pickle_recv(self.bot_controller_socket)
                except ValueError:
                    # TODO HACK, fix this, assumes the bot won  ####################
                    dones = {1: True, -1: True, "__all__": True}
                    obs = {1: self._get_current_obs(player=1), -1: self._get_current_obs(player=-1)}

                    infos = {1: {}, -1: {}}

                    infos[1]['game_result_was_invalid'] = False
                    infos[-1]['game_result_was_invalid'] = False

                    player_1_reward = 1 if self.bot_player_num == 1 else -1
                    player_2_reward = 1 if self.bot_player_num == -1 else -1

                    if player_1_reward == 1:
                        infos[1]['game_result'] = 'won'
                        infos[-1]['game_result'] = 'lost'
                    elif player_1_reward == -1:
                        infos[1]['game_result'] = 'lost'
                        infos[-1]['game_result'] = 'won'
                    else:
                        infos[1]['game_result'] = 'tied'
                        infos[-1]['game_result'] = 'tied'

                    rewards = {1: player_1_reward,
                               -1: player_2_reward}

                    return obs, rewards, dones, infos

                    #####################################################


                assert cmd == 'bot made move'
                # print(f"Step(): Got bot move: {bot_move}")
                assert self.base_env.is_move_valid_by_position(self.state, self.player, *bot_move, allow_piece_oscillation=True)
                bot_move = self.base_env.get_action_positions_from_player_perspective(self.bot_player_num, *bot_move)
                bot_move_action_index = self.base_env.get_action_1d_index_from_positions(*bot_move)
                return self.step(action_dict={self.bot_player_num: bot_move_action_index}, allow_piece_oscillation=True)

            dones = {self.player: False, "__all__": False}
            obs = {self.player: self._get_current_obs(player=self.player)}
            rewards = {self.player: 0}
            infos = {}
        else:
            dones = {1: True, -1: True, "__all__": True}
            obs = {1: self._get_current_obs(player=1), -1: self._get_current_obs(player=-1)}

            infos = {1: {}, -1: {}}

            if self.base_env.get_game_result_is_invalid(self.state):
                rewards = {1: 0, -1: 0}
                infos[1]['game_result_was_invalid'] = True
                infos[-1]['game_result_was_invalid'] = True
                infos[1]['game_result'] = 'tied'
                infos[-1]['game_result'] = 'tied'
            else:
                infos[1]['game_result_was_invalid'] = False
                infos[-1]['game_result_was_invalid'] = False

                player_1_reward = self.base_env.get_game_ended(self.state, player=1)
                player_2_reward = self.base_env.get_game_ended(self.state, player=-1)

                if player_1_reward == 1:
                    infos[1]['game_result'] = 'won'
                    infos[-1]['game_result'] = 'lost'
                elif player_1_reward == -1:
                    infos[1]['game_result'] = 'lost'
                    infos[-1]['game_result'] = 'won'
                else:
                    infos[1]['game_result'] = 'tied'
                    infos[-1]['game_result'] = 'tied'

                rewards = {1: player_1_reward,
                           -1: player_2_reward}

            if self.vs_human:
                obs.pop(self.human_player_num)
                rewards.pop(self.human_player_num)
                dones.pop(self.human_player_num)
                infos.pop(self.human_player_num)

        return obs, rewards, dones, infos


def make_stratego_env(env_config):
    return StrategoMultiAgentEnv(env_config)


register_env(STRATEGO_ENV, make_stratego_env)


if __name__ == '__main__':

    stratego_env_config = {
        'version': TINY,
        'observation_includes_internal_state': True,
        'observation_mode': FULLY_OBSERVABLE,
        'repeat_games_from_other_side': True,
        'vs_human': True,
        'human_player_num': -1
    }

    env = make_stratego_env(env_config=stratego_env_config)

    obs = env.reset()
    while True:
        valid_moves = obs[1][VALID_ACTIONS_MASK]
        action_probs = valid_moves / np.sum(valid_moves)
        action = np.random.choice(range(len(valid_moves)), p=action_probs)
        obs, rewards, dones, infos = env.step({1: action})

        if dones['__all__']:
            obs = env.reset()