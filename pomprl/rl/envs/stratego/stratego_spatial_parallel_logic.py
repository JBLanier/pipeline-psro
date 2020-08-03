import numpy as np
from numba import types, jit, prange

from pomprl.games.stratego.stratego_game_logic import _get_state_from_player_perspective, _get_valid_moves_as_1d_mask, \
    _get_partially_observable_observation, _get_fully_observable_observation, _get_valid_moves_as_spatial_mask, \
    _get_action_1d_index_from_player_perspective, _get_next_state, _get_game_ended, _get_partially_observable_observation_extended_channels, \
    _get_fully_observable_observation_extended_channels, _get_heuristic_rewards_from_move

STSH = "int64[:, :, ::1]"
OTSH = "float32[:, :, ::1]"
COTSH = types.Optional(types.float32[:, :, ::1])
MSTSH = "int64[:, :, :, ::1]"
MOTSH = "float32[:, :, :, :, ::1]"

VTSH = "int64[:, :, ::1]"
MVTSH = "int64[:, :, :, :, ::1]"

@jit(f"void(int64, int64, int64, int64, int64, "
     f"float32[:, :, ::1], float32[:, :, ::1], float32[:, :, ::1], float32[:, :, ::1], boolean, boolean, boolean, boolean,"
     f"{STSH}, {VTSH}, {OTSH}, {OTSH}, {OTSH})", nopython=True, cache=True)
def _fill_observation(player, action_size, max_actions, rows, columns,
                      p_obs_mids, p_ops_range, f_obs_mids, f_obs_range, partial_out, full_out, internal_out, extended_channels,
                      state, valid_actions_mask, partial_observation, full_observation, internal_state):
    pstate = _get_state_from_player_perspective(state=state, player=player)

    valid_actions_mask[:] = _get_valid_moves_as_spatial_mask(state=pstate, player=np.int64(1))
    if partial_out:
        if extended_channels:
            partial_observation[:] = _get_partially_observable_observation_extended_channels(state=pstate, player=np.int64(1),
                                                                           rows=rows, columns=columns)
        else:
            partial_observation[:] = _get_partially_observable_observation(state=pstate, player=np.int64(1),
                                                                       rows=rows, columns=columns)

        partial_observation[:] -= p_obs_mids
        partial_observation[:] /= p_ops_range

    if full_out:
        if extended_channels:
            full_observation[:] = _get_fully_observable_observation_extended_channels(state=pstate, player=np.int64(1),
                                                                    rows=rows, columns=columns)
        else:
            full_observation[:] = _get_fully_observable_observation(state=pstate, player=np.int64(1),
                                                                rows=rows, columns=columns)

        full_observation[:] -= f_obs_mids
        full_observation[:] /= f_obs_range

    if internal_out:
        internal_state[:] = pstate


@jit(f"Tuple((float32, boolean))(int64, int64, int64, int64, int64, int64, "
     f"float32[:, :, ::1], float32[:, :, ::1], float32[:, :, ::1], float32[:, :, ::1], boolean, boolean, boolean, boolean,"
     f"{STSH}, {VTSH}, {OTSH}, {OTSH}, {OTSH},"
     f"boolean, float32[:, ::1])", nopython=True, cache=True)
def _step_single_env(player, action, action_size, max_actions, rows, columns,
                     p_obs_mids, p_ops_range, f_obs_mids, f_obs_range, partial_out, full_out, internal_out, extended_channels,
                     state, valid_actions_mask, partial_observation, full_observation, internal_state,
                     add_heuristic_rewards, heuristic_reward_matrix):

    # Convert action
    action = _get_action_1d_index_from_player_perspective(action_index=action, player=player,
                                                          action_size=action_size, rows=rows, columns=columns,
                                                          max_possible_actions_per_start_position=max_actions)

    if add_heuristic_rewards:
        heuristic_reward = _get_heuristic_rewards_from_move(state=state, player=player, action_index=action,
                                                            action_size=action_size,
                                                            max_possible_actions_per_start_position=max_actions,
                                                            allow_piece_oscillation=False,
                                                            reward_matrix=heuristic_reward_matrix)
    else:
        heuristic_reward = 0.0

    # Update state buffer in-place
    state[:] = _get_next_state(state=state, player=player, action_index=action,
                               action_size=action_size, max_possible_actions_per_start_position=max_actions,
                               allow_piece_oscillation=False)

    # Update player
    player = -player

    # Generate NEXT player's observation
    _fill_observation(player, action_size, max_actions, rows, columns,
                      p_obs_mids, p_ops_range, f_obs_mids, f_obs_range,
                      partial_out, full_out, internal_out, extended_channels,
                      state, valid_actions_mask, partial_observation,
                      full_observation, internal_state)

    orig_reward = _get_game_ended(state=state, player=player)
    done = orig_reward != 0
    if done:
        heuristic_reward = 0.0
    reward = orig_reward - heuristic_reward

    return reward, done


@jit(f"void(int64[::1], int64[::1], float32[::1], boolean[::1], int64, int64, int64, int64, "
     f"float32[:, :, ::1], float32[:, :, ::1], float32[:, :, ::1], float32[:, :, ::1], boolean, boolean, boolean, boolean,"
     f"{MSTSH}, {MVTSH}, {MOTSH}, {MOTSH}, {MOTSH},"
     f"boolean, float32[:, ::1])", nopython=True, parallel=True, cache=True)
def _step_multi_envs(players, actions, reward_buffer, done_buffer, action_size, max_actions, rows, columns,
                     p_obs_mids, p_ops_range, f_obs_mids, f_obs_range,
                     partial_out, full_out, internal_out, extended_channels,
                     states, valid_actions_masks, partial_observations, full_observations, internal_states,
                     add_heuristic_rewards, heuristic_reward_matrix):
    num_envs = players.shape[0]
    for i in prange(num_envs):
        # Extract current environments variables
        player = players[i]
        action = actions[i]
        state = states[i, :]

        # This is confusing as fuck
        # We need the observation buffers for the NEXT PLAYER
        obs_idx = (player + 1) // 2
        valid_actions_mask = valid_actions_masks[obs_idx, i, :]
        partial_observation = partial_observations[obs_idx, i, :]
        full_observation = full_observations[obs_idx, i, :]
        internal_state = internal_states[obs_idx, i, :]

        reward, done = _step_single_env(player, action, action_size, max_actions, rows, columns,
                                  p_obs_mids, p_ops_range, f_obs_mids, f_obs_range,
                                  partial_out, full_out, internal_out, extended_channels,
                                  state, valid_actions_mask, partial_observation, full_observation, internal_state,
                                  add_heuristic_rewards, heuristic_reward_matrix)

        players[i] = -player
        reward_buffer[i] = reward
        done_buffer[i] = done
        if done:
            # player = -player
            obs_idx = (-player + 1) // 2
            valid_actions_mask = valid_actions_masks[obs_idx, i, :]
            partial_observation = partial_observations[obs_idx, i, :]
            full_observation = full_observations[obs_idx, i, :]
            internal_state = internal_states[obs_idx, i, :]

            _fill_observation(player, action_size, max_actions, rows, columns,
                              p_obs_mids, p_ops_range, f_obs_mids, f_obs_range,
                              partial_out, full_out, internal_out, extended_channels,
                              state, valid_actions_mask, partial_observation,
                              full_observation, internal_state)