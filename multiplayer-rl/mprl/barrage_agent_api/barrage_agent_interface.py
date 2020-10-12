
import numpy as np
import random

from stratego_env import StrategoMultiAgentEnv, ObservationComponents
from stratego_env.game.stratego_procedural_impl import SP, StateLayers
from mprl.barrage_agent_api.create_initial_state import STRATEGO_ENV_BARRAGE_INTERFACE_CONFIG, P2SRO_AGENT_PLAYER_ID, \
    OUTSIDE_AGENT_PLAYER_ID, create_manual_barrage_initial_state
from mprl.barrage_agent_api.p2sro_agent_wrapper import BarrageAgentPolicyPopulationWrapper


class BarrageAgentInterface:

    def __init__(self, disable_piece_oscillation_checks: bool = False):

        self.stratego_env = StrategoMultiAgentEnv(env_config=STRATEGO_ENV_BARRAGE_INTERFACE_CONFIG)
        self._p2sro_agent = BarrageAgentPolicyPopulationWrapper(stratego_env_config=STRATEGO_ENV_BARRAGE_INTERFACE_CONFIG)
        self._disable_piece_oscillation_checks = disable_piece_oscillation_checks
        self._current_episode_move_count = 0

    def reset(self, p2sro_agent_goes_first: bool, initial_state_override: np.ndarray = None):
        """Starts a new game of 10x10 Barrage Stratego against the P2SRO agent.

        If the p2sro agent is chosen to go first, it will make its first move before this function returns.

        Args:
            p2sro_agent_goes_first (bool): Whether or not the p2sro agent takes the first move in the new episode
            initial_state_override (ndarray, optional): Manually specified stratego env initial state for the purpose of
                specifying starting piece positions. If not provided, random human piece initializations will be used
                for both sides.

        Returns:
            ndarray: The internal state representation used by the stratego game environment with the outside agent
                appearing as player 1 and p2sro agent as player -1 i.e. player 2.
                A detailed breakdown for the contents of the stratego env state can be found at
                https://github.com/JBLanier/stratego_env/blob/master/stratego_env/game/stratego_procedural_impl.py
            ndarray: The observation for the outside agent in the same format as used by the p2sro agent.
                It is assumed that the outside agent code will create its own observation from the internal state,
                likely in a different format than the p2sro agent's. This is returned in the interest of redundancy.

            It is also possible that the user may not need this return information, as they may be tracking the game
            state through their own stratego game logic implementation.
        """

        # Sample a new policy from the P2SRO agent's policy population for this episode
        self._p2sro_agent.sample_new_policy_from_metanash()

        # Reset the stratego env to a new episode
        starting_player = P2SRO_AGENT_PLAYER_ID if p2sro_agent_goes_first else OUTSIDE_AGENT_PLAYER_ID
        obs_dict = self.stratego_env.reset(first_player_override=starting_player,
                                           initial_state_override=initial_state_override)
        acting_player_observation = obs_dict[self.stratego_env.player]

        # Print debug information and view of board from the first acting player
        print("-"*70)
        print("NEW GAME")
        print(f"Acting Player is {self.stratego_env.player} ({'P2SRO Agent' if self.stratego_env.player == P2SRO_AGENT_PLAYER_ID else 'Outside Agent'})")
        print(f"Player {self.stratego_env.player}'s view of board before move:")
        acting_player_perspective_state_before_move = self.stratego_env.base_env.get_state_from_player_perspective(
            state=self.stratego_env.state,
            player=self.stratego_env.player)
        self.stratego_env.base_env.print_board_to_console(state=acting_player_perspective_state_before_move, partially_observable=False)
        print("-"*70)

        # If the P2SRO agent goes first, have it make a move now.
        if self.stratego_env.player == P2SRO_AGENT_PLAYER_ID:
            p2sro_agent_action_as_1d_index = self._p2sro_agent.get_action(
                extended_channels_observation=acting_player_observation)
            p2sro_agent_action_as_positions = self.stratego_env.base_env.get_action_positions_from_1d_index(
                action_index=p2sro_agent_action_as_1d_index)
            acting_player_perspective_state, acting_player_observation, _, _, _ = self.make_move(*p2sro_agent_action_as_positions)
        else:
            acting_player_perspective_state = self.stratego_env.base_env.get_state_from_player_perspective(
                state=self.stratego_env.state,
                player=self.stratego_env.player)

        self._current_episode_move_count = 0

        return acting_player_perspective_state, acting_player_observation

    def make_move(self, start_row: int, start_col: int, end_row: int, end_col: int):
        """Executes a move on behalf of the outside agent.

        If the game is not over as a result of said move, the p2sro agent will also make a move before this function
        returns.

        Args:
            start_row (int): Starting row to pick up a piece from
            start_col (int): Starting column to pick up a piece from
            end_row (int): Destination row to place the moved piece
            end_col (int): Destination column to place the moved piece

        Returns:
            ndarray: The new internal state used by the stratego game environment with the outside agent
                appearing as player 1 and p2sro agent as player -1 i.e. player 2.
                A detailed breakdown for the contents of the stratego env state can be found at
                https://github.com/JBLanier/stratego_env/blob/master/stratego_env/game/stratego_procedural_impl.py
            ndarray: The observation for the outside agent in the same format as used by the p2sro agent.
                It is assumed that the outside agent code will create its own observation from the internal state,
                likely in a different format than the p2sro agent's. This is returned in the interest of redundancy.
            bool: Whether the game is now over
            bool: Whether the outside agent won
            bool: Whether the game resulted in a tie
        """
        # Print Pre-Move Debug information
        print("-"*70)
        print(f"Move {self._current_episode_move_count}")
        print(f"Acting Player is {self.stratego_env.player} ({'P2SRO Agent' if self.stratego_env.player == P2SRO_AGENT_PLAYER_ID else 'Outside Agent'})")
        print(f"Player {self.stratego_env.player}'s view of board before move:")
        acting_player_perspective_state_before_move = self.stratego_env.base_env.get_state_from_player_perspective(
            state=self.stratego_env.state,
            player=self.stratego_env.player)
        self.stratego_env.base_env.print_board_to_console(state=acting_player_perspective_state_before_move, partially_observable=True)
        piece_type_being_moved = self._get_piece_type_name_at_coordinate_for_player(row=start_row, col=start_col,
                                                                                    player=self.stratego_env.player)
        destination_piece_type = self._get_piece_type_name_at_coordinate_for_player(row=end_row, col=end_col,
                                                                                    player=self.stratego_env.player)
        print(f"Player {self.stratego_env.player}'s Move: {(start_row, start_col)}, {piece_type_being_moved} -> {(end_row, end_col)}, {destination_piece_type}")

        # Check if the move is considered valid by the stratego environment.
        # If it isn't valid, we give the outside agent the benefit of the doubt.
        # - the game will end and the outside agent is declared the winner.
        if not self.stratego_env.base_env.is_move_valid_by_position(
                state=acting_player_perspective_state_before_move, player=1,
                start_r=start_row, start_c=start_col, end_r=end_row, end_c=end_col,
                allow_piece_oscillation=self._disable_piece_oscillation_checks):
            new_acting_player_perspective_state_after_move = None
            new_acting_player_observation_after_move = None
            game_ended = True
            outside_agent_won = True
            tie = False
            print(f"Move by player {self.stratego_env.player} isn't considered valid by the stratego environment. "
                  f"Game is ended.")
        else:
            # Perform the valid move in the stratego environment
            action_as_1d_index = self.stratego_env.base_env.get_action_1d_index_from_positions(
                start_r=start_row, start_c=start_col, end_r=end_row, end_c=end_col)
            obs_dict, rewards, dones, infos = self.stratego_env.step(
                action_dict={self.stratego_env.player: action_as_1d_index},
                allow_piece_oscillation=self._disable_piece_oscillation_checks,
                is_spatial_index=False)
            new_acting_player_observation_after_move = obs_dict[self.stratego_env.player]

            # Check if the game is over and if there is a winner
            if dones["__all__"]:
                game_ended = True
                outside_agent_won = rewards[OUTSIDE_AGENT_PLAYER_ID] == 1
                tie = rewards[OUTSIDE_AGENT_PLAYER_ID] != 1 and rewards[OUTSIDE_AGENT_PLAYER_ID] != -1
            else:
                game_ended = False
                outside_agent_won = False
                tie = False

            # Print Post-Move Debug information
            print(f"Player {self.stratego_env.player}'s view of board after player {-1 * self.stratego_env.player}'s move:")
            new_acting_player_perspective_state_after_move = self.stratego_env.base_env.get_state_from_player_perspective(
                state=self.stratego_env.state,
                player=self.stratego_env.player)
            self.stratego_env.base_env.print_board_to_console(state=new_acting_player_perspective_state_after_move,
                                                              partially_observable=True)
        print("-"*70)
        self._current_episode_move_count += 1

        # If it is now the P2SRO agent's turn, have it make a move now.
        if self.stratego_env.player == P2SRO_AGENT_PLAYER_ID and not game_ended:
            p2sro_agent_action_as_1d_index = self._p2sro_agent.get_action(
                extended_channels_observation=new_acting_player_observation_after_move)
            p2sro_agent_action_as_positions = self.stratego_env.base_env.get_action_positions_from_1d_index(
                action_index=p2sro_agent_action_as_1d_index)
            return self.make_move(*p2sro_agent_action_as_positions)

        return new_acting_player_perspective_state_after_move, new_acting_player_observation_after_move, game_ended, outside_agent_won, tie

    def _get_piece_type_name_at_coordinate_for_player(self, row, col, player):
        player_perspective_state = self.stratego_env.base_env.get_state_from_player_perspective(
            state=self.stratego_env.state,
            player=player)
        # Check piece values from both player's pieces.
        piece_type_num = player_perspective_state[StateLayers.PLAYER_1_PIECES.value][row, col]
        if piece_type_num == SP.NOPIECE.value:
            piece_type_num = player_perspective_state[StateLayers.PLAYER_2_PIECES.value][row, col]
        piece_type_name = SP(piece_type_num).name
        return piece_type_name


if __name__ == '__main__':
    # Example usage with the outside agent executing random actions:

    # BarrageAgentInterface only needs to be initialized once to play multiple games sequentially.
    interface = BarrageAgentInterface()

    def decompose_observation(obs):
        board_observation = obs[ObservationComponents.PARTIAL_OBSERVATION.value]

        valid_actions_mask_as_spatial_indexes = obs[ObservationComponents.VALID_ACTIONS_MASK.value]
        valid_actions_as_positions_list = []
        for spatial_idx, is_move_valid in np.ndenumerate(valid_actions_mask_as_spatial_indexes):
            if is_move_valid:
                start_r, start_c, end_r, end_c = interface.stratego_env.base_env.get_action_positions_from_spatial_index(
                    spatial_index=spatial_idx)
                valid_actions_as_positions_list.append([start_r, start_c, end_r, end_c])

        return board_observation, valid_actions_as_positions_list

    # The ps2sro agent will use a random piece initialization from a large set of recorded human inits.
    # If you also want to use a random human init, pass None for initial_state_override in interface.reset()
    # If you want to specify your own piece initialization, follow the example below:
    outside_agent_uses_random_human_init = False
    if outside_agent_uses_random_human_init:
        initial_state = None
    else:
        # Creates a stratego env initial state with a random human piece initialization for the p2sro agent
        # and the specified piece locations for the outside agent:
        initial_state = create_manual_barrage_initial_state(
            spy_locations_list=[[0, 3]],
            scout_locations_list=[[0, 5], [2, 5]],
            miner_locations_list=[[2, 4]],
            sergeant_locations_list=[],
            lieutenant_locations_list=[],
            captain_locations_list=[],
            major_locations_list=[],
            colonel_locations_list=[],
            general_locations_list=[[3, 8]],
            marshall_locations_list=[[3, 5]],
            flag_locations_list=[[0, 7]],
            bomb_locations_list=[[3, 0]],
            specify_pieces_for_player=OUTSIDE_AGENT_PLAYER_ID
        )

    # Start a new game
    game_state, observation_for_outside_agent = interface.reset(p2sro_agent_goes_first=np.random.random() < 0.5,
                                                                initial_state_override=initial_state)
    game_finished = False
    while not game_finished:
        board_observation, valid_actions_as_position_list = decompose_observation(observation_for_outside_agent)
        random_action_as_positions = random.choice(valid_actions_as_position_list)
        move_result = interface.make_move(*random_action_as_positions)
        game_state, observation_for_outside_agent, game_finished, outside_agent_won, tie = move_result

        if game_finished and not tie:
            print(f"{'Outside agent won.' if outside_agent_won else 'P2SRO agent won.'}")
        elif game_finished and tie:
            print("Tie game.")
