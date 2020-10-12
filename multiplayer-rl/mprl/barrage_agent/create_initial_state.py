import numpy as np
from stratego_env import GameVersions, ObservationModes
from stratego_env.game.stratego_procedural_impl import SP, StateLayers, INT_DTYPE_NP
from stratego_env.game.stratego_procedural_env import StrategoProceduralEnv
from stratego_env.stratego_multiagent_env import VERSION_CONFIGS
from stratego_env.game.util import create_initial_positions_from_human_data
from stratego_env.game.inits.barrage_human_inits import BARRAGE_INITS as HUMAN_INITS

OUTSIDE_AGENT_PLAYER_ID = 1
P2SRO_AGENT_PLAYER_ID = -1


STRATEGO_ENV_BARRAGE_INTERFACE_CONFIG = {
    'version': GameVersions.BARRAGE,
    'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,
}


def create_manual_barrage_initial_state(
        spy_locations_list,
        scout_locations_list,
        miner_locations_list,
        sergeant_locations_list,
        lieutenant_locations_list,
        captain_locations_list,
        major_locations_list,
        colonel_locations_list,
        general_locations_list,
        marshall_locations_list,
        flag_locations_list,
        bomb_locations_list,
        specify_pieces_for_player=OUTSIDE_AGENT_PLAYER_ID):
    """Creates an initial stratego env state with manually specified piece positions for a single player
        and a random human initialization for the other player.

    Example initial board piece positions.
    Positive pieces belong to player 1, negative to player -1:

        COL   9     8     7     6     5     4     3     2     1     0
           -------------------------------------------------------------
    Row  9 |     |     |     |     |     |     |     |  -1 |     |     |
           -------------------------------------------------------------
    Row  8 |     |  -2 |     |     |     |     |     |     |     |     |
           -------------------------------------------------------------
    Row  7 |     |     |     | -10 |     |     |     |     |  -2 |     |
           -------------------------------------------------------------
    Row  6 |     |     |     |     |     |  -9 | -11 |  -3 |     | -12 |
           -------------------------------------------------------------
    Row  5 |     |     |   R |   R |     |     |   R |   R |     |     |
           -------------------------------------------------------------
    Row  4 |     |     |   R |   R |     |     |   R |   R |     |     |
           -------------------------------------------------------------
    Row  3 |     |   9 |     |     |  10 |     |     |     |     |  12 |
           -------------------------------------------------------------
    Row  2 |     |     |     |     |   2 |   3 |     |     |     |     |
           -------------------------------------------------------------
    Row  1 |     |     |     |     |     |     |     |     |     |     |
           -------------------------------------------------------------
    Row  0 |     |     |  11 |   2 |     |   1 |     |     |     |     |
           -------------------------------------------------------------

    Args:
        spy_locations_list: list of 2d coordinates where spies will be placed
        scout_locations_list: list of 2d coordinates where scouts will be placed
        miner_locations_list: list of 2d coordinates where miners will be placed
        sergeant_locations_list: list of 2d coordinates where sergeants will be placed
        lieutenant_locations_list: list of 2d coordinates where lieutenants will be placed
        captain_locations_list: list of 2d coordinates where captains will be placed
        major_locations_list: list of 2d coordinates where majors will be placed
        colonel_locations_list: list of 2d coordinates where colonels will be placed
        general_locations_list: list of 2d coordinates where generals will be placed
        marshall_locations_list: list of 2d coordinates where marshals will be placed
        flag_locations_list: list of 2d coordinates (1 element) where the flag will be placed
        bomb_locations_list: list of 2d coordinates where bombs will be placed
        specify_pieces_for_player (int, optional): The player (1 or -1) for which piece positions are specified

    Returns:
        ndarray: The resulting initial state representation for use in the stratego env.

    Example Usage:

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
            specify_pieces_for_player=OUTSIDE_AGENT_PLAYER_ID)

    """
    game_version = STRATEGO_ENV_BARRAGE_INTERFACE_CONFIG['version']
    game_version_config = VERSION_CONFIGS[game_version]
    board_shape = (game_version_config['rows'], game_version_config['columns'])
    procedural_env = StrategoProceduralEnv(*board_shape)

    # Verify inputs and fill a 2d ndarray with specified player piece values.
    if not (specify_pieces_for_player == 1 or specify_pieces_for_player == -1):
        raise ValueError("specify_pieces_for_player must be 1 or -1")

    allowed_piece_rows_for_player = [0, 1, 2, 3] if specify_pieces_for_player == 1 else [6, 7, 8, 9]
    specified_player_initial_piece_map = np.zeros(shape=board_shape, dtype=INT_DTYPE_NP)

    manual_piece_locations = {
        SP.SPY: spy_locations_list,
        SP.SCOUT: scout_locations_list,
        SP.MINER: miner_locations_list,
        SP.SERGEANT: sergeant_locations_list,
        SP.LIEUTENANT: lieutenant_locations_list,
        SP.CAPTAIN: captain_locations_list,
        SP.MAJOR: major_locations_list,
        SP.COLONEL: colonel_locations_list,
        SP.GENERAL: general_locations_list,
        SP.MARSHALL: marshall_locations_list,
        SP.FLAG: flag_locations_list,
        SP.BOMB: bomb_locations_list
    }

    for piece_type, locations_list in manual_piece_locations.items():
        if len(locations_list) > 0 and \
                (len(np.shape(locations_list)) != 2 or
                    (len(np.shape(locations_list)) == 2 and np.shape(locations_list)[1] != 2)):
            raise ValueError(f"Each locations list must be a list of 2d coordinates. Examples: [] or [[1,2], [2,5]].\n"
                             f"For {piece_type.name}, {locations_list} was passed.")

        if len(locations_list) != game_version_config['piece_amounts'][piece_type]:
            allowed_piece_amounts = {pc_type.name: amt for pc_type, amt in game_version_config['piece_amounts'].items()}
            raise ValueError(f"{len(locations_list)} {piece_type.name} piece locations were provided when "
                             f"{game_version.name} requires the following piece amounts: \n{allowed_piece_amounts}")

        for location in locations_list:
            row, column = location
            if (not 0 <= column < board_shape[1]) or (row not in allowed_piece_rows_for_player):
                raise ValueError(f"The out-of-range location {location} for {piece_type.name} was provided. "
                                 f"Locations are in the format, (row, column). "
                                 f"Rows take values in {allowed_piece_rows_for_player} for player {specify_pieces_for_player}. "
                                 f"Columns must be in the range [0, {board_shape[1]}].")
            if specified_player_initial_piece_map[row, column] != 0:
                raise ValueError(f"The location {location} was specified for more than one piece.")

            # Set piece value for location
            specified_player_initial_piece_map[row, column] = piece_type.value

    # Grab a random human initialization for the non-specified player.
    # Human inits have been downloaded from the Gravon Archive (https://www.gravon.de/gravon/stratego/strados2.jsp)
    random_human_init_spec_str = np.random.choice(HUMAN_INITS)
    player_1_random_human_piece_map, player_2_random_human_piece_map = create_initial_positions_from_human_data(
        player1_string=random_human_init_spec_str, player2_string=random_human_init_spec_str,
        game_version_config=game_version_config)

    # Set obstacle locations
    obstacle_map = np.zeros(shape=board_shape, dtype=INT_DTYPE_NP)
    for obstacle_location in VERSION_CONFIGS[game_version]['obstacle_locations']:
        obstacle_map[obstacle_location] = 1.0

    # Create the initial state
    initial_state = procedural_env.create_initial_state(
        obstacle_map=obstacle_map,
        player_1_initial_piece_map=specified_player_initial_piece_map if specify_pieces_for_player == 1 else player_1_random_human_piece_map,
        player_2_initial_piece_map=specified_player_initial_piece_map if specify_pieces_for_player == -1 else player_2_random_human_piece_map,
        max_turns=game_version_config['max_turns'])

    return initial_state



