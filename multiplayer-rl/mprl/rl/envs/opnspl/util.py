import pyspiel
import numpy as np
from typing import Optional
from open_spiel.python.algorithms import get_all_states


def policy_to_dict_but_we_can_actually_use_it(player_policy,
                                              game,
                                              all_states=None,
                                              state_to_information_state=None,
                                              player_id:Optional = None):
    """Converts a Policy instance into a tabular policy represented as a dict.

    This is compatible with the C++ TabularExploitability code (i.e.
    pyspiel.exploitability, pyspiel.TabularBestResponse, etc.).

    While you do not have to pass the all_states and state_to_information_state
    arguments, creating them outside of this funciton will speed your code up
    dramatically.

    Args:
      player_policy: The policy you want to convert to a dict.
      game: The game the policy is for.
      all_states: The result of calling get_all_states.get_all_states. Can be
        cached for improved performance.
      state_to_information_state: A dict mapping str(state) to
        state.information_state for every state in the game. Can be cached for
        improved performance.

    Returns:
      A dictionary version of player_policy that can be passed to the C++
      TabularBestResponse, Exploitability, and BestResponse functions/classes.
    """
    if all_states is None:
        all_states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=False,
            include_chance_states=False)
        state_to_information_state = {
            state: str(
                np.asarray(all_states[state].information_state_as_normalized_vector(), dtype=np.float32).tolist()) for
            state in all_states
        }
    tabular_policy = dict()
    for state in all_states:
        if player_id is not None and all_states[state].current_player() != player_id:
            continue

        information_state = state_to_information_state[state]
        tabular_policy[information_state] = list(
            player_policy.action_probabilities(all_states[state]).items())
    return tabular_policy
