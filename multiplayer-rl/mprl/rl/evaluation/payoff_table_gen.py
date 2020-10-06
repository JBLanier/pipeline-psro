from mprl.rl.envs.stratego.stratego_spatial_multiagent_env import SpatialStrategoMultiAgentEnv
from progress.bar import Bar
import numpy as np
import dill
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))

def _eval_policy_matchup(get_policy_fn_a, get_policy_fn_b, env, stratego_env_config, games_per_matchup):
    resample_policy_fn_a = False
    if isinstance(get_policy_fn_a, tuple):
        get_policy_fn_a, resample_policy_fn_a = get_policy_fn_a

    policy_a_name, policy_a_get_action_index = get_policy_fn_a(stratego_env_config)

    resample_policy_fn_b = False
    if isinstance(get_policy_fn_b, tuple):
        get_policy_fn_b, resample_policy_fn_b = get_policy_fn_b

    policy_b_name, policy_b_get_action_index = get_policy_fn_b(stratego_env_config)
    policy_funcs = [policy_a_get_action_index, policy_b_get_action_index]

    policy_a_state = None
    policy_b_state = None
    policy_states = [policy_a_state, policy_b_state]

    def policy_index(agent_id):
        if agent_id == 1:
            return 0
        else:
            return 1

    policy_a_total_return = 0
    ties = 0
    # with Bar('Evaluating {} vs {}'.format(policy_a_name, policy_b_name), max=games_per_matchup) as bar:
    for game in range(games_per_matchup):

        if resample_policy_fn_a:
            policy_a_get_action_index(None, None, resample=True)

        if resample_policy_fn_b:
            policy_b_get_action_index(None, None, resample=True)

        obs = env.reset()
        dones = {}
        infos = {}
        game_length = 0
        while True:
            if "__all__" in dones:
                if dones["__all__"]:
                    break
            game_length += 1
            assert len(obs) == 1
            acting_agent_id, acting_agent_observation = list(obs.items())[0]
            acting_policy_fn = policy_funcs[policy_index(acting_agent_id)]
            acting_policy_state = policy_states[policy_index(acting_agent_id)]

            action_index, new_policy_state = acting_policy_fn(acting_agent_observation, acting_policy_state)
            policy_states[policy_index(acting_agent_id)] = new_policy_state

            obs, rewards, dones, infos = env.step(action_dict={acting_agent_id: action_index})

        player_a_won = infos[1]['game_result'] == 'won'
        tied = infos[1]['game_result'] == 'tied'

        if player_a_won:
            policy_a_total_return += 1
        elif not tied:
            policy_a_total_return -= 1
        elif tied:
            ties += 1

        # print(f"game length: {game_length}")

        # bar.next()

    policy_a_expected_payoff = policy_a_total_return / games_per_matchup
    tie_percentage = ties / games_per_matchup

    return policy_a_name, policy_b_name, policy_a_expected_payoff, tie_percentage


def generate_payoff_table(get_policy_fn_list,
                          games_per_matchup,
                          stratego_env_config,
                          policies_also_play_against_self=True,
                          return_matrix=False,
                          num_processes=0):

    env = SpatialStrategoMultiAgentEnv(env_config=stratego_env_config)

    payoff_table_dict = {}
    tie_dict = {}
    results_dict = {}
    payoff_table_matrix = np.zeros(shape=(len(get_policy_fn_list),
                                          len(get_policy_fn_list)))

    payoff_matrix_i_names = [None] * len(get_policy_fn_list)
    payoff_matrix_j_names = [None] * len(get_policy_fn_list)

    if num_processes == 0:
        num_processes = cpu_count()

    pool = Pool(processes=num_processes)

    for i in range(len(get_policy_fn_list)):

        get_policy_fn_a = get_policy_fn_list[i]

        if policies_also_play_against_self:
            j_start = i
        else:
            j_start = i + 1

        for j in range(j_start, len(get_policy_fn_list)):
            get_policy_fn_b = get_policy_fn_list[j]

            res = apply_async(pool, _eval_policy_matchup, (get_policy_fn_a, get_policy_fn_b, env, stratego_env_config, games_per_matchup))

            if i not in results_dict:
                results_dict[i] = {}
            results_dict[i][j] = res
            print(f"submitted {i} vs {j}")

    for i in range(len(get_policy_fn_list)):
        print("waiting for and processing results now...")
        if policies_also_play_against_self:
            j_start = i
        else:
            j_start = i + 1
        for j in range(j_start, len(get_policy_fn_list)):
            policy_a_name, policy_b_name, policy_a_expected_payoff, tie_percentage = results_dict[i][j].get()
            payoff_matrix_i_names[i] = policy_a_name
            payoff_matrix_j_names[j] = policy_b_name

            if policy_a_name not in payoff_table_dict:
                payoff_table_dict[policy_a_name] = {}
                tie_dict[policy_a_name] = {}
            payoff_table_dict[policy_a_name][policy_b_name] = policy_a_expected_payoff
            tie_dict[policy_a_name][policy_b_name] = tie_percentage

            payoff_table_matrix[i, j] = policy_a_expected_payoff
            print(f"got {i} ({policy_a_name}) vs {j} ({policy_b_name})")


    if return_matrix:
        return payoff_table_dict, tie_dict, payoff_table_matrix, payoff_matrix_i_names

    return payoff_table_dict, tie_dict



def generate_single_player_payoff_table(get_policy_fn_list,
                        play_as_agent_id,
                          games_per_matchup,
                          stratego_env_config,
                        resample_policy_every_game=False):

    env = SpatialStrategoMultiAgentEnv(env_config=stratego_env_config)

    payoff_table_dict = {}
    tie_dict = {}

    payoff_table_matrix = np.zeros(shape=(len(get_policy_fn_list)))

    for i in range(len(get_policy_fn_list)):

        get_policy_fn_a = get_policy_fn_list[i]

        policy_a_name, policy_a_get_action_index = get_policy_fn_a(stratego_env_config)
        policy_func = policy_a_get_action_index

        policy_a_state = None
        policy_state = policy_a_state

        policy_a_total_return = 0
        ties = 0
        with Bar('Evaluating {}'.format(policy_a_name), max=games_per_matchup) as bar:
            for game in range(games_per_matchup):

                if resample_policy_every_game:
                    policy_func(None, None, resample=True)

                obs = env.reset()
                dones = {}
                infos = {}
                # env.base_env.print_fully_observable_board_to_console(state=env.state)

                while True:
                    if "__all__" in dones:
                        if dones["__all__"]:
                            break
                    assert len(obs) == 1
                    acting_agent_id, acting_agent_observation = list(obs.items())[0]
                    assert acting_agent_id == play_as_agent_id
                    acting_policy_fn = policy_func
                    acting_policy_state = policy_state

                    action_index, new_policy_state = acting_policy_fn(acting_agent_observation, acting_policy_state)
                    policy_state = new_policy_state

                    obs, rewards, dones, infos = env.step(action_dict={acting_agent_id: action_index})

                player_a_won = infos[play_as_agent_id]['game_result'] == 'won'
                tied = infos[play_as_agent_id]['game_result'] == 'tied'

                if player_a_won:
                    policy_a_total_return += 1
                elif not tied:
                    policy_a_total_return -= 1
                elif tied:
                    ties +=1
                bar.next()

        policy_a_expected_payoff = policy_a_total_return / games_per_matchup
        tie_percentage = ties/games_per_matchup

        payoff_table_dict[policy_a_name] = policy_a_expected_payoff
        tie_dict[policy_a_name] = tie_percentage

        payoff_table_matrix[i] = policy_a_expected_payoff

    return payoff_table_dict, tie_dict
