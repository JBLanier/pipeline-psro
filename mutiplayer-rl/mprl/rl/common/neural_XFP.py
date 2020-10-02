"""Implementations of classical fictitious play.
See https://en.wikipedia.org/wiki/Fictitious_play.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import itertools
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import Sequential
from keras.layers import Dense
from open_spiel.python import rl_environment

import pandas
import pyspiel
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import tensorflow.compat.v1 as tf
# import sonnet as snt


from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import openspiel_utils

def _uniform_policy(state):
    legal_actions = state.legal_actions()
    return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


def _callable_tabular_policy(tabular_policy):
    """Turns a tabular policy into a callable.
    Args:
      tabular_policy: A dictionary mapping information state key to a dictionary
      of action probabilities (action -> prob).
    Returns:
      A function `state` -> list of (action, prob)
    """

    def wrap(state):
        infostate_key = state.information_state(state.current_player())
        assert infostate_key in tabular_policy
        ap_list = []
        for action in state.legal_actions():
            assert action in tabular_policy[infostate_key]
            ap_list.append((action, tabular_policy[infostate_key][action]))
        return ap_list

    return wrap


def _joint_policy(policies):
    """Turns a table of callables (indexed by player) into a single callable.
  Args:
    policies: A dictionary mapping player number to a function `state` -> list
      of (action, prob).
  Returns:
    A function `state` -> list of (action, prob)
  """

    def wrap(state):
        player = state.current_player()
        return policies[player](state)

    return wrap


def _full_best_response_policy(br_infoset_dict):
    """Turns a dictionary of best response action selections into a full policy.
  Args:
    br_infoset_dict: A dictionary mapping information state to a best response
      action.
  Returns:
    A function `state` -> list of (action, prob)
  """

    def wrap(state):
        infostate_key = state.information_state(state.current_player())
        br_action = br_infoset_dict[infostate_key]
        ap_list = []
        for action in state.legal_actions():
            ap_list.append((action, 1.0 if action == br_action else 0.0))
        return ap_list

    return wrap


def _policy_dict_at_state(callable_policy, state):
    """Turns a policy function into a dictionary at a specific state.
  Args:
    callable_policy: A function from `state` -> lis of (action, prob),
    state: the specific state to extract the policy from.
  Returns:
    A dictionary of action -> prob at this state.
  """

    infostate_policy_list = callable_policy(state)
    infostate_policy = {}
    for ap in infostate_policy_list:
        print(ap)
        infostate_policy[ap[0]] = ap[1]
    return infostate_policy


class XFPSolver(object):
    """An implementation of extensive-form fictitious play (XFP).
  XFP is Algorithm 1 in (Heinrich, Lanctot, and Silver, 2015, "Fictitious
  Self-Play in Extensive-Form Games"). Refer to the paper for details:
  http://mlanctot.info/files/papers/icml15-fsp.pdf.
  """

    def __init__(self, 
                 game_name, 
                 state_representation_size, 
                 num_actions, 
                 hidden_layers_sizes,
                 train_epochs,
                 num_trajectories,
                 save_oracles=False):
        """Initialize the XFP solver.
        Arguments:
          game: the open_spiel game object.
          save_oracles: a boolean, indicating whether or not to save all the BR
            policies along the way (including the initial uniform policy). This
            could take up some space, and is only used when generating the meta-game
            for analysis.
        """

        self._game_name = game_name
        self._game = pyspiel.load_game(game_name)
        self._num_players = self._game.num_players()
        self._num_actions = num_actions

        # A set of callables that take in a state and return a list of
        # (action, probability) tuples.
        self._oracles = [] if save_oracles else None

        # A set of callables that take in a state and return a list of
        # (action, probability) tuples.
        self._policies = []
        for _ in range(self._num_players):
            self._policies.append(_uniform_policy)
            if save_oracles:
                self._oracles.append([_uniform_policy])

        self._best_responses = [None] * self._num_players
        self._rl_best_responses = [None] * self._num_players
        self._iterations = 0
        self._delta_tolerance = 1e-5
        self._average_policy_tables = []
        self._average_policy_tables_mlp = []
        
        self._info_sets_inputs0 = []
        self._info_sets_targets0 = []
        self._info_sets_inputs1 = []
        self._info_sets_targets1 = []
        
        self._train_epochs = train_epochs
        self._layer_sizes = hidden_layers_sizes
        self._num_trajectories = num_trajectories
        

                
        self._avg_network = self._get_MLP(state_representation_size, self._layer_sizes)
#         self._avg_policy = self._avg_network(self._info_state_ph)
#         self._avg_policy_probs = tf.nn.softmax(self._avg_policy)
        


    def _act(self, info_state, legal_actions):
        info_state = np.reshape(info_state, [1, -1])
        action_values, action_probs = self._session.run(
            [self._avg_policy, self._avg_policy_probs],
            feed_dict={self._info_state_ph: info_state})

        self._last_action_values = action_values[0]
        # Remove illegal actions, normalize probs
        probs = np.zeros(self._num_actions)
        probs[legal_actions] = action_probs[0][legal_actions]
        probs /= sum(probs)
        action = np.random.choice(len(probs), p=probs)
        return action, probs
    
    def _get_exploitability(self):
        tabular_policy = policy.TabularPolicy(self._game)
        for player_id in range(2):
            for info_state, state_policy in self.average_policy_tables()[player_id].items():
                policy_to_update_tabular = tabular_policy.policy_for_key(info_state)
                for action, probability in state_policy.items():
                    policy_to_update_tabular[action] = probability
        average_policy_values = expected_game_score.policy_value(
            self._game.new_initial_state(), [tabular_policy, tabular_policy])
#         print("Kuhn 2P average values after %s iterations" %iters)
#         print("P0: {}".format(average_policy_values[0]))
#         print("P1: {}".format(average_policy_values[1]))
        exp = exploitability.exploitability(game, tabular_policy)
        print("exploitability: {}".format(exp))
        return exp
        
    def _get_MLP(self, input_size, output_sizes):
        model = Sequential()
        model.add(Dense(output_sizes[0], activation='relu', input_shape=(input_size,)))
        for i in range(len(output_sizes)-1):
            model.add(Dense(output_sizes[i+1], activation='relu'))
        model.add(Dense(self._num_actions, activation='softmax'))
        model.compile(loss=keras.losses.kullback_leibler_divergence,
              optimizer='Adam',
              metrics=['accuracy'])
        print(model.summary())
        return model

    def average_policy_tables(self):
        """Returns a dictionary of information state -> dict of action -> prob.
        This is a joint policy (policy for all players).
        """
        return self._average_policy_tables
    
    def average_policy_tables_mlp(self):
        return self._average_policy_tables_mlp
    
    def average_network(self):
        return self._avg_network

    def average_policy_callable(self):
        """Returns a function of state -> list of (action, prob).
        This is a joint policy (policy for all players).
        """
        return _joint_policy(self._policies)

    def iteration(self, train_epochs):
        self._iterations += 1
        self._train_epochs = train_epochs
        
        self.compute_best_responses()
        self.compute_approx_best_responses()
#         self.update_average_policies()
        self.update_average_policies_sampling()
        self.train_MLP()
        
    def compute_best_responses(self):
        """Updates self._oracles to hold best responses for each player."""
        for i in range(self._num_players):
          # Compute a best response policy to pi_{-i}.
          # First, construct pi_{-i}.
            joint_policy = _joint_policy(self._policies)
            br_info = exploitability.best_response(
              self._game, policy.PolicyFromCallable(self._game, joint_policy), i)
            full_br_policy = _full_best_response_policy(
              br_info["best_response_action"])
            self._best_responses[i] = full_br_policy
            if self._oracles is not None:
                self._oracles[i].append(full_br_policy)
                
    def compute_approx_best_responses(self):
        tabular_policy = policy.TabularPolicy(self._game)
        if len(self.average_policy_tables()) > 0:
            for player_id in range(2):
                for info_state, state_policy in self.average_policy_tables()[player_id].items():
                    policy_to_update_tabular = tabular_policy.policy_for_key(info_state)
                    for action, probability in state_policy.items():
                        policy_to_update_tabular[action] = probability

        for player_id in range(2):
            self._rl_best_responses[player_id] = openspiel_utils.get_openspeil_format_rl_br_policy(game_name=self._game_name,
                                                         br_player_id=player_id,
                                                         policy_to_exploit_player_id=1-player_id,
                                                         policy_to_exploit=tabular_policy)
            
            print(self._rl_best_responses[0])
#             self._rl_best_responses[player_id] = _callable_tabular_policy(self._rl_best_responses[player_id])
            print(self._rl_best_responses[0])
            print(self._best_responses[0])
                    
    
    def update_average_policies(self):
        """Update the average policies given the newly computed best response."""

        br_reach_probs = np.ones(self._num_players)
        avg_reach_probs = np.ones(self._num_players)
        self._average_policy_tables = [{} for _ in range(self._num_players)]
#         self._average_policy_tables_mlp = [{} for _ in range(self._num_players)]
        
        self._info_sets_inputs0 = []
        self._info_sets_targets0 = []
        self._info_sets_inputs1 = []
        self._info_sets_targets1 = []
        
        self._recursively_update_average_policies(self._game.new_initial_state(),
                                                  avg_reach_probs, br_reach_probs)
        for i in range(self._num_players):
            self._policies[i] = _callable_tabular_policy(
                self._average_policy_tables[i])
            
    def update_average_policies_sampling(self):
        """
        Instead of recursively updating every infoset, sample game trajectories
        - get game trajectories
        - do update like in _recursively_update_avg_policies
        
        """
        infos0 = []
        infos1 = []
        self._info_sets_inputs0 = []
        self._info_sets_targets0 = []
        self._info_sets_inputs1 = []
        self._info_sets_targets1 = []
        self._average_policy_tables = [{} for _ in range(self._num_players)]

        for i in range(self._num_trajectories):
            inf0, inf1 = self._get_trajectory()
            for i0 in inf0:
#                 print(i0)
#                 print(i0[2])
                infos0.append(i0)
            for i1 in inf1:
                infos1.append(i1)
               
        
        for i in range(len(infos0)):
            self._update_infostate(0, infos0[i])
        for i in range(len(infos1)):
            self._update_infostate(1, infos1[i])
        
        #recursively update average policy tables using MLP policy to get tabular policy for best response
        br_reach_probs = np.ones(self._num_players)
        avg_reach_probs = np.ones(self._num_players)
        self._average_policy_tables = [{} for _ in range(self._num_players)]

        self._recursively_update_average_policies(self._game.new_initial_state(),
                                          avg_reach_probs, br_reach_probs)
        
        for i in range(self._num_players):
            self._policies[i] = _callable_tabular_policy(
                self._average_policy_tables[i])
        
            
    def _update_infostate(self, player, 
                          info):
        info_vector = info[0]
        infostate_key = info[1]
        state = info[2]
#         br_policy = _policy_dict_at_state(self._best_responses[player], state)
        print(self._rl_best_responses[player][infostate_key], "rl best response")
        br_policy = _policy_dict_at_state(self._rl_best_responses[player], state)
        
        
        legal_actions = state.legal_actions()
        
        br_reach_probs = info[3] 
        avg_reach_probs = info[4]

        avg_policy = self._avg_network.predict(np.array([info_vector]))
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        avg_policy = np.multiply(avg_policy, legal_actions_mask)[0]
        avg_policy = avg_policy/avg_policy.sum()

        
        if infostate_key not in self._average_policy_tables[player]:
            alpha = 1 / (self._iterations + 1)
            self._average_policy_tables[player][infostate_key] = {}

            pr_sum = 0.0

            target = np.zeros(self._num_actions)

            for action in legal_actions:

                pr = (
                  avg_policy[action] + (alpha * br_reach_probs *
                                        (br_policy[action] - avg_policy[action])) /
                  ((1.0 - alpha) * avg_reach_probs + 
                   alpha * br_reach_probs))
#                 self._average_policy_tables[player][infostate_key][action] = pr
                target[action] = pr

                pr_sum += pr

            if player == 0:
                self._info_sets_inputs0.append(info_vector)
                self._info_sets_targets0.append(target)
            else:
                self._info_sets_inputs1.append(info_vector)
                self._info_sets_targets1.append(target)

            
            assert (1.0 - self._delta_tolerance <= pr_sum <=
                    1.0 + self._delta_tolerance), pr_sum
        
    def _get_trajectory(self):
        """
        Returns lists of infosets for both players
        (should also eventually include action probs so we don't double compute them) 
        """
        state = self._game.new_initial_state()
        infos0 = []
        infos1 = []
        br_reach_probs = np.ones(self._num_players)
        avg_reach_probs = np.ones(self._num_players)
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:

                player = state.current_player()
#                 br_policy = _policy_dict_at_state(self._best_responses[player], state)
                print(self._rl_best_responses[player][state.information_state(player)], "rl best response")
                br_policy = _policy_dict_at_state(self._rl_best_responses[player], state)

                
                legal_actions = state.legal_actions()
                info_vector = state.information_state_as_normalized_vector(player)
                infostate_key = state.information_state(player)
                avg_policy = self._avg_network.predict(np.array([info_vector]))
#                 noise = np.random.uniform(size=self._num_actions)
#                 avg_policy = avg_policy + 0.03*abs(noise)
#                 avg_policy /= avg_policy.sum()
                legal_actions_mask = np.zeros(self._num_actions)
                legal_actions_mask[legal_actions] = 1.0
                avg_policy = np.multiply(avg_policy, legal_actions_mask)[0]
                avg_policy = avg_policy/avg_policy.sum()

                
#                 avg_policy = np.multiply(noise/noise.sum(), legal_actions_mask)

               
                if player == 0:
                    infos0.append((info_vector, infostate_key, state.clone(), br_reach_probs[0], avg_reach_probs[0]))
                else:
                    infos1.append((info_vector, infostate_key, state.clone(), br_reach_probs[1], avg_reach_probs[1]))
                
                action = np.random.choice(avg_policy.shape[0], p=avg_policy)
                br_reach_probs[player] *= br_policy[action]
                avg_reach_probs[player] *= avg_policy[action]
                state.apply_action(action)
        
       
        return infos0, infos1
                
 
    def _recursively_update_average_policies(self, state, avg_reach_probs,
                                           br_reach_probs):
        """Recursive implementation of the average strategy update."""

        if state.is_terminal():
            return
        elif state.is_chance_node():
            for action, _ in state.chance_outcomes():
                new_state = state.clone()
                new_state.apply_action(action)
                                
                self._recursively_update_average_policies(new_state, avg_reach_probs,
                                                      br_reach_probs)
        else:
            player = state.current_player()
            avg_policy_tabular = _policy_dict_at_state(self._policies[player], state)
             
#             br_policy = _policy_dict_at_state(self._best_responses[player], state)
            br_policy = _policy_dict_at_state(self._rl_best_responses[player], state)
            
            legal_actions = state.legal_actions()
            infostate_key = state.information_state(player)
            info_vector = state.information_state_as_normalized_vector(player)
            
            avg_policy = self._avg_network.predict(np.array([info_vector]))
            legal_actions_mask = np.zeros(self._num_actions)
            legal_actions_mask[legal_actions] = 1.0
            avg_policy = np.multiply(avg_policy, legal_actions_mask)[0]
            avg_policy /= avg_policy.sum()

            
            # First traverse the subtrees.
            
            """
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]
            action, probs = self._act(info_state, legal_actions)
            """
            
            for action in legal_actions:
                assert action in br_policy
                assert action in avg_policy_tabular
                new_state = state.clone()
                new_state.apply_action(action)
                new_avg_reach = np.copy(avg_reach_probs)
                new_avg_reach[player] *= avg_policy[action]
                new_br_reach = np.copy(br_reach_probs)
                new_br_reach[player] *= br_policy[action]
                self._recursively_update_average_policies(new_state, new_avg_reach,
                                                          new_br_reach)
          # Now, do the updates.
            if infostate_key not in self._average_policy_tables[player]:
                alpha = 1 / (self._iterations + 1)
                self._average_policy_tables[player][infostate_key] = {}
#                 self._average_policy_tables_mlp[player][infostate_key] = {}
            
                pr_sum = 0.0
                
                target = np.zeros(self._num_actions)
#                 print(avg_policy)
#                 print(avg_policy_tabular)

                for action in legal_actions:
                                        
                    pr = (
                      avg_policy[action] + (alpha * br_reach_probs[player] *
                                            (br_policy[action] - avg_policy[action])) /
                      ((1.0 - alpha) * avg_reach_probs[player] +
                       alpha * br_reach_probs[player]))
                    self._average_policy_tables[player][infostate_key][action] = pr
#                     self._average_policy_tables_mlp[player][infostate_key] = pr
                    target[action] = pr

                    pr_sum += pr
                
                if player == 0:
                    self._info_sets_inputs0.append(info_vector)
                    self._info_sets_targets0.append(target)
                else:
                    self._info_sets_inputs1.append(info_vector)
                    self._info_sets_targets1.append(target)
                    
                assert (1.0 - self._delta_tolerance <= pr_sum <=
                        1.0 + self._delta_tolerance)

    
    def train_MLP(self):
        """
        make dataset of infosets -> action probs (maybe do this in the recursive part?)
        train MLP on dataset
        """
        
        infosets0 = np.array(self._info_sets_inputs0)
        infosets1 = np.array(self._info_sets_inputs1)
        targets0 = np.array(self._info_sets_targets0)
        targets1 = np.array(self._info_sets_targets1)
        inputs = np.vstack((infosets0, infosets1))
        targets = np.vstack((targets0, targets1))
        self._avg_network.fit(inputs, targets,
                    batch_size=12,
                    epochs=self._train_epochs,
                    verbose=1)
        print(self._avg_network.predict(infosets0))
        print(targets0)
        

                
    def sample_episode(self, state, policies):
        """Samples an episode according to the policies, starting from state.
        Args:
          state: Pyspiel state representing the current state.
          policies: List of policy representing the policy executed by each player.
        Returns:
          The result of the call to returns() of the final state in the episode.
            Meant to be a win/loss integer.
        """

        if state.is_terminal():
            return np.array(state.returns(), dtype=np.float32)
        elif state.is_chance_node():
            outcomes = []
            probs = []
            for action, prob in state.chance_outcomes():
                outcomes.append(action)
                probs.append(prob)
            outcome = np.random.choice(outcomes, p=probs)
            state.apply_action(outcome)
            return self.sample_episode(state, policies)
        else:
            player = state.current_player()
            state_policy = _policy_dict_at_state(policies[player], state)
            actions = []
            probs = []
            for action in state_policy:
                actions.append(action)
                probs.append(state_policy[action])
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
            return self.sample_episode(state, policies)

    def sample_episodes(self, policies, num):
        """Samples episodes and averages their returns.
        Args:
          policies: A list of policies representing the policies executed by each
            player.
          num: Number of episodes to execute to estimate average return of policies.
        Returns:
          Average episode return over num episodes.
        """

        totals = np.zeros(self._num_players)
        for _ in range(num):
            totals += self.sample_episode(self._game.new_initial_state(), policies)
        return totals / num

    def get_empirical_metagame(self, sims_per_entry, seed=None):
        """Gets a meta-game tensor of utilities from episode samples.
        The tensor is a cross-table of all the saved oracles and initial uniform
        policy.
        Args:
            sims_per_entry: number of simulations (episodes) to perform per entry in
              the tables, i.e. each is a crude Monte Carlo estimate
            seed: the seed to set for random sampling, for reproducibility
        Returns:
            the K^n (KxKx...K, with dimension n) meta-game tensor where n is the
            number of players and K is the number of strategies (one more than the
            number of iterations of fictitious play since the initial uniform
            policy is included).
        """

        if seed is not None:
            np.random.seed(seed=seed)
        assert self._oracles is not None
        num_strategies = len(self._oracles[0])
        # Each metagame will be (num_strategies)^self._num_players.
        # There are self._num_player metagames, one per player.
        meta_games = []
        for _ in range(self._num_players):
            shape = [num_strategies] * self._num_players
            meta_game = np.ndarray(shape=shape, dtype=np.float32)
            meta_games.append(meta_game)
        for coord in itertools.product(
            range(num_strategies), repeat=self._num_players):
            policies = []
            for i in range(self._num_players):
                iteration = coord[i]
                policies.append(self._oracles[i][iteration])
            utility_estimates = self.sample_episodes(policies, sims_per_entry)
            for i in range(self._num_players):
                meta_games[i][coord] = utility_estimates[i]
        return meta_games

