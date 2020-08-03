from open_spiel.python.rl_environment import Environment, TimeStep, StepType


class ValidActionsMaskEnvironment(Environment):

    def step(self, actions):
        """Updates the environment according to `actions` and returns a `TimeStep`.

        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `actions`
        will be ignored.

        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `actions` will be ignored.

        Args:
          actions: a list containing one action per player, following specifications
            defined in `action_spec()`.

        Returns:
          A `TimeStep` namedtuple containing:
            observation: list of dicts containing one observations per player, each
              corresponding to `observation_spec()`.
            reward: list of rewards at this timestep, or None if step_type is
              `StepType.FIRST`.
            discount: list of discounts in the range [0, 1], or None if step_type is
              `StepType.FIRST`.
            step_type: A `StepType` value.
        """
        assert len(actions) == self.num_actions_per_step, (
            "Invalid number of actions! Expected {}".format(self.num_players))
        if self._should_reset:
            return self.reset()

        if self.is_turn_based:
            self._state.apply_action(actions[0])
        else:
            self._state.apply_actions(actions)
        self._sample_external_events()

        observations = {"info_state": [], "legal_actions": [], "current_player": []}
        rewards = []
        step_type = StepType.LAST if self._state.is_terminal() else StepType.MID
        self._should_reset = step_type == StepType.LAST

        cur_rewards = self._state.rewards()
        for player_id in range(self.num_players):
            rewards.append(cur_rewards[player_id])
            observations["info_state"].append(
                self._state.observation_as_normalized_vector(player_id) if self
                    ._use_observation else self._state
                    .information_state_as_normalized_vector(player_id))

            # CHANGED HERE TO USE LEGAL ACTIONS MASK
            observations["legal_actions"].append(self._state.legal_actions_mask(player_id))
        observations["current_player"] = self._state.current_player()

        return TimeStep(
            observations=observations,
            rewards=rewards,
            discounts=self._discounts,
            step_type=step_type)

    def reset(self):
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
          A `TimeStep` namedtuple containing:
            observations: list of dicts containing one observations per player, each
              corresponding to `observation_spec()`.
            rewards: list of rewards at this timestep, or None if step_type is
              `StepType.FIRST`.
            discounts: list of discounts in the range [0, 1], or None if step_type
              is `StepType.FIRST`.
            step_type: A `StepType` value.
        """
        self._should_reset = False
        self._state = self._game.new_initial_state()
        self._sample_external_events()

        observations = {"info_state": [], "legal_actions": [], "current_player": []}
        for player_id in range(self.num_players):
            observations["info_state"].append(
                self._state.observation_as_normalized_vector(player_id) if self
                    ._use_observation else self._state
                    .information_state_as_normalized_vector(player_id))

            # CHANGED HERE TO USE LEGAL ACTIONS MASK
            observations["legal_actions"].append(self._state.legal_actions_mask(player_id))
        observations["current_player"] = self._state.current_player()

        return TimeStep(
            observations=observations,
            rewards=None,
            discounts=None,
            step_type=StepType.FIRST)