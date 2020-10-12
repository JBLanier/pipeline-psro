[barrage_agent_interface.py](/multiplayer-rl/mprl/barrage_agent_api/barrage_agent_interface.py)
\- A single player gym-esque interface for playing games against the P2SRO Barrage agent. A usage example is included in this file's \_\_main\_\_ and can be run with:
```shell script
python barrage_agent_interface.py
```
The code is densely commented to explain how to interact with the agent through example.

[create_initial_state.py](/multiplayer-rl/mprl/barrage_agent_api/create_initial_state.py) \- A utility function to specify starting piece positions for the outside (non-paper) agent in each game. The P2SRO agent uses random human piece initializations downloaded from the [Gravon Archive](Gravon Archive (https://www.gravon.de/gravon/stratego/strados2.jsp)). Using this function is optional, and you can avoid it if you also want to use random human inits.

[p2sro_agent_wrapper.py](/multiplayer-rl/mprl/barrage_agent_api/p2sro_agent_wrapper.py) \- A wrapper over the P2SRO agent (comprising of a metanash population of policies, with a new one selected each game). Provides `get_action` and `sample_new_policy_from_metanash` methods.

The stratego environment enforces the two-squares rule, but the more-squares rule is currently not enforced.