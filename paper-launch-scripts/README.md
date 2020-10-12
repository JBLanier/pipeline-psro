# Running Poker Experiments

## Launching Experiments in Tmux sessions
To launch Pipeline PSRO on Leduc poker with 3 rl-learners and 3 evaluator-workers, run: 
```shell script
# (with your conda environment active)
cd paper-launch-scripts
./launch_exp.sh pipe.sh 1 3 3 leduc_poker
```

(More commands like this can be found in [example_commands.txt](/paper-launch-scripts/example_commands.txt))

This will launch multiple processes in a tmux session. You change between tmux windows using your mouse or with `ctrl+b, w`

You can detached from the session with `ctrl+b, d`. When detached, the processes will continue running in the background unless killed. You can reattach to the session with `tmux attach` (or `tmux attach -t <session-name>` if you are running multiple experiments).

To stop an experiment, either use `ctrl+c` to kill each process and exit every shell in the session, or detach from the session (using `ctrl+b, d`) and kill the session with `tmux kill-session -t <session-name>` (or `tmux kill-server` to kill all sessions).

Quick guides to using tmux:
- [A Quick and Easy Guide to tmux](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/)
- [Tactical tmux: The 10 Most Important Commands](https://danielmiessler.com/study/tmux/)

## Experiment Processes Breakdown
Our experiment code supports multiple algorithms for training parallel learners in 2-player games. The stack is designed such that an algorithm can be validated on toy games like Kuhn & Leduc poker on a single node and then used in large-scale games with learners spanning multiple nodes with the same code.

The basic building block processes that you may see launched for each experiment are:

#### Manager
Maintains a checkpointable payoff table data structure for keeping track of an empirical payoff matrix and metadata for frozen/finished policies. Maintains a list of offline payoff evaluations needed and provides an interface to request ad-hoc evaluations between policy checkpoints from different learners.

#### Evaluator
Launches multiple eval workers to query needed offline policy match-up payoff evaluations from the manager, run them, and report payoff results back to the manager.

#### Lock Server
Provides an interface for learners to post, mutate, and query named values in atomic operations. The lock server is used for parallel learners to coordinate and synchronize operations. 

#### Learner
Independent reinforcement learning processes. Each learner trains a single policy until completion. Learners are individually responsible for determining which other policies to train against and when to "finish" and submit final checkpoints to the manager for inclusion in the payoff matrix.

#### Dashboard
HTTP webserver to host and display statuses from the manager. After entries to emperical payoff matrix are added, displays a graph of the population metanash probabilities over time.

## Graphing Results

See [graph_results.ipyb](/paper-launch-scripts/graph_results.ipynb) for an example of parsing the payoff table checkpoints saved by experiment manager processes and graphing the exploitability over time.

When launching a new experiment, the manager will log the location where it's latest payoff table checkpoint will be saved. This can be loaded in [graph_results.ipyb](/paper-launch-scripts/graph_results.ipynb) to graph the experiment's results.

Example manager log output:
```text
INFO:mprl.utility_services.manager.manager:Latest Manager Payoff Table Checkpoint will always be at kuhn_poker_pipe_3_workers_poker_ps/kuhn_pipeline_psro/goku_pid_753661_12_09_18PM_Oct-12-2020/payoff_tables/latest.dill (local file path /home/user/git/pipeline-psro/multiplayer-rl/mprl/data/kuhn_poker_pipe_3_workers_poker_ps/kuhn_pipeline_psro/goku_pid_753661_12_09_18PM_Oct-12-2020/payoff_tables/latest.dill)
```
(Hence in this example, the payoff table checkpoint will be available at `/home/user/git/pipeline-psro/multiplayer-rl/mprl/data/kuhn_poker_pipe_3_workers_poker_ps/kuhn_pipeline_psro/goku_pid_753661_12_09_18PM_Oct-12-2020/payoff_tables/latest.dill`)