#!/usr/bin/env bash

THIS_SCRIPTS_DIR="`dirname \"$0\"`"
cd "$THIS_SCRIPTS_DIR" || (echo "Couldn't cd into $THIS_SCRIPTS_DIR" && exit)

if [[ -z ${1+x} ]]; then echo "EXP_CONFIG_FILE (\$1) is unset" && exit 1; else echo "EXP_CONFIG_FILE is set to '$1'"; fi
if [[ -z ${2+x} ]]; then echo "SEED_NUM (\$2) is unset" && exit 1; else echo "SEED_NUM is set to '$2'"; fi
if [[ -z ${3+x} ]]; then echo "NUM_WORKERS (\$3) is unset" && exit 1; else echo "NUM_WORKERS is set to '$3'"; fi
if [[ -z ${4+x} ]]; then echo "NUM_EVAL_WORKERS (\$4) is unset" && exit 1; else echo "NUM_EVAL_WORKERS is set to '$4'"; fi
if [[ -z ${5+x} ]]; then echo "POKER_GAME_VERSION (\$5) is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '$5'"; fi


EXP_CONFIG_FILE=$1
SEED_NUM=$2
NUM_WORKERS=$3
export NUM_EVAL_WORKERS=$4
export POKER_GAME_VERSION=$5

if [[ $POKER_GAME_VERSION == "kuhn_poker" ]]; then
    export POKER_GAME_PORT_PREFIX=8;
elif [[ $POKER_GAME_VERSION == "leduc_poker" ]]; then
    export POKER_GAME_PORT_PREFIX=7;
elif [[ $POKER_GAME_VERSION == "fives" ]]; then
    export POKER_GAME_PORT_PREFIX=9;
elif [[ $POKER_GAME_VERSION == "barrage" ]]; then
    export POKER_GAME_PORT_PREFIX=6;
else
    echo "unknown poker game version: \"${POKER_GAME_VERSION}\"" && exit 1
fi

source ${EXP_CONFIG_FILE}

export CLOUD_PREFIX="${POKER_GAME_VERSION}_${ALGO_NAME}_${NUM_WORKERS}_workers_"

manager_cmd="python manager_main.py -c=launch_configs/${MANAGER_CONFIG_PATH}"
lock_server_cmd="python lock_server_main.py -c=launch_configs/${LOCK_SERVER_CONFIG_PATH}"
evaluator_cmd="python evaluator.py"

session_name="${POKER_GAME_VERSION}_${ALGO_NAME}_${SEED_NUM}"
echo "session name is ${session_name}"

tmux new-session -s "${session_name}" -n "manager" -d -c /home/deploy/population_server/population_server/manager
echo "starting manager"
tmux send-keys "${manager_cmd}" Enter

if [[ -n "${LOCK_SERVER_PORT}" ]]; then
    tmux new-window -n "locks" -c /home/deploy/population_server/population_server/lock_server
    sleep 1
    echo "starting locks"
    tmux send-keys "${lock_server_cmd}" Enter
fi

tmux new-window -n evals -c /home/deploy/partially_observable_rl/po/scripts/poker_pop_server/evaluators
sleep 1
echo "starting evals"
tmux send-keys "${evaluator_cmd}" Enter

if [[ -n "$EXTRA_UTIL_PATH" ]]; then
    tmux new-window -n "extra_util" -c /home/deploy/partially_observable_rl/po/scripts/poker_pop_server/
    sleep 1
    echo "starting ${EXTRA_UTIL_PATH}"
    tmux send-keys "python ${EXTRA_UTIL_PATH}" Enter
fi

if [[ -n "$SPECIAL_WORKER_PATH" ]]; then
    let NUM_WORKERS=NUM_WORKERS-1
    tmux new-window -n "special_worker" -c /home/deploy/partially_observable_rl/po/scripts/poker_pop_server/learners
    sleep 1
    echo "starting ${SPECIAL_WORKER_PATH}"
    tmux send-keys "python ${SPECIAL_WORKER_PATH}" Enter
fi

if [[ -n "$MAIN_WORKER_PATH" ]]; then
    for i in $( seq 1 $NUM_WORKERS )
    do
        tmux new-window -n "main_worker_${i}" -c /home/deploy/partially_observable_rl/po/scripts/poker_pop_server/learners
        sleep 1
        echo "starting ${MAIN_WORKER_PATH}"
        tmux send-keys "python ${MAIN_WORKER_PATH}" Enter
    done
fi

tmux a -t ${session_name}