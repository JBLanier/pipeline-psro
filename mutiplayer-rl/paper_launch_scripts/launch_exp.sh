#!/usr/bin/env bash

# We will be launching multiple tmux windows which may not by default be using the same python environment as the current shell.
PYTHON_EXECUTABLE="$(command -v python)"
echo "Using current python executable at ${PYTHON_EXECUTABLE}"

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

echo "sourcing experiment config file at ${EXP_CONFIG_FILE}"
source ${EXP_CONFIG_FILE}

export CLOUD_PREFIX="${POKER_GAME_VERSION}_${ALGO_NAME}_${NUM_WORKERS}_workers_"

MPRL_BASE_DIR=$(${PYTHON_EXECUTABLE} -c "import os, mprl; print(os.path.dirname(mprl.__file__))")
echo "Using mprl base directory at ${MPRL_BASE_DIR}"

LEARNERS_DIR="${MPRL_BASE_DIR}/scripts/poker_parallel_algos/learners"

MANAGER_CMD="${PYTHON_EXECUTABLE} manager_main.py -c=launch_configs/${MANAGER_CONFIG_PATH}"
MANAGER_DIR="${MPRL_BASE_DIR}/utility_services/manager"

LOCK_SERVER_CMD="${PYTHON_EXECUTABLE} lock_server_main.py -c=launch_configs/${LOCK_SERVER_CONFIG_PATH}"
LOCK_SERVER_DIR="${MPRL_BASE_DIR}/utility_services/lock_server"

DASHBOARD_CMD="${PYTHON_EXECUTABLE} dashboard_main.py -c=launch_configs/${DASHBOARD_CONFIG_PATH}"
DASHBOARD_DIR="${MPRL_BASE_DIR}/utility_services/dashboard"

EVALUATOR_LOCK_SERVER_CMD="${PYTHON_EXECUTABLE} evaluator.py"
EVALUATOR_DIR="${MPRL_BASE_DIR}/scripts/poker_parallel_algos/evaluators"

export TUNE_SAVE_DIR="${MPRL_BASE_DIR}/logs/ray_results"
echo "Set tune local save dir to ${TUNE_SAVE_DIR}"

session_name="${POKER_GAME_VERSION}_${ALGO_NAME}_${SEED_NUM}"
echo "tmux session name is ${session_name}"

tmux new-session -s "${session_name}" -n "manager" -d -c "${MANAGER_DIR}"
echo "starting manager"
tmux send-keys "${MANAGER_CMD}" Enter

if [[ -n "${LOCK_SERVER_PORT}" ]]; then
    tmux new-window -n "locks" -c "${LOCK_SERVER_DIR}"
    sleep 1
    echo "starting locks"
    tmux send-keys "${LOCK_SERVER_CMD}" Enter
fi

tmux new-window -n evals -c "${EVALUATOR_DIR}"
sleep 1
echo "starting evals"
tmux send-keys "${EVALUATOR_LOCK_SERVER_CMD}" Enter

if [[ -n "$EXTRA_UTIL_PATH" ]]; then
    tmux new-window -n "extra_util" -c "${MPRL_BASE_DIR}/scripts/poker_parallel_algos"
    sleep 1
    echo "starting ${EXTRA_UTIL_PATH}"
    tmux send-keys "${PYTHON_EXECUTABLE} ${EXTRA_UTIL_PATH}" Enter
fi

if [[ -n "$SPECIAL_WORKER_PATH" ]]; then
    let NUM_WORKERS=NUM_WORKERS-1
    tmux new-window -n "special_rl_worker" -c "${LEARNERS_DIR}"
    sleep 1
    echo "starting ${SPECIAL_WORKER_PATH}"
    tmux send-keys "${PYTHON_EXECUTABLE} ${SPECIAL_WORKER_PATH}" Enter
fi

if [[ -n "$MAIN_WORKER_PATH" ]]; then
    for i in $( seq 1 $NUM_WORKERS )
    do
        tmux new-window -n "main_rl_worker_${i}" -c "${LEARNERS_DIR}"
        sleep 1
        echo "starting ${MAIN_WORKER_PATH}"
        tmux send-keys "${PYTHON_EXECUTABLE} ${MAIN_WORKER_PATH}" Enter
    done
fi

tmux new-window -n dashboard -c "${DASHBOARD_DIR}"
sleep 1
echo "starting dashboard"
tmux send-keys "${DASHBOARD_CMD}" Enter

tmux a -t ${session_name}