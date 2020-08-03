#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=rect
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}005
export LOCK_SERVER_PORT=2${POKER_GAME_PORT_PREFIX}505
export LOCK_SERVER_CONFIG_PATH="${POKER_GAME_VERSION}_rectified_psro.yaml"
export MANAGER_CONFIG_PATH="poker/${POKER_GAME_VERSION}_psro_rectified.yaml"
export MAIN_WORKER_PATH=rectified_psro/sac_arch1_rectified_psro_learner.py
export SPECIAL_WORKER_PATH=
export EXTRA_UTIL_PATH=learners/rectified_psro/rectified_psro_job_scheduler.py