#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=naive
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}004
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_psro_naive.yaml
export LOCK_SERVER_PORT=
export LOCK_SERVER_CONFIG_PATH=
export DASHBOARD_PORT=2${POKER_GAME_PORT_PREFIX}404
export DASHBOARD_CONFIG_PATH=${POKER_GAME_VERSION}_psro_naive.yaml
export MAIN_WORKER_PATH=naive_parallel_psro/sac_arch1_psro_naive_parallel_psro.py
export SPECIAL_WORKER_PATH=
export EXTRA_UTIL_PATH=