#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=subsets
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}007
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_subsets.yaml
export LOCK_SERVER_PORT=2${POKER_GAME_PORT_PREFIX}507
export LOCK_SERVER_CONFIG_PATH=${POKER_GAME_VERSION}_subsets.yaml
export DASHBOARD_PORT=2${POKER_GAME_PORT_PREFIX}407
export DASHBOARD_CONFIG_PATH=${POKER_GAME_VERSION}_subsets.yaml
export MAIN_WORKER_PATH=metanash_subset_psro/sac_arch1_psro_metanash_subset.py
export SPECIAL_WORKER_PATH=metanash_subset_psro/sac_arch1_psro_naive_parallel_psro.py
export EXTRA_UTIL_PATH=