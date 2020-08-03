#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=singles
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}008
export LOCK_SERVER_PORT=2${POKER_GAME_PORT_PREFIX}508
export LOCK_SERVER_CONFIG_PATH=${POKER_GAME_VERSION}_singles.yaml
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_single_policies.yaml
export MAIN_WORKER_PATH=single_pol_exploiter_psro/sac_arch1_psro_single_policy_exploit.py
export SPECIAL_WORKER_PATH=single_pol_exploiter_psro/sac_arch1_psro_naive_parallel_psro.py
export EXTRA_UTIL_PATH=