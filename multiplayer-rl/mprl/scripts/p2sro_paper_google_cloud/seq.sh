#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=seq
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}002
export LOCK_SERVER_PORT=
export LOCK_SERVER_CONFIG_PATH=
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_psro_sequential.yaml
export MAIN_WORKER_PATH=
export SPECIAL_WORKER_PATH=sequential_prso/sac_arch1_psro_sequential.py
export EXTRA_UTIL_PATH=