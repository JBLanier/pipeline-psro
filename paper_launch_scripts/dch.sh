#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=dch
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}006
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_dch.yaml
export LOCK_SERVER_PORT=2${POKER_GAME_PORT_PREFIX}506
export LOCK_SERVER_CONFIG_PATH=${POKER_GAME_VERSION}_dch.yaml
export DASHBOARD_PORT=2${POKER_GAME_PORT_PREFIX}406
export DASHBOARD_CONFIG_PATH=${POKER_GAME_VERSION}_dch.yaml
export MAIN_WORKER_PATH=dch/sac_arch1_dch_learner.py
export SPECIAL_WORKER_PATH=
export EXTRA_UTIL_PATH=