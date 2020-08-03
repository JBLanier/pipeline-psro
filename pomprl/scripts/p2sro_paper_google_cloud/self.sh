#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=self_play
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}009
export LOCK_SERVER_PORT=
export LOCK_SERVER_CONFIG_PATH=
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_self_play.yaml
export MAIN_WORKER_PATH=
export SPECIAL_WORKER_PATH=self_play/sac_arch1_self_play.py
export EXTRA_UTIL_PATH=