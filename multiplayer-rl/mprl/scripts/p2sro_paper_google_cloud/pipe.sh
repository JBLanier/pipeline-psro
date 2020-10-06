#!/usr/bin/env bash

if [[ -z ${POKER_GAME_VERSION+x} ]]; then echo "POKER_GAME_VERSION is unset" && exit 1; else echo "POKER_GAME_VERSION is set to '${POKER_GAME_VERSION}'"; fi
if [[ -z ${POKER_GAME_PORT_PREFIX+x} ]]; then echo "POKER_GAME_PORT_PREFIX is unset" && exit 1; else echo "POKER_GAME_PORT_PREFIX is set to '${POKER_GAME_PORT_PREFIX}'"; fi


export ALGO_NAME=pipe
export MANAGER_PORT=2${POKER_GAME_PORT_PREFIX}003
export LOCK_SERVER_PORT=2${POKER_GAME_PORT_PREFIX}503
export LOCK_SERVER_CONFIG_PATH=${POKER_GAME_VERSION}_pipeline_psro.yaml
export MANAGER_CONFIG_PATH=poker/${POKER_GAME_VERSION}_pipeline_psro.yaml
export MAIN_WORKER_PATH=pipeline_psro/sac_arch1_pipeline_psro_learner.py
export SPECIAL_WORKER_PATH=
export EXTRA_UTIL_PATH=