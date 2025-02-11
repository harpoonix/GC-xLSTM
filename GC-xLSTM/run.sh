#!/bin/bash
export PYTHONPATH=../

if [ -z "$1" ]; then
    echo "Error: CONFIG_FILE argument is required."
    exit 1
fi

CONFIG_FILE=$1
GPU=${2:-0}

CUDA_VISIBLE_DEVICES=$GPU python xlstm_neural_gc.py --config "configs/$CONFIG_FILE"
