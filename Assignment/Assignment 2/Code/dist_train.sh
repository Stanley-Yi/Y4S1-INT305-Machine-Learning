#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python main.py --batch-size 128 --epochs 2
