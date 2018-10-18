#!/usr/bin/env bash

set -e

nohup python train.py > train.log 2>&1 &