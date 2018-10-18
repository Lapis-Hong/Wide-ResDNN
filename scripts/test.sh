#!/usr/bin/env bash

set -e

nohup python test.py > test.log 2>&1 &