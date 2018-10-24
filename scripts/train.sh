#!/usr/bin/env bash

set -ex

cur_dir=$(cd `dirname $0`; pwd)
dir=`dirname ${cur_dir}`

cd ${dir}
nohup python train.py > log/train.log 2>&1 &