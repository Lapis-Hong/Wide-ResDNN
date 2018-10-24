#!/usr/bin/env bash

set -ex

cur_dir=$(cd `dirname $0`; pwd)
dir=`dirname ${cur_dir}`

cd ${dir}
nohup python test.py > log/test.log 2>&1 &