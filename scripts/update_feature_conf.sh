#!/usr/bin/env bash
# The scripts proviede pipeline for update feature conf.
set -ex

cur_dir=$(cd `dirname $0`; pwd)
conf_dir=`dirname ${cur_dir}`/conf
data_file=`dirname ${cur_dir}`/data/criteo/train

bash cal_feat_val_cnt.sh ${data_file} > ${conf_dir}/feat_val_cnt.txt
bash cal_mean_std.sh ${data_file} 2 14 > ${conf_dir}/feat_mean_std.txt

python update_feature_conf.py

