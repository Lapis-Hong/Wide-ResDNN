#!/usr/bin/env bash
# The scripts calculate each feature distinct value count.
# Usage:
#   bash cal_feat_val_cnt.sh data_file
file=$1

if [ $# -ne 1 ] ;then
    echo 'Usage: bash cal_feature_value_cnt.sh data_file';
    exit 0
fi

num_features=`head -1 ${file} | awk -F '\t' '{print NF}'`

#for i in $(seq 1 ${num_features})
#for i in {1..39}
for((i=1;i<=${num_features};i++));
do
    echo -e "${i}\t`cut -f ${i} ${file} | sort -u|wc -l`"
done
