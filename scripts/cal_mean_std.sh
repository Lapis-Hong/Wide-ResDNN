#!/usr/bin/env bash
# This scripts calculate continuous features mean and standard deviation.
# Usage:
#   bash cal_mean_std.sh data_file feature_index [feature_index2]

# awk '{sum+=$NF} END {print sum, sum/NR}'
begin_t=$(date +%s)

if [ $# -eq 2 ] ;then
    awk -F '\t' '{v[NR]=$"'$2'";sum+=$"'$2'"} END {avg=sum/NR;for(n in v)sd+=(v[n]-avg)^2;sd=sqrt(sd/NR);printf("%d\t%f\t%f\n","'$2'",avg,sd)}' $1

elif [ $# -eq 3 ] ;then
    for i in $(seq $2 $3)
    do
        awk -v featureIndex=${i} -F '\t' '{v[NR]=$featureIndex;sum+=$featureIndex} END {avg=sum/NR;for(n=1;n<=NR;n++)sd+=(v[n]-avg)^2;sd=sqrt(sd/NR);printf("%d\t%f\t%f\n",featureIndex,avg,sd)}' $1
    done

else
    echo "Usage: bash cal_mean_std.sh data_file feature_index [feature_index2]"
    exit
fi

end_t=$(date +%s)
cost_t=$(($end_t - $begin_t))
echo
echo "Take $cost_t sec."
