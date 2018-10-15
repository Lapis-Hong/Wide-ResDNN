#!/bin/bash
# The scripts to synchronize wide and deep project between servers.
# Usage:
#		1. bash rsync.sh

#hosts=(dinghongquan@10.172.110.162 dinghongquan@10.120.180.212 dinghongquan@10.120.180.213
#dinghongquan@10.120.180.214 dinghongquan@10.120.180.215)
hosts=hongquan@202.120.45.45

cur_dir=$(cd `dirname $0`; pwd)
local_dir=`dirname ${cur_dir}`
remote_dir=/home/hongquan

for host in ${hosts[@]}
do
	rsync -ravz -e'ssh -p 10000' \
	    --exclude ".git" \
	    --exclude "model" \
	    --exclude "log" \
	    $local_dir $host:$remote_dir
done
