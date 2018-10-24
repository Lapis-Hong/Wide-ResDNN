#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/10/18
"""This scripts for randomly sample data."""
import sys
import os
import random


def sample(data_file, ratio):
    data_dir = os.path.dirname(os.path.abspath(data_file))
    sample_file = os.path.join(data_dir, "sample")
    cnt = 0
    with open(sample_file, "w") as f1:
        for line in open(data_file):
            p = random.random()
            if p < ratio:
                f1.writelines(line)
                cnt += 1
    print("Sample {} samples from `{}`.\nSee results in `{}`.".format(cnt, data_file, sample_file))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        exit("Usage:\n\tpython sample.py $data_file $sample_ratio")
    data_file = sys.argv[1]
    try:
        sample_ratio = float(sys.argv[2])
    except Exception as e:
        print("Can not parse argv `{}`, {}".format(sys.argv[2], e))
    else:
        sample(data_file, sample_ratio)