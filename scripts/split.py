#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/10/17
"""This script for randomly split train data into train and test sets.
Note here the test set actually means the validation set
We usually split train data into train set and dev set, and with another test data."""

import sys
import os
import random


def split(data_file, train_ratio=0.9):
    data_dir = os.path.dirname(os.path.abspath(data_file))
    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "dev.csv")
    train_cnt, test_cnt = 0, 0
    with open(train_file, "w") as f1, \
                open(test_file, "w") as f2:
        for line in open(data_file):
            p = random.random()
            if p < train_ratio:
                f1.writelines(line)
                train_cnt += 1
            else:
                f2.writelines(line)
                test_cnt += 1
    print("Split data into {} train samples and {} validation samples.\nSee results in `{}`.".format(
        train_cnt, test_cnt, data_dir))


if __name__ == '__main__':
    if not (len(sys.argv) == 2 or len(sys.argv) == 3):
        exit("Usage:\n\t1. python split.py $data_file\n\t2. python split.py $data_file $train_ratio")
    data_file = sys.argv[1]
    try:
        train_ratio = float(sys.argv[2])
    except Exception as e:
        print("Using defaults train:dev=0.9:0.1.")
        split(data_file)
    else:
        split(data_file, train_ratio)


