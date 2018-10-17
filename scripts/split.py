#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/10/17
"""This scripts for randomly split data into train, validation and test sets."""
import sys
import os
import random


def split(data_file, validation_ratio=0.05, test_ratio=0.05):
    data_dir = os.path.dirname(os.path.abspath(data_file))
    train_file = os.path.join(data_dir, "train.csv")
    validation_file = os.path.join(data_dir, "dev.csv")
    test_file = os.path.join(data_dir, "test.csv")
    train_cnt, dev_cnt, test_cnt = 0, 0, 0
    with open(train_file, "w") as f1, \
            open(validation_file, "w") as f2, \
                open(test_file, "w") as f3:
        for line in open(data_file):
            p = random.random()
            if p < 1 - validation_ratio - test_ratio:
                f1.writelines(line)
                train_cnt += 1
            elif 1 - validation_ratio - test_ratio <= p < 1 - test_ratio:
                f2.writelines(line)
                dev_cnt += 1
            else:
                f3.writelines(line)
                test_cnt += 1
    print("Split data into {} train samples, {} dev samples and {} test samples.\nSee results in `{}`.".format(
        train_cnt, dev_cnt, test_cnt, data_dir))


if __name__ == '__main__':
    if not (len(sys.argv) == 2 or len(sys.argv) == 4):
        exit("Usage:\n\t1. python split.py $data_file\n\t2. python split.py $data_file $valid_ratio $test_ratio")
    data_file = sys.argv[1]
    try:
        valid_ratio = float(sys.argv[2])
        test_ratio = float(sys.argv[3])
    except Exception as e:
        print(e)
        print("Can not parse argv, using defaults ratio 0.9:0.05:0.05.")
        split(data_file)
    else:
        split(data_file, valid_ratio, test_ratio)


