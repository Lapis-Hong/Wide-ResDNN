#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/3/25
"""Deprecated
Auto update feature dynamic config for feature.advance.yaml using feat_val_cnt.txt and mean_std.txt results"""
from __future__ import unicode_literals

import os
import yaml

conf_dir = '../conf/criteo'

feat_val_cnt_file = os.path.join(conf_dir, 'feat_val_cnt.txt')
feat_mean_std_file = os.path.join(conf_dir, 'feat_mean_std.txt')


def size2hash_bucket_size(feature_size):
    """Category feature size to hash_bucket_size
    Args:
        feature_size: category feature distinct value count
    Returns:
        hash_bucket_size
    """
    def up_round(value):
        """ Up round an Int to its head number+1 and padding with zero.
        >>> up_round(1234)
        2000
        >>> up_round(65)
        70
        """
        num_len = len(str(value))
        head_num = int(str(value)[0])
        return int(str(head_num+1)+'0'*(num_len-1))

    if feature_size <= 10:
        return up_round(4*feature_size)
    elif 10 < feature_size <= 1000:
        return up_round(3*feature_size)
    else:
        return up_round(2*feature_size)


def read_feat_val_cnt():
    feat_val_cnt_dict = {}
    for nu, line in enumerate(open(feat_val_cnt_file)):
        val = line.strip()
        feat_val_cnt_dict['f{}'.format(nu)] = int(val)
    return feat_val_cnt_dict


def read_feat_mean_std():
    feat_mean_std_dict = {}
    for line in open(feat_mean_std_file):
        f, mean, std = line.strip().split('\t')
        feat_mean_std_dict[f] = {'mean': round(float(mean), 2), 'std': round(float(std), 2)}
    return feat_mean_std_dict


def update_feature_conf():
    with open(os.path.join(conf_dir, 'feature.yaml'), 'r') as fi, open(os.path.join(conf_dir, 'feature.advance.yaml'), 'w') as fo:
        feature_conf = yaml.load(fi)
        print(feature_conf)
        feat_val_cnt_dict = read_feat_val_cnt()
        feat_mean_std_dict = read_feat_mean_std()

        for feat, conf in feature_conf.items():
            if conf["type"] == 'continuous':
                if feat in feat_mean_std_dict:
                    feature_conf[feat]['parameter']['mean'] = feat_mean_std_dict[feat]['mean']
                    feature_conf[feat]['parameter']['std'] = feat_mean_std_dict[feat]['std']
            else:
                if feat in feat_val_cnt_dict:
                    feature_conf[feat]['parameter']['hash_bucket_size'] = size2hash_bucket_size(feat_val_cnt_dict[feat])
                    feature_conf[feat]['parameter']['embedding_dim'] = 'auto'
        # feature_conf = OrderedDict(sorted(feature_conf.items(), key=lambda d: int(d[0][1:])))
        yaml.dump(feature_conf, fo, default_flow_style=False)

if __name__ == '__main__':
    update_feature_conf()