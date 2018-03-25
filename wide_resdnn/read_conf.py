#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
"""Read All Configuration from ../conf/*.yaml"""
import os
import yaml


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'conf')
FEATURE_CONF_FILE = 'feature.basic.yaml'
MODEL_CONF_FILE = 'model.yaml'
TRAIN_CONF_FILE = 'train.yaml'


class Config(object):
    """Config class"""
    def __init__(self,
                 feature_conf_file=FEATURE_CONF_FILE,
                 model_conf_file=MODEL_CONF_FILE,
                 train_conf_file=TRAIN_CONF_FILE):
        self._feature_conf_file = os.path.join(BASE_DIR, feature_conf_file)
        self._model_conf_file = os.path.join(BASE_DIR, model_conf_file)
        self._train_conf_file = os.path.join(BASE_DIR, train_conf_file)

    @staticmethod
    def _check_feature_conf(feature, f_type, f_param):
        if f_type is None:
            raise ValueError("feature type are required in feature conf, "
                             "found empty value for feature `{}`.".format(feature))
        assert f_type in {'category', 'continuous'}, (
            "Invalid type `{}` for feature `{}` in feature conf, "
            "must be `category` or `continuous`.".format(f_type, feature))
        # check transform and parameter
        if f_type == 'category':
            hash_bucket_size, embedding_dim = f_param['hash_bucket_size'], f_param['embedding_dim']
            if hash_bucket_size is None:
                raise ValueError("Hash_bucket_size are required, "
                                 "found empty value for feature `{}` in feature conf.".format(feature))
            if not isinstance(hash_bucket_size, int):
                raise TypeError('Hash_bucket_size must be an integer, found `{}` for feature '
                                '`{}` in feature conf'.format(hash_bucket_size, feature))
            if embedding_dim:
                if not isinstance(embedding_dim, int):
                    raise TypeError('Embedding_dim must be an integer, found `{}` for feature '
                                    '`{}` in feature conf'.format(embedding_dim, feature))
        else:
            mean, std, boundaries = f_param['mean'], f_param['std'], f_param['boundaries']
            if mean and not isinstance(mean, (float, int)):
                raise TypeError('Mean must be int or float, found `{}` for feature '
                                '`{}` in feature conf.'.format(mean, feature))
            if std and (not isinstance(std, (float, int)) or std <= 0):
                    raise TypeError('Std must be a positive value, found `{}` for feature '
                                    '`{}` in feature conf.'.format(std, feature))
            if boundaries:
                if not isinstance(boundaries, (tuple, list)):
                    raise TypeError('Boundaries parameter must be a list, found `{}` for feature '
                                    '`{}` in feature conf.'.format(boundaries, feature))
                for v in boundaries:
                    if not isinstance(v, (int, float)):
                        raise TypeError('boundaries parameter element must be integer or float,'
                                        'found `{}` for feature `{}` in feature conf.'.format(boundaries, feature))

    def read_feature_conf(self):
        with open(self._feature_conf_file) as f:
            feature_conf = yaml.load(f)
            for feature, conf in feature_conf.items():
                type_, param = conf["type"], conf["parameter"]
                self._check_feature_conf(feature.lower(), type_, param)
            return feature_conf

    def _read_model_conf(self):
        with open(self._model_conf_file) as f:
            return yaml.load(f)

    def _read_train_conf(self):
        with open(self._train_conf_file) as f:
            return yaml.load(f)

    @property
    def config(self):
        return self._read_train_conf()

    @property
    def train(self):
        return self._read_train_conf()["train"]

    @property
    def distribution(self):
        return self._read_train_conf()["distribution"]

    @property
    def runconfig(self):
        return self._read_train_conf()["runconfig"]

    @property
    def model(self):
        return self._read_model_conf()

if __name__ == '__main__':
    print(Config().read_feature_conf())




