#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""This module for building estimator for tf.estimators API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os

import numpy as np
import tensorflow as tf

from wide_resdnn.read_conf import Config
from wide_resdnn.joint import WideAndDeepClassifier


# wide columns
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
bucketized_column = tf.feature_column.bucketized_column
# deep columns
numeric_column = tf.feature_column.numeric_column
embedding_column = tf.feature_column.embedding_column


def _embed_dim(dim):
    """Empirical embedding dim"""
    return int(np.power(2, np.ceil(np.log(dim ** 0.25))))


def _build_model_columns():
    """
    Build wide and deep feature columns.
        wide_columns: category features + [discretized continuous features]
        deep_columns: continuous features + [embedded category features]
    Return: 
        _CategoricalColumn and _DenseColumn in tf.estimators API
    """
    feature_conf_dic = Config().feature
    tf.logging.info('Total feature classes: {}'.format(len(feature_conf_dic)))

    wide_columns, deep_columns = [], []
    wide_dim, deep_dim = 0, 0
    for feature, conf in feature_conf_dic.items():
        f_type, f_param = conf["type"], conf["parameter"]
        if f_type == 'category':  # category features
                hash_bucket_size, embedding_dim = f_param['hash_bucket_size'], f_param["embedding_dim"]
                col = categorical_column_with_hash_bucket(feature, hash_bucket_size=hash_bucket_size)
                wide_columns.append(col)
                wide_dim += hash_bucket_size
                if embedding_dim:  # embedding category feature for deep input
                    if isinstance(embedding_dim, str):  # auto set embedding dim
                        embedding_dim = _embed_dim(hash_bucket_size)
                    deep_columns.append(embedding_column(col, dimension=embedding_dim, combiner='mean'))
                    deep_dim += embedding_dim
        else:  # continuous features
            mean, std, boundaries = f_param["mean"] or 0, f_param["std"] or 1, f_param["boundaries"]
            col = numeric_column(feature, shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=lambda x: (x - mean) / std)
            deep_columns.append(col)
            deep_dim += 1
            if boundaries:
                wide_columns.append(bucketized_column(col, boundaries=boundaries))
                wide_dim += (len(f_param)+1)

    # add columns logging info
    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.logging.debug('Wide columns: {}'.format(col))
    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.debug('Deep columns: {}'.format(col))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))

    return wide_columns, deep_columns


def _build_distribution():
    TF_CONFIG = Config().distribution
    if TF_CONFIG["is_distribution"]:
        cluster_spec = TF_CONFIG["cluster"]
        job_name = TF_CONFIG["job_name"]
        task_index = TF_CONFIG["task_index"]
        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster_spec,
             'task': {'type': job_name, 'index': task_index}})
        run_config = tf.estimator.RunConfig()
        if job_name in ["ps", "chief", "worker"]:
            assert run_config.master == 'grpc://' + cluster_spec[job_name][task_index]  # grpc://10.120.180.212
            assert run_config.task_type == job_name
            assert run_config.task_id == task_index
            assert run_config.num_ps_replicas == len(cluster_spec["ps"])
            assert run_config.num_worker_replicas == len(cluster_spec["worker"]) + len(cluster_spec["chief"])
            assert run_config.is_chief == (job_name == "chief")
        elif job_name == "evaluator":
            assert run_config.master == ''
            assert run_config.evaluator_master == ''
            assert run_config.task_id == 0
            assert run_config.num_ps_replicas == 0
            assert run_config.num_worker_replicas == 0
            assert run_config.cluster_spec == {}
            assert run_config.task_type == 'evaluator'
            assert not run_config.is_chief


# def build_estimator(model_dir, model_type):
#     """Build an estimator appropriate for the given model type."""
#     runconfig = Config().runconfig
#     conf = Config().model
#     wide_columns, deep_columns = _build_model_columns()
#     _build_distribution()
#     # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
#     # trains faster than GPU for this model.
#     run_config = tf.estimator.RunConfig(**runconfig).replace(
#         session_config=tf.ConfigProto(device_count={'GPU': 0}))
#
#     if model_type == 'wide':
#         return tf.estimator.LinearClassifier(
#             model_dir=model_dir,
#             feature_columns=wide_columns,
#             weight_column=None,
#             optimizer=tf.train.FtrlOptimizer(
#                 learning_rate=conf["wide_learning_rate"],
#                 l1_regularization_strength=conf["wide_l1"],
#                 l2_regularization_strength=conf["wide_l2"]),
#             partitioner=None,
#             config=run_config)
#     elif model_type == 'deep':
#         return tf.estimator.DNNClassifier(
#             model_dir=model_dir,
#             feature_columns=deep_columns,
#             hidden_units=conf["hidden_units"],
#             optimizer=tf.train.ProximalAdagradOptimizer(
#                 learning_rate=conf["deep_learning_rate"],
#                 l1_regularization_strength=conf["deep_l1"],
#                 l2_regularization_strength=conf["deep_l2"]),
#             activation_fn=eval(conf["activation_function"]),
#             dropout=conf["dropout"],
#             weight_column=None,
#             input_layer_partitioner=None,
#             config=run_config)
#     else:
#         return tf.estimator.DNNLinearCombinedClassifier(
#             model_dir=model_dir,  # self._model_dir = model_dir or self._config.model_dir
#             linear_feature_columns=wide_columns,
#             linear_optimizer=tf.train.FtrlOptimizer(
#                 learning_rate=conf["wide_learning_rate"],
#                 l1_regularization_strength=conf["wide_l1"],
#                 l2_regularization_strength=conf["wide_l2"]),
#             dnn_feature_columns=deep_columns,
#             dnn_optimizer=tf.train.ProximalAdagradOptimizer(
#                 learning_rate=conf["deep_learning_rate"],
#                 l1_regularization_strength=conf["deep_l1"],
#                 l2_regularization_strength=conf["deep_l2"]),
#             dnn_hidden_units=conf["hidden_units"],
#             dnn_activation_fn=eval(conf["activation_function"]),
#             dnn_dropout=conf["dropout"],
#             n_classes=2,
#             weight_column=None,
#             label_vocabulary=None,
#             input_layer_partitioner=None,
#             config=run_config)
def _build_opt(opt, lr, l1, l2, lr_decay):
    if lr_decay:
        return lambda: opt(
            learning_rate=tf.train.exponential_decay(
                learning_rate=lr,
                global_step=tf.train.get_global_step(),
                decay_steps=10000,
                decay_rate=0.96),
            l1_regularization_strength=l1,
            l2_regularization_strength=l2)
    else:
        return opt(
            learning_rate=lr,
            l1_regularization_strength=l1,
            l2_regularization_strength=l2)


def build_estimator(model_dir, model_type):
    """Build an estimator for wide & resdnn model."""
    runconfig = Config().runconfig
    conf = Config().model
    wide_columns, deep_columns = _build_model_columns()
    _build_distribution()
    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig(**runconfig).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    # Optimizer with regularization and learning rate decay.
    wide_opt = _build_opt(
        tf.train.FtrlOptimizer, conf["wide_learning_rate"], conf["wide_l1"], conf["wide_l2"], conf["wide_lr_decay"])
    deep_opt = _build_opt(
        tf.train.ProximalAdagradOptimizer, conf["deep_learning_rate"], conf["deep_l1"], conf["deep_l2"], conf["deep_lr_decay"])

    return WideAndDeepClassifier(
        model_type=model_type,
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        linear_optimizer=wide_opt,
        dnn_feature_columns=deep_columns,
        dnn_optimizer=deep_opt,
        dnn_hidden_units=conf["hidden_units"],
        dnn_connect_mode=conf["connect_mode"],
        dnn_residual_mode=conf["residual_mode"],
        dnn_activation_fn=eval(conf["activation_function"]),
        dnn_dropout=conf["dropout"],
        dnn_batch_norm=conf["batch_normalization"],
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        input_layer_partitioner=None,
        config=run_config)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    model = build_estimator('./model', 'wide_deep')
    # print(model.config)  # <tensorflow.python.estimator.run_config.RunConfig object at 0x118de4e10>
    # print(model.model_dir)  # ./model
    # print(model.model_fn)  # <function public_model_fn at 0x118de7b18>
    # print(model.params)  # {}
    # print(model.get_variable_names())
    # print(model.get_variable_value('dnn/hiddenlayer_0/bias'))
    # print(model.latest_checkpoint())  # another 4 method is export_savedmodel,train evaluate predict
