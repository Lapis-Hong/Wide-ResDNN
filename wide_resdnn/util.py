#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/2
"""Provide some utility function."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import six
from functools import wraps

import tensorflow as tf


def timer(info=''):
    """parameter decarotor"""
    def _timer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):  # passing params to func
            s_time = time.time()
            func(*args, **kwargs)
            e_time = time.time()
            period = (e_time-s_time) / 60.0
            print(info + '-> elapsed time: %.3f minutes' % period)
        return wrapper
    return _timer


def elapse_time(start_time):
    return round((time.time()-start_time) / 60)


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def _check_no_sync_replicas_optimizer(optimizer):
    if isinstance(optimizer, tf.train.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear model.')


def get_optimizer_instance(opt, learning_rate=None):
    """Returns an optimizer instance.
    Supports the following types for the given `opt`:
        * An `Optimizer` instance: Returns the given `opt`.
        * A string: Creates an `Optimizer` subclass with the given `learning_rate`.
      Supported strings:
        * 'Adagrad': Returns an `AdagradOptimizer`.
        * 'Adam': Returns an `AdamOptimizer`.
        * 'Ftrl': Returns an `FtrlOptimizer`.
        * 'RMSProp': Returns an `RMSPropOptimizer`.
        * 'SGD': Returns a `GradientDescentOptimizer`.
    Args:
      opt: An `Optimizer` instance, or string, as discussed above.
      learning_rate: A float. Only used if `opt` is a string.
    Returns:
      An `Optimizer` instance.
    Raises:
      ValueError: If `opt` is an unsupported string.
      ValueError: If `opt` is a supported string but `learning_rate` was not specified.
      ValueError: If `opt` is none of the above types.
    """
    """Methods related to optimizers used in canned_estimators."""
    _OPTIMIZER_CLS_NAMES = {
        'Adagrad': tf.train.AdagradOptimizer,
        'Adam': tf.train.AdamOptimizer,
        'Ftrl': tf.train.FtrlOptimizer,
        'RMSProp': tf.train.RMSPropOptimizer,
        'SGD': tf.train.GradientDescentOptimizer
    }
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
            if not learning_rate:
                raise ValueError('learning_rate must be specified when opt is string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
        raise ValueError('Unsupported optimizer name: {}. Supported names are: {}'.format(
            opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))
    if not isinstance(opt, tf.train.Optimizer):
        raise ValueError('The given object is not an Optimizer instance. Given: {}'.format(opt))
    return opt
