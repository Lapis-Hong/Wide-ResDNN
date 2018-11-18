#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module contains DNN and its variants ResDNN logit builder for deep model part.
The code is based on tf.estimator.DNNClassifier.

DNN logit builder. 
Extend DNN architecture, support several patterns of connections even arbitrary connections between layers.
Extend single DNN to Multi-DNN, which can have different architectures. DNN_1, DNN_2, ...
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import tensorflow as tf

from wide_resdnn.util import add_layer_summary

_LEARNING_RATE = 0.05


def _dnn_logit_fn(features, mode, model_id, units,
                  hidden_units, shortcut, aggregation, feature_columns,
                  activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Deep Neural Network logit_fn.
    (This is general DNN called ResDNN, which support residual connections.)
    Args:
        features: This is the first item returned from the `input_fn`
            passed to `train`, `evaluate`, and `predict`. This should be a
            single `Tensor` or `dict` of same.
        mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.
        model_id: An int indicating the model index of Multi DNN.
        units: An int indicating the dimension of the logit layer.  In the
            MultiHead case, this should be the sum of all component Heads' logit
            dimensions.
        hidden_units: Iterable of integer number of hidden units per layer.
        shortcut: String or list, support several patterns.
            `normal`: use normal DNN with no residual connections
            `first_dense`: add addition connections from first input layer to all hidden layers.
            `last_dense`: add addition connections from all previous layers to last layer.
            `dense`: add addition connections between all layers, similar to DenseNet.
            `resnet`: add addition connections between adjacent layers, similar to ResNet.
            arbitrary connections string: add addition connections between layer0 to layer1 like '01', separated by comma
                eg: '01,03,12'  index start from zero (input_layer), max index is len(hidden_units), smaller index first.
        aggregation: `sum` or `concat`, `sum` can only used for same hidden size architecture.
        feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
        activation_fn: Activation function applied to each layer.
        dropout: When not `None`, the probability we will drop out a given coordinate.
        batch_norm: Bool, Whether to use BN in dnn.
        input_layer_partitioner: Partitioner for input layer.
    Returns:
        A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    Raises:
        AssertError: If aggregation is not one of `sum` or `concat`;
        ValueError: If aggregation is `sum` and hidden_units is different;
        AssertError: If shortcut is string, but not one of `first_dense`, `last_dense`, `dense` or `resnet`;
    """
    aggregation = aggregation or "concat"  # default use concat
    assert aggregation in {"sum", "concat"}, "Invalid aggregation: {}".format(aggregation)
    if len(set(hidden_units)) != 1 and aggregation == "sum":
        raise ValueError("Can not set `sum` aggregation for different hidden units.")
    if shortcut[0].isalpha():
        assert shortcut in {'normal', 'first_dense', 'last_dense', 'dense', 'resnet'}, (
            'Invalid connect mode: {}'.format(shortcut))

    def residual_fn(nets, mode):
        if mode == "sum":
            return tf.add_n(nets)
            # return tf.reduce_prod(nets, axis=0)
        elif mode == "concat":
            return tf.concat(nets, axis=1)

    with tf.variable_scope(
            'input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner,
            reuse=tf.AUTO_REUSE):
        net = tf.feature_column.input_layer(
            features=features,
            feature_columns=feature_columns)
    layers = [net]

    # `sum` mode residual connection start from 1st hidden layer, not input layer (different dim).
    if shortcut != "normal" and aggregation == "sum":
        with tf.variable_scope(
                'dnn_{}/hiddenlayer_{}'.format(model_id, 1), values=(net,)) as hidden_layer_scope:
            net = tf.layers.dense(
                net,
                units=hidden_units[0],
                activation=activation_fn,
                kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                name=hidden_layer_scope)
            add_layer_summary(net, hidden_layer_scope.name)
            # Add dropout and BN.
            if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                net = tf.layers.dropout(net, rate=dropout, training=True)  # dropout rate
            if batch_norm:
                net = tf.layers.batch_normalization(net)
        hidden_units = hidden_units[1:]
        layers = [net]  # 1st hidden layer

    for layer_id, num_hidden_units in enumerate(hidden_units):
        layer_id += (aggregation == "sum")  # If sum mode, layer_id += 1
        with tf.variable_scope(
                'dnn_{}/hiddenlayer_{}'.format(model_id, layer_id+1), values=(net,)) as hidden_layer_scope:
            net = tf.layers.dense(
                net,
                units=num_hidden_units,
                activation=activation_fn,
                kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                name=hidden_layer_scope)

            # This is ResDNN architecture.
            if shortcut != 'normal':
                if shortcut == 'first_dense':
                    # This connect input features to each DNN layers
                    net = residual_fn([net, layers[0]], aggregation)
                elif shortcut == 'last_dense':
                    if layer_id == len(hidden_units) - 1:
                        # This concat each DNN layers to final feature layer.
                        net = residual_fn([net]+layers, aggregation)
                elif shortcut == 'dense':
                    # This concat all layers between each other.
                    net = residual_fn([net]+layers, aggregation)
                elif shortcut == 'resnet':
                    # This concat previous layers in turn. eg: 0-1, 1-2, 2-3, ...
                    net = residual_fn([net, layers[-1]], aggregation)
                else:
                    # This is arbitrary connections, '01,03,13', small index layer first
                    # map each layer index to its previous connect layer index: {1: [0], 3: [0, 1]}
                    connect_pair = shortcut.split(",")
                    connect_map = {}
                    for i, j in connect_pair:  # ['01','03','13']
                        if int(j) not in connect_map:
                            connect_map[int(j)] = [int(i)]
                        else:
                            connect_map[j].append(int(i))
                    if layer_id + 1 in connect_map:
                        previous_layers = [net for idx, net in enumerate(layers) if idx in connect_map[layer_id + 1]]
                    else:
                        previous_layers = []
                    net = residual_fn([net]+previous_layers, aggregation)

                layers.append(net)

            # Add dropout and BN.
            if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                net = tf.layers.dropout(net, rate=dropout, training=True)  # dropout rate
            if batch_norm:
                net = tf.layers.batch_normalization(net)  # add bn layer, it has been added in high version tf
        add_layer_summary(net, hidden_layer_scope.name)

    with tf.variable_scope('dnn_{}/logits'.format(model_id), values=(net,)) as logits_scope:
        logits = tf.layers.dense(
                net,
                units=units,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)
    add_layer_summary(logits, logits_scope.name)

    return logits


def multidnn_logit_fn_builder(units, hidden_units_list, shortcut_list, aggregation_list,
                              feature_columns, activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Multi DNN logit function builder.
    Args:
        hidden_units_list: 1D iterable obj for single DNN or 2D for Multi DNN.
            eg: [128, 64, 32] or [[128, 64, 32], [64, 32]]
        shortcut_list: iterable obj of {`normal`, `first_dense`, `last_dense`, `dense`, `resnet`, or connect string} 
            consistent with above hidden_units_list. 
            eg: `normal` or [`normal`, `first_dense`]
        aggregation_list: iterable obj of {`sum`, `concat`}
    Returns:
        Multi DNN logit fn.
    """
    if not isinstance(units, int):
        raise ValueError('units must be an int. Given type: {}'.format(type(units)))

    # Compatible for single DNN input hidden_units, shortcut, aggregation
    if not isinstance(hidden_units_list[0], (list, tuple)):
        hidden_units_list = [hidden_units_list]
    if isinstance(shortcut_list, basestring) or shortcut_list is None:
        shortcut_list = [shortcut_list]
    if isinstance(aggregation_list, basestring) or aggregation_list is None:
        aggregation_list = [aggregation_list]

    assert len(hidden_units_list) == len(shortcut_list) == len(aggregation_list), (
        "Hidden units, shortcut and aggregation must have same length.")

    def multidnn_logit_fn(features, mode):
        logits = []
        for idx, (hidden_units, shortcut, aggregation) in enumerate(
                zip(hidden_units_list, shortcut_list, aggregation_list)):
            logits.append(
                _dnn_logit_fn(
                    features,
                    mode,
                    idx + 1,
                    units,
                    hidden_units,
                    shortcut,
                    aggregation,
                    feature_columns,
                    activation_fn,
                    dropout,
                    batch_norm,
                    input_layer_partitioner))
        logits = tf.add_n(logits)  # Adds all input tensors element-wise.

        return logits

    return multidnn_logit_fn
