#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module is based on tf.estimator.DNNClassifier.
Dnn logits builder. 
Extend dnn architecture, add BN layer, add arbitrary connections between layers.
Extend dnn to multi joint dnn.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from wide_resdnn.util import add_layer_summary, get_optimizer_instance

_LEARNING_RATE = 0.001  # 0.05


def _dnn_logit_fn(features, mode, model_id, units,
                  hidden_units, connected_mode, feature_columns,
                  activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Deep Neural Network logit_fn.
    Args:
        features: This is the first item returned from the `input_fn`
            passed to `train`, `evaluate`, and `predict`. This should be a
            single `Tensor` or `dict` of same.
        mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.
        model_id: An int indicating the model index of multi dnn.
        units: An int indicating the dimension of the logit layer.  In the
            MultiHead case, this should be the sum of all component Heads' logit
            dimensions.
        hidden_units: Iterable of integer number of hidden units per layer.
        connected_mode: one of {`simple`, `first_dense`, `last_dense`, `dense`, `resnet`}
            or arbitrary connections index tuples.
            1. `simple`: normal dnn architecture.
            2. `first_dense`: add addition connections from first input layer to all hidden layers.
            3. `last_dense`: add addition connections from all previous layers to last layer.
            4. `dense`: add addition connections between all layers, similar to DenseNet.
            5. `resnet`: add addition connections between adjacent layers, similar to ResNet.
            6. arbitrary connections list: add addition connections from layer_0 to layer_1 like 0-1.
                eg: [0-1,0-3,1-2]  index start from zero (input_layer), max index is len(hidden_units), smaller index first.
        feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
        activation_fn: Activation function applied to each layer.
        dropout: When not `None`, the probability we will drop out a given coordinate.
        batch_norm: Bool, Whether to use BN in dnn.
        input_layer_partitioner: Partitioner for input layer.
    Returns:
        A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    Raises:
        AssertError: If connected_mode is string, but not one of `simple`, `first_dense`, `last_dense`, 
            `dense` or `resnet`
    """
    if isinstance(connected_mode, str):
        assert connected_mode in {'simple', 'first_dense', 'lase_dense', 'dense', 'resnet'}, (
            'Invalid connected_mode: {}'.format(connected_mode)
        )
    with tf.variable_scope(
            'input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner,
            reuse=tf.AUTO_REUSE):
        net = tf.feature_column.input_layer(
            features=features,
            feature_columns=feature_columns)
    input_layer = net
    if connected_mode == 'simple':
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'first_dense':
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net = tf.concat([net, input_layer], axis=1)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'last_dense':
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)
        net = tf.concat(net_collections, axis=1)  # Concatenates the list of tensors `values` along dimension `axis`

    elif connected_mode == 'dense':
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)  # rate=0.1 would drop out 10% of input units.
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net_collections.append(net)
                net = tf.concat(net_collections, axis=1)
            add_layer_summary(net, hidden_layer_scope.name)

    elif connected_mode == 'resnet':  # connect layers in turn 0-1; 1-2; 2-3;
        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                net = tf.concat([net, net_collections[layer_id + 1 - 1]], axis=1)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)

    else:  # arbitrary connections, ['0-1','0-3','1-3'], small index layer first
        connected_mode = [map(int, s.split('-')) for s in connected_mode]
        # map each layer index to its early connected layer index: {1: [0], 2: [1], 3: [0]}
        connected_mapping = {}
        for i, j in connected_mode:
            if j not in connected_mapping:
                connected_mapping[j] = [i]
            else:
                connected_mapping[j] = connected_mapping[j].append(i)

        net_collections = [input_layer]
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with tf.variable_scope('dnn_{}/hiddenlayer_{}'.format(model_id, layer_id),
                    values=(net,)) as hidden_layer_scope:
                net = tf.layers.dense(
                    net,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                    name=hidden_layer_scope)
                if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                    net = tf.layers.dropout(net, rate=dropout, training=True)
                if batch_norm:
                    net = tf.layers.batch_normalization(net)
                connect_net_collections = [net for idx, net in enumerate(net_collections) if idx in connected_mapping[layer_id + 1]]
                connect_net_collections.append(net)
                net = tf.concat(connect_net_collections, axis=1)
                net_collections.append(net)
            add_layer_summary(net, hidden_layer_scope.name)

    with tf.variable_scope('dnn_{}/logits'.format(model_id), values=(net,)) as logits_scope:
        logits = tf.layers.dense(
                net,
                units=units,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)
    add_layer_summary(logits, logits_scope.name)
    return logits


def multidnn_logit_fn_builder(units, hidden_units_list,
                               connected_mode_list, feature_columns, activation_fn,
                               dropout, batch_norm, input_layer_partitioner):
    """Multi dnn logit function builder.
    Args:
        hidden_units_list: 1D iterable list for single dnn or 2D for multi dnn.
            if use single format, default to use same hidden_units in all multi dnn.
            eg: [128, 64, 32] or [[128, 64, 32], [64, 32]]
        connected_mode_list: iterable list of {`simple`, `first_dense`, `last_dense`, `dense`, `resnet`} 
            consistent with above hidden_units_list. 
            if use single format, default to use same connected_mode in all multi dnn.
            eg: `simple` or [`simple`, `first_dense`] or [0-1, 0-3] or [[0-1, 0-3], [0-1]]
    Returns:
        multidnn logit fn.
    """
    if not isinstance(units, int):
        raise ValueError('units must be an int. Given type: {}'.format(type(units)))
    if not isinstance(hidden_units_list[0], (list, tuple)):
        hidden_units_list = [hidden_units_list]  # compatible for single dnn input hidden_units
        # raise ValueError('multi dnn hidden_units must be a 2D list or tuple. Given: {}'.format(hidden_units_list))
    if isinstance(connected_mode_list, str) or \
            (isinstance(connected_mode_list[0], str) and len(connected_mode_list[0]) == 3):  # `simple`
        connected_mode_list = [connected_mode_list] * len(hidden_units_list)

    def multidnn_logit_fn(features, mode):
        logits = []
        for idx, (hidden_units, connected_mode) in enumerate(zip(hidden_units_list, connected_mode_list)):
            logits.append(
                _dnn_logit_fn(
                    features,
                    mode,
                    idx + 1,
                    units,
                    hidden_units,
                    connected_mode,
                    feature_columns,
                    activation_fn,
                    dropout,
                    batch_norm,
                    input_layer_partitioner))
        logits = tf.add_n(logits)  # Adds all input tensors element-wise.
        return logits
    return multidnn_logit_fn
