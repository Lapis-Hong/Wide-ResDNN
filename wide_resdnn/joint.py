#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/7
"""
TensorFlow Custom Estimators for Wide and ResDNN Joint Training Models.
This is a more general and flexible framework than Wide & Deep Learning, 
which can use residual connections or arbitrary connections, even with Multi DNN 
instead of a single plain DNN.

There are two ways to build custom estimator.
    1. Write model_fn function to pass `tf.estimator.Estimator` to generate an instance.
        easier to build but with less flexibility. 
    2. Write subclass of `tf.estimator.Estimator` like premade(canned) estimators.
        much suitable for official project. 

This module is based on tf.estimator.DNNLinearCombinedClassifier.
It combines `wide`, `deep`, `wide_deep` three types model into one class 
`WideAndDeepClassifier` by argument model_type 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib

from wide_resdnn.linear import linear_learning_rate, linear_logit_fn_builder
from wide_resdnn.dnn import multidnn_logit_fn_builder
from wide_resdnn.util import add_layer_summary, _check_no_sync_replicas_optimizer, get_optimizer_instance

# The default learning rates are a historical artifact of the initial implementation.
_DNN_LEARNING_RATE = 0.001
_LINEAR_LEARNING_RATE = 0.005


def _wide_deep_combined_model_fn(
        features, labels, mode, head,
        model_type,
        linear_feature_columns=None,
        linear_optimizer='Ftrl',
        dnn_feature_columns=None,
        dnn_optimizer='Adagrad',
        dnn_hidden_units=None,
        dnn_shortcut=None,
        dnn_aggregation=None,
        dnn_activation_fn=tf.nn.relu,
        dnn_dropout=None,
        dnn_batch_norm=None,
        input_layer_partitioner=None,
        config=None):
    """Wide and Deep combined model_fn. (Linear, ResDNN)
    Args:
        features: dict of `Tensor`.
        labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
            `int32` or `int64` in the range `[0, n_classes)`.
      mode: Defines whether this is training, evaluation or prediction. See `ModeKeys`.
      head: A `Head` instance.
      model_type: one of `wide`, `deep`, `wide_deep`.
      linear_feature_columns: An iterable containing all the feature columns used
          by the Linear model.
      linear_optimizer: String, `Optimizer` object, or callable that defines the
          optimizer to use for training the Linear model. Defaults to the Ftrl
          optimizer.
      dnn_feature_columns: An iterable containing all the feature columns used by
        the DNN model.
      dnn_optimizer: String, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN model. Defaults to the Adagrad
        optimizer.
      dnn_hidden_units: List of hidden units per DNN layer, nested lists for Multi DNN.
      dnn_shortcut: String or list for connect mode, list for Multi DNN.
      dnn_aggregation: `sum` or `concat`, list for Multi DNN.
      dnn_activation_fn: Activation function applied to each DNN layer. If `None`,
          will use `tf.nn.relu`.
      dnn_dropout: When not `None`, the probability we will drop out a given DNN
          coordinate.
      dnn_batch_norm: Bool, add BN layer after each DNN layer
      input_layer_partitioner: Partitioner for input layer.
          config: `RunConfig` object to configure the runtime settings.
    Returns:
        `ModelFnOps`
    Raises:
        ValueError: If both `linear_feature_columns` and `dnn_features_columns`
            are empty at the same time, or `input_layer_partitioner` is missing,
            or features has the wrong type.
    """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = input_layer_partitioner or (
        tf.min_max_variable_partitioner(max_partitions=num_ps_replicas,
                                        min_slice_size=64 << 20))
    # Build DNN Logits.
    dnn_parent_scope = 'dnn'
    if model_type == 'wide' or not dnn_feature_columns:
        dnn_logits = None
    else:
        dnn_optimizer = get_optimizer_instance(
            dnn_optimizer, learning_rate=_DNN_LEARNING_RATE)
        if model_type == 'wide_deep':
            _check_no_sync_replicas_optimizer(dnn_optimizer)
        dnn_partitioner = tf.min_max_variable_partitioner(max_partitions=num_ps_replicas)
        with tf.variable_scope(
                dnn_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=dnn_partitioner):
            dnn_logit_fn = multidnn_logit_fn_builder(
                units=head.logits_dimension,
                hidden_units_list=dnn_hidden_units,
                shortcut_list=dnn_shortcut,
                aggregation_list=dnn_aggregation,
                feature_columns=dnn_feature_columns,
                activation_fn=dnn_activation_fn,
                dropout=dnn_dropout,
                batch_norm=dnn_batch_norm,
                input_layer_partitioner=input_layer_partitioner
            )
            dnn_logits = dnn_logit_fn(features=features, mode=mode)

    # Build Linear Logits.
    linear_parent_scope = 'linear'
    if model_type == 'deep' or not linear_feature_columns:
        linear_logits = None
    else:
        linear_optimizer = get_optimizer_instance(
            linear_optimizer, learning_rate=linear_learning_rate(len(linear_feature_columns)))
        _check_no_sync_replicas_optimizer(linear_optimizer)
        with tf.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=input_layer_partitioner) as scope:
            logit_fn = linear_logit_fn_builder(units=head.logits_dimension,
                                               feature_columns=linear_feature_columns)
            linear_logits = logit_fn(features=features)
            add_layer_summary(linear_logits, scope.name)

    # Combine logits and build full model.
    logits_combine = []
    # _BinaryLogisticHeadWithSigmoidCrossEntropyLoss, logits_dimension=1
    for logits in [dnn_logits, linear_logits]:  # shape: [batch_size, 1]
        if logits is not None:
            logits_combine.append(logits)
    logits = tf.add_n(logits_combine)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        train_ops = []
        global_step = tf.train.get_global_step()
        # BN, when training, the moving_mean and moving_variance need to be updated. By default the
        # update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if dnn_logits is not None:
                train_ops.append(
                    dnn_optimizer.minimize(
                        loss,
                        var_list=tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=dnn_parent_scope)))
            if linear_logits is not None:
                train_ops.append(
                    linear_optimizer.minimize(
                        loss,
                        var_list=tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES,
                            scope=linear_parent_scope)))
            # Create an op that groups multiple ops. When this op finishes,
            # all ops in inputs have finished. This op has no output.
            train_op = tf.group(*train_ops)
        with tf.control_dependencies([train_op]):
            # Returns a context manager that specifies an op to colocate with.
            with tf.colocate_with(global_step):
                return tf.assign_add(global_step, 1)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


class WideAndDeepClassifier(tf.estimator.Estimator):
    """An custom estimator for TensorFlow Wide and ResDNN joined classification models.
    The usage and behavior is exactly same with tf.estimator.DNNLinearCombinedClassifier.
    """

    def __init__(self,
                 model_type=None,
                 model_dir=None,
                 linear_feature_columns=None,
                 linear_optimizer='Ftrl',
                 dnn_feature_columns=None,
                 dnn_optimizer='Adagrad',
                 dnn_hidden_units=None,
                 dnn_shortcut=None,
                 dnn_aggregation=None,
                 dnn_activation_fn=tf.nn.relu,
                 dnn_dropout=None,
                 dnn_batch_norm=None,
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 input_layer_partitioner=None,
                 config=None):
        """Initializes a WideDeepCombinedClassifier instance.
        Args:
            model_dir: Directory to save model parameters, graph and etc. This can
                also be used to load checkpoints from the directory into a estimator
                to continue training a previously saved model.
            linear_feature_columns: An iterable containing all the feature columns
                used by linear part of the model. All items in the set must be
                instances of classes derived from `FeatureColumn`.
            linear_optimizer: An instance of `tf.Optimizer` used to apply gradients to
                the linear part of the model. Defaults to FTRL optimizer.
            dnn_feature_columns: An iterable containing all the feature columns used
                by deep part of the model. All items in the set must be instances of
                classes derived from `FeatureColumn`.
            dnn_optimizer: An instance of `tf.Optimizer` used to apply gradients to
                the deep part of the model. Defaults to Adagrad optimizer.
            dnn_hidden_units: List of hidden units per layer. All layers are fully
                connected.
            dnn_aggregation: `sum` or `concat` residual connections.
            dnn_activation_fn: Activation function applied to each layer. If None,
                will use `tf.nn.relu`.
            dnn_dropout: When not None, the probability we will drop out
                a given coordinate.
            n_classes: Number of label classes. Defaults to 2, namely binary
                classification. Must be > 1.
            weight_column: A string or a `_NumericColumn` created by
                `tf.feature_column.numeric_column` defining feature column representing
                weights. It is used to down weight or boost examples during training. It
                will be multiplied by the loss of the example. If it is a string, it is
                used as a key to fetch weight tensor from the `features`. If it is a
                `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
                then weight_column.normalizer_fn is applied on it to get weight tensor.
            label_vocabulary: A list of strings represents possible label values. If
                given, labels must be string type and have any value in
                `label_vocabulary`. If it is not given, that means labels are
                already encoded as integer or float within [0, 1] for `n_classes=2` and
                encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
                Also there will be errors if vocabulary is not provided and labels are
                string.
            input_layer_partitioner: Partitioner for input layer. Defaults to
                `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
            config: RunConfig object to configure the runtime settings.
        Raises:
            ValueError: If both linear_feature_columns and dnn_features_columns are
                empty at the same time.
        """
        if not linear_feature_columns and not dnn_feature_columns:
            raise ValueError('Either linear_feature_columns or dnn_feature_columns must be defined.')
        if model_type is None:
            raise ValueError("Model type must be defined. one of `wide`, `deep`, `wide_deep`.")
        else:
            assert model_type in {'wide', 'deep', 'wide_deep'}, (
                "Invalid model type, must be one of `wide`, `deep`, `wide_deep`.")
            if model_type == 'wide':
                if not linear_feature_columns:
                    raise ValueError('Linear_feature_columns must be defined for wide model.')
            elif model_type == 'deep':
                if not dnn_feature_columns:
                    raise ValueError('Dnn_feature_columns must be defined for deep model.')
        if dnn_feature_columns and not dnn_hidden_units:
            raise ValueError('dnn_hidden_units must be defined when dnn_feature_columns is specified.')

        if n_classes == 2:
            # units = 1
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)
        else:
            # units = n_classes
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
                n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)

        def _model_fn(features, labels, mode, config):
            return _wide_deep_combined_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                model_type=model_type,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                dnn_feature_columns=dnn_feature_columns,
                dnn_shortcut=dnn_shortcut,
                dnn_aggregation=dnn_aggregation,
                dnn_optimizer=dnn_optimizer,
                dnn_hidden_units=dnn_hidden_units,
                dnn_activation_fn=dnn_activation_fn,
                dnn_dropout=dnn_dropout,
                dnn_batch_norm=dnn_batch_norm,
                input_layer_partitioner=input_layer_partitioner,
                config=config)

        super(WideAndDeepClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config)

