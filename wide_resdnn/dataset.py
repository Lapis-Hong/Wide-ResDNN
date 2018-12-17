#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def _csv_column_defaults(conf):
    """Parse csv columns name and defaults from config.
    Returns:
        (column_names, column_defaults) 
    """
    schema = conf.schema
    feature_conf = conf.feature

    column_names, column_defaults = schema.values(), []
    # column_names,  column_defaults = ["clicked"], [[0]]
    for f in column_names:
        if f == "label":
            column_defaults.append([0])  # label
        else:
            if feature_conf[f]["type"] == "continuous":
                column_defaults.append([0.0])  # continuous features
            else:
                column_defaults.append([''])  # category features
    return column_names, column_defaults


def input_fn(conf, data_file, num_epochs, batch_size, shuffle=True, shuffle_buffer_size=None):
    """Input function for train or evaluation.
    Args:
        conf: Config instance.
        data_file: can be both file or directory.
    Returns:
        (features, label) 
        `features` is a dictionary in which each value is a batch of values for
        that feature; `labels` is a batch of labels.
    """
    # check file exsits
    assert tf.gfile.Exists(data_file), (
        'train or test data file not found. Please make sure you have either '
        'default data_file or set both arguments --train_data and --test_data.')

    column_names, column_defaults = _csv_column_defaults(conf)

    def _parse_csv(value):  # value: Tensor("arg0:0", shape=(), dtype=string)
        tf.logging.info('Parsing input files: {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=column_defaults, field_delim=conf.train["field_delim"], na_value='')
        features = dict(zip(column_names, columns))
        labels = features.pop('label')
        return features, tf.equal(labels, 1)

    # Extract lines from input files using the Dataset API.
    if conf.train["skip_lines"]:
        dataset = tf.data.TextLineDataset(data_file).skip(conf.train["skip_lines"])
    else:
        dataset = tf.data.TextLineDataset(data_file)
    # if self.is_distribution:  # allows each worker to read a unique subset.
    #     dataset = dataset.shard(self.num_workers, self.worker_index)
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    if shuffle:
        shuffle_buffer_size = shuffle_buffer_size or batch_size * 1000  # Why it determines the train speed ???
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=123)  # set of Tensor object
    # Call repeat after shuffling, to prevent separate epochs from blending together.
    dataset = dataset.map(_parse_csv, num_parallel_calls=4).repeat(num_epochs).batch(batch_size)
    dataset = dataset.prefetch(2)
    # To use a Dataset in the input_fn of a tf.estimator.Estimator,
    # simply return the Dataset and the framework will take care of creating an iterator and initializing it for you.
    # dataset.make_one_shot_iterator().get_next() in older version API
    return dataset
