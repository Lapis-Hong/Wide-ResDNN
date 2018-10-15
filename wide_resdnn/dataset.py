#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from wide_resdnn.read_conf import Config


def _csv_column_defaults():
    """Parse csv columns name and defaults from config.
    Returns:
        (column_names, column_defaults) 
    """
    feature_conf = Config().feature
    num_features = Config().num_features

    column_names,  column_defaults = ["clicked"], [[0]]  # first field is label
    for i in range(num_features):
        feature = "f{}".format(i+1)
        column_names.append(feature)
        if feature_conf[feature]["type"] == "continuous":
            column_defaults.append([0.0])  # continuous features
        else:
            column_defaults.append([''])  # category features
    return column_names, column_defaults


def input_fn(data_file, num_epochs, batch_size, shuffle=True, shuffle_buffer_size=None):
    """Input function for train or evaluation.
    Args:
        data_file: can be both file or directory.
    Returns:
        (features, label) 
        `features` is a dictionary in which each value is a batch of values for
        that feature; `labels` is a batch of labels.
    """

    column_names, column_defaults = _csv_column_defaults()

    def _parse_csv(value):  # value: Tensor("arg0:0", shape=(), dtype=string)
        tf.logging.info('Parsing input files: {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=column_defaults, field_delim='\t', na_value='')
        features = dict(zip(column_names, columns))
        labels = features.pop('clicked')
        return features, tf.equal(labels, 1)

    # check file exsits
    assert tf.gfile.Exists(data_file), (
        'train or test data file not found. Please make sure you have either '
        'default data_file or set both arguments --train_data and --test_data.')
    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    # if self.is_distribution:  # allows each worker to read a unique subset.
    #     dataset = dataset.shard(self.num_workers, self.worker_index)
    if shuffle:
        shuffle_buffer_size = shuffle_buffer_size or batch_size * 1000
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=123)  # set of Tensor object
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    dataset = dataset.map(_parse_csv, num_parallel_calls=5).repeat(num_epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    input_fn()
