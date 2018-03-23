#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

features = ['f'+str(i) for i in range(1, 40)]
label = ["clicked"]
_CSV_COLUMNS = label + features

# 1 int label, 13 ints, 26 strings
continuous_defaults = [[0.0]] * 13
category_defaults = [['']] * 26
label_defaults = [[0]]
_CSV_COLUMN_DEFAULTS = label_defaults + continuous_defaults + category_defaults

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def input_fn(data_file, num_epochs, batch_size, shuffle=True):
    """Input function for train or evaluation.
    Args:
        data_file: can be both file or directory.
    Returns:
        (features, label) 
        `features` is a dictionary in which each value is a batch of values for
        that feature; `labels` is a batch of labels.
    """

    def _parse_csv(value):  # value: Tensor("arg0:0", shape=(), dtype=string)
        tf.logging.info('Parsing input files: {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim='\t')
        features = dict(zip(_CSV_COLUMNS, columns))
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
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'], seed=123)  # set of Tensor object
    # Use `Dataset.map()` to build a pair of a feature dictionary
    # and a label tensor for each example.
    dataset = dataset.map(_parse_csv, num_parallel_calls=5).repeat(num_epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    input_fn()
