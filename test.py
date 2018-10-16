#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Wide and Deep Model Evaluation"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import sys
import time

import tensorflow as tf

from wide_resdnn.build_estimator import build_estimator
from wide_resdnn.dataset import input_fn
from wide_resdnn.read_conf import Config
from wide_resdnn.util import elapse_time

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Evaluate Wide and Deep Model.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Model checkpoint dir for evaluating.')

parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--test_data', type=str, default=CONFIG["test_data"],
    help='Evaluating data dir.')

parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')

parser.add_argument(
    '--checkpoint_path', type=str, default=CONFIG["checkpoint_path"],
    help="Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.")


def main(_):
    print("Using TensorFlow version %s, neee TensorFlow 1.4 or later." % tf.__version__)
    # assert "1.4" <= tf.__version__, "TensorFlow r1.4 or later is needed"
    print('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('Model directory: {}'.format(model_dir))

    model = build_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))

    tf.logging.info('='*30+' START TESTING'+'='*30)
    s_time = time.time()
    results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, FLAGS.batch_size, False),
                             steps=None,  # Number of steps for which to evaluate model.
                             hooks=None,
                             checkpoint_path=FLAGS.checkpoint_path,  # If None, the latest checkpoint is used.
                             name=None)
    tf.logging.info('='*30+'FINISH TESTING, TAKE {}'.format(elapse_time(s_time))+'='*30)
    # Display evaluation metrics
    print('-' * 80)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
