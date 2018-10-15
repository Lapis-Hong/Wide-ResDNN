#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Training Wide and Deep Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import shutil
import sys
import time

import tensorflow as tf

from wide_resdnn.build_estimator import build_estimator, build_custom_estimator
from wide_resdnn.dataset import input_fn
from wide_resdnn.read_conf import Config
from wide_resdnn.util import elapse_time

CONFIG = Config().train
parser = argparse.ArgumentParser(description='Train Wide and Deep Model.')
parser.add_argument(
    '--train_data', type=str, default=CONFIG["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--test_data', type=str, default=CONFIG["test_data"],
    help='Path to the test data.')
parser.add_argument(
    '--model_dir', type=str, default=CONFIG["model_dir"],
    help='Base directory for the model.')
parser.add_argument(
    '--model_type', type=str, default=CONFIG["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")
parser.add_argument(
    '--train_epochs', type=int, default=CONFIG["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=CONFIG["epochs_per_eval"],
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=CONFIG["batch_size"],
    help='Number of examples per batch.')
parser.add_argument(
    '--keep_train', type=int, default=CONFIG["keep_train"],
    help='Whether to keep training on previous trained model.')
# parser.add_argument(
#     '--checkpoint_path', type=int, default=CONFIG["checkpoint_path"],
#     help='Model checkpoint path for testing.')


def train(model):
    print('Evaluate every {} epochs'.format(FLAGS.epochs_per_eval))
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tf.logging.info('START TRAIN AT EPOCH {}'.format(FLAGS.epochs_per_eval*n + 1))
        t0 = time.time()
        model.train(input_fn=lambda: input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, FLAGS.batch_size),
                    hooks=None,
                    steps=None,
                    max_steps=None,
                    saving_listeners=None)
        tf.logging.info('Finish train {} epochs, take {} mins'.format(n + 1, FLAGS.epochs_per_eval, elapse_time(t0)))
        print('-' * 80)

        t0 = time.time()
        results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, FLAGS.batch_size, False),
                                 steps=None,  # Number of steps for which to evaluate model.
                                 hooks=None,
                                 checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                                 name=None)
        tf.logging.info('Finish evaluation, take {} mins'.format(n + 1, elapse_time(t0)))
        print('-' * 80)

        # Display evaluation metrics
        for key in sorted(results):
            print('{}: {}'.format(key, results[key]))


def train_and_eval(model):
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(FLAGS.train_data, 1, FLAGS.batch_size), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(FLAGS.eval_data, 1, FLAGS.batch_size, False))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def main(_):
    print("Using TensorFlow version %s, neee TensorFlow 1.4 or later." % tf.__version__)
    # assert "1.4" <= tf.__version__, "Need TensorFlow r1.4 or later."
    CONFIG = Config()
    print('Model type: {}'.format(FLAGS.model_type))
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.model_type)
    print('Model directory: {}'.format(model_dir))
    if not FLAGS.keep_train:
        # Clean up the model directory if not keep training
        shutil.rmtree(model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(model_dir))
    model = build_custom_estimator(model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))

    if CONFIG.distribution["is_distribution"]:
        print("Using PID: {}".format(os.getpid()))
        cluster = CONFIG.distribution["cluster"]
        job_name = CONFIG.distribution["job_name"]
        task_index = CONFIG.distribution["task_index"]
        print("Using Distributed TensorFlow. Local host: {} Job_name: {} Task_index: {}"
              .format(cluster[job_name][task_index], job_name, task_index))
        cluster = tf.train.ClusterSpec(CONFIG.distribution["cluster"])
        server = tf.train.Server(cluster,
                                 job_name=job_name,
                                 task_index=task_index)
        if job_name == 'ps':
            # wait for incoming connection forever
            server.join()
        else:
            train(model)
            # train_and_eval(model)
    else:
        train(model)  # local run

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
