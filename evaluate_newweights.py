#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the KittiSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

import tensorvision.train as train
import tensorvision.analyze as ana
import tensorvision.utils as utils

from evaluation import kitti_test

flags.DEFINE_string('RUN', 'KittiSeg_2019_11_15_16.15',
                    'Modifier for model parameters.')
flags.DEFINE_string('hypes', 'hypes/KittiSeg.json',
                    'File storing model parameters.')
flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

tf.app.flags.DEFINE_boolean(
        'save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))



def main(_):
    utils.set_gpus_to_use()

    try:
        import tensorvision.train
        import tensorflow_fcn.utils
    except ImportError:
        logging.error("Could not import the submodules.")
        logging.error("Please execute:"
                      "'git submodule update --init --recursive'")
        exit(1)

    with open(tf.app.flags.FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)
    utils.load_plugins()

    runs_dir = 'RUNS'

    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    utils._add_paths_to_sys(hypes)

    logging.info("Evaluating on Validation data.")
    logdir = os.path.join(runs_dir, FLAGS.RUN)
    # logging.info("Output images will be saved to {}".format)
    ana.do_analyze(logdir)

    logging.info("Creating output on test data.")
    kitti_test.do_inference(logdir)

    logging.info("Analysis for pretrained model complete.")
    logging.info("For evaluating your own models I recommend using:"
                 "`tv-analyze --logdir /path/to/run`.")
    logging.info("tv-analysis has a much cleaner interface.")


if __name__ == '__main__':
    tf.app.run()
