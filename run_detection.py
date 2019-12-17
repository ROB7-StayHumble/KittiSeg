#!/usr/bin/env python2

"""
Detects Cars in an image using KittiSeg.
Input: Image
Output: Image (with Cars plotted in Green)
Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.
Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]
--------------------------------------------------------------------------------
The MIT License (MIT)
Copyright (c) 2017 Marvin Teichmann
Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image



# runs_dir = 'RUNS'
runs_dir = '/home/zacefron/Desktop/golfcart-workspace/catkin_ws/src/lanedetection/include/KittiSeg/RUNS'
default_run = 'KittiSeg_2019_11_15_16.15'
logdir = os.path.join(runs_dir, default_run)
tv_utils.set_gpus_to_use()
# Loading hyperparameters from logdir
hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

logging.info("Hypes loaded successfully.")

# Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
modules = tv_utils.load_modules_from_logdir(logdir)
logging.info("Modules loaded successfully. Starting to build tf graph.")


os.chdir('src/lanedetection/include/KittiSeg')
print(os.getcwd())



def run_detection(img):
    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        print("starting KittiSeg inference")
        print("la1")
        image_pl = tf.placeholder(tf.float32)
        print("la21")
        image = tf.expand_dims(image_pl, 0)
        print("la")

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules, image=image)

        print("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        print("Weights loaded successfully.")

        print("Starting inference")

        # Load and resize input image
        image = img
        if hypes['jitter']['reseize_image']:
            # Resize input only, if specified in hypes
            image_height = hypes['jitter']['image_height']
            image_width = hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')

        # Run KittiSeg model on image
        feed = {image_pl: image}
        softmax = prediction['softmax']
        output = sess.run([softmax], feed_dict=feed)

        # Reshape output from flat vector to 2D Image
        shape = image.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        rb_image = seg.make_overlay(image, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold

        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(image, street_prediction)

        logging.info("--> Done with detection")

        return green_image



