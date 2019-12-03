#!/usr/bin/env python

###############################################################################
# Author: Dr. Asiri I.B. Obeysekara
# Date: 20/11/2019
#
#  {WORK IN PROGRESS}
###############################################################################

from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
tf.__version__

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
import matplotlin.pyplot as plt

import GANerr

#this is discrimitator class
class GAN2D_D():
    def __init__(self,main):
        self.main=main

    def discriminator():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
        return model

#this is the generator class
class GAN2D_G():
    def __init__(self,main):
        self.main=main

    def generator():
        model.tf.keras.Sequential()
        model.add(laters.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.LeakyReLU())
        return model


if __name__ == '__main__':
    print('this is home of the main GAN generator + discriminator wrappers')
