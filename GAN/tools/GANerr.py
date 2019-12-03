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

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Error_Estimator:
    def __init__(self, real, fake):
        self.real=real
        self.fake=fake

    def loss_discriminator(self):
        real_loss= cross_entropy(tf.ones_like(self.real), self.real)
        fake_loss= cross_entropy(tf.zeros_like(self.fake), self.fake)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(self.fake), self.fake)

if __name__ == '__main__':
    print('this is home of the main GAN error estimator wrapper')
