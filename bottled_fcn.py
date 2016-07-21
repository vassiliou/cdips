from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
import fcn16_vgg
from fcn16_vgg import FCN16VGG


class bottledFCN16(FCN16VGG):
    def __init__(self,vgg_path = None):
        FCN16VGG.__init__(self,vgg_path)

    def build(self, inputs,train = False,num_classes=2, random_init_fc8=False,debug=False):
        """ argument inputs is tuple of (fc6_output, pool4_output)  """
        
        self.fc7 = self._fc_layer(inputs[0], "fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)

        self.pred = tf.argmax(self.score_fr, dimension=3)

        self.upscore2 = self._upscore_layer(self.score_fr,
                                            shape=tf.shape(inputs[1]),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2)

        self.score_pool4 = self._score_layer(inputs[1], "score_pool4",
                                             num_classes=num_classes)

        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        
        
        self.upscore32 = self._upscore_layer(self.fuse_pool4,
                                             shape=None,
                                             num_classes=num_classes,
                                             debug=debug, name='upscore32',
                                             ksize=32, stride=16)

        self.pred_up = tf.argmax(self.upscore32, dimension=3)
 
