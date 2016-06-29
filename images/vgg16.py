# -*- coding: utf-8 -*-

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.

VGG 16-layers convolutional with semantic segmentation

References:
    Fully Convolutional Networks for Semantic Segmentation
    Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015.

Links:
    https://arxiv.org/abs/1605.06211
"""

from __future__ import division, print_function, absolute_import

import tflearn


def vgg16(placeholderX=None):

    x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                           placeholder=placeholderX)

    x = tflearn.conv_2d(x, 64, 3, activation='relu', name='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', name='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='pool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', name='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', name='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='pool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', name='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='pool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='pool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', name='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='pool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', name='fc6')
    x = tflearn.dropout(x, 0.5)

    x = tflearn.fully_connected(x, 4096, activation='relu', name='fc7')
    x = tflearn.dropout(x, 0.5)

    return x
