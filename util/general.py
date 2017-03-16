# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from scipy.signal import lfilter

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
# Implementation from: https://github.com/awjuliani/DeepRL-Agents
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# How to process raw frames before input layer.
def process_frame(frame):
    return frame / 255.0

# Discounting function used to calculate discounted returns.
# Implementation from: https://github.com/awjuliani/DeepRL-Agents
def discount(x, gamma):
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers
# Implementation from: https://github.com/awjuliani/DeepRL-Agents
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
