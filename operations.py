import tensorflow as tf
import numpy as np
import time, os

'''
No batch normalization needed
'''

def conv2d(x, output_dim, filter_height=3, filter_width=3, stride_hor=2, stride_ver=2, name='conv2d'):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter', [filter_height, filter_width, x.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        convolution = tf.nn.conv2d(x, filter, strides=[1,stride_hor, stride_ver,1], padding='SAME')
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0))
        weighted_sum = convolution + bias
        return weighted_sum

def deconv2d(x, output_shape, filter_height=3, filter_width=3, stride_hor=2, stride_ver=2, name='deconv2d'):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter', [filter_height, filter_width, output_shape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        deconvolution = tf.nn.conv2d_transpose(x, filter, output_shape=output_shape, strides=[1,stride_hor, stride_ver,1])
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0))
        weighted_sum = deconvolution + bias
        return weighted_sum

def linear(x, hidden, name='linear'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [hidden], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x, weight) + bias
        return weighted_sum

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, x*leak)

if __name__ == "__main__":
    x = tf.get_variable('test', [3,5, 5, 3])
    a = conv2d(x, 7)
    print(a.get_shape())