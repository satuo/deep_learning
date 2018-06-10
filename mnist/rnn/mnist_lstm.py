#!/usr/bin/env python
# -*- coding: gb18030 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as inputdata

mnist = inputdata.read_data_sets("../MNIST_data/", one_hot=True)

input_size = 28
time_steps = 28
num_units = 64 
num_classes = 10
num_cell_layer = 3
batch_size = 64 

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_reshape = tf.reshape(x, [-1, time_steps, input_size])

# lstm - cell
def cell(num_units):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return lstm_cell

# lstm - net
rnn_layers = [cell(num_units) for _ in range(num_cell_layer)]
cells  = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
outputs, states = tf.nn.dynamic_rnn(cells, inputs=x_reshape, initial_state=None, dtype=tf.float32, time_major=False)

# mlp
W_fc1 = tf.Variable(tf.random_normal([num_units, num_classes])) 
b_fc1 = tf.Variable(tf.random_normal([num_classes])) 
y_logtis = tf.matmul(outputs[:, -1, :], W_fc1) + b_fc1

# acc
y_score = tf.nn.softmax(y_logtis)
correct_pred = tf.equal(tf.argmax(y_score, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loss
losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logtis, labels=y_)
loss = tf.reduce_mean(losses)

learning_rate = 1e-4
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# train
session_conf = tf.ConfigProto(allow_soft_placement=True,
        log_device_placement=False)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config=session_conf)
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for idx in range(20001):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cur_loss, cur_acc = sess.run([train_op, loss, acc], \
                feed_dict={x : batch_x, y_ : batch_y, keep_prob:1.0})
        if idx % 100 == 0:
            #print >> sys.stderr, 'Train, idx: %s, loss: %s, acc: %s, learn_rate: %s' % (idx, cur_loss, cur_acc, cur_learning_rate)
            print >> sys.stderr, 'Train, idx: %s, loss: %s, acc: %s' % (idx, cur_loss, cur_acc)
            # val - op
            _, cur_loss, cur_acc = sess.run(\
                    [train_op, loss, acc], feed_dict={x : mnist.validation.images, y_: mnist.validation.labels, keep_prob:1.0})
            print >> sys.stderr, "Val, idx: %s, loss: %s, acc: %s" % (idx, cur_loss, cur_acc)
    # test
    _, cur_loss, cur_acc = sess.run(\
            [train_op, loss, acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0})
    print >> sys.stderr, 'Test-Result, loss: %s, acc: %s' % (cur_loss, cur_acc) 


