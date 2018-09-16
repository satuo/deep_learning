#!/usr/bin/env python
# -*- coding: gb18030 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as inputdata

mnist = inputdata.read_data_sets("../MNIST_data/", one_hot=True)

input_size = 28
time_steps = 28
num_units = 128 
num_classes = 10
batch_size = 128 
model_type = 'lstm'
learning_rate = 1e-2

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_reshape = tf.reshape(x, [-1, time_steps, input_size])

# lstm - cell
def get_cell(num_units):
    assert model_type in ['lstm', 'gru', 'rnn']
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
    if model_type == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
    elif model_type == 'rnn':
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

# lstm - net
# concat + wx + b
# Test-Result, loss: 0.0653677, acc: 0.9813
fw_cell = get_cell(num_units)
bw_cell = get_cell(num_units)
outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x_reshape, dtype=tf.float32, time_major=False)
fw_output, bw_output = outputs
outputs_concat = tf.concat(outputs, 2)
W_fc1 = tf.Variable(tf.random_normal([2 * num_units, num_classes])) 
b_fc1 = tf.Variable(tf.random_normal([num_classes])) 
y_logtis = tf.matmul(outputs_concat[:, -1, :], W_fc1) + b_fc1

# spilt wx + b
# Test-Result, loss: 0.0683187, acc: 0.9803
#fw_cell = get_cell(num_units)
#bw_cell = get_cell(num_units)
#outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x_reshape, dtype=tf.float32, time_major=False)
#fw_output = outputs[0][:, -1, :] 
#bw_output = outputs[1][:, -1, :] 
#W_fc1 = tf.Variable(tf.random_normal([num_units, num_classes])) 
#b_fc1 = tf.Variable(tf.random_normal([num_classes])) 
#y_logtis = tf.add(tf.add(tf.matmul(fw_output, W_fc1), tf.matmul(bw_output, W_fc1)), b_fc1)

# wx1+b, wx2+b
# Test-Result, loss: 0.084897, acc: 0.9753
#fw_cell = get_cell(num_units)
#bw_cell = get_cell(num_units)
#outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x_reshape, dtype=tf.float32, time_major=False)
#fw_output = outputs[0][:, -1, :]
#bw_output = outputs[1][:, -1, :]
#fw_bw_output = fw_output + bw_output
#W_fc1 = tf.Variable(tf.random_normal([num_units, num_classes])) 
#b_fc1 = tf.Variable(tf.random_normal([num_classes])) 
#y_logtis = tf.matmul(fw_bw_output, W_fc1) + b_fc1
#
# acc
y_score = tf.nn.softmax(y_logtis)
correct_pred = tf.equal(tf.argmax(y_score, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loss
losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logtis, labels=y_)
loss = tf.reduce_mean(losses)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# train
session_conf = tf.ConfigProto(allow_soft_placement=True,
        log_device_placement=False)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config=session_conf)
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for idx in range(1001):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cur_loss, cur_acc = sess.run([train_op, loss, acc], \
                feed_dict={x : batch_x, y_ : batch_y, keep_prob:1.0})
        if idx % 10 == 0:
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


