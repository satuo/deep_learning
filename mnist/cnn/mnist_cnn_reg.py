#!/usr/bin/env python
# -*- coding: gb18030 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as inputdata

mnist = inputdata.read_data_sets("../MNIST_data/", one_hot=True)
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2_2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

# conv1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2_2(h_conv1)

# conv2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2_2(h_conv2)

# fc1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# fc2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_logtis = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# acc
y_score = tf.nn.softmax(y_logtis)
correct_pred = tf.equal(tf.argmax(y_score, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loss
losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logtis, labels=y_)
loss = tf.reduce_mean(losses)

#regularizers = tf.nn.l2_loss(b_conv1) + tf.nn.l2_loss(b_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
#regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) + \
#        tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) + \
#        tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2)
#regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))
regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
loss += 5e-4 * regularizers
#train_op = tf.train.AdagradOptimizer(1e-4).minimize(loss)

#base_learn_rate = 1e-4
batch_size = 100
#batch = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(\
#        base_learn_rate,  # Base learning rate.
#        batch * batch_size,
#        mnist.train.num_examples,  # Decay step.
#        0.85,  # Decay rate
#        staircase=True)
#train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
#train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
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
        # train - op
        #_, cur_loss, cur_acc, cur_learning_rate = sess.run([train_op, loss, acc, learning_rate], \
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


