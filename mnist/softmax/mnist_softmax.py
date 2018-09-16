#!/usr/bin/env python
# -*- coding: gb18030 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as inputdata

mnist = inputdata.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_logtis = tf.matmul(x, W) + b
y_score = tf.nn.softmax(y_logtis)

# calc loss
losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logtis, labels=y_true)
loss = tf.reduce_mean(losses)

# acc 
correct_pred = tf.equal(tf.argmax(y_score, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# optimize loss
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#with tf.Graph().as_default():
session_conf = tf.ConfigProto(allow_soft_placement=True,
        log_device_placement=False)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config=session_conf)
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for idx in range(5000):
        batch_x, batch_y = mnist.train.next_batch(100)
        # train - op
        _, cur_loss, cur_acc = sess.run([train_op, loss, acc], feed_dict={x : batch_x, y_true : batch_y})
        if idx % 100 == 0:
            print >> sys.stderr, 'Train, idx: %s, loss: %s, acc: %s' % (idx, cur_loss, cur_acc)
            # val - op
            _, cur_loss, cur_acc = sess.run(\
                    [train_op, loss, acc], feed_dict={x : mnist.validation.images, y_true: mnist.validation.labels})
            print >> sys.stderr, "Val, idx: %s, loss: %s, acc: %s" % (idx, cur_loss, cur_acc)
    # test
    _, cur_loss, cur_acc = sess.run(\
            [train_op, loss, acc], feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
    print >> sys.stderr, 'Test-Result, loss: %s, acc: %s' % (cur_loss, cur_acc) 


