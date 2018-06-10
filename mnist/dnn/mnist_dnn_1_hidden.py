#!/usr/bin/env python
# -*- coding: gb18030 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as inputdata

tf.logging.set_verbosity(tf.logging.WARN)

mnist = inputdata.read_data_sets("../MNIST_data/", one_hot=True)

dropout_keep_prob = 0.5

def hidden(data, input_feature_size, output_feature_size, is_dropout=False):
    W = tf.Variable(tf.truncated_normal([input_feature_size, output_feature_size], stddev=0.1))
    b = tf.Variable(tf.zeros([output_feature_size]))
    if is_dropout:
        return tf.nn.dropout(tf.nn.relu(tf.matmul(data, W) + b), dropout_keep_prob)
    else:
        return tf.nn.relu(tf.matmul(data, W) + b)

# net
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

hidden_1 = hidden(x, 784, 256, is_dropout=True)
y_logtis = hidden(hidden_1, 256, 10, is_dropout=False)

# eval result 
y_score = tf.nn.softmax(y_logtis)
correct_pred = tf.equal(tf.argmax(y_score, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# nn-net train
losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logtis, labels=y_true)
loss = tf.reduce_mean(losses)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

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
        if idx % 200 == 0:
            print >> sys.stderr, 'Train, idx: %s, loss: %s, acc: %s' % (idx, cur_loss, cur_acc)
            # val - op
            _, cur_loss, cur_acc = sess.run(\
                    [train_op, loss, acc], feed_dict={x : mnist.validation.images, y_true: mnist.validation.labels})
            print >> sys.stderr, "Val, idx: %s, loss: %s, acc: %s" % (idx, cur_loss, cur_acc)
    # test
    _, cur_loss, cur_acc = sess.run(\
            [train_op, loss, acc], feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
    print >> sys.stderr, 'Test-Result, loss: %s, acc: %s' % (cur_loss, cur_acc) 





