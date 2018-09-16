#!/usr/bin/env python
# -*- coding: gb18030 -*-

import sys
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as inputdata

mnist = inputdata.read_data_sets("../MNIST_data/", one_hot=True)
dropout_keep_prob = 1.0

# net
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

#flag = "truncated_normal"
#flag = "random_normal"
#flag = "random_uniform"
#flag = "normal"
flag = "truncated_normal"

# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None) 
# tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) 
# tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None) 
if flag == "truncated_normal":
    W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=0.01))
    b1 = tf.Variable(tf.truncated_normal([256], stddev=0.01))
elif flag == "random_normal":
    W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.random_normal([256], stddev=0.1))
elif flag == "random_uniform":
    W1 = tf.Variable(tf.random_uniform([784, 256], minval=-1, maxval=1))
    b1 = tf.Variable(tf.random_uniform([256], minval=-1, maxval=1))
elif flag == "normal":
    W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
elif flag == "zero":
    W1 = tf.Variable(tf.zeros([784, 256]))
    b1 = tf.Variable(tf.zeros([256]))

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, dropout_keep_prob)

if flag == "truncated_normal":
    W2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.01))
    b2 = tf.Variable(tf.truncated_normal([10], stddev=0.01))
elif flag == "random_normal":
    W2 = tf.Variable(tf.random_normal([256, 10], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([10]))
elif flag == "random_uniform":
    W2 = tf.Variable(tf.random_uniform([256, 10], minval=-1, maxval=1))
    b2 = tf.Variable(tf.random_uniform([10], minval=-1, maxval=1))
elif flag == "normal":
    W2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
    b2 = tf.Variable(tf.zeros([10]))
elif flag == "zero":
    W2 = tf.Variable(tf.zeros([256, 10]))
    b2 = tf.Variable(tf.zeros([10]))


y_logtis = tf.matmul(hidden1_drop, W2) + b2

# eval result 
y_score = tf.nn.softmax(y_logtis)
correct_pred = tf.equal(tf.argmax(y_score, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# nn-net train
#loss = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_score), reduction_indices=[1]))
losses = tf.nn.softmax_cross_entropy_with_logits(logits=y_logtis, labels=y_true)
loss = tf.reduce_mean(losses)
train_op = tf.train.AdagradOptimizer(0.3).minimize(loss)

session_conf = tf.ConfigProto(allow_soft_placement=True,
        log_device_placement=False)
session_conf.gpu_options.allow_growth = True
sess = tf.Session(config=session_conf)
with sess.as_default():
    sess.run(tf.global_variables_initializer())
    for idx in range(3001):
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


