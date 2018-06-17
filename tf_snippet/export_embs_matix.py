# -*- coding: utf-8 -*-
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import codecs
import sys
np.set_printoptions(threshold=np.inf) 

# Load data
X_train, y_train = data_helpers.load_data_and_labels(FLAGS.train_data_file, FLAGS.class_num, shuffle=True)
X_val, y_val = data_helpers.load_data_and_labels(FLAGS.val_data_file, FLAGS.class_num, shuffle=True)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in X_train])
max_document_length = min(1024, max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=2)
vocab_processor.fit(X_train)
X_train = np.array(list(vocab_processor.transform(X_train)))
X_val = np.array(list(vocab_processor.transform(X_val)))

# 直接把每个样本的词向量打印到文件中
with tf.Session() as sess:
    _, loss, acc, embs, = sess.run([train_op, rnn.loss, rnn.acc, rnn.embs], feed_dict)
    with open('embedding.txt', 'w') as file_:
        for i in range(max_document_length):
            embed = embs[i, :]
            word = vocab_processor.vocabulary_._reverse_mapping[i]
            file_.write('%s %s\n' % (word.encode('gb18030'), ' '.join(map(str, embed))))

# 直接每次训练的词向量导入graph
saver = tf.train.Saver([embs])
with tf.Session() as sess:
      saver.save(sess, 'path/to/checkpoint')

