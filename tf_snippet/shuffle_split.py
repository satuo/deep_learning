#!/usr/bin/env python
# -*- coding: gb18030 -*-

import os
import re
import sys
import numpy as np

# Randomly shuffle data
np.random.seed(10) # seed() 用于生成随机数时所用算法开始的整数值
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

np.random.shuffle(train_set)
np.random.shuffle(test_set)

# Split train/test set
# dev_sample_percentage: 表示采样比例
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


