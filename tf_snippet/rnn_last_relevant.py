import tensorflow as tf


class TextRNN(object):
    """
    A RNN for text classification.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, \
            cell_type, num_units, num_cell_layer, l2_reg_lambda=0.0):

        # 0. 输入
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # 1. 求非填充值个数数组[100, 80, 90, ...]
        text_length = self._length(self.input_x)
        # 2. 当参数传入dynamic_rnn
        with tf.name_scope("rnn"):
            cell = self._get_cell(num_units, cell_type)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.embedded_chars,
                                               sequence_length=text_length,
                                               dtype=tf.float32)
        # 3. 根据非填充值个数数组，求出每个样本最后一个非填充值的输出
            self.h_outputs = self.last_relevant(all_outputs, text_length)


    def _length(self, seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.

    # export from https://github.com/roomylee/rnn-text-classification-tf 
    # export from https://danijar.com/variable-sequence-lengths-in-tensorflow/
    # export from https://blog.csdn.net/u013061183/article/details/80641457
    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    # 注意: code来自于roomylee/rnn-text-classification-tf, danijar 博客和该代码有点diff，思路是一致的。
    # danijar 博客处理的是三维数组 
    # roomylee/rnn-text-classification-tf处理的是二维数组
