# encoding: utf-8
#!/usr/bin/env python

import tensorflow as tf
import numpy as np

train_text_list = [u"你 我 他", u"山东 人", u"河南 人", u"我 是 河南 人"]
# max_document_length = 4
max_document_length = max([len(train_text.split(" ")) for train_text in train_text_list])
# 截断句子的词个数 且 只保留频率大于1的词, 存储 [我, 人, 河南]  
# max_document_length: 截断句子的词个数, 句子中后面的词忽略
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=1) 
x_transform_train = vocab_processor.fit_transform(train_text_list)
# 打印list
print np.array(list(x_transform_train))
# 把符合条件[]的词, 存储到vocab中, 这个vocab非明文的
vocab_processor.save('vocab')

# 在预测时, 也需要把待预测的句子转换为id list
# 首先需要先加载vocab, 之后继续执行transform
new_vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore('vocab')
tokens = new_vocab.transform([u"你 我 他 山东 河南 湖北"]) 
# max_document_length = 4, 第4个词之后都忽略 
print list(tokens) # [[0, 2, 0, 0]]

# 打印词长度
print len(vocab_processor.vocabulary_)
# 打印词 以及 其对应id
for word, word_id in vocab_processor.vocabulary_._mapping.items():
    print word.encode('gb18030'), word_id 

# 最终的使用:  Sentence -> Word ID -> Word Embedding
# 1. 读入正负样本, 格式: [['我 是 帅哥 哈哈'], ['你 我 他 哈哈']] [[0, 1], [1, 0]]
x_train, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# 2. 求出所有中最大词的个数, 这里是4
max_document_length = max([len(x.split(" ")) for x in x_text])
# 3. 生成词处理器: 只保留最大词个数 以及 词频率>2的词
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=2)
# 4. 把每个句子的词都转化为id list, 类似:  [[0, 2, 0, 0], [1,0,0,3]]
x_train = np.array(list(vocab_processor.fit_transform(x_text)))
# 5. 最终词id list都用到embedding_lookup, 用于查询id对应的向量 
self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

