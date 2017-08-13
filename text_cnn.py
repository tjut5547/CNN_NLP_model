import tensorflow as tf
import numpy as np
import time

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
          embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        '''
        sequence_length : 句子的长度，句子的长度被padding到59
        num_classes : 输出层种类，这里只有两种，正类和负类
        vocab_size : 词汇的大小(词汇ont-hot向量的大小)，这需要我们定义embeddding层，
                     shape = [vocabulary_size, embedding_size]
        embedding_size : embedding 向量的大小
        filter_sizes : 卷积核的大小
        num_filters ：卷积核的个数
        '''
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        # Embedding layer
        '''
        tf.device('/cpu:0') ：强制操作在CPU上面执行，TensorFlow默认在GPU上面执行，但是
                              embedding implementation没有被GPU版本的，执行GPU会出错。
        tf.name_scope ：创建了一个新的名字是"embedding"的name scope，这个scope将所有的
                        operation加入了成为"embedding"最高层次的节点中，这会使得我们能够
                        可视化的TensorBoard中获得一个很好的层次。
        W ：是我们需要学习的embdding矩阵，我们使用标准正态分布去初始化
        tf.nn.embedding_lookup ：创建了一个实际的embedding 操作，这个embedding operation
                                 是三维我的tensor，shape = [None, sequence_length, embedding_size].

        '''
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # print(self.input_x.eval()); time.sleep(10000)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        '''
        filter_shape : filter_size * embedding_size的卷积核，一个通道, num_filters个卷积核
                        采用truncated_normal初始化（截断正态分布），标准差0.1
        conv ：卷积层
               self.embedded_chars_expanded ：输入
               W ：filter，卷积核
               strides ：strides有四个参数：
                        第一个是矩阵数量，第二是是矩阵高度
                        第三个是矩阵宽度，第四个是通道个数
                padding ："VALID" : 不进行填充
                        "SAME" ：用0进行填充
        h ：修正线性单元，max(x，0)
        pooled ：池化层，
                 ksize当中，第一，四维必须是1，其它是需要池化区域的大小
                 strides : 类比于上一个strides的意思
                 padding ：类比于上一个padding
        '''
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                '''
                Maxpooling over the outputs
                sequence_length - filter_size + 1 : 卷积层输出长度
                                                    句子的长度减去选中单词的大小 + 1
                '''
                pooled = tf.nn.max_pool( 
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        '''
        self.h_pool : 维度变成以前的三倍

        在这里W是我们的卷积核矩阵，h是对卷积层输出使用relu这个线性修正单元的输出
        每一个卷积核划过所有的embedding向量，其中最重要的是选择多少个单词。

        '''

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        print (self.h_pool); time.sleep(100)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")