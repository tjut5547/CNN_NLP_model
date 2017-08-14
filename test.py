import tensorflow as tf
import numpy as np
import time

class App(object):
    def __init__(self, sequence_length, num_classes):
        self.input_xx = tf.placeholder(tf.int32, [sequence_length])
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.W = tf.Variable(
            tf.random_uniform([10, 10], -1.0, 1.0),
            name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        self.test = tf.nn.embedding_lookup(self.W, self.input_xx)
        print (self.test)

    def sess_run(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        feed_dict = {
            self.input_x : [[1, 2, 6, 3], [1, 2, 3, 4]],
            self.input_y: [[1, 2, 6, 2, 3]],
            self.input_xx : [1, 2, 3, 4]
        }
        a, b, c, d, e = sess.run([self.input_x, self.input_y, self.W, self.embedded_chars, self.test], feed_dict)

if __name__ == "__main__":
    app = App(4, 5)
    app.sess_run()