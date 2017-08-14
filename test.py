import tensorflow as tf
import numpy as np
import time

class App(object):
    def __init__(self, sequence_length, num_classes):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.W = tf.Variable(
            tf.random_uniform([10, 10], -1.0, 1.0),
            name="W")
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        print (self.input_x)
        print (self.input_y)
        print (self.W)
        print (self.embedded_chars)


        


if __name__ == "__main__":
    app = App(4, 5)