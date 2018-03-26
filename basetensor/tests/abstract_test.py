#!/usr/bin/env python3
"""Tensorflow utility functions"""
########################################################################
# File: utils.py
#  executable: utils.py

# Author: Andrew Bailey
# History: 12/07/17 Created
########################################################################
import unittest
import numpy as np
import tensorflow as tf
from basetensor.abstract import CreateTFDataset, CreateTFNetwork


class TestCreateTFNetwork(unittest.TestCase):
    """Test the static functions in CreateTFNetwork"""

    @classmethod
    def setUpClass(cls):
        super(TestCreateTFNetwork, cls).setUpClass()
        cls.dataset = PassDataset()
        cls.network = NetworkPass(cls.dataset)

    def test_lstm(self):
        """Test lstm cell works"""
        input_vector = tf.placeholder(tf.float32, shape=[None, None, 10])
        sequence_length = tf.placeholder(tf.int32, shape=[None])
        n_hidden = 10
        layer_name = "test"
        forget_bias = 5
        output_keep_prob = 1
        state_keep_prob = 1
        input_keep_prob = 1
        output = self.network.lstm(input_vector=input_vector, sequence_length=sequence_length, n_hidden=n_hidden,
                                   layer_name=layer_name, forget_bias=forget_bias, output_keep_prob=output_keep_prob,
                                   state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 10
            data_intput = np.random.randint(0, 10, [batch_size, n_hidden, 10])
            seq_length = np.ones(batch_size) * 10
            output_tensor = sess.run(output, feed_dict={sequence_length: seq_length, input_vector: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (batch_size, n_hidden, 10))

    def test_decoder_lstm(self):
        """Test decoder lstm cells"""
        # single input tensor because this method loops output into the input
        input_vector = tf.placeholder(tf.float32, shape=[None, 10])
        sequence_length = 10
        n_hidden = 10
        layer_name = "test"
        forget_bias = 5
        output_keep_prob = 1
        state_keep_prob = 1
        input_keep_prob = 1
        batch_size = 10
        output = self.network.decoder_lstm(input_vector=input_vector, batch_size=batch_size,
                                           sequence_length=sequence_length, n_hidden=n_hidden,
                                           layer_name=layer_name, forget_bias=forget_bias,
                                           output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            data_intput = np.random.randint(0, 10, [batch_size, 10])
            output_tensor = sess.run(output, feed_dict={input_vector: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (batch_size, n_hidden, 10))

    def test_fulconn_layer(self):
        """Test the fully connected layer"""
        # single input tensor because this method loops output into the input
        input_data = tf.placeholder(tf.float32, shape=[None, 10])
        output = self.network._fulconn_layer(input_data, 10, seq_len=1, activation_func=None)

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 10
            data_intput = np.random.randint(0, 10, [batch_size, 10])
            output_tensor, weights, bias = sess.run(output, feed_dict={input_data: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (batch_size, 10))

    def test_fully_connected_layer(self):
        """Test tensorflow implemented fully connected layer"""
        # single input tensor because this method loops output into the input
        input_data = tf.placeholder(tf.float32, shape=[None, 10])
        output = self.network.fully_connected_layer(input_data, num_outputs=10)

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 10
            data_intput = np.random.randint(0, 10, [batch_size, 10])
            output_tensor = sess.run(output, feed_dict={input_data: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (batch_size, 10))

    def test_blstm(self):
        """Test lstm cell works"""
        input_vector = tf.placeholder(tf.float32, shape=[None, None, 10])
        sequence_length = tf.placeholder(tf.int32, shape=[None])
        n_hidden = 20
        layer_name = "test"
        forget_bias = 5
        output_keep_prob = 1
        state_keep_prob = 1
        input_keep_prob = 1
        output = self.network.blstm(input_vector=input_vector, sequence_length=sequence_length, n_hidden=n_hidden,
                                    layer_name=layer_name, forget_bias=forget_bias, output_keep_prob=output_keep_prob,
                                    state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob, concat=True)
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 10
            seq_len = 30
            data_intput = np.random.randint(0, 10, [batch_size, seq_len, 10])
            seq_length = np.ones(batch_size) * 10
            output_tensor = sess.run(output, feed_dict={sequence_length: seq_length, input_vector: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (batch_size, seq_len, n_hidden * 2))

    def test_batch_normalization(self):
        """Test batch_normalization"""
        input_vector = tf.placeholder(tf.float32, shape=[None, None, 10])
        batch_norm = self.network.batch_normalization(input_vector, name="test")

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            data_intput = np.random.randint(0, 10, [1, 20, 10])
            output_tensor = sess.run(batch_norm, feed_dict={input_vector: data_intput})

            self.assertSequenceEqual(output_tensor.shape, (1, 20, 10))

    def test_conv_layer(self):
        """Test conv_layer"""
        input_vector = tf.placeholder(tf.float32, shape=[None, 10, 10, 1])
        k_size = [2, 2]
        padding = "SAME"
        filters = 10
        # conv_layer(self, input_vector, ksize, padding, layer_name="convolutional_layer", dilate=1, strides=(1, 1),
        #            bias_term=False, active=tf.nn.relu, bn=True, regularizer=None)
        cov = self.network.conv_layer(input_vector, filters, k_size, padding=padding, layer_name="convolutional_layer",
                                      dilate=1,
                                      strides=(1, 1), bias_term=True, active=tf.nn.relu, bn=True,
                                      regularizer=None)
        cov2 = self.network.conv_layer(input_vector, filters, k_size, padding="VALID",
                                       layer_name="convolutional_layer2",
                                       dilate=1, strides=(1, 1), bias_term=False, active=None, bn=False)

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            data_intput = np.random.randint(0, 10, [3, 10, 10, 1])
            output_tensor = sess.run(cov, feed_dict={input_vector: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (3, 10, 10, 10))

            output_tensor = sess.run(cov2, feed_dict={input_vector: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (3, 9, 9, 10))

    def test_residual_layer(self):
        """Test residual layer"""
        input_vector = tf.placeholder(tf.float32, shape=[None, 10, 10, 1])
        out_channel = 3
        res_layer = self.network.residual_layer(input_vector, out_channel, ksizes=([2, 2], [2, 2], [2, 4], [2, 2]),
                                                layer_name="residual_layer",
                                                i_bn=True, regularizer=None)

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            data_intput = np.random.randint(0, 10, [3, 10, 10, 1])
            output_tensor = sess.run(res_layer, feed_dict={input_vector: data_intput})
            self.assertSequenceEqual(output_tensor.shape, (3, 10, 10, 3))

    def test_inception_layer(self):
        """Test inception_layer"""
        input_vector = tf.placeholder(tf.float32, shape=[None, 10, 10, 1])
        out_channel = 10
        res_layer = self.network.inception_layer(input_vector, times=out_channel, layer_name="residual_layer")

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            data_intput = np.random.randint(0, 10, [3, 10, 10, 1])
            output_tensor = sess.run(res_layer, feed_dict={input_vector: data_intput})
            # 30 from max pool, 30 from 1x1, 30 from 3x3, 30 from 5x5
            self.assertSequenceEqual(output_tensor.shape, (3, 10, 10, 120))


class BaseTensorTests(unittest.TestCase):
    """Test the functions in all of basetensor"""

    @classmethod
    def setUpClass(cls):
        super(BaseTensorTests, cls).setUpClass()
        cls.dataset = PassDataset()

    def test_abstractfunctions(self):
        """test_CreateTFDataset"""
        with self.assertRaises(TypeError):
            DatasetFail()
            NetworkFail(self.dataset)

    def test_createnetwork(self):
        """test_createnetwork"""
        NetworkPass(self.dataset)
        with self.assertRaises(AssertionError):
            NetworkPass("somethingelse")


class PassDataset(CreateTFDataset):
    def __init__(self):
        super(PassDataset, self).__init__()

    def create_dataset(self):
        pass

    def create_iterator(self):
        pass

    def test(self):
        pass

    def load_training_data(self):
        pass

    def process_graph_output(self):
        pass

    def load_inference_data(self):
        pass


class DatasetFail(CreateTFDataset):
    def __init__(self):
        super(DatasetFail, self).__init__()

    def create_iterator(self):
        pass

    def test(self):
        pass

    def load_training_data(self):
        pass

    def process_graph_output(self):
        pass

    def load_inference_data(self):
        pass


class NetworkPass(CreateTFNetwork):
    def __init__(self, dataset):
        super(NetworkPass, self).__init__(dataset)

    def create_model(self):
        pass

    def create_ops(self):
        pass


class NetworkFail(CreateTFNetwork):
    def __init__(self, dataset):
        super(NetworkFail, self).__init__(dataset)

    def create_train_ops(self):
        pass

    def create_inference_op(self):
        pass


if __name__ == '__main__':
    unittest.main()
