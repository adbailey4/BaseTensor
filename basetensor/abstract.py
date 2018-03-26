#!/usr/bin/env python3
"""Abstract classes for defining datasets, graphs and training models"""
########################################################################
# File: abstract.py
#  executable: abstract.py

# Author: Andrew Bailey
# History: 12/08/17 Created
########################################################################

import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


class BasicTFTraining(ABC):
    """Boilerplate abstract class for running tensorflow models"""

    def __init__(self, model):
        assert isinstance(model, CreateTFNetwork)
        super(BasicTFTraining, self).__init__()

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def run_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    @abstractmethod
    def read_parameters(self):
        pass


class GanTFTraining(ABC):
    """Boilerplate abstract class for running tensorflow models"""

    def __init__(self, models):
        for model in models:
            assert isinstance(model, CreateTFNetwork)
        super(GanTFTraining, self).__init__()

    @abstractmethod
    def train_gan(self):
        pass

    @abstractmethod
    def run_generator(self):
        pass

    @abstractmethod
    def read_parameters(self):
        pass

    @abstractmethod
    def pretrain_generator(self):
        pass

    @abstractmethod
    def pretrain_discriminator(self):
        pass

        # def profile_training(self, sess, writer, run_metadata, run_options):
        #     """Expensive profile step so we can track speed of operations of the model"""
        #     _, summary, global_step = sess.run(
        #         [self.train_op, self.summaries, self.global_step],
        #         run_metadata=run_metadata, options=run_options)
        #     # add summary statistics
        #     writer.add_summary(summary, global_step)
        #     writer.add_run_metadata(run_metadata, "step{}_train".format(global_step))
        #     if self.args.save_trace:
        #         self.chrome_trace(run_metadata, self.args.trace_name)
        #
        # @staticmethod
        # def chrome_trace(metadata_proto, f_name):
        #     """Save a chrome trace json file.
        #     To view json vile go to - chrome://tracing/
        #     """
        #     time_line = timeline.Timeline(metadata_proto.step_stats)
        #     ctf = time_line.generate_chrome_trace_format()
        #     with open(f_name, 'w') as file1:
        #         file1.write(ctf)
        #
        #
        # def test_time(self):
        #     """Return true if it is time to save the model"""
        #     delta = (datetime.now() - self.start).total_seconds()
        #     if delta > self.args.save_model:
        #         self.start = datetime.now()
        #         return True
        #     return False
        #
        # def get_model_files(self, *files):
        #     """Collect neccessary model files for upload"""
        #     file_list = [self.model_path + ".data-00000-of-00001", self.model_path + ".index"]
        #     for file1 in files:
        #         file_list.append(file1)
        #     return file_list
        #
        #
        # def average_gradients(tower_grads):
        #     """Calculate the average gradient for each shared variable across all towers.
        #     Note that this function provides a synchronization point across all towers.
        #     Args:
        #       tower_grads: List of lists of (gradient, variable) tuples. The outer list
        #         is over individual gradients. The inner list is over the gradient
        #         calculation for each tower.
        #     Returns:
        #        List of pairs of (gradient, variable) where the gradient has been averaged
        #        across all towers.
        #     """
        #     average_grads = []
        #     # # print(tower_grads)
        #     for grad_and_vars in zip(*tower_grads):
        #         # Note that each grad_and_vars looks like the following:
        #         #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        #         # print(grad_and_vars)
        #         grads = []
        #         for g, v in grad_and_vars:
        #             # print(g)
        #             # print(v)
        #             # print("Another gradient and variable")
        #             # Add 0 dimension to the gradients to represent the tower.
        #             expanded_g = tf.expand_dims(g, 0)
        #
        #             # Append on a 'tower' dimension which we will average over below.
        #             grads.append(expanded_g)
        #
        #         # Average over the 'tower' dimension.
        #         grad = tf.concat(axis=0, values=grads)
        #         grad = tf.reduce_mean(grad, 0)
        #         #
        #         # # Keep in mind that the Variables are redundant because they are shared
        #         # # across towers. So .. we will just return the first tower's pointer to
        #         # # the Variable.
        #         v = grad_and_vars[0][1]
        #         grad_and_var = (grad, v)
        #         average_grads.append(grad_and_var)
        #     return average_grads
        #
        #
        # def test_for_nvidia_gpu(num_gpu):
        #     assert type(num_gpu) is int, "num_gpu option must be integer"
        #     if num_gpu == 0:
        #         return False
        #     else:
        #         try:
        #             utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
        #                                      subprocess.check_output(["nvidia-smi", "-q"]),
        #                                      flags=re.MULTILINE | re.DOTALL)
        #             indices = [i for i, x in enumerate(utilization) if x == ('0', '0')]
        #             assert len(indices) >= num_gpu, "Only {0} GPU's are available, change num_gpu parameter to {0}".format(
        #                 len(indices))
        #             return indices[:num_gpu]
        #         except OSError:
        #             log.info("No GPU's found. Using CPU.")
        #             return False
        #
        #
        # def optimistic_restore(session, save_file):
        #     """ Implementation from: https://github.com/tensorflow/tensorflow/issues/312 """
        #     print('Restoring model from:', save_file)
        #     reader = tf.train.NewCheckpointReader(save_file)
        #     saved_shapes = reader.get_variable_to_shape_map()
        #     var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
        #                         if var.name.split(':')[0] in saved_shapes])
        #     restore_vars = []
        #     name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        #     with tf.variable_scope('', reuse=True):
        #         for var_name, saved_var_name in var_names:
        #             curr_var = name2var[saved_var_name]
        #             var_shape = curr_var.get_shape().as_list()
        #             if var_shape == saved_shapes[saved_var_name]:
        #                 restore_vars.append(curr_var)
        #     saver = tf.train.Saver(restore_vars)
        #     saver.restore(session, save_file)


class CreateTFNetwork(ABC):
    def __init__(self, dataset):
        assert isinstance(dataset, CreateTFDataset)
        self.dataset = dataset
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        super(CreateTFNetwork, self).__init__()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_ops(self):
        pass

    @staticmethod
    def _fulconn_layer(input_data, output_dim, seq_len=1, activation_func=None):
        """Create a fully connected layer using matrix multiplication
        source:
        https://stackoverflow.com/questions/39808336/tensorflow-bidirectional-dynamic-rnn-none-values-error/40305673
        """
        # get input dimensions
        input_dim = int(input_data.get_shape()[1])
        weight = tf.get_variable(name="weights", shape=[input_dim, output_dim * seq_len],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / (2 * output_dim))))
        bias = tf.get_variable(name="bias", shape=[output_dim * seq_len],
                               initializer=tf.zeros_initializer)
        if activation_func:
            output = activation_func(tf.nn.bias_add(tf.matmul(input_data, weight), bias))
        else:
            output = tf.nn.bias_add(tf.matmul(input_data, weight), bias)
        return output, weight, bias

    @staticmethod
    def fully_connected_layer(inputs, num_outputs, activation_fn=tf.nn.relu, normalizer_fn=None, normalizer_params=None,
                              weights_initializer=initializers.xavier_initializer(), weights_regularizer=None,
                              biases_initializer=tf.zeros_initializer(), biases_regularizer=None, reuse=None,
                              variables_collections=None, outputs_collections=None, trainable=True, scope=None):
        """Fully connected layer wrapper so we don't have to

        :param inputs: A tensor of at least rank 2 and static value for the last dimension;
          i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
        :param num_outputs: Integer or long, the number of output units in the layer.
        :param activation_fn: Activation function. The default value is a ReLU function.
          Explicitly set it to None to skip it and maintain a linear activation.
        :param normalizer_fn: Normalization function to use instead of `biases`. If
          `normalizer_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
          default set to None for no normalizer function
        :param normalizer_params: Normalization function parameters.
        :param weights_initializer: An initializer for the weights.
        :param weights_regularizer: Optional regularizer for the weights.
        :param biases_initializer: An initializer for the biases. If None skip biases.
        :param biases_regularizer: Optional regularizer for the biases.
        :param reuse: Whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        :param variables_collections: Optional list of collections for all the variables or
          a dictionary containing a different list of collections per variable.
        :param outputs_collections: Collection to add the outputs.
        :param trainable: If `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        :param scope: Optional scope for variable_scope.
        :return The tensor variable representing the result of the series of operations.
        """

        return tf.contrib.layers.fully_connected(inputs, num_outputs, activation_fn=activation_fn,
                                                 normalizer_fn=normalizer_fn,
                                                 normalizer_params=normalizer_params,
                                                 weights_initializer=weights_initializer,
                                                 weights_regularizer=weights_regularizer,
                                                 biases_initializer=biases_initializer,
                                                 biases_regularizer=biases_regularizer,
                                                 reuse=reuse,
                                                 variables_collections=variables_collections,
                                                 outputs_collections=outputs_collections,
                                                 trainable=trainable,
                                                 scope=scope)

    def blstm(self, input_vector, sequence_length, layer_name="blstm_layer1", n_hidden=128, forget_bias=5.0,
              concat=True, state_keep_prob=1.0, input_keep_prob=1.0, output_keep_prob=1.0):
        """Create a bidirectional LSTM using code from the example at
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py

        :param input_vector: a tensor of inputs
        :param sequence_length: length of input without padding
        :param n_hidden: number of the hidden nodes
        :param forget_bias: bias on the forget gate. Increase to forget less
        :param layer_name: name of the layer for the graph
        :param output_keep_prob: probability of output nodes being kept
        :param state_keep_prob: probability of keeping a state node
        :param input_keep_prob: probability of keeping a input node
        :param concat: concatenate outputs, otherwise return forward and backward states together
        """
        with tf.variable_scope(layer_name):
            # Forward direction cell
            lstm_fw_cell = self._lstm_cell(n_hidden, forget_bias=forget_bias, output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)
            # Backward direction cell
            lstm_bw_cell = self._lstm_cell(n_hidden, forget_bias=forget_bias, output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                     lstm_bw_cell, input_vector,
                                                                     dtype=tf.float32,
                                                                     sequence_length=sequence_length)
            # concat two output layers so we can treat as single output layer
            if concat:
                output = tf.concat(outputs, 2)
            else:
                output = outputs
        return output

    def lstm(self, input_vector, sequence_length, n_hidden, layer_name, forget_bias=5.0, output_keep_prob=1.0,
             state_keep_prob=1.0, input_keep_prob=1.0):
        """Define basic dynamic long short term memory layer

        :param input_vector: a tensor of inputs
        :param sequence_length: length of input without padding
        :param n_hidden: number of the hidden nodes
        :param forget_bias: bias on the forget gate. Increase to forget less
        :param layer_name: name of the layer for the graph
        :param output_keep_prob: probability of output nodes being kept
        :param state_keep_prob: probability of keeping a state node
        :param input_keep_prob: probability of keeping a input node

        :return: output tensor
        """
        with tf.variable_scope(layer_name):
            # define LSTM
            rnn_cell = self._lstm_cell(n_hidden, forget_bias=forget_bias, output_keep_prob=output_keep_prob,
                                       state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)
            # dynamic rnn to control variable sequence lengths
            output, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                          inputs=input_vector,
                                          dtype=tf.float32,
                                          time_major=False,
                                          sequence_length=sequence_length)
        return output

    # TODO add L2 and L1 regularization to LSTM models
    # # # This depends on the
    @staticmethod
    def _lstm_cell(n_hidden, forget_bias=5.0, output_keep_prob=1.0, state_keep_prob=1.0, input_keep_prob=1.0):
        """Helper function to create a LSTM cell with dropoutwrapper

        :param n_hidden: number of the hidden nodes
        :param forget_bias: bias on the forget gate. Increase to forget less
        :param output_keep_prob: probability of output nodes being kept
        :param state_keep_prob: probability of keeping a state node
        :param input_keep_prob: probability of keeping a input node

        :return lstm cell
        """
        # define LSTM cell

        rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias)
        # create dropout option
        if output_keep_prob < 1 or state_keep_prob < 1 or state_keep_prob < 1:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_cell, output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)
        return rnn_cell

    def decoder_lstm(self, input_vector, n_hidden, sequence_length, batch_size, layer_name="decoder_lstm",
                     forget_bias=5, output_keep_prob=1.0, state_keep_prob=1.0, input_keep_prob=1.0):
        """Feeds output from lstm into input of same lstm cell

        :param input_vector: a tensor of inputs
        :param batch_size: batch size
        :param sequence_length: must be set because of
        :param n_hidden: number of the hidden nodes
        :param forget_bias: bias on the forget gate. Increase to forget less
        :param layer_name: name of the layer for the graph
        :param output_keep_prob: probability of output nodes being kept
        :param state_keep_prob: probability of keeping a state node
        :param input_keep_prob: probability of keeping a input node
        :return: stacked outputs from each cell
        """

        with tf.variable_scope(layer_name):

            state = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=forget_bias).zero_state(batch_size, tf.float32)
            outputs = []
            # loops through and creates outputs up to sequence length
            for time_step in range(sequence_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # create a cell
                rnn_cell = self._lstm_cell(n_hidden, forget_bias=forget_bias, output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)
                # call the cell with inputs and cell state
                (cell_output, state) = rnn_cell(inputs=input_vector, state=state)
                outputs.append(cell_output)
                # use output as new input
                input_vector = cell_output
            # return outputs
            output = tf.stack(outputs, 1)
        return output

    def conv_layer(self, input_vector, filters, ksize, padding="SAME", layer_name="convolutional_layer", dilate=1,
                   strides=(1, 1), bias_term=False, active=tf.nn.relu, bn=True, regularizer=None):
        """A standard convolutional layer
        :param input_vector: input data shape: [batch, in_height, in_width, in_channels]
        :param filters: output size
        :param ksize: kernel size eg [2,2]
        :param padding: "SAME" pads to create same shape of output,  "VALID" has no padding
        :param layer_name: name the layer with a variable scope
        :param dilate: size of dilation is (dilate-1)
        :param strides: size of stride in each dimension
        :param bias_term: boolean for including a bias term
        :param active: activation function, set to None if no activation
        :param bn: batch normalization option
        :param regularizer: regularizer for kernel regularization
        :return: output of convolution
        """
        with tf.variable_scope(layer_name):
            # filters is number of output nodes
            # kernel_size eg [2,2]
            # strides are (1,1)
            conv_out = tf.layers.conv2d(inputs=input_vector, filters=filters, kernel_size=ksize,
                                        strides=strides, padding=padding, name=layer_name,
                                        use_bias=bias_term, bias_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=regularizer, activation=active,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        dilation_rate=dilate)
        # batch normalization`
        if bn:
            with tf.variable_scope(layer_name + '_bn'):
                conv_out = self.batch_normalization(conv_out, name=layer_name + '_bn')

        return conv_out

    @staticmethod
    def _variable_with_weight_decay(name, shape, wd, dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer()):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        :param name: name of the variable
        :param shape: list of ints
        :param wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        :param initializer: initialization function

        :return Variable Tensor
        """
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    @staticmethod
    def batch_normalization(input_vector, name):
        """Batch normalization as implemented by http://arxiv.org/abs/1502.03167

        :param input_vector: input data
        :param name: batchnorm name
        :return batch normalized layer of same shape as input_vector
        """
        # get input shape and size of each set of nodes
        ksize = input_vector.get_shape().as_list()
        ksize = [ksize[-1]]
        # get mini batch mean and variance
        mean, variance = tf.nn.moments(input_vector, [0, 1, 2], name=name + '_moments')
        # create learned variables scale and offset
        scale = tf.get_variable(name + "_scale",
                                shape=ksize, initializer=tf.contrib.layers.variance_scaling_initializer())
        offset = tf.get_variable(name + "_offset",
                                 shape=ksize, initializer=tf.contrib.layers.variance_scaling_initializer())
        return tf.nn.batch_normalization(input_vector, mean=mean, variance=variance, scale=scale, offset=offset,
                                         variance_epsilon=1e-5)

    def inception_layer(self, input_vector, layer_name="inception_layer", times=16, regularizer=None, max_pool=True,
                        dilate=False):
        """Inception module with dilate conv layer from http://arxiv.org/abs/1512.00567
        :param input_vector: input data
        :param layer_name: layer name
        :param times: number of filters
        :param regularizer: optional regularizer
        :param max_pool: if False, use average pool for one of the branches otherwise uses max_pool
        :param dilate: boolean option
        :return: tensor of concatenation of all 6 branches
        """
        with tf.variable_scope(layer_name):
            if max_pool:
                with tf.variable_scope('branch1_MaxPooling'):
                    max_pool = tf.layers.max_pooling2d(input_vector, pool_size=(1, 3), strides=(1, 1),
                                                       padding='SAME', name='max_pool0a1x3')
                    conv1a = self.conv_layer(max_pool, filters=times * 3, ksize=[1, 1], padding='SAME',
                                             layer_name='conv1a_1x1', regularizer=regularizer)

            else:
                with tf.variable_scope('branch1_AvgPooling'):
                    avg_pool = tf.layers.average_pooling2d(input_vector, pool_size=(1, 3), strides=(1, 1),
                                                           padding='SAME', name='avg_pool0a1x3')
                    conv1a = self.conv_layer(avg_pool, filters=times * 3, ksize=[1, 1], padding='SAME',
                                             layer_name='conv1a_1x1', regularizer=regularizer)
            with tf.variable_scope('branch2_1x1'):
                conv0b = self.conv_layer(input_vector, filters=times * 3, ksize=[1, 1], padding='SAME',
                                         layer_name='conv0b_1x1', regularizer=regularizer)
            with tf.variable_scope('branch3_1x3'):
                conv0c = self.conv_layer(input_vector, filters=times * 2, ksize=[1, 1], padding='SAME',
                                         layer_name='conv0c_1x1', regularizer=regularizer)
                conv1c = self.conv_layer(conv0c, filters=times * 3, ksize=[1, 3], padding='SAME',
                                         layer_name='conv1c_1x3', regularizer=regularizer)
            with tf.variable_scope('branch4_1x5'):
                conv0d = self.conv_layer(input_vector, filters=times * 2, ksize=[1, 1], padding='SAME',
                                         layer_name='conv0d_1x1', regularizer=regularizer)
                conv1d = self.conv_layer(conv0d, filters=times * 3, ksize=[1, 5], padding='SAME',
                                         layer_name='conv1d_1x5', regularizer=regularizer)
            if dilate:
                with tf.variable_scope('branch5_1x3_dilate_2'):
                    conv0e = self.conv_layer(input_vector, filters=times * 2, ksize=[1, 1], padding='SAME',
                                             layer_name='conv0e_1x1', regularizer=regularizer)
                    conv1e = self.conv_layer(conv0e, filters=times * 3, ksize=[1, 3], padding='SAME',
                                             layer_name='conv1e_1x3_d2', dilate=2, regularizer=regularizer)
                with tf.variable_scope('branch6_1x3_dilate_3'):
                    conv0f = self.conv_layer(input_vector, filters=times * 2, ksize=[1, 1], padding='SAME',
                                             layer_name='conv0f_1x1', regularizer=regularizer)
                    conv1f = self.conv_layer(conv0f, filters=times * 3, ksize=[1, 3], padding='SAME',
                                             layer_name='conv1f_1x3_d3', dilate=3, regularizer=regularizer)
                output_layer = tf.concat([conv1a, conv0b, conv1c, conv1d, conv1e, conv1f], axis=-1, name='concat')
            else:
                output_layer = tf.concat([conv1a, conv0b, conv1c, conv1d], axis=-1, name='concat')

        return output_layer

    def residual_layer(self, indata, out_channel, ksizes=([1, 1], [1, 1], [1, 3], [1, 1]),
                       layer_name="residual_layer", i_bn=False, regularizer=None):
        """Residual layer with batch normalization option.

        https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
        :param indata: input data shape: [batch, in_height, in_width, in_channels]
        :param out_channel: size of out_channel
        :param ksizes: kernal sizes of [branch1, layer1, layer2, layer3]
        :param i_bn: batch normalization on middle layer
        :param regularizer: optional weight regularizer
        :param layer_name: variable scope for layer
        :return: output tensor
        """
        # fea_shape = indata.get_shape().as_list()
        # in_channel = fea_shape[-1]
        with tf.variable_scope(layer_name):
            with tf.variable_scope('branch1'):
                indata_cp = self.conv_layer(indata, filters=out_channel, ksize=ksizes[0],
                                            padding='SAME', layer_name='conv1', bn=i_bn, active=None,
                                            regularizer=regularizer)
            with tf.variable_scope('branch2'):
                conv_out1 = self.conv_layer(indata, filters=out_channel, ksize=ksizes[1],
                                            padding='SAME', layer_name='conv2a', bias_term=False,
                                            regularizer=regularizer)
                # 1x3 on the conv layer
                conv_out2 = self.conv_layer(conv_out1, filters=out_channel, ksize=ksizes[2],
                                            padding='SAME', layer_name='conv2b', bias_term=False,
                                            regularizer=regularizer)
                conv_out3 = self.conv_layer(conv_out2, filters=out_channel, ksize=ksizes[3],
                                            padding='SAME', layer_name='conv2c', bias_term=False, active=None,
                                            regularizer=regularizer)
            with tf.variable_scope('plus'):
                relu_out = tf.nn.relu(indata_cp + conv_out3, name='final_relu')

        return relu_out

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        source: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            summary = tf.summary.scalar('mean', mean)
            self.summaries.append(summary)


class CreateTFDataset(ABC):
    def __init__(self):
        self.iterator = "iterator"
        super(CreateTFDataset, self).__init__()

    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def create_iterator(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def load_training_data(self):
        pass

    @abstractmethod
    def process_graph_output(self):
        pass

    @abstractmethod
    def load_inference_data(self):
        pass


if __name__ == "__main__":
    print("This is a library file. Nothing to execute")
    raise SystemExit
