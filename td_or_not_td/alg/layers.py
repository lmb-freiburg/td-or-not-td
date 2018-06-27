import tensorflow as tf
from functools import reduce
from operator import mul


class NetworkCreator:
    """call public functions after each other to create a network"""

    def __init__(self,
                 network_name, data_format="NHWC", reuse=False, input_layer=None,
                 print_shape=False):
        self.network_name = network_name
        assert data_format in ("NHWC", "NCHW")
        self._data_format = data_format
        self._reuse = reuse  # call this class with same network_name & reuse=True to reuse weights
        self._connector = input_layer
        self._print_shape = print_shape

        self._layer_number = 0

        if self._print_shape:
            print("\nNetwork:", network_name, self._data_format)

        self._flatting_channels = None
        self._flatting_dimension = None

    def relu(self, input_layer=None):
        input_layer = self._config_input(input_layer)
        if self._print_shape:
            print("ReLu")
        return self._config_output(tf.nn.relu(input_layer))

    def conv_layer(self, output_channels, filter_size, stride, input_layer=None):
        with tf.variable_scope(self.network_name):
            with tf.variable_scope(self._get_name(), reuse=self._reuse):
                input_layer = self._config_input(input_layer)

                if self._data_format == "NHWC":
                    input_channels = input_layer.get_shape().as_list()[3]
                    input_size_h = input_layer.get_shape().as_list()[1]
                    stride_shape = [1, stride, stride, 1]
                else:
                    input_channels = input_layer.get_shape().as_list()[1]
                    input_size_h = input_layer.get_shape().as_list()[2]
                    stride_shape = [1, 1, stride, stride]

                init = tf.contrib.layers.variance_scaling_initializer(factor=1., uniform=True)
                shape = [filter_size, filter_size, input_channels, output_channels]
                w_conv = tf.get_variable("w", shape=shape, dtype=tf.float32, initializer=init)

                init = tf.constant_initializer(0.0)
                shape = [output_channels]
                b_conv = tf.get_variable("b", shape=shape, dtype=tf.float32, initializer=init)

                conv = tf.nn.conv2d(input_layer, w_conv, stride_shape,
                                    padding='VALID',
                                    data_format=self._data_format)

                output = tf.nn.bias_add(conv, b_conv, data_format=self._data_format)

                if self._print_shape:
                    if (input_size_h - filter_size) % stride != 0:
                        print("unclean convolution:")
                    print(self._layer_number, "Conv",
                          input_layer.get_shape().as_list(),
                          " >> {",
                          [filter_size, filter_size, input_channels, output_channels],
                          [stride],
                          "} >> ",
                          output.get_shape().as_list())
                return self._config_output(output)

    def make_unflat(self, input_layer=None, unflatting_channels=None):
        input_layer = self._config_input(input_layer)
        input_size = input_layer.get_shape().as_list()[1]

        if unflatting_channels is None:  # use value from last flatting:
            if self._flatting_channels is None:
                raise ValueError("input_channels unknown for upconv unflatting")
            output_size = self._flatting_channels
            self._flatting_channels = None
        else:
            output_size = unflatting_channels

        assert input_size % output_size == 0
        dim = self._get_square(input_size // output_size)
        assert dim != -1
        if self._data_format == "NHWC":
            output_shape = [-1, dim, dim, output_size]
        else:
            output_shape = [-1, output_size, dim, dim]
        output = tf.reshape(input_layer, output_shape)
        if self._print_shape:
            print("unflat",
                  input_layer.get_shape().as_list(),
                  " >> ",
                  output.get_shape().as_list())
        return self._config_output(output)

    # input_channels needed if first upconv after fc layer (and no size defined by _makeflat)
    def upconv_layer(self, output_channels, filter_size, stride,
                     input_layer=None, input_channels=None):
        with tf.variable_scope(self.network_name):
            with tf.variable_scope(self._get_name(), reuse=self._reuse):
                input_layer = self._config_input(input_layer)
                if len(input_layer.get_shape().as_list()) == 2:
                    if input_channels is None:  # use value from last flatting:
                        if self._flatting_channels is None:
                            raise ValueError("input_channels unknown for upconv unflatting")
                        unflatting_channels = self._flatting_channels
                        self._flatting_channels = None
                    else:
                        unflatting_channels = input_channels

                    input_layer = self._make_unflat(unflatting_channels, input_layer)

                if self._data_format == "NHWC":
                    input_channels = input_layer.get_shape().as_list()[3]
                    input_size_h = input_layer.get_shape().as_list()[1]
                    input_size_w = input_layer.get_shape().as_list()[2]
                    stride_shape = [1, stride, stride, 1]
                    output_size_h = (input_size_h - 1) * stride + filter_size
                    output_size_w = (input_size_w - 1) * stride + filter_size
                    output_shape = tf.stack([tf.shape(input_layer)[0],
                                             output_size_h, output_size_w,
                                             output_channels])
                else:
                    input_channels = input_layer.get_shape().as_list()[1]
                    input_size_h = input_layer.get_shape().as_list()[2]
                    input_size_w = input_layer.get_shape().as_list()[3]
                    stride_shape = [1, 1, stride, stride]
                    output_size_h = (input_size_h - 1) * stride + filter_size
                    output_size_w = (input_size_w - 1) * stride + filter_size
                    output_shape = tf.stack([tf.shape(input_layer)[0],
                                             output_channels,
                                             output_size_h, output_size_w])

                init = tf.contrib.layers.variance_scaling_initializer(factor=1., uniform=True)
                shape = [filter_size, filter_size, output_channels, input_channels]
                w_upconv = tf.get_variable("w", shape=shape, dtype=tf.float32, initializer=init)

                init = tf.constant_initializer(0.0)
                shape = [output_channels]
                b_upconv = tf.get_variable("b", shape=shape, dtype=tf.float32, initializer=init)

                upconv = tf.nn.conv2d_transpose(input_layer, w_upconv, output_shape, stride_shape,
                                                padding='VALID',
                                                data_format=self._data_format)

                output = tf.nn.bias_add(upconv, b_upconv, data_format=self._data_format)

                output = tf.reshape(output, output_shape)

                if self._print_shape:
                    print(self._layer_number, "upconv",
                          input_layer.get_shape().as_list(),
                          " >> {",
                          [filter_size, filter_size, input_channels, output_channels],
                          [stride],
                          "} >> ",
                          output.get_shape().as_list())

                return self._config_output(output)

    def fc_layer(self, output_size,
                 input_layer=None):  # output_size==-1: use size from last flatting
        with tf.variable_scope(self.network_name):
            with tf.variable_scope(self._get_name(), reuse=self._reuse):
                input_layer = self._config_input(input_layer)
                if len(input_layer.get_shape().as_list()) == 4:
                    input_layer = self._make_flat(input_layer)
                input_size = input_layer.get_shape().as_list()[1]

                if output_size == -1:
                    if self._flatting_dimension is None:
                        raise ValueError("input_dimension unknown for unflatting fc_layer")
                    output_size = self._flatting_dimension
                    self._flatting_dimension = None

                init = tf.contrib.layers.variance_scaling_initializer(factor=1., uniform=True)
                shape = [input_size, output_size]
                w_fc = tf.get_variable("w", shape=shape, dtype=tf.float32, initializer=init)

                init = tf.constant_initializer(0.0)
                shape = [output_size]
                b_fc = tf.get_variable("b", shape=shape, dtype=tf.float32, initializer=init)

                output = tf.matmul(input_layer, w_fc) + b_fc

                if self._print_shape:
                    print(self._layer_number, "fc",
                          input_layer.get_shape().as_list(),
                          " >> ",
                          output.get_shape().as_list())

                return self._config_output(output)
    
    def extend(self, new_input, input_layer=None):
        input_layer = self._config_input(input_layer)
        if len(input_layer.get_shape().as_list()) == 4:
            input_layer = self._make_flat(input_layer)
        if len(new_input.get_shape().as_list()) == 4:
            new_input = self._make_flat(new_input)
        output = tf.concat([input_layer, new_input], 1)
        if self._print_shape:
            print("comb",
                  input_layer.get_shape().as_list(),
                  "+",
                  new_input.get_shape().as_list(),
                  " >> ",
                  output.get_shape().as_list())
        return self._config_output(output)

    def _make_flat(self, input_layer=None):
        """called when fc_layer gets a conv layer output
        also saves parameters for potential unflatting"""
        input_layer = self._config_input(input_layer)
        # saving number of channels in case of a "_make_unflat" call in the future:
        if self._data_format == "NHWC":
            self._flatting_channels = input_layer.get_shape().as_list()[3]
        else:
            self._flatting_channels = input_layer.get_shape().as_list()[1]
        dim = reduce(mul, input_layer.get_shape().as_list()[1:])
        self._flatting_dimension = dim
        output = tf.reshape(input_layer, [-1, dim])
        if self._print_shape:
            print("flat",
                  input_layer.get_shape().as_list(),
                  " >> ",
                  output.get_shape().as_list())
        return self._config_output(output)

    def _make_unflat(self, output_size, input_layer=None):
        """called when upconv layer gets flat input"""
        input_layer = self._config_input(input_layer)
        input_size = input_layer.get_shape().as_list()[1]
        assert input_size % output_size == 0
        dim = self._get_square(input_size // output_size)
        assert dim != -1
        if self._data_format == "NHWC":
            output_shape = [-1, dim, dim, output_size]
        else:
            output_shape = [-1, output_size, dim, dim]
        output = tf.reshape(input_layer, output_shape)
        if self._print_shape:
            print("unflat",
                  input_layer.get_shape().as_list(),
                  " >> ",
                  output.get_shape().as_list())
        return self._config_output(output)

    def _get_name(self):
        """gives every layer a different name)"""
        self._layer_number += 1
        layer_name = "layer_{}".format(self._layer_number)
        return layer_name

    def _config_input(self, input_layer):
        if input_layer is None:  # try use previous output:
            assert self._connector is not None
            return self._connector
        else:
            return input_layer

    def _config_output(self, output):
        self._connector = output
        return self._connector

    @staticmethod
    def _get_square(input_int):
        assert input_int > 0
        x = input_int // 2
        seen = [x]
        while x * x != input_int:
            x = (x + (input_int // x)) // 2
            if x in seen:
                return -1
            seen.append(x)
        return x
