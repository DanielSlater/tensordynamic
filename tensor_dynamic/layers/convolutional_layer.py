from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender
import tensorflow as tf


class ConvolutionalLayer(BaseLayer):
    def __init__(self,
                 input_layer,
                 convolution_dimensions,#3d (width, height, convolutions)
                 stride=1,
                 padding='SAME',#only same currently supported
                 session=None,
                 weight_extender_func=noise_weight_extender,
                 name=None,
                 freeze=False,
                 non_liniarity=tf.nn.relu):
        assert len(input_layer.output_nodes) == 3, "expected input to have 3 dimensions"
        assert len(convolution_dimensions) == 3, "expected output to have 3 dimensions"
        output_nodes = (input_layer.output_nodes[0]/stride,
                        input_layer.output_nodes[1]/stride,
                        convolution_dimensions[2])

        super(ConvolutionalLayer, self).__init__(input_layer,
                                                 output_nodes,
                                                 session=session,
                                                 weight_extender_func=weight_extender_func,
                                                 freeze=freeze,
                                                 name=name)

        self._weights = tf.Variable(tf.random_normal([convolution_dimensions[0], convolution_dimensions[1],
                                                      self.input_layer.output_nodes[2], convolution_dimensions[2]]))
        self._bias = tf.Variable(tf.random_normal([convolution_dimensions[2]]))
        self._stride = stride
        self._padding = padding
        self._non_liniarity = non_liniarity

    def _activation_process(self, input_tensor, train):
        x = tf.nn.conv2d(input_tensor, self._weights, strides=[1, self._stride, self._stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, self._bias)
        return self._non_liniarity(x)

    @property
    def bactivation_train(self):
        raise NotImplementedError()

    @lazyprop
    def activation_predict(self): #TODO: Move to base class
        return self._activation_process(self.input_layer.activation_predict, False)

    def resize(self, new_output_nodes=None, output_nodes_to_prune=None, input_nodes_to_prune=None, split_output_nodes=None,
               split_input_nodes=None, split_nodes_noise_std=.1):
        pass

    @property
    def bactivation_predict(self):
        raise NotImplementedError()

    @lazyprop
    def activation_train(self): #TODO: Move to base class
        return self._activation_process(self.input_layer.activation_predict, True)