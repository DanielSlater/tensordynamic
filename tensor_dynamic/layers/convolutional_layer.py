from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender
import tensorflow as tf


class ConvolutionalLayer(BaseLayer):
    def __init__(self,
                 input_layer,
                 convolution_dimensions,  # 3d (width, height, convolutions)
                 stride=(1, 1),
                 padding='SAME',  # only same currently supported
                 session=None,
                 weights=None,
                 bias=None,
                 weight_extender_func=noise_weight_extender,
                 name='ConvolutionalLayer',
                 freeze=False,
                 non_liniarity=tf.nn.relu):
        assert len(input_layer.output_nodes) == 3, "expected input to have 3 dimensions"
        assert len(convolution_dimensions) == 3, "expected output to have 3 dimensions"
        self._convolution_dimensions = convolution_dimensions

        output_nodes = (input_layer.output_nodes[0] / stride[0],
                        input_layer.output_nodes[1] / stride[1],
                        convolution_dimensions[2])

        super(ConvolutionalLayer, self).__init__(input_layer,
                                                 output_nodes,
                                                 session=session,
                                                 weight_extender_func=weight_extender_func,
                                                 freeze=freeze,
                                                 name=name)

        self._weights = self._create_variable("weights",
                                              (convolution_dimensions[0], convolution_dimensions[1],
                                               BaseLayer.INPUT_DIM_3_BOUND_VALUE, BaseLayer.OUTPUT_DIM_3_BOUND_VALUE),
                                              weights if weights is not None else tf.random_normal(
                                                  [convolution_dimensions[0], convolution_dimensions[1],
                                                   self.input_layer.output_nodes[2], convolution_dimensions[2]]))
        self._bias = self._create_variable("bias",
                                           (BaseLayer.OUTPUT_DIM_3_BOUND_VALUE,),
                                           bias if bias is not None else tf.zeros((convolution_dimensions[2],)))
        self._stride = stride
        self._padding = padding
        self._non_liniarity = non_liniarity

    @property
    def convolutional_nodes(self):
        return self._convolution_dimensions

    def _layer_activation(self, input_activation, is_train):
        x = tf.nn.conv2d(input_activation, self._weights, strides=[1, self._stride[0], self._stride[1], 1], padding='SAME')
        x = tf.nn.bias_add(x, self._bias)
        return self._non_liniarity(x)

    def resize(self, new_output_nodes=None,
               output_nodes_to_prune=None,
               input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None, split_nodes_noise_std=.1):
        if isinstance(new_output_nodes, int):
            temp = list(self.output_nodes)
            temp[2] = new_output_nodes
            new_output_nodes = tuple(temp)

        super(ConvolutionalLayer, self).resize(new_output_nodes,
                                               output_nodes_to_prune=output_nodes_to_prune,
                                               input_nodes_to_prune=input_nodes_to_prune,
                                               split_output_nodes=split_output_nodes,
                                               split_input_nodes=split_input_nodes)

    def clone(self, session=None):
        """Produce a clone of this layer AND all connected upstream layers

        Args:
            session (tensorflow.Session): If passed in the clone will be created with all variables initialised in this session
                                          If None then the current session of this layer is used

        Returns:
            tensorflow_dynamic.BaseLayer: A copy of this layer and all upstream layers
        """
        new_self = self.__class__(self.input_layer.clone(session or self.session),
                                  self.convolutional_nodes,
                                  session=session or self._session,
                                  **self.kwargs)

        return new_self