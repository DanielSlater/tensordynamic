import numpy as np

import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.node_importance import node_importance_optimal_brain_damage


class ConvolutionalLayer(BaseLayer):
    MINIMUM_GROW_AMOUNT = 1

    def __init__(self,
                 input_layer,
                 convolution_dimensions,  # 3d (width, height, convolutions)
                 stride=(1, 1, 1),
                 padding='SAME',  # only same currently supported
                 session=None,
                 weights=None,
                 bias=None,
                 weight_extender_func=None,
                 weight_initializer_func=None,
                 bias_initializer_func=None,
                 node_importance_func=None,
                 name='ConvolutionalLayer',
                 freeze=False,
                 layer_noise_std=None,
                 drop_out_prob=None,
                 batch_normalize_input=False,
                 non_liniarity=tf.nn.relu):
        assert len(input_layer.output_nodes) == 3, "expected input to have 3 dimensions"
        assert len(convolution_dimensions) == 3, "expected output to have 3 dimensions"
        self._convolution_dimensions = convolution_dimensions

        output_nodes = (input_layer.output_nodes[0] / stride[0],
                        input_layer.output_nodes[1] / stride[1],
                        convolution_dimensions[2] / stride[2])

        super(ConvolutionalLayer, self).__init__(input_layer,
                                                 output_nodes,
                                                 session=session,
                                                 weight_extender_func=weight_extender_func,
                                                 weight_initializer_func=weight_initializer_func,
                                                 bias_initializer_func=bias_initializer_func,
                                                 freeze=freeze,
                                                 layer_noise_std=layer_noise_std,
                                                 drop_out_prob=drop_out_prob,
                                                 batch_normalize_input=batch_normalize_input,
                                                 name=name)

        self._weights = self._create_variable("weights",
                                              (convolution_dimensions[0], convolution_dimensions[1],
                                               BaseLayer.INPUT_DIM_3_BOUND_VALUE, BaseLayer.OUTPUT_DIM_3_BOUND_VALUE),
                                              weights)
        self._bias = self._create_variable("bias",
                                           (BaseLayer.OUTPUT_DIM_3_BOUND_VALUE,),
                                           bias)
        self._node_importance_func = self._get_property_or_default(node_importance_func,
                                                                   '_node_importance_func',
                                                                   node_importance_optimal_brain_damage)
        self._stride = stride
        self._padding = padding
        self._non_liniarity = non_liniarity

    @property
    def convolutional_nodes(self):
        return self._convolution_dimensions

    def _layer_activation(self, input_activation, is_train):
        x = tf.nn.conv2d(input_activation, self._weights, strides=(1,) + self._stride,
                         padding=self._padding)
        x = tf.nn.bias_add(x, self._bias)
        return self._non_liniarity(x)

    def resize(self, new_output_nodes=None,
               output_nodes_to_prune=None,
               input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None,
               data_set_train=None,
               data_set_validation=None,
               no_splitting_or_pruning=False,
               split_nodes_noise_std=.1):
        if isinstance(new_output_nodes, int):
            temp = list(self.output_nodes)
            temp[2] = new_output_nodes / self._stride[2]
            new_output_nodes = tuple(temp)

        super(ConvolutionalLayer, self).resize(new_output_nodes,
                                               output_nodes_to_prune=output_nodes_to_prune,
                                               input_nodes_to_prune=input_nodes_to_prune,
                                               split_output_nodes=split_output_nodes,
                                               split_input_nodes=split_input_nodes,
                                               split_nodes_noise_std=split_nodes_noise_std,
                                               data_set_train=data_set_train,
                                               data_set_validation=data_set_validation,
                                               no_splitting_or_pruning=no_splitting_or_pruning)

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

    def has_resizable_dimension(self):
        return True

    def get_resizable_dimension_size(self):
        return self.convolutional_nodes[2]

    def _get_node_importance(self):
        # simplest way to do this just sum the weights in each convolution
        weights = self.session.run(self._weights)
        bias = self.session.run(self._bias)

        # group everything by convolutional output node
        weights = weights.transpose(3, 0, 1, 2).reshape(self.convolutional_nodes[2], -1)
        importance = [np.sum(weights[i]) + bias[i] for i in range(self.convolutional_nodes[2])]
        return importance

    def get_resizable_dimension(self):
        return 2

    @property
    def kwargs(self):
        kwargs = super(ConvolutionalLayer, self).kwargs

        kwargs['stride'] = self._stride
        kwargs['padding'] = self._padding
        kwargs['non_liniarity'] = self._non_liniarity

        return kwargs