import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import clear_all_lazyprops
import math


class MaxPoolLayer(BaseLayer):
    def __init__(self,
                 input_layer,
                 ksize=(2, 2, 1),
                 strides=(2, 2, 1),
                 padding="SAME",
                 input_noise_std=None,
                 session=None,
                 name='MaxPoolLayer'):
        assert len(input_layer.output_nodes) == 3, "expected 3 output dimensions"
        assert len(ksize) == 3, "expected 3 ksize dimensions"
        assert len(strides) == 3, "expected 3 strides dimensions"

        output_nodes = self._calculate_output_nodes(input_layer, strides)

        super(MaxPoolLayer, self).__init__(input_layer,
                                           output_nodes,
                                           input_noise_std=input_noise_std,
                                           session=session,
                                           name=name)

        self._strides = strides
        self._ksize = ksize
        self._padding = padding

    @staticmethod
    def _calculate_output_nodes(input_layer, strides):
        return (int(math.ceil(input_layer.output_nodes[0] / float(strides[0]))),
                int(math.ceil(input_layer.output_nodes[1] / float(strides[1]))),
                int(math.ceil(input_layer.output_nodes[2] / float(strides[2]))))

    def _layer_activation(self, input_tensor, is_train):
        return tf.nn.max_pool(input_tensor, ksize=(1,) + self._strides,
                              strides=(1,) + self._ksize,
                              padding=self._padding)

    def resize(self, new_output_nodes=None,
               output_nodes_to_prune=None,
               input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None, split_nodes_noise_std=.1):
        output_nodes = self._calculate_output_nodes(self.input_layer, self._strides)

        if self.output_nodes != output_nodes:
            self._output_nodes = output_nodes

            clear_all_lazyprops(self)
            for layer in self.downstream_layers:
                clear_all_lazyprops(layer)

            if self.next_layer is not None and self.next_layer.resize_needed():
                # TODO: D.S make sure resize is consistant, i.e new nodes are not just created on the end...
                # Must do this at some point
                self._next_layer.resize(input_nodes_to_prune=output_nodes_to_prune,
                                        split_input_nodes=split_output_nodes)

    def clone(self, session=None):
        """Produce a clone of this layer AND all connected upstream layers

        Args:
            session (tensorflow.Session): If passed in the clone will be created with all variables initialised in this session
                                          If None then the current session of this layer is used

        Returns:
            tensorflow_dynamic.BaseLayer: A copy of this layer and all upstream layers
        """
        new_self = self.__class__(self.input_layer.clone(session or self.session),
                                  session=session or self._session,
                                  name=self._name)

        return new_self

    @property
    def kwargs(self):
        kwargs = super(MaxPoolLayer, self).kwargs

        kwargs['stride'] = self._stride
        kwargs['padding'] = self._padding
        kwargs['ksize'] = self._ksize

        return kwargs