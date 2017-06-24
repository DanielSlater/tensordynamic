import functools

import operator

from tensor_dynamic.layers.base_layer import BaseLayer
import tensorflow as tf

from tensor_dynamic.lazyprop import lazyprop, clear_all_lazyprops


class FlattenLayer(BaseLayer):
    def __init__(self,
                 input_layer,
                 session=None,
                 name='FlattenLayer'):
        assert len(input_layer.output_nodes) > 1, "expected multiple input dims"
        output_nodes = functools.reduce(operator.mul, input_layer.output_nodes)

        super(FlattenLayer, self).__init__(input_layer,
                                           (output_nodes,),
                                           session=session,
                                           name=name)

    def _layer_activation(self, input_activation, is_train):
        # TODO can this be done using tf.shape? like input noise?
        return tf.reshape(input_activation, [-1, self.output_nodes[0]])

    def resize(self, new_output_nodes=None,
               output_nodes_to_prune=None,
               input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None, split_nodes_noise_std=.1):
        output_nodes = (functools.reduce(operator.mul, self.input_layer.output_nodes),)

        if self.output_nodes != output_nodes:
            self._output_nodes = output_nodes

            # can't resize the tf.reshape so just regen everything
            clear_all_lazyprops(self)
            for layer in self.downstream_layers:
                clear_all_lazyprops(layer)

            if self.next_layer is not None and self.next_layer.resize_needed():
                # TODO: D.S make sure resize is consistant, i.e new nodes are not just created on the end...
                # Must do this at some point
                self._next_layer.resize(input_nodes_to_prune=output_nodes_to_prune, split_input_nodes=split_output_nodes)

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
