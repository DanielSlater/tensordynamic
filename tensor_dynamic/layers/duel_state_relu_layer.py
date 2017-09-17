import math

import operator
import tensorflow as tf
import numpy as np
from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender


class DuelStateReluLayer(HiddenLayer):
    ACTIVE_THRESHOLD = 0.25  # 0.2

    def __init__(self,
                 input_layer,
                 output_nodes,
                 width_binarizer_constant=1e-4,
                 width_regularizer_constant=1e-2,
                 inactive_nodes_to_leave=3,
                 session=None,
                 weights=None,
                 bias=None,
                 width=None,
                 layer_noise_std=None,
                 batch_normalize_input=False,
                 drop_out_prob=None,
                 non_liniarity=tf.nn.relu,
                 weight_extender_func=noise_weight_extender,
                 name='DuelStateReluLayer',
                 freeze=False):
        super(DuelStateReluLayer, self).__init__(input_layer, output_nodes,
                                                 session=session, weight_extender_func=weight_extender_func,
                                                 weights=weights,
                                                 bias=bias,
                                                 bactivate=False,
                                                 layer_noise_std=layer_noise_std,
                                                 batch_normalize_input=batch_normalize_input,
                                                 drop_out_prob=drop_out_prob,
                                                 non_liniarity=non_liniarity,
                                                 name=name,
                                                 freeze=freeze)
        self._width = self._create_variable("width",
                                            (BaseLayer.OUTPUT_BOUND_VALUE,),
                                            width if width is not None else np.ones(self.output_nodes,
                                                                                    dtype=np.float32))
        self._width_regularizer_constant = width_regularizer_constant
        self._width_binarizer_constant = width_binarizer_constant
        self._inactive_nodes_to_leave = inactive_nodes_to_leave

    def _layer_activation(self, input_activation, is_train):
        activation = super(DuelStateReluLayer, self)._layer_activation(input_activation, is_train)
        return activation * self._width

    def unsupervised_cost_train(self):
        return tf.reduce_sum(self._width * (1 - self._width)) * self._width_binarizer_constant + \
               tf.reduce_sum(self._width) * self._width_regularizer_constant

    @property
    def kwargs(self):
        kwargs = super(DuelStateReluLayer, self).kwargs

        # bactivate is not optional for these layers
        del kwargs['bactivate']
        del kwargs['bactivation_loss_func']

        kwargs['width_regularizer_constant'] = self._width_regularizer_constant
        kwargs['width_binarizer_constant'] = self._width_binarizer_constant
        kwargs['inactive_nodes_to_leave'] = self._inactive_nodes_to_leave

        return kwargs

    def width(self):
        """
        return a 1D array of the widths used for the layer
        """
        return self.session.run(self._width)

    def active_nodes(self):
        return len([x for x in np.abs(self.width()) if x > self.ACTIVE_THRESHOLD])

    def inactive_nodes(self):
        return self._output_nodes - self.active_nodes()

    def prune(self, inactive_nodes_to_leave=3):
        """
        Removes inactive nodes from the layer

        Parameters
        ----------
        inactive_nodes_to_leave: int
            Number of inactive nodes we want left after pruning

        Returns
        -------
        bool: True we we pruned nodes, otherwise False
        """
        # may need to validate we aren't the output layer...
        active_nodes = self.active_nodes()
        nodes_to_prune = self.output_nodes - (active_nodes + inactive_nodes_to_leave)
        if nodes_to_prune <= 0:
            # no need to prune if we have only 1 inactive node
            # TODO resize so as to leave 1 inactive node?
            return False

        # find the nodes_to_prune least active nodes
        width = self.width()
        width_abs = np.abs(width)
        width_sorted = sorted(width_abs, reverse=False)
        prune_below_width = width_sorted[nodes_to_prune - 1]
        prune_indexes = [i for i, x in enumerate(width_abs) if x <= prune_below_width]

        self.resize(output_nodes_to_prune=prune_indexes)

        print("layer %s pruned node size now %s" % (self.layer_number, self.output_nodes))

        return True

    def grow(self, inactive_nodes_to_leave=3):
        active_nodes = len([x for x in np.abs(self.width()) if x > self.active_nodes()])
        inactive_nodes = self.output_nodes - active_nodes
        if inactive_nodes >= inactive_nodes_to_leave:
            return False

        width = self.width()
        width_abs = np.abs(width)
        max_index = max(enumerate(width_abs), key=operator.itemgetter(1))[0]

        # add some nodes
        self.resize(split_output_nodes=[max_index])

        print("layer %s added node size now %s" % (self.layer_number, self.output_nodes))

        # set newly created node to active
        width = np.append(width, 1.0)
        self._session.run(self._width.assign(width))

    def resize(self, new_output_nodes=None, output_nodes_to_prune=None, input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None,
               split_nodes_noise_std=.01):
        width = self.width()
        output_nodes_increase = (new_output_nodes or self._output_nodes[0]) - self._output_nodes[0]

        super(DuelStateReluLayer, self).resize(new_output_nodes, output_nodes_to_prune, input_nodes_to_prune,
                                               split_output_nodes, split_input_nodes, split_nodes_noise_std)

        if output_nodes_increase > 0:
            # set newly created node to active
            width = np.append(width, [1.0]*output_nodes_increase)
            self.session.run(tf.assign(self._width, width, validate_shape=False))

    @property
    def assign_op(self):
        return self._width.assign(tf.clip_by_value(self._width, 0.01, 0.99))
