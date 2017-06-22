import numpy as np
import sys

import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import xavier_init, create_hessian_variable_op
from tensor_dynamic.weight_functions import noise_weight_extender


class Layer(BaseLayer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 bactivate=False,
                 freeze=False,
                 non_liniarity=tf.nn.sigmoid,
                 weight_extender_func=noise_weight_extender,
                 unsupervised_cost=1.,
                 supervised_cost=1.,
                 noise_std=None,
                 bactivation_loss_func=squared_loss,
                 name='Layer'):
        super(Layer, self).__init__(input_layer,
                                    output_nodes,
                                    session=session,
                                    weight_extender_func=weight_extender_func,
                                    freeze=freeze,
                                    name=name)
        self._non_liniarity = non_liniarity
        self._bactivate = bactivate
        self._unsupervised_cost = unsupervised_cost
        self._supervised_cost = supervised_cost
        self._noise_std = noise_std
        self._bactivation_loss_func = bactivation_loss_func

        self._weights = self._create_variable("weights",
                                              (BaseLayer.INPUT_BOUND_VALUE, BaseLayer.OUTPUT_BOUND_VALUE),
                                              weights if weights is not None else xavier_init(self._input_nodes,
                                                                                              self._output_nodes))

        self._bias = self._create_variable("bias",
                                           (BaseLayer.OUTPUT_BOUND_VALUE,),
                                           bias if bias is not None else tf.zeros(self._output_nodes))

        if self.bactivate:
            self._back_bias = self._create_variable("back_bias",
                                                    (BaseLayer.INPUT_BOUND_VALUE,),
                                                    back_bias if back_bias is not None else tf.zeros(
                                                        self._input_nodes))
        else:
            self._back_bias = None

    @property
    def bactivate(self):
        return self._bactivate

    @property
    def kwargs(self):
        kwargs = super(Layer, self).kwargs

        kwargs['bactivate'] = self.bactivate
        kwargs['bactivation_loss_func'] = self._bactivation_loss_func
        kwargs['non_liniarity'] = self._non_liniarity
        kwargs['unsupervised_cost'] = self._unsupervised_cost
        kwargs['supervised_cost'] = self._supervised_cost
        kwargs['noise_std'] = self._noise_std

        return kwargs

    @property
    def has_bactivation(self):
        return self.bactivate

    @lazyprop
    def activation_corrupted(self):
        if self._noise_std is None:
            raise Exception("No corrupted activation without noise std")
        return self.input_layer.activation_train + tf.random_normal(
            tf.shape(self.input_layer.activation_train),
            stddev=self._noise_std)

    def _layer_activation(self, input_activation, is_train):
        if self._noise_std is not None and is_train:
            input_activation = self.activation_corrupted

        return self._non_liniarity(tf.matmul(input_activation, self._weights) + self._bias)

    def _layer_bactivation(self, activation, is_train):
        if self.bactivate:
            return self._non_liniarity(
                tf.matmul(activation, tf.transpose(self._weights)) + self._back_bias)

    @property
    def non_liniarity(self):
        return self._non_liniarity

    def supervised_cost_train(self, targets):
        if not self.next_layer:
            return tf.reduce_mean(tf.reduce_sum(tf.square(self.activation_train - targets), 1))
        else:
            return None

    @lazyprop
    def bactivation_loss_train(self):
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.bactivation_train - self.input_layer.activation_train), 1))

    @lazyprop
    def bactivation_loss_predict(self):
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(self.bactivation_predict - self.input_layer.activation_predict), 1))

    def unsupervised_cost_train(self):
        if self.bactivate:
            return self.bactivation_loss_train * self._unsupervised_cost
        else:
            return None

    def unsupervised_cost_predict(self):
        if self.bactivate:
            return self.bactivation_loss_predict
        else:
            return None

    def has_resizable_dimension(self):
        return True

    def get_resizable_dimension_size(self):
        return self.output_nodes[0]

    def _choose_nodes_to_split(self, desired_size):
        current_size = self.get_resizable_dimension_size()

        if desired_size >= current_size:
            return None

        importance = self._get_node_importance()

        to_split = {}

        while desired_size < current_size + len(to_split):
            max_node = np.argmax(importance)
            importance[max_node] = -sys.float_info.max
            to_split.add(max_node)

        return to_split

    def _get_node_importance(self):
        importance = self._session.run(self.activation_predict,
                                       feed_dict={self.input_placeholder:
                                                      np.ones(shape=(1,) + self.input_layer.output_nodes,
                                                              dtype=np.float32)})[0]
        return importance

    def _get_node_importance_hessian(self, features, labels):
        # TODO: Lazy prop these?
        weights_hessian_op = create_hessian_variable_op(self.last_layer.target_loss_op_train,
                                                        self._weights)

        bias_hessian_op = create_hessian_variable_op(self.last_layer.target_loss_op_train,
                                                     self._bias)

        weights_hessian, bias_hessian = self.session.run(
            [weights_hessian_op, bias_hessian_op],
            feed_dict={self.input_placeholder: features,
                       self.target_placeholder: labels}
        )

        node_importance = []

        for i in range(len(self.output_nodes[-1])):
            sum = bias_hessian[i]
            # TODO: Optimal brian damage recommends multiplying this by the value squared...
            for j in range(len(self.input_nodes[-1])):
                sum += weights_hessian[j][i]

            node_importance.append(sum)

        return node_importance

    def _choose_nodes_to_prune(self, desired_size):
        current_size = self.get_resizable_dimension_size()

        if desired_size <= current_size:
            return None

        importance = self._get_node_importance()

        to_prune = {}

        while desired_size < current_size - len(to_prune):
            min_node = np.argmin(importance)
            importance[min_node] = sys.float_info.max
            to_prune.add(min_node)

        return to_prune


if __name__ == '__main__':
    with tf.Session() as session:
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, 20, session=session)
        layer.activation.get_shape()
