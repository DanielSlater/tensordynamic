import tensorflow as tf
from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import xavier_init
from tensor_dynamic.weight_functions import noise_weight_extender


class BackWeightLayer(Layer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 back_weights=None,
                 back_bias=None,
                 freeze=False,
                 non_liniarity=tf.nn.relu,
                 bactivation_loss_func=squared_loss,
                 weight_extender_func=noise_weight_extender,
                 unsupervised_cost=1.,
                 supervised_cost=1.,
                 noise_std=None,
                 name='BackWeightLayer'):
        super(BackWeightLayer, self).__init__(input_layer, output_nodes,
                                              session=session,
                                              bias=bias,
                                              weights=weights,
                                              back_bias=back_bias,
                                              bactivate=True,
                                              freeze=freeze,
                                              non_liniarity=non_liniarity,
                                              weight_extender_func=weight_extender_func,
                                              bactivation_loss_func=bactivation_loss_func,
                                              unsupervised_cost=unsupervised_cost,
                                              supervised_cost=supervised_cost,
                                              noise_std=noise_std,
                                              name=name)
        self._back_weights = self._create_variable("back_weights",
                                                   (BaseLayer.OUTPUT_BOUND_VALUE, BaseLayer.INPUT_BOUND_VALUE),
                                                   back_weights if back_weights is not None else xavier_init(
                                                       self._output_nodes,
                                                       self._input_nodes))

    def _layer_bactivation(self, activation):
        return self._non_liniarity(
            tf.matmul(activation, self._back_weights) + self._back_bias)

    @property
    def kwargs(self):
        kwargs = super(BackWeightLayer, self).kwargs

        # bactivate is not optional for these layers
        del kwargs['bactivate']

        return kwargs
