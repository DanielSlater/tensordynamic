from layer import Layer
from tensor_dynamic.weight_functions import noise_weight_extender
import tensorflow as tf
import numpy as np


class ResiduleLayer(Layer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None, weights=None, back_bias=None,
                 bactivate=False,
                 freeze=False,
                 non_liniarity=tf.nn.sigmoid,
                 weight_extender_func=noise_weight_extender):
        super(ResiduleLayer).__init__(input_layer, output_nodes,
                                      session=session,
                                      bias=bias, weights=weights, back_bias=back_bias,
                                      bactivate=bactivate,
                                      freeze=freeze,
                                      non_liniarity=non_liniarity,
                                      weight_extender_func=weight_extender_func)
