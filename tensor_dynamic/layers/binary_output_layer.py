import tensorflow as tf

from tensor_dynamic.layers.output_layer import OutputLayer
from tensor_dynamic.lazyprop import lazyprop


class BinaryOutputLayer(OutputLayer):
    def __init__(self, input_layer,
                 session=None,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 freeze=False,
                 weight_extender_func=None,
                 layer_noise_std=None,
                 regularizer_weighting=0.01,
                 name='BinaryOutputLayer'):
        super(BinaryOutputLayer, self).__init__(input_layer, (1,),
                                                session=session,
                                                bias=bias,
                                                weights=weights,
                                                back_bias=back_bias,
                                                freeze=freeze,
                                                weight_extender_func=weight_extender_func,
                                                layer_noise_std=layer_noise_std,
                                                regularizer_weighting=regularizer_weighting,
                                                name=name)

    @lazyprop
    def accuracy_op(self):
        correct_prediction = tf.equal(
            tf.round(tf.abs(self.activation_predict - self.target_placeholder)), 0)

        return tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))