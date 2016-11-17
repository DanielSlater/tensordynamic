import tensorflow as tf
from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import xavier_init
from tensor_dynamic.weight_functions import noise_weight_extender


class DenoisingSourceLayer(Layer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 freeze=False,
                 a1=None,
                 a2=None,
                 a3=None,
                 a4=None,
                 a5=None,
                 a6=None,
                 a7=None,
                 a8=None,
                 a9=None,
                 a10=None,
                 non_liniarity=tf.nn.relu,
                 bactivation_loss_func=squared_loss,
                 weight_extender_func=noise_weight_extender,
                 unsupervised_cost=1.,
                 supervised_cost=1.,
                 noise_std=None,
                 name='BackWeightLayer'):
        super(DenoisingSourceLayer, self).__init__(input_layer, output_nodes,
                                                   session=session,
                                                   bias=bias,
                                                   weights=weights,
                                                   bactivate=True,
                                                   freeze=freeze,
                                                   non_liniarity=non_liniarity,
                                                   weight_extender_func=weight_extender_func,
                                                   bactivation_loss_func=bactivation_loss_func,
                                                   unsupervised_cost=unsupervised_cost,
                                                   supervised_cost=supervised_cost,
                                                   noise_std=noise_std,
                                                   name=name)

        self._a1 = self._create_variable('a1', (BaseLayer.INPUT_BOUND_VALUE,), a1 if a1 is not None else tf.zeros([self.input_nodes]))
        self._a2 = self._create_variable('a2', (BaseLayer.INPUT_BOUND_VALUE,), a2 if a2 is not None else tf.ones([self.input_nodes]))
        self._a3 = self._create_variable('a3', (BaseLayer.INPUT_BOUND_VALUE,), a3 if a3 is not None else tf.zeros([self.input_nodes]))
        self._a4 = self._create_variable('a4', (BaseLayer.INPUT_BOUND_VALUE,), a4 if a4 is not None else tf.zeros([self.input_nodes]))
        self._a5 = self._create_variable('a5', (BaseLayer.INPUT_BOUND_VALUE,), a5 if a5 is not None else tf.zeros([self.input_nodes]))
        self._a6 = self._create_variable('a6', (BaseLayer.INPUT_BOUND_VALUE,), a6 if a6 is not None else tf.zeros([self.input_nodes]))
        self._a7 = self._create_variable('a7', (BaseLayer.INPUT_BOUND_VALUE,), a7 if a7 is not None else tf.ones([self.input_nodes]))
        self._a8 = self._create_variable('a8', (BaseLayer.INPUT_BOUND_VALUE,), a8 if a8 is not None else tf.zeros([self.input_nodes]))
        self._a9 = self._create_variable('a9', (BaseLayer.INPUT_BOUND_VALUE,), a9 if a9 is not None else tf.zeros([self.input_nodes]))
        self._a10 = self._create_variable('a10', (BaseLayer.INPUT_BOUND_VALUE,), a10 if a10 is not None else tf.zeros([self.input_nodes]))

    def _gaussian_denoise(self, input_corrupted, activation):
        mu = self._a1 * tf.sigmoid(self._a2 * activation + self._a3) + self._a4 * activation + self._a5
        v = self._a6 * tf.sigmoid(self._a7 * activation + self._a8) + self._a9 * activation + self._a10

        z_est = (input_corrupted - mu) * v + mu

        return z_est

    @lazyprop
    def bactivation_train(self):
        return self._gaussian_denoise(self._activation_corrupted, self.activation_train)

    @lazyprop
    def bactivation_predict(self):
        return self._gaussian_denoise(self.activation_predict, self.activation_predict)

    @property
    def kwargs(self):
        kwargs = super(DenoisingSourceLayer, self).kwargs

        # bactivate is not optional for these layers
        del kwargs['bactivate']

        return kwargs
