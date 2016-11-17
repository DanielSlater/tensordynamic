import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import xavier_init
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
                                           bias if bias is not None else tf.zeros((self._output_nodes,)))

        if self.bactivate:
            self._back_bias = self._create_variable("back_bias",
                                                    (BaseLayer.INPUT_BOUND_VALUE,),
                                                    back_bias if back_bias is not None else tf.zeros(
                                                        (self._input_nodes,)))
        else:
            self._back_bias = None

        if self._noise_std is not None:
            self._activation_corrupted = self.input_layer.activation_train + tf.random_normal(
                tf.shape(self.input_layer.activation_train),
                stddev=self._noise_std)
        else:
            self._activation_corrupted = self.input_layer.activation_train

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
    def activation_predict(self):
        return self._non_liniarity(tf.matmul(self.input_layer.activation_predict, self._weights) + self._bias)

    @lazyprop
    def activation_train(self):
        return self._non_liniarity(tf.matmul(self._activation_corrupted, self._weights) + self._bias)

    @lazyprop
    def bactivation_train(self):
        if self.bactivate:
            return self._non_liniarity(
                tf.matmul(self.activation_train, tf.transpose(self._weights)) + self._back_bias)
        else:
            return None

    @lazyprop
    def bactivation_predict(self):
        if self.bactivate:
            return self._non_liniarity(
                tf.matmul(self.activation_predict, tf.transpose(self._weights)) + self._back_bias)
        else:
            return None

    @property
    def non_liniarity(self):
        return self._non_liniarity

    def get_layers_list(self):
        result = [self]
        if self.next_layer is not None:
            result.extend(self.next_layer.get_layers_list())
        return result

    def supervised_cost_train(self, targets):
        if not self.next_layer:
            #return tf.reduce_mean(tf.square(self.activation_train - targets)) * self._supervised_cost
            return tf.reduce_mean(tf.reduce_sum(tf.square(self.activation_train - targets), 1))
        else:
            return None

    @lazyprop
    def bactivation_loss_train(self):
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.bactivation_train - self.input_layer.activation_train), 1))#self._bactivation_loss_func(self.bactivation_train,
               #                            self.input_layer.activation_train)

    @lazyprop
    def bactivation_loss_predict(self):
        return tf.reduce_mean(tf.reduce_sum(tf.square(self.bactivation_predict - self.input_layer.activation_predict), 1))
        #return self._bactivation_loss_func(self.bactivation_predict,
        #                                   self.input_layer.activation_predict)

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


if __name__ == '__main__':
    with tf.Session() as session:
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, 20, session=session)
        layer.activation.get_shape()
