import tensorflow as tf

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop, clear_all_lazyprops
from tensor_dynamic.utils import xavier_init, tf_resize
from tensor_dynamic.weight_functions import noise_weight_extender


class Layer(BaseLayer):
    def __init__(self, input_layer, output_nodes,
                 session,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 bactivate=False,
                 freeze=False,
                 non_liniarity=tf.nn.sigmoid,
                 weight_extender_func=noise_weight_extender,
                 name='Layer'):
        super(Layer, self).__init__(input_layer, weight_extender_func=weight_extender_func, freeze=freeze, name=name)

        self.output_nodes = output_nodes
        self._non_liniarity = non_liniarity
        self.bactivate = bactivate

        self.input_nodes = int(self.input_shape[-1])

        self._weights = self._create_variable((self.input_nodes, self.output_nodes),
                                              weights if weights is not None else xavier_init(self.input_nodes,
                                                                                              self.output_nodes),
                                              "weights")

        self._bias = self._create_variable((self.output_nodes,),
                                           bias if bias is not None else tf.zeros((self.output_nodes,)),
                                           "bias")

        if self.bactivate:
            self._back_bias = self._create_variable((self.input_nodes,),
                                                    back_bias if back_bias is not None else tf.zeros((self.input_nodes,)),
                                                    "back_bias")
        else:
            self._back_bias = None

        if session:
            session.run(tf.initialize_variables([self._bias, self._weights]))
            if self._back_bias:
                session.run(tf.initialize_variables([self._back_bias]))

    @property
    def kwargs(self):
        kwargs = super(Layer, self).kargs()

        kwargs['bactivate'] = self.bactivate
        kwargs['non_liniarity'] = self._non_liniarity

        return kwargs

    @property
    def has_bactivation(self):
        return self.bactivate

    @lazyprop
    def activation(self):
        return self._non_liniarity(tf.matmul(self.input_layer.activation, self._weights) + self._bias)

    @lazyprop
    def bactivation(self):
        if self.bactivate:
            return self._non_liniarity(
                tf.matmul(self.activation, tf.transpose(self._weights)) + self._back_bias)
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

    def clone(self, input_layer, old_session, new_session=None, bias=None, weight=None, back_bias=None):
        if new_session is None:
            new_session = old_session
        if bias is None:
            bias = old_session.run(self._bias)
        if weight is None:
            weight = old_session.run(self._weights)
        if back_bias is None and self._back_bias is not None:
            back_bias = old_session.run(self._back_bias)

        new_self = self.__class__(input_layer,
                                  self.output_nodes,
                                  weights=weight,
                                  bias=bias,
                                  back_bias=back_bias,
                                  bactivate=self.bactivate,
                                  freeze=self.freeze,
                                  session=new_session,
                                  non_liniarity=self._non_liniarity,
                                  weight_extender_func=self._weight_extender_func)

        return new_self

    def supervised_cost(self, targets):
        if not self.next_layer:
            return tf.reduce_sum(tf.square(self.activation - targets))
        else:
            return None

    def unsupervised_cost(self):
        if self.bactivate:
            return tf.reduce_mean(tf.square(self.bactivation - self.input_layer.activation))
        else:
            return None

    def resize_needed(self):
        if self._input_layer.output_shape[-1] != self.input_nodes:
            return True
        return False

    def resize(self, new_output_nodes, session):
        new_input_nodes = int(self.input_shape[-1])

        tf_resize(session, self._weights, (new_input_nodes, new_output_nodes),
                  self._weight_extender_func(session.run(self._weights), (new_input_nodes, new_output_nodes)))

        tf_resize(session, self._bias, (new_output_nodes,),
                  self._weight_extender_func(session.run(self._bias), (new_output_nodes,)))

        tf_resize(session, self.activation, (None, new_output_nodes))

        if self.bactivate and new_input_nodes != self.input_nodes:
            tf_resize(session, self._back_bias, (new_input_nodes,),
                      self._weight_extender_func(session.run(self._back_bias),
                                                 (self.input_nodes,)))

            tf_resize(session, self.bactivation, (None, new_input_nodes))

        self.output_nodes = new_output_nodes
        self.input_nodes = new_input_nodes

        if self._next_layer and self._next_layer.resize_needed():
            self._next_layer.resize(None, session)


if __name__ == '__main__':
    with tf.Session() as session:
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, 20, session=session)
        layer.activation.get_shape()
