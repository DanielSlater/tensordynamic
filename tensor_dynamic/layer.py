import tensorflow as tf
import numpy as np
import math

from tensor_dynamic.utils import xavier_init


class Layer(object):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None, weights=None, back_bias=None,
                 bactivate=False,
                 freeze=False):

        if isinstance(input_layer, Layer):
            input_layer.output_layer = self
            self.input_layer_activation = input_layer.activation
        else:
            self.input_layer_activation = input_layer
        self.input_nodes = int(self.input_layer_activation.get_shape()[-1])

        self.output_layer = None
        self.output_nodes = output_nodes

        self.freeze = freeze
        trainable = not freeze
        if weights is None:
            self._weights = tf.Variable(xavier_init(self.input_nodes, self.output_nodes),
                                        trainable=trainable)
        else:
            if weights.shape[1] != self.output_nodes:
                raise Exception("Weight shape must equal output nodes")

            if self.input_nodes > weights.shape[0]:
                weights = np.append(weights,
                                    np.random.normal(scale=1.0 / math.sqrt(float(weights.shape[1])),
                                                     size=(self.input_nodes - weights.shape[0], weights.shape[1]))
                                    .astype(weights.dtype),
                                    axis=0)
            if self.output_nodes > weights.shape[1]:
                weights = np.append(weights,
                                    np.random.normal(scale=1.0 / math.sqrt(float(weights.shape[0])),
                                                     size=(weights.shape[0], self.output_nodes - weights.shape[1]))
                                    .astype(weights.dtype),
                                    axis=1)

            self._weights = tf.Variable(weights, trainable=trainable)

        if bias is None:
            self._bias = tf.Variable(tf.zeros([self.output_nodes]), trainable=trainable)
        else:
            self._bias = tf.Variable(bias, trainable=trainable)

        self.activation = tf.nn.sigmoid(tf.matmul(self.input_layer_activation, self._weights) + self._bias)
        self.bactivate = bactivate
        if bactivate:
            if back_bias is None:
                self._back_bias = tf.Variable(tf.zeros([self.input_nodes]), trainable=trainable)
            else:
                if self.input_nodes > back_bias.shape[-1]:
                    back_bias = np.append(back_bias, tf.zeros([self.input_nodes - back_bias.shape[-1]])) \
                        .astype(back_bias.dtype)
                self._back_bias = tf.Variable(back_bias, trainable=trainable)

            self.bactivation = tf.nn.sigmoid(tf.matmul(self.activation, tf.transpose(self._weights)) + self._back_bias)
        else:
            self._back_bias = None

        if session:
            if self._back_bias:
                session.run(tf.initialize_variables([self._bias, self._weights, self._back_bias]))
            else:
                session.run(tf.initialize_variables([self._bias, self._weights]))

    def get_output_layer_activation(self):
        if self.output_layer is not None:
            return self.output_layer.get_output_layer_activation()
        return self.activation

    def get_layers_list(self):
        result = [self]
        if self.output_layer is not None:
            result.extend(self.output_layer.get_layers_list())
        return result

    def add_output(self, session):
        weights = session.run(self._weights)

        new_weights = np.append(weights,
                                np.random.normal(scale=1.0 / math.sqrt(float(weights.shape[0])),
                                                 size=(weights.shape[0], 1)).astype(weights.dtype),
                                axis=1)
        bias = session.run(self._bias)
        bias = np.append(bias, 0.0).astype(bias.dtype)

        if self._back_bias is not None:
            back_bias = session.run(self._back_bias)
        else:
            back_bias = None

        new_self = Layer(self.input_layer_activation,
                         self.output_nodes+1,
                         weights=new_weights,
                         bias=bias,
                         back_bias=back_bias,
                         bactivate=self.bactivate,
                         freeze=self.freeze,
                         session=session)

        return new_self

    def clone(self, session, input_layer, bias=None, weight=None, back_bias=None):
        if bias is None:
            bias = session.run(self._bias)
        if weight is None:
            weight = session.run(self._weights)
        if back_bias is None and self._back_bias is not None:
            back_bias = session.run(self._back_bias)

        new_self = Layer(input_layer,
                         self.output_nodes,
                         weights=weight,
                         bias=bias,
                         back_bias=back_bias,
                         bactivate=self.bactivate,
                         freeze=self.freeze,
                         session=session)

        return new_self


if __name__ == '__main__':
    with tf.Session() as session:
        input_p = tf.placeholder("float", (None, 10))
        layer = Layer(input_p, 20, session=session)
        layer.activation.get_shape()
