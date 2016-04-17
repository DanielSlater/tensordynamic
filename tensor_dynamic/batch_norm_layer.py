import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop


class BatchNormLayer(BaseLayer):
    def __init__(self, input_layer,
                 session,
                 name='BatchNormLayer'):
        super(BatchNormLayer, self).__init__(input_layer,
                                             input_layer.output_nodes,
                                             session,
                                             name=name)
        self._running_mean = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros((self.input_nodes,)),
                                                   "running_mean")
        self._running_var = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.ones((self.input_nodes,)),
                                                  "running_var")

        self._ewma = tf.train.ExponentialMovingAverage(decay=.99)

        with self.name_scope():
            self._training = tf.Variable(True, name="training")
            self._session.run(tf.initialize_variables([self._training]))

    @property
    def predict(self):
        return not self._session.run(self._training)

    @predict.setter
    def predict(self, value):
        self._session.run([tf.assign(self._training, not value)])

    @lazyprop
    def activation(self):
        return control_flow_ops.cond(self._training, self.update_batch_normalization, self.eval)

    def update_batch_normalization(self):
        "batch normalize + update average mean and variance of layer"
        mean, var = tf.nn.moments(self._input_layer.activation, axes=[0])
        assign_mean = self._running_mean.assign(mean)
        assign_var = self._running_var.assign(var)
        # TODO: add this back in?
        assign_op = self._ewma.apply([self._running_mean, self._running_var])

        # need to init the variables the ewma creates
        self._session.run(tf.initialize_variables(self._ewma._averages.values()))

        with tf.control_dependencies([assign_mean, assign_var, assign_op]):
            return self.batch_normalize(self._input_layer.activation, mean, var)

    def eval(self):
        mean = self._ewma.average(self._running_mean)
        var = self._ewma.average(self._running_var)
        return self.batch_normalize(self._input_layer.activation, mean, var)

    @staticmethod
    def batch_normalize(batch, mean, var):
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
