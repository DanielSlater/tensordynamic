import math
import tensorflow as tf
import numpy as np

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.utils import xavier_init
from tensor_dynamic.weight_functions import noise_weight_extender


class LadderLayer(BaseLayer):
    NOISE_STD = 0.3

    def __init__(self, input_layer, output_nodes,
                 denoising_cost=1.,
                 session=None,
                 beta=None,
                 weights=None,
                 back_weights=None,
                 non_liniarity=tf.nn.relu,
                 weight_extender_func=noise_weight_extender,
                 freeze=False,
                 name="ladder"):
        super(LadderLayer, self).__init__(input_layer,
                                          weight_extender_func=weight_extender_func,
                                          freeze=freeze,
                                          name=name)
        self._denoising_cost = denoising_cost
        self.output_nodes = output_nodes
        self.input_nodes = self.input_shape[-1]
        self._non_liniarity = non_liniarity

        self._weights = self._create_variable((self.input_nodes, self.output_nodes),
                                              weights or xavier_init(self.input_nodes, self.output_nodes),
                                              "weights")

        self._back_weights = self._create_variable((self.output_nodes, self.input_nodes),
                                                   back_weights or xavier_init(self.output_nodes, self.input_nodes),
                                                   "back_weights")

        self._beta = self._create_variable((self.output_nodes,),
                                           beta or tf.zeros([self.output_nodes]),
                                           "beta")
        """values generating mean of output"""

        self._running_mean = tf.Variable(tf.constant(0.0, shape=[self.output_nodes]), trainable=False,
                                         name="running_mean")
        self._running_var = tf.Variable(tf.constant(1.0, shape=[self.output_nodes]), trainable=False,
                                        name="running_var")

        self._ewma = tf.train.ExponentialMovingAverage(decay=0.99)

        self._session = session

        self.z_pre = tf.matmul(self.input_layer.activation, self._weights)

        self.mean, self.variance = tf.nn.moments(self.z_pre, axes=[0])

        self.z = self.batch_normalization(self.z_pre, self.mean, self.variance) + tf.random_normal(tf.shape(self.z_pre),
                                                                                                   stddev=self.NOISE_STD)

        if session:
            session.run(tf.initialize_variables([self._weights, self._back_weights, self._beta, self._running_mean,
                                                 self._running_var]))

    @lazyprop
    def activation_noisy(self):
        print "Corrupt Act ", self.layer_number, ": ", self.input_nodes, " -> ", self.output_nodes
        return self._activation_method(self.z)

    @lazyprop
    def activation(self):
        print "Clean Act ", self.layer_number, ": ", self.input_nodes, " -> ", self.output_nodes
        z = self.update_batch_normalization(self.z_pre)
        return self._activation_method(z)

    @lazyprop
    def z_est(self):
        print "Layer ", self.layer_number, ": ", self.output_nodes, " -> ", self.input_nodes, ", denoising cost: ", self._denoising_cost
        u = self._bactivation_target()
        u = self.batch_normalization(u)
        return self.g_gauss(self.input_layer.z, u, self.input_nodes)

    @lazyprop
    def z_est_bn(self):
        if isinstance(self.input_layer, LadderLayer):
            return (self.z_est - self.input_layer.mean) / self.input_layer.variance
        else:
            # no norm that layer
            return self.z_est / 1-1e-10

    @property
    def bactivation(self):
        return self.z_est

    @staticmethod
    def batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    def update_batch_normalization(self, batch):
        "batch normalize + update average mean and variance of layer"
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = self._running_mean.assign(mean)
        assign_var = self._running_var.assign(var)
        # TODO: add this back in?
        # bn_assigns.append(self._ewma.apply([self._running_mean, self._running_var]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    def unsupervised_cost(self):
        mean = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - self.input_layer.z), 1))
        # TODO: input_nodes may change...
        return (mean / self.input_nodes) * self._denoising_cost

    def g_gauss(self, z_c, u, size):
        """gaussian denoising function proposed in the original paper"""

        def wi(inits, name):
            return tf.Variable(inits * tf.ones([size]), name=name)

        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu

        if self._session:
            self._session.run(tf.initialize_variables([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]))

        return z_est

    def _activation_method(self, z):
        return self._non_liniarity(z + self._beta)

    def _bactivation_target(self):
        if self.next_layer:
            return tf.matmul(self.next_layer.bactivation, self._back_weights)
        else:
            return self.input_layer.activation_noisy

class LadderGammaLayer(LadderLayer):
    def __init__(self, input_layer, output_nodes,
                 denoising_cost,
                 session=None,
                 beta=None,
                 gamma=None,
                 weights=None,
                 back_weights=None,
                 freeze=False,
                 non_liniarity=tf.nn.softmax,
                 weight_extender_func=noise_weight_extender,
                 name="ladderOutput"):
        super(LadderGammaLayer, self).__init__(input_layer, output_nodes,
                                                denoising_cost,
                                                session=session,
                                                beta=beta,
                                                weights=weights,
                                                back_weights=back_weights,
                                                freeze=freeze,
                                                non_liniarity=non_liniarity,
                                                weight_extender_func=weight_extender_func,
                                                name=name)
        if gamma is None:
            gamma = tf.ones([self.output_nodes])
        else:
            gamma = self._weight_extender_func(gamma, (self.output_nodes,))
        self._gamma = tf.Variable(gamma, name="gamma")
        """values for generating std dev of output"""

        if session:
            session.run(tf.initialize_variables([self._gamma]))

    def _activation_method(self, z):
        return self._non_liniarity(self._gamma * (z + self._beta))


class LadderOutputLayer(BaseLayer):
    def __init__(self, input_layer, output_nodes,
                 denoising_cost,
                 session=None,
                 beta=None,
                 gamma=None,
                 weights=None,
                 back_weights=None,
                 freeze=False,
                 non_liniarity=tf.nn.softmax,
                 weight_extender_func=noise_weight_extender,
                 name="ladderOutput"):
        super(LadderOutputLayer, self).__init__(input_layer,
                                                freeze=freeze,
                                                weight_extender_func=weight_extender_func,
                                                name=name)
        self._session = session
        self.output_nodes = output_nodes
        self._denoising_cost = denoising_cost
        self.input_nodes = int(self.input_layer.activation.get_shape()[-1])
        #self.mean, self.variance = tf.nn.moments(self.activation, axes=[0])

    @property
    def activation(self):
        return self.input_layer.activation

    @lazyprop
    def z_est_bn(self):
        return (self.z_est - self.input_layer.mean) / self.input_layer.variance

    @lazyprop
    def z_est(self):
        u = self.input_layer.activation_noisy
        u = LadderLayer.batch_normalization(u)
        return self.g_gauss(self.input_layer.z, u, self.input_nodes)

    @property
    def bactivation(self):
        print "Layer ", self.layer_number, ": ", self.output_nodes, " -> ", self.input_nodes, ", denoising cost: ", self._denoising_cost
        return self.z_est_bn

    def unsupervised_cost(self):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - self.activation), 1))
        # TODO: input_nodes may change...
        return (cost / self.input_nodes) * self._denoising_cost

    def supervised_cost(self, targets):
        # todo may have to do something around the labelled vs unlabelled data
        #num_unlabeled_items = tf.shape(self.activation)[0] - tf.shape(targets)[0]
        #D.S todo remove hard coded 100...
        labeled_activations = tf.slice(self.activation, [0, 0], [100, -1])
        return -tf.reduce_mean(tf.reduce_sum(targets * tf.log(labeled_activations), 1))

    def g_gauss(self, z_c, u, size):
        """gaussian denoising function proposed in the original paper"""

        def wi(inits, name):
            return tf.Variable(inits * tf.ones([size]), name=name)

        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu

        if self._session:
            self._session.run(tf.initialize_variables([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]))

        return z_est