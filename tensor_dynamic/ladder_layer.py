import math
import tensorflow as tf

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

        self._weights = self._create_variable(
            (self.input_nodes, self.output_nodes),
            weights if weights is not None else tf.random_normal((self.input_nodes, self.output_nodes),
                                                                 stddev=1. / math.sqrt(self.input_nodes))
            , "weights")

        self._back_weights = self._create_variable(
            (self.output_nodes, self.input_nodes),
            back_weights if back_weights is not None else tf.random_normal((self.output_nodes, self.input_nodes),
                                                                           stddev=1. / math.sqrt(self.output_nodes)),
            "back_weights")

        self._beta = self._create_variable((self.output_nodes,),
                                           beta if beta is not None else tf.zeros([self.output_nodes]),
                                           "beta")
        """values generating mean of output"""

        self._running_mean = tf.Variable(tf.constant(0.0, shape=[self.output_nodes]), trainable=False,
                                         name="running_mean")
        self._running_var = tf.Variable(tf.constant(1.0, shape=[self.output_nodes]), trainable=False,
                                        name="running_var")

        self._ewma = tf.train.ExponentialMovingAverage(decay=0.99)

        self._session = session

        self.z_pre_corrupted = tf.matmul(self._input_corrupted, self._weights)

        self.z_pre_clean = tf.matmul(self.input_layer.activation, self._weights)

        self.mean_corrupted, self.variance_corrupted = tf.nn.moments(self.z_pre_corrupted, axes=[0])
        self.mean_clean, self.variance_clean = tf.nn.moments(self.z_pre_clean, axes=[0])

        self.z_corrupted = self.batch_normalization(self.z_pre_corrupted, self.mean_corrupted,
                                                    self.variance_corrupted) + \
                           tf.random_normal(tf.shape(self.z_pre_corrupted),
                                            stddev=self.NOISE_STD)

        self.z_clean = self._update_batch_normalization(self.z_pre_clean, self.mean_clean, self.variance_clean)

        if session:
            session.run(tf.initialize_variables([self._weights, self._back_weights, self._beta, self._running_mean,
                                                 self._running_var]))

    @lazyprop
    def _input_corrupted(self):
        if isinstance(self.input_layer, LadderLayer):
            return self.input_layer.activation_corrupted
        else:
            return self.input_layer.activation + tf.random_normal(tf.shape(self.input_layer.activation),
                                                                  stddev=self.NOISE_STD)

    @lazyprop
    def activation_corrupted(self):
        print "Corrupt Act ", self.layer_number, ": ", self.input_nodes, " -> ", self.output_nodes
        return self._activation_method(self.z_corrupted)

    @lazyprop
    def activation(self):
        print "Clean Act ", self.layer_number, ": ", self.input_nodes, " -> ", self.output_nodes
        # z = self.update_batch_normalization(self)
        # TODO: add back in update batch norm
        return self._activation_method(self.z_clean)

    @lazyprop
    def z_est(self):
        print "Layer ", self.layer_number, ": ", self.output_nodes, " -> ", self.input_nodes, ", denoising cost: ", self._denoising_cost

        u = tf.matmul(self.next_layer.z_est, self._back_weights)
        u = self.batch_normalization(u)
        return self._g_gauss(self.input_z_corrupted, u, self.input_nodes)

    @lazyprop
    def z_est_bn(self):
        if isinstance(self.input_layer, LadderLayer):
            return (self.z_est - self.input_layer.mean_clean) / self.input_layer.variance_clean
        else:
            # no norm that layer
            return self.z_est / 1 - 1e-10

    @property
    def bactivation(self):
        return self.z_est

    @staticmethod
    def batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + 1e-10)

    def _update_batch_normalization(self, batch, mean, var):
        "batch normalize + update average mean and variance of layer"
        assign_mean = self._running_mean.assign(mean)
        assign_var = self._running_var.assign(var)
        # TODO: add this back in?
        ewma_apply = self._ewma.apply([self._running_mean, self._running_var])
        with tf.control_dependencies([assign_mean, assign_var, ewma_apply]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    @property
    def input_z_clean(self):
        if isinstance(self.input_layer, LadderLayer):
            return self.input_layer.z_clean
        else:
            return self.input_layer.activation

    @property
    def input_z_corrupted(self):
        if isinstance(self.input_layer, LadderLayer):
            return self.input_layer.z_corrupted
        else:
            return self.input_layer.activation + tf.random_normal(tf.shape(self.input_layer.activation),
                                                                  stddev=self.NOISE_STD)

    def unsupervised_cost(self):
        mean = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - self.input_z_clean), 1))
        # TODO: input_nodes may change...
        return (mean / self.input_nodes) * self._denoising_cost

    def _g_gauss(self, z_c, u, size):
        """gaussian denoising function proposed in the original paper"""

        def wi(inits, name):
            return tf.Variable(inits * tf.ones([size]), name=name)

        a1 = self._create_variable((size,), tf.zeros([size]), name='a1')
        a2 = self._create_variable((size,), tf.ones([size]), name='a2')
        a3 = self._create_variable((size,), tf.zeros([size]), name='a3')
        a4 = self._create_variable((size,), tf.zeros([size]), name='a4')
        a5 = self._create_variable((size,), tf.zeros([size]), name='a5')
        a6 = self._create_variable((size,), tf.zeros([size]), name='a6')
        a7 = self._create_variable((size,), tf.ones([size]), name='a7')
        a8 = self._create_variable((size,), tf.zeros([size]), name='a8')
        a9 = self._create_variable((size,), tf.zeros([size]), name='a9')
        a10 = self._create_variable((size,), tf.zeros([size]), name='a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu

        if self._session:
            self.initialize_variables(self._session)

        return z_est

    def _activation_method(self, z):
        return self._non_liniarity(z + self._beta)


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
        self._gamma = self._create_variable((self.output_nodes,),
                                            gamma if gamma is not None else tf.ones([self.output_nodes]), name="gamma")
        """values for generating std dev of output"""

        if session:
            session.run(tf.initialize_variables([self._gamma]))

    def _activation_method(self, z):
        return self._non_liniarity(self._gamma * (z + self._beta))
