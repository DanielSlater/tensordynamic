import math
import tensorflow as tf

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender

join = lambda l, u: tf.concat(0, [l, u], name="join")
labeled = lambda x: tf.slice(x, [0, 0], [100, -1], name="slice_unlabeled") if x is not None else x
unlabeled = lambda x: tf.slice(x, [100, 0], [-1, -1], name="slice_labeled") if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))


class LadderLayer(BaseLayer):
    NOISE_STD = 0.3

    def __init__(self, input_layer,
                 output_nodes,
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
                                          output_nodes,
                                          session=session,
                                          weight_extender_func=weight_extender_func,
                                          freeze=freeze,
                                          name=name)
        self._denoising_cost = denoising_cost
        self._non_liniarity = non_liniarity

        self._weights = self._create_variable(
            (BaseLayer.INPUT_BOUND_VALUE, BaseLayer.OUTPUT_BOUND_VALUE),
            weights if weights is not None else tf.random_normal((self.input_nodes, self.output_nodes),
                                                                 stddev=1. / math.sqrt(self.input_nodes))
            , "weights")

        self._back_weights = self._create_variable(
            (BaseLayer.OUTPUT_BOUND_VALUE, BaseLayer.INPUT_BOUND_VALUE),
            back_weights if back_weights is not None else tf.random_normal((self.output_nodes, self.input_nodes),
                                                                           stddev=1. / math.sqrt(self.output_nodes)),
            "back_weights")

        self._beta = self._create_variable((BaseLayer.OUTPUT_BOUND_VALUE,),
                                           beta if beta is not None else tf.zeros([self.output_nodes]),
                                           "beta")
        """values generating mean of output"""

        self.bn_assigns = []

        with tf.name_scope(str(self.layer_number) + "_" + self._name):
            self._running_mean = tf.Variable(tf.constant(0.0, shape=[self.output_nodes]), trainable=False,
                                             name="running_mean")
            self._running_var = tf.Variable(tf.constant(1.0, shape=[self.output_nodes]), trainable=False,
                                            name="running_var")

            self._ewma = tf.train.ExponentialMovingAverage(decay=0.99)

            self.z_pre_corrupted = tf.matmul(self._input_corrupted, self._weights, name="z_pre_corrupted")

            z_pre_corrupted_labeled, z_pre_corrupted_unlabeled = split_lu(self.z_pre_corrupted)

            self.z_pre_clean = tf.matmul(self.input_layer.activation, self._weights, name="z_pre_clean")

            z_pre_clean_labeled, z_pre_clean_unlabeled = split_lu(self.z_pre_clean)

            self.mean_corrupted_unlabeled, self.variance_corrupted_unlabeled = tf.nn.moments(z_pre_corrupted_unlabeled,
                                                                                             axes=[0])

            self.mean_clean_unlabeled, self.variance_clean_unlabeled = tf.nn.moments(z_pre_clean_unlabeled, axes=[0])

            self.z_corrupted = join(self.batch_normalization(z_pre_corrupted_labeled),
                                    self.batch_normalization(z_pre_corrupted_unlabeled, self.mean_corrupted_unlabeled,
                                                             self.variance_corrupted_unlabeled)) + \
                               tf.random_normal(tf.shape(self.z_pre_corrupted),
                                                stddev=self.NOISE_STD)

            self.z_clean = join(self._update_batch_normalization(z_pre_clean_labeled),
                                self.batch_normalization(z_pre_clean_unlabeled, self.mean_clean_unlabeled,
                                                         self.variance_clean_unlabeled))

        if session:
            session.run(tf.initialize_variables([self._running_mean,
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

        u = tf.matmul(self.next_layer.z_est, self._back_weights, name="u")
        u = self.batch_normalization(u)
        # self._input_corrupted ?? this is changed?
        return self._g_gauss(unlabeled(self.input_z_corrupted), u)

    @lazyprop
    def z_est_bn(self):
        if isinstance(self.input_layer, LadderLayer):
            return (self.z_est - self.input_layer.mean_clean_unlabeled) / self.input_layer.variance_clean_unlabeled
        else:
            # no norm that layer
            return self.z_est / 1 - 1e-10

    @property
    def bactivation(self):
        return self.z_est

    @staticmethod
    def batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0], name="batch_normalization")
        return (batch - mean) / tf.sqrt(var + 1e-10)

    def _update_batch_normalization(self, batch):
        "batch normalize + update average mean and variance of layer"
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = self._running_mean.assign(mean)
        assign_var = self._running_var.assign(var)
        # TODO: add this back in?
        self.bn_assigns.append(self._ewma.apply([self._running_mean, self._running_var]))
        with tf.control_dependencies([assign_mean, assign_var]):
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
        mean = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - unlabeled(self.input_z_clean)), 1))
        # TODO: input_nodes may change...
        return (mean / self.input_nodes) * self._denoising_cost

    def _g_gauss(self, z_c, u):
        """gaussian denoising function proposed in the original paper"""
        a1 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a1')
        a2 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.ones([self.input_nodes]), name='a2')
        a3 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a3')
        a4 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a4')
        a5 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a5')
        a6 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a6')
        a7 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.ones([self.input_nodes]), name='a7')
        a8 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a8')
        a9 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a9')
        a10 = self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]), name='a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu

        return z_est

    def _activation_method(self, z):
        with tf.name_scope(str(self.layer_number) + "_" + self._name):
            return self._non_liniarity(z + self._beta, name="activation")


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
        self._gamma = self._create_variable((BaseLayer.OUTPUT_BOUND_VALUE,),
                                            gamma if gamma is not None else tf.ones([self.output_nodes]), name="gamma")
        """values for generating std dev of output"""

    def _activation_method(self, z):
        with tf.name_scope(str(self.layer_number) + "_" + self._name):
            return self._non_liniarity(self._gamma * (z + self._beta), name="activation_gamma")
