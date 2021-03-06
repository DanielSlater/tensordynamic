import math

import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.layers.input_layer import SemiSupervisedInputLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender

join = lambda l, u: tf.concat(0, [l, u], name="join")
labeled = lambda x, labeled_size: tf.slice(x, [0, 0], [labeled_size, -1], name="slice_unlabeled") if x is not None else x
unlabeled = lambda x, labeled_size: tf.slice(x, [labeled_size, 0], [-1, -1], name="slice_labeled") if x is not None else x
split_lu = lambda x, labeled_size: (labeled(x, labeled_size), unlabeled(x, labeled_size))


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
        if not isinstance(self.first_layer, SemiSupervisedInputLayer):
            raise Exception("To use a ladder network you must have a tensor_dynamic.layers.input_layer.SemiSupervisedInputLayer as the input layer to the network")

        self._denoising_cost = denoising_cost
        self._non_liniarity = non_liniarity

        self._weights = self._create_variable("weights",
                                              (BaseLayer.INPUT_BOUND_VALUE, BaseLayer.OUTPUT_BOUND_VALUE),
                                              weights if weights is not None else tf.random_normal(
                                                  (self.input_nodes, self.output_nodes),
                                                  stddev=1. / math.sqrt(self.input_nodes)))

        self._back_weights = self._create_variable("back_weights",
                                                   (BaseLayer.OUTPUT_BOUND_VALUE, BaseLayer.INPUT_BOUND_VALUE),
                                                   back_weights if back_weights is not None else tf.random_normal(
                                                       (self.output_nodes, self.input_nodes),
                                                       stddev=1. / math.sqrt(self.output_nodes)))

        self._beta = self._create_variable("beta",
                                           (BaseLayer.OUTPUT_BOUND_VALUE,),
                                           beta if beta is not None else tf.zeros([self.output_nodes]))
        """values generating mean of output"""

        self.bn_assigns = []

        with self.name_scope():
            # self._running_mean = tf.Variable(tf.constant(0.0, shape=[self.output_nodes]), trainable=False,
            #                                  name="running_mean")
            # self._running_var = tf.Variable(tf.constant(1.0, shape=[self.output_nodes]), trainable=False,
            #                                 name="running_var")
            # self._ewma = tf.train.ExponentialMovingAverage(decay=0.99)

            self.z_pre_corrupted = tf.matmul(self._input_corrupted, self._weights, name="z_pre_corrupted")

            z_pre_corrupted_labeled, z_pre_corrupted_unlabeled = split_lu(self.z_pre_corrupted, self.first_layer.labeled_input_size)

            self.z_pre_clean = tf.matmul(self.input_layer.activation_predict, self._weights, name="z_pre_clean")

            z_pre_clean_labeled, z_pre_clean_unlabeled = split_lu(self.z_pre_clean, self.first_layer.labeled_input_size)

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

        # if session:
        #     session.run(tf.initialize_variables([self._running_mean,
        #                                          self._running_var]))

    @lazyprop
    def _input_corrupted(self):
        if isinstance(self.input_layer, LadderLayer):
            return self.input_layer.activation_train
        else:
            return self.input_layer.activation_predict + tf.random_normal(tf.shape(self.input_layer.activation_predict),
                                                                          stddev=self.NOISE_STD)

    @lazyprop
    def activation_train(self):
        print "Corrupt Act ", self.layer_number, ": ", self.input_nodes, " -> ", self.output_nodes
        return self._activation_method(self.z_corrupted)

    @lazyprop
    def activation_predict(self):
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
        return self._g_gauss(unlabeled(self.input_z_corrupted, self.first_layer.labeled_input_size), u)

    @lazyprop
    def z_est_bn(self):
        if isinstance(self.input_layer, LadderLayer):
            return (self.z_est - self.input_layer.mean_clean_unlabeled) / self.input_layer.variance_clean_unlabeled
        else:
            # no norm that layer
            return self.z_est / 1 - 1e-10

    @property
    def bactivation_train(self):
        return self.z_est

    @property
    def bactivation_predict(self):
        #maybe this should be from an uncorrupted forward pass?
        return self.z_est

    @staticmethod
    def batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0], name="batch_normalization")
        return (batch - mean) / tf.sqrt(var + 1e-10)

    def _update_batch_normalization(self, batch):
        "batch normalize + update average mean and variance of layer"
        # mean, var = tf.nn.moments(batch, axes=[0])
        # assign_mean = self._running_mean.assign(mean)
        # assign_var = self._running_var.assign(var)

        # self.bn_assigns.append(self._ewma.apply([self._running_mean, self._running_var]))
        # with tf.control_dependencies([assign_mean, assign_var]):
        #     return (batch - mean) / tf.sqrt(var + 1e-10)
        return self.batch_normalization(batch)

    # @property
    # def assign_op(self):
    #     return self.bn_assigns

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
            return self.input_layer.activation_predict + tf.random_normal(tf.shape(self.input_layer.activation_predict),
                                                                          stddev=self.NOISE_STD)

    def unsupervised_cost_train(self):
        mean = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - unlabeled(self.input_z_clean, self.first_layer.labeled_input_size)), 1))
        # TODO: input_nodes may change...
        return (mean / self.input_nodes) * self._denoising_cost

    def _g_gauss(self, z_c, u):
        """gaussian denoising function proposed in the original paper"""
        a1 = self._create_variable('a1', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a2 = self._create_variable('a2', (BaseLayer.INPUT_BOUND_VALUE,), tf.ones([self.input_nodes]))
        a3 = self._create_variable('a3', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a4 = self._create_variable('a4', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a5 = self._create_variable('a5', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a6 = self._create_variable('a6', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a7 = self._create_variable('a7', (BaseLayer.INPUT_BOUND_VALUE,), tf.ones([self.input_nodes]))
        a8 = self._create_variable('a8', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a9 = self._create_variable('a9', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))
        a10 = self._create_variable('a10', (BaseLayer.INPUT_BOUND_VALUE,), tf.zeros([self.input_nodes]))

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu

        return z_est

    def _activation_method(self, z):
        with self.name_scope():
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
                 name="ladder_gamma_layer"):
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
        self._gamma = self._create_variable("gamma",
                                            (BaseLayer.OUTPUT_BOUND_VALUE,),
                                            gamma if gamma is not None else tf.ones([self.output_nodes]))
        """values for generating std dev of output"""

    def _activation_method(self, z):
        with self.name_scope():
            return self._non_liniarity(self._gamma * (z + self._beta), name="activation_gamma")
