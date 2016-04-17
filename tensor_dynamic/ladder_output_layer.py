import tensorflow as tf

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.ladder_layer import LadderLayer, unlabeled, labeled
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender


class LadderOutputLayer(BaseLayer):
    def __init__(self, input_layer,
                 denoising_cost,
                 session=None,
                 freeze=False,
                 weight_extender_func=noise_weight_extender,
                 name="ladderOutput"):
        super(LadderOutputLayer, self).__init__(input_layer,
                                                input_layer.output_nodes,
                                                session=session,
                                                freeze=freeze,
                                                weight_extender_func=weight_extender_func,
                                                name=name)
        self._denoising_cost = denoising_cost

    @property
    def activation(self):
        return labeled(self.input_layer.activation)

    @property
    def bactivation(self):
        print "Layer ", self.layer_number, ": ", None, " -> ", self.input_nodes, ", denoising cost: ", self._denoising_cost
        return self.z_est_bn

    @lazyprop
    def z_est(self):
        print "Layer ", self.layer_number, ": ", self.output_nodes, " -> ", self.input_nodes, ", denoising cost: ", self._denoising_cost
        u = unlabeled(self.input_layer.activation_corrupted)
        u = LadderLayer.batch_normalization(u, self.input_layer.mean_clean_unlabeled, self.input_layer.variance_clean_unlabeled)
        return self._g_gauss(unlabeled(self.input_layer.z_corrupted), u)

    @lazyprop
    def z_est_bn(self):
        return (self.z_est - self.input_layer.mean_clean_unlabeled) / self.input_layer.variance_clean_unlabeled

    def unsupervised_cost(self):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - unlabeled(self.input_layer.z_clean)), 1))
        # TODO: input_nodes may change...
        return (cost / self.input_nodes) * self._denoising_cost

    def supervised_cost(self, targets):
        # todo may have to do something more around the labelled vs unlabelled data

        labeled_activations_corrupted = labeled(self.input_layer.activation_corrupted) #tf.slice(self.activation, [0, 0], tf.shape(targets))
        return -tf.reduce_mean(tf.reduce_sum(targets * tf.log(labeled_activations_corrupted), 1))

    # def train(self, unlabeled_input, labeled_input, labeled_targets):
    #
    # def prediction(self):
    #

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
