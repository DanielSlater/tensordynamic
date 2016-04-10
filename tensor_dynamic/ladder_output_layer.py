import tensorflow as tf

from tensor_dynamic.base_layer import BaseLayer
from tensor_dynamic.ladder_layer import LadderLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.weight_functions import noise_weight_extender


class LadderOutputLayer(BaseLayer):
    def __init__(self, input_layer,
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
        self._denoising_cost = denoising_cost
        self.input_nodes = int(self.input_layer.activation.get_shape()[-1])

    @property
    def activation(self):
        return self.input_layer.activation

    @lazyprop
    def z_est_bn(self):
        return (self.z_est - self.input_layer.mean_clean) / self.input_layer.variance_clean

    @lazyprop
    def z_est(self):
        u = self.input_layer.activation_corrupted
        u = LadderLayer.batch_normalization(u)
        return self._g_gauss(self.input_layer.z_corrupted, u, self.input_nodes)

    @property
    def bactivation(self):
        print "Layer ", self.layer_number, ": ", None, " -> ", self.input_nodes, ", denoising cost: ", self._denoising_cost
        return self.z_est_bn

    def unsupervised_cost(self):
        cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_est_bn - self.input_layer.z_clean), 1))
        # TODO: input_nodes may change...
        return (cost / self.input_nodes) * self._denoising_cost

    def supervised_cost(self, targets):
        # todo may have to do something around the labelled vs unlabelled data
        # num_unlabeled_items = tf.shape(self.activation)[0] - tf.shape(targets)[0]
        # D.S todo remove hard coded 100...
        tf.concat()
        labeled_activations = tf.slice(self.activation, [0, 0], [100, -1])
        return -tf.reduce_mean(tf.reduce_sum(targets * tf.log(labeled_activations), 1))

    def _g_gauss(self, z_c, u, size):
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