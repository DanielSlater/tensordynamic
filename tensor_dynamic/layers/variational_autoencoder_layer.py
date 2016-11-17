import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.utils import xavier_init
from tensor_dynamic.weight_functions import noise_weight_extender


class VariationalAutoencoderLayer(BaseLayer):
    def __init__(self, input_layer, output_nodes,
                 hidden_recog_nodes_1,
                 hidden_recog_nodes_2,
                 hidden_generation_nodes_1,
                 hidden_generation_nodes_2,
                 session=None,
                 hidden_recog_weights_1=None,
                 hidden_recog_weights_2=None,
                 hidden_recog_bias_1=None,
                 hidden_recog_bias_2=None,
                 hidden_generation_weights_1=None,
                 hidden_generation_weights_2=None,
                 hidden_generation_bias_1=None,
                 hidden_generation_bias_2=None,
                 output_mean_weights=None,
                 output_mean_bias=None,
                 output_var_weights=None,
                 output_var_bias=None,
                 reconstruction_mean_weights=None,
                 reconstruction_mean_bias=None,
                 freeze=False,
                 non_liniarity=tf.nn.softplus,
                 weight_extender_func=noise_weight_extender,
                 unsupervised_cost=1.,
                 supervised_cost=1.,
                 name='VariationalAutoencoderLayer'):
        super(VariationalAutoencoderLayer, self).__init__(input_layer, output_nodes,
                                                          session=session,
                                                          freeze=freeze,
                                                          weight_extender_func=weight_extender_func,
                                                          name=name)
        self._unsupervised_cost = unsupervised_cost
        self._non_linarity = non_liniarity
        self._hidden_recog_nodes_1 = hidden_recog_nodes_1
        self._hidden_recog_nodes_2 = hidden_recog_nodes_2
        self._hidden_generation_nodes_1 = hidden_generation_nodes_1
        self._hidden_generation_nodes_2 = hidden_generation_nodes_2

        self._hidden_recog_weights_1 = self._create_variable("hidden_recog_weights_1",
                                                             (BaseLayer.INPUT_BOUND_VALUE, self._hidden_recog_nodes_1),
                                                             hidden_recog_weights_1 if hidden_recog_weights_1 is not None else xavier_init(
                                                                 self._input_nodes,
                                                                 self._hidden_recog_nodes_1))
        self._hidden_recog_bias_1 = self._create_variable("hidden_recog_bias_1",
                                                          (self._hidden_recog_nodes_1,),
                                                          hidden_recog_bias_1 if hidden_recog_bias_1 is not None else tf.zeros(
                                                              (self._hidden_recog_nodes_1,)))
        self._hidden_recog_weights_2 = self._create_variable("hidden_recog_weights_2",
                                                             (self._hidden_recog_nodes_1, self._hidden_recog_nodes_2),
                                                             hidden_recog_weights_2 if hidden_recog_weights_2 is not None else xavier_init(
                                                                 self._hidden_recog_nodes_1,
                                                                 self._hidden_recog_nodes_2))
        self._hidden_recog_bias_2 = self._create_variable("hidden_recog_bias_2",
                                                          (self._hidden_recog_nodes_2,),
                                                          hidden_recog_bias_2 if hidden_recog_bias_2 is not None else tf.zeros(
                                                              (self._hidden_recog_nodes_2,)))
        self._hidden_generation_weights_1 = self._create_variable("hidden_generation_weights_1",
                                                                  (BaseLayer.OUTPUT_BOUND_VALUE,
                                                                   self._hidden_generation_nodes_1),
                                                                  hidden_generation_weights_1 if hidden_generation_weights_1 is not None else xavier_init(
                                                                      self._output_nodes,
                                                                      self._hidden_generation_nodes_1))
        self._hidden_generation_bias_1 = self._create_variable("hidden_generation_bias_1",
                                                               (self._hidden_generation_nodes_1,),
                                                               hidden_generation_bias_1 if hidden_generation_bias_1 is not None else tf.zeros(
                                                                   (self._hidden_generation_nodes_1,)))
        self._hidden_generation_weights_2 = self._create_variable("hidden_generation_weights_2",
                                                                  (self._hidden_generation_nodes_1,
                                                                   self._hidden_generation_nodes_2),
                                                                  hidden_generation_weights_2 if hidden_generation_weights_2 is not None else xavier_init(
                                                                      self._hidden_generation_nodes_1,
                                                                      self._hidden_generation_nodes_2))
        self._hidden_generation_bias_2 = self._create_variable("hidden_generation_bias_2",
                                                               (self._hidden_generation_nodes_2,),
                                                               hidden_generation_bias_2 if hidden_generation_bias_2 is not None else tf.zeros(
                                                                   (self._hidden_generation_nodes_2,)))
        self._output_mean_weights = self._create_variable("output_mean_weights",
                                                          (self._hidden_recog_nodes_2, BaseLayer.OUTPUT_BOUND_VALUE),
                                                          output_mean_weights if output_mean_weights is not None else xavier_init(
                                                              self._hidden_recog_nodes_2,
                                                              self._output_nodes))
        self._output_mean_bias = self._create_variable("output_mean_bias",
                                                       (BaseLayer.OUTPUT_BOUND_VALUE,),
                                                       output_mean_bias if output_mean_bias is not None else tf.zeros(
                                                           (self._output_nodes,)))
        self._output_var_weights = self._create_variable("output_var_weights",
                                                         (self._hidden_recog_nodes_2, BaseLayer.OUTPUT_BOUND_VALUE),
                                                         output_var_weights if output_var_weights is not None else xavier_init(
                                                             self._hidden_recog_nodes_2,
                                                             self._output_nodes))
        self._output_var_bias = self._create_variable("output_var_bias",
                                                      (BaseLayer.OUTPUT_BOUND_VALUE,),
                                                      output_var_bias if output_var_bias is not None else tf.zeros(
                                                          (self._output_nodes,)))
        self._reconstruction_mean_weights = self._create_variable("reconstruction_mean_weights",
                                                                  (
                                                                      self._hidden_generation_nodes_2,
                                                                      BaseLayer.INPUT_BOUND_VALUE),
                                                                  reconstruction_mean_weights if reconstruction_mean_weights is not None else xavier_init(
                                                                      self._hidden_generation_nodes_2,
                                                                      self._input_nodes))
        self._reconstruction_mean_bias = self._create_variable("reconstruction_mean_bias",
                                                               (BaseLayer.INPUT_BOUND_VALUE,),
                                                               reconstruction_mean_bias if reconstruction_mean_bias is not None else tf.zeros(
                                                                   (self._input_nodes,)))

        self._z_mean_train, self._z_log_sigma_sq_train = self.recognition(self.input_layer.activation_train)
        self._z_mean_predict, self._z_log_sigma_sq_predict = self.recognition(self.input_layer.activation_predict)

        eps = tf.random_normal(tf.shape(self._z_mean_train), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self._z_train = tf.add(self._z_mean_train,
                               tf.mul(tf.sqrt(tf.exp(self._z_log_sigma_sq_train)), eps))

        self._x_reconstruction_train = self.generator(self._z_train)
        self._x_reconstruction_predict = self.generator(self._z_mean_predict)

    def recognition(self, input):
        layer_1 = self._non_linarity(tf.add(tf.matmul(input, self._hidden_recog_weights_1),
                                            self._hidden_recog_bias_1))
        layer_2 = self._non_linarity(tf.add(tf.matmul(layer_1, self._hidden_generation_weights_2),
                                            self._hidden_recog_bias_2))
        z_mean = tf.add(tf.matmul(layer_2, self._output_mean_weights),
                        self._output_mean_bias)
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, self._output_var_weights),
                   self._output_var_bias)
        return z_mean, z_log_sigma_sq

    def generator(self, input):
        layer_1 = self._non_linarity(tf.add(tf.matmul(input, self._hidden_generation_weights_1),
                                            self._hidden_generation_bias_1))
        layer_2 = self._non_linarity(tf.add(tf.matmul(layer_1, self._hidden_generation_weights_2),
                                            self._hidden_generation_bias_2))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self._reconstruction_mean_weights),
                                 self._reconstruction_mean_bias))
        return x_reconstr_mean

    @lazyprop
    def activation_train(self):
        return self._z_train

    @lazyprop
    def activation_predict(self):
        return self._z_mean_predict

    @lazyprop
    def bactivation_train(self):
        return self._x_reconstruction_train

    @lazyprop
    def bactivation_predict(self):
        return self._x_reconstruction_predict

    @lazyprop
    def bactivation_loss_train(self):
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.input_layer.activation_train * tf.log(1e-10 + self.bactivation_train)
                           + (1 - self.input_layer.activation_train) * tf.log(1e-10 + 1 - self.bactivation_train),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self._z_log_sigma_sq_train
                                           - tf.square(self._z_mean_train)
                                           - tf.exp(self._z_log_sigma_sq_train), 1)
        return tf.reduce_mean(reconstr_loss + latent_loss)
        #return reconstr_loss + latent_loss

    @lazyprop
    def bactivation_loss_predict(self):
        reconstr_loss = \
            -tf.reduce_sum(self.input_layer.activation_predict * tf.log(1e-10 + self.bactivation_predict)
                           + (1 - self.input_layer.activation_predict) * tf.log(1e-10 + 1 - self.bactivation_predict),
                           1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self._z_log_sigma_sq_predict
                                           - tf.square(self._z_mean_predict)
                                           - tf.exp(self._z_log_sigma_sq_predict), 1)
        return tf.reduce_mean(reconstr_loss + latent_loss)
        #return reconstr_loss + latent_loss

    def unsupervised_cost_train(self):
        return self.bactivation_loss_train * self._unsupervised_cost

    def unsupervised_cost_predict(self):
        return self.bactivation_loss_predict * self._unsupervised_cost

    @property
    def kwargs(self):
        kwargs = super(VariationalAutoencoderLayer, self).kwargs

        return kwargs
