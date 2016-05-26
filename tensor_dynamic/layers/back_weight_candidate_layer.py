import tensorflow as tf
import numpy as np

from tensor_dynamic.layers.back_weight_layer import BackWeightLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import xavier_init, tf_resize
from tensor_dynamic.weight_functions import noise_weight_extender


class BackWeightCandidateLayer(BackWeightLayer):
    CANDIDATES = 1
    CANDIDATE_TRAIN_DISCOUNT = 0.1

    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 back_weights=None,
                 back_bias=None,
                 freeze=False,
                 non_liniarity=tf.nn.relu,
                 weight_extender_func=noise_weight_extender,
                 bactivation_loss_func=squared_loss,
                 unsupervised_cost=1.,
                 supervised_cost=1.,
                 noise_std=None,
                 name='BackWeightLayer'):
        super(BackWeightCandidateLayer, self).__init__(input_layer, output_nodes,
                                                       session=session,
                                                       bias=bias,
                                                       weights=weights,
                                                       back_weights=back_weights,
                                                       back_bias=back_bias,
                                                       freeze=freeze,
                                                       bactivation_loss_func=bactivation_loss_func,
                                                       non_liniarity=non_liniarity,
                                                       weight_extender_func=weight_extender_func,
                                                       unsupervised_cost=unsupervised_cost,
                                                       supervised_cost=supervised_cost,
                                                       noise_std=noise_std,
                                                       name=name)

        self._candidate_bias = self._create_variable("candidate_bias",
                                                     (self.CANDIDATES,),
                                                     np.zeros(self.CANDIDATES, dtype=np.float32),
                                                     is_kwarg=False)
        self._candidate_weights = self._create_variable("candidate_weights",
                                                        (self.INPUT_BOUND_VALUE, self.CANDIDATES),
                                                        xavier_init(
                                                            self.input_nodes,
                                                            1),
                                                        is_kwarg=False)
        self._candidate_back_bias = self._create_variable("candidate_back_bias",
                                                          (self.CANDIDATES,),
                                                          np.zeros(self.CANDIDATES, dtype=np.float32),
                                                          is_kwarg=False)
        self._candidate_back_weights = self._create_variable("candidate_back_weights",
                                                             (self.CANDIDATES, self.INPUT_BOUND_VALUE),
                                                             xavier_init(
                                                                 1,
                                                                 self.input_nodes),
                                                             is_kwarg=False)

        self._candidate_bactivation_predict = self._non_liniarity(
            tf.matmul(
                self._non_liniarity(
                    tf.matmul(self.input_layer.activation_predict, self._candidate_weights) + self._candidate_bias),
                self._candidate_back_weights) + self._candidate_back_bias)

        self._candidate_bactivation_train = self._non_liniarity(
            tf.matmul(
                self._non_liniarity(
                    tf.matmul(self.input_layer.activation_train, self._candidate_weights) + self._candidate_bias),
                self._candidate_back_weights) + self._candidate_back_bias)

        self.session.run(tf.initialize_variables([self._candidate_weights, self._candidate_bias,
                                                  self._candidate_back_bias, self._candidate_back_weights]))

    @lazyprop
    def bactivation_loss_train(self):
        return tf.reduce_mean(tf.square(
            (self.bactivation_train + self._candidate_bactivation_train * self.CANDIDATE_TRAIN_DISCOUNT)
            - self.input_layer.activation_train))

    @lazyprop
    def bactivation_loss_predict(self):
        return tf.reduce_mean(tf.square(
            (self.bactivation_predict + self._candidate_bactivation_predict * self.CANDIDATE_TRAIN_DISCOUNT)
            - self.input_layer.activation_predict))

    def resize(self, new_output_nodes=None):
        if new_output_nodes is None or new_output_nodes > self.output_nodes:
            # promote the candidate
            tf_resize(self._session, self._weights,
                      new_values=np.append(self.session.run(self._weights), self.session.run(self._candidate_weights),
                                           axis=1).astype(np.float32))

            tf_resize(self._session, self._back_weights,
                      new_values=np.append(self.session.run(self._back_weights),
                                           self.session.run(self._candidate_back_weights), axis=0).astype(np.float32), )

            tf_resize(self._session, self._bias,
                      new_values=np.append(self.session.run(self._bias), self.session.run(self._candidate_bias)).astype(
                          np.float32))

            tf_resize(self._session, self._back_bias,
                      new_values=np.append(self.session.run(self._back_bias),
                                           self.session.run(self._candidate_back_bias)).astype(np.float32))

        super(BackWeightCandidateLayer, self).resize(new_output_nodes=new_output_nodes)
