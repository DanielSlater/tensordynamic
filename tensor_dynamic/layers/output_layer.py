import logging

from tensor_dynamic.layers.layer import Layer
import tensorflow as tf

from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import get_tf_optimizer_variables, train_till_convergence
from tensor_dynamic.weight_functions import noise_weight_extender

logger = logging.getLogger(__name__)


class OutputLayer(Layer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 freeze=False,
                 non_liniarity=tf.sigmoid,
                 bactivate=False,
                 bactivation_loss_func=squared_loss,
                 weight_extender_func=noise_weight_extender,
                 unsupervised_cost=1.,
                 supervised_cost=1.,
                 noise_std=None,
                 name='OutputLayer'):
        super(OutputLayer, self).__init__(input_layer, output_nodes,
                                          session=session,
                                          bias=bias,
                                          weights=weights,
                                          back_bias=back_bias,
                                          bactivate=bactivate,
                                          freeze=freeze,
                                          non_liniarity=non_liniarity,
                                          weight_extender_func=weight_extender_func,
                                          bactivation_loss_func=bactivation_loss_func,
                                          unsupervised_cost=unsupervised_cost,
                                          supervised_cost=supervised_cost,
                                          noise_std=noise_std,
                                          name=name)
        with self.name_scope():
            self._target_placeholder = tf.placeholder('float', shape=(None,) + self.output_nodes, name='target')

    @property
    def target_placeholder(self):
        return self._target_placeholder

    def _squared_loss(self, activation):
        return tf.reduce_mean(tf.reduce_sum(tf.square(activation - self._target_placeholder), 1))

    @lazyprop
    def squared_loss_predict(self):
        return self._squared_loss(self.activation_predict)

    @lazyprop
    def squared_loss_train(self):
        return self._squared_loss(self.activation_train)

    def resize(self, **kwargs):
        assert kwargs.get('new_output_nodes') is None, "Can't change output nodes for Output layer"
        assert kwargs.get('split_output_nodes') is None, "Can't change output nodes for Output layer"
        super(OutputLayer, self).resize(**kwargs)

    def has_resizable_dimension(self):
        return False

    def get_resizable_dimension_size(self):
        return None

    def train_till_convergence(self, data_set_train, data_set_validation=None, mini_batch_size=100,
                               continue_epochs=2, learning_rate=0.0001):
        """Train this network until we stopping seeing an improvement in the error of the validation set

        Args:
            learning_rate (float): Learning rate to be used in adam optimizer
            continue_epochs (int): Number of epochs without improvement to go before stopping
            mini_batch_size (int): Number of items per mini-batch
            data_set_train (tensor_dynamic.data.data_set.DataSet): Used for training
            data_set_validation (tensor_dynamic.data.data_set.DataSet): If passed used for checking error rate each
                iteration

        Returns:
            float: Error/Accuracy we finally converged on
        """
        optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                              name="prop_for_%s" % (str(self.get_resizable_dimension_size_all_layers())
                                                                    .replace('(', '_').replace(')', '_')
                                                                    .replace('[', '_').replace(']', '_')
                                                                    .replace(',', '_').replace(' ', '_'),))
        train_op = optimizer.minimize(self.loss)

        self._session.run(tf.initialize_variables(list(get_tf_optimizer_variables(optimizer))))
        print(optimizer._name)

        iterations = [0]

        def train():
            iterations[0] += 1
            error = 0.

            for features, labels in data_set_train.one_iteration_in_batches(mini_batch_size):
                _, batch_error = self._session.run([train_op, self.loss],
                                                   feed_dict={self.input_placeholder: features,
                                                              self.target_placeholder: labels})

                error += batch_error

            if data_set_validation is not None and data_set_validation is not data_set_train:
                error, acc = self._session.run([self.loss, self._accuracy],
                                               feed_dict={
                                                   self.input_placeholder: data_set_validation.features,
                                                   self.target_placeholder: data_set_validation.labels})
                print(error, acc)

            return error

        error = train_till_convergence(train, log=False, continue_epochs=continue_epochs)

        logger.info("iterations = %s error = %s", iterations[0], error)

        return error


class CategoricalOutputLayer(OutputLayer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 freeze=False,
                 weight_extender_func=noise_weight_extender,
                 noise_std=None,
                 regularizer_weighting=0.01,
                 target_weighting=0.99,
                 name='CategoricalOutputLayer'):
        super(CategoricalOutputLayer, self).__init__(input_layer, output_nodes,
                                                     session=session,
                                                     bias=bias,
                                                     weights=weights,
                                                     back_bias=back_bias,
                                                     freeze=freeze,
                                                     weight_extender_func=weight_extender_func,
                                                     noise_std=noise_std,
                                                     name=name)
        self._target_weighting = target_weighting
        self._regularizer_weighting = regularizer_weighting

    def _layer_activation(self, input_activation, is_train):
        return tf.nn.softmax(tf.matmul(input_activation, self._weights) + self._bias)

    @lazyprop
    def target_loss(self):
        return tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.activation_train, labels=self._target_placeholder),
        )

    @lazyprop
    def regularizer_loss(self):
        weights_squared = [tf.reduce_sum(tf.square(variable)) for variable in self.variables_all_layers]
        # TODO improve
        chain_weights_squared = weights_squared[0]
        for x in weights_squared[1:]:
            chain_weights_squared = chain_weights_squared + x

        return chain_weights_squared

    @lazyprop
    def loss(self):
        return self.target_loss * .5 * self._target_weighting + \
               self.regularizer_loss * .5 * self._regularizer_weighting

    @lazyprop
    def _accuracy(self):
        return tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.activation_predict, tf.argmax(self.target_placeholder, 1), 1),
                                      tf.float32))

    def accuracy(self, data, targets):
        """Get the accuracy of our predictions against the real targets, returns a value in the range 0. to 1.

        Args:
            data (np.array):
            targets (np.array):

        Returns:
            float: accuracy in the range 0. to 1.
        """
        return self.session.run(self._accuracy,
                                feed_dict={self.input_placeholder: data, self._target_placeholder: targets})
