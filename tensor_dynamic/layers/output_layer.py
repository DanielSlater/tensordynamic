import logging
import math
import random

import tensorflow as tf

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.tf_loss_functions import squared_loss
from tensor_dynamic.utils import get_tf_optimizer_variables, train_till_convergence
from tensor_dynamic.weight_functions import noise_weight_extender

logger = logging.getLogger(__name__)


class OutputLayer(HiddenLayer):
    def __init__(self, input_layer, output_nodes,
                 session=None,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 freeze=False,
                 non_liniarity=None,
                 bactivate=False,
                 bactivation_loss_func=None,
                 weight_extender_func=None,
                 input_noise_std=None,
                 regularizer_weighting=0.01,
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
                                          input_noise_std=input_noise_std,
                                          name=name)
        self._regularizer_weighting = regularizer_weighting

        with self.name_scope():
            self._target_placeholder = tf.placeholder('float', shape=(None,) + self.output_nodes, name='target')

    @property
    def target_placeholder(self):
        return self._target_placeholder

    def _target_loss_op(self, input_tensor):
        return tf.reduce_mean(tf.reduce_sum(tf.square(input_tensor - self._target_placeholder), 1))

    @lazyprop
    def target_loss_op_train(self):
        return self._target_loss_op(self.activation_train)

    @lazyprop
    def target_loss_op_predict(self):
        return self._target_loss_op(self.activation_predict)

    @lazyprop
    def loss_op_train(self):
        if self._regularizer_weighting > 0.:
            return self.target_loss_op_train * (1. - self._regularizer_weighting) + \
                   self.regularizer_2_loss_op * self._regularizer_weighting
        else:
            return self.target_loss_op_train

    @lazyprop
    def loss_op_predict(self):
        if self._regularizer_weighting > 0.:
            return self.target_loss_op_train * (1. - self._regularizer_weighting) + \
                   self.regularizer_2_loss_op * self._regularizer_weighting
        else:
            return self.target_loss_op_train

    @lazyprop
    def regularizer_2_loss_op(self):
        weights_squared = [tf.reduce_sum(tf.square(variable)) for variable in self.variables_all_layers]
        # TODO improve
        chain_weights_squared = weights_squared[0]
        for x in weights_squared[1:]:
            chain_weights_squared = chain_weights_squared + x

        return chain_weights_squared

    @lazyprop
    def accuracy_op(self):
        return self.target_loss_op_predict  # TODO accuracy doesn't make sense here...

    def resize(self, **kwargs):
        assert kwargs.get('new_output_nodes') is None, "Can't change output nodes for Output layer"
        assert kwargs.get('split_output_nodes') is None, "Can't change output nodes for Output layer"
        super(OutputLayer, self).resize(**kwargs)

    def has_resizable_dimension(self):
        return False

    def get_resizable_dimension_size(self):
        return None

    def train_till_convergence(self, data_set_train, data_set_validation=None, mini_batch_size=100,
                               continue_epochs=2, learning_rate=0.0001,
                               optimizer=tf.train.AdamOptimizer,
                               on_iteration_complete_func=None,
                               on_mini_batch_complete_func=None):
        """Train this network until we stopping seeing an improvement in the error of the validation set

        Args:
            optimizer (tf.train.Optimizer): Type of optimizer to use, e.g. Adam or RMSProp
            learning_rate (float): Learning rate to be used in adam optimizer
            continue_epochs (int): Number of epochs without improvement to go before stopping
            mini_batch_size (int): Number of items per mini-batch
            data_set_train (tensor_dynamic.data.data_set.DataSet): Used for training
            data_set_validation (tensor_dynamic.data.data_set.DataSet): If passed used for checking error rate each
                iteration

        Returns:
            float: Error/Accuracy we finally converged on
        """
        assert isinstance(data_set_train, DataSet)
        if data_set_validation is not None:
            assert isinstance(data_set_validation, DataSet)

        optimizer_instance = optimizer(learning_rate,
                                       name="prop_for_%s" % (str(self.get_resizable_dimension_size_all_layers())
                                                             .replace('(', '_').replace(')', '_')
                                                             .replace('[', '_').replace(']', '_')
                                                             .replace(',', '_').replace(' ', '_'),))
        train_op = optimizer_instance.minimize(self.loss_op_train)

        self._session.run(tf.initialize_variables(list(get_tf_optimizer_variables(optimizer_instance))))
        print(optimizer_instance._name)

        iterations = [0]

        validation_size = data_set_validation.num_examples if data_set_validation is not None else data_set_train.num_examples

        validation_part_size = validation_size / int(math.ceil(validation_size / 1000.))

        def train():
            iterations[0] += 1
            train_error = 0.
            test_error = None

            for features, labels in data_set_train.one_iteration_in_batches(mini_batch_size):
                _, batch_error = self._session.run([train_op, self.loss_op_train],
                                                   feed_dict={self.input_placeholder: features,
                                                              self.target_placeholder: labels})

                if on_mini_batch_complete_func is not None:
                    on_mini_batch_complete_func(self, iterations[0], batch_error)

                train_error += batch_error

            if data_set_validation is not None and data_set_validation is not data_set_train:
                # we may have to break this into equal parts
                test_error, acc = 0., 0.
                parts = 0
                for features, labels in data_set_validation.one_iteration_in_batches(validation_part_size):
                    parts += 1
                    batch_error, batch_acc = self._session.run([self.loss_op_predict, self.accuracy_op],
                                                               feed_dict={
                                                                   self.input_placeholder: features,
                                                                   self.target_placeholder: labels})
                    test_error += batch_error
                    acc += batch_acc

                test_error /= parts
                print(test_error, acc / parts)

            if on_iteration_complete_func is not None:
                on_iteration_complete_func(self, iterations[0], train_error=train_error, test_error=test_error)

            return test_error or train_error

        error = train_till_convergence(train, log=False, continue_epochs=continue_epochs)

        logger.info("iterations = %s error = %s", iterations[0], error)

        return error

    def evaluation_stats(self, dataset):
        """Returns stats related to run

        Args:
            dataset (DataSet):

        Returns:
            (float, float, float): log_probability of the targets given the data, accuracy, target_loss
        """
        log_prob, accuracy, target_loss = self.session.run([self.log_probability_of_targets_op,
                                                            self.accuracy_op,
                                                            self.target_loss_op_train],
                                                           feed_dict={self.input_placeholder: dataset.features,
                                                                      self._target_placeholder: dataset.labels})

        return log_prob, accuracy, target_loss

    @property
    def kwargs(self):
        kwargs = super(OutputLayer, self).kwargs

        kwargs['regularizer_weighting'] = self._regularizer_weighting

        return kwargs

    def learn_structure_random(self, data_set_train, data_set_validate, start_learn_rate=0.01, continue_learn_rate=0.0001):
        rejected_changes = 0
        self.train_till_convergence(data_set_train, data_set_validate, learning_rate=start_learn_rate)
        number_of_convergences = 1

        best_log_prob, accuracy, target_loss = self.evaluation_stats(data_set_validate)
        print(best_log_prob, accuracy, target_loss)

        best_param = self.get_parameters_all_layers()
        best_model_weight = best_log_prob - math.log(self.get_parameters_all_layers())

        # make random change
        while rejected_changes <= 4:
            layer_to_resize = random.choice(list(self.get_all_resizable_layers()))
            network_start_state = self.get_network_state()
            node_change = random.choice([self.GROWTH_MULTIPLYER, self.SHRINK_MULTIPLYER])
            start_size = layer_to_resize.get_resizable_dimension_size()
            new_node_count = layer_to_resize._get_new_node_count(node_change)
            layer_to_resize.resize(new_node_count)
            try:
                self.train_till_convergence(data_set_train, data_set_validate, learning_rate=continue_learn_rate)
            except Exception as ex:
                print (ex)
            number_of_convergences += 1

            # did it work?
            new_log_prob, _, _ = self.evaluation_stats(data_set_validate)

            new_model_weight = new_log_prob - math.log(self.get_parameters_all_layers())
            if new_model_weight <= best_model_weight:
                rejected_changes += 1
                print("REJECTED change of layer %s" % (layer_to_resize.layer_number,))
                print("from size:%s log_prob:%s param:%s score:%s" % (start_size, best_log_prob, best_param,
                                                                      best_model_weight))
                print("To size:%s log_prob:%s param:%s score:%s score change" % (new_node_count,
                                                                                 new_log_prob,
                                                                                 self.get_parameters_all_layers(),
                                                                                 new_model_weight))
                self.set_network_state(network_start_state)
            else:
                rejected_changes = 0
                print("ACCEPTED change of layer %s" % (layer_to_resize.layer_number,))
                print("from size:%s log_prob:%s param:%s score:%s" % (start_size, best_log_prob, best_param,
                                                                      best_model_weight))
                print("To size:%s log_prob:%s param:%s score:%s score change" % (new_node_count,
                                                                                 new_log_prob,
                                                                                 self.get_parameters_all_layers(),
                                                                                 new_model_weight))
                best_param = self.get_parameters_all_layers()
                best_model_weight = new_model_weight
                best_log_prob = new_log_prob


class BinaryOutputLayer(OutputLayer):
    def __init__(self, input_layer,
                 session=None,
                 bias=None,
                 weights=None,
                 back_bias=None,
                 freeze=False,
                 weight_extender_func=None,
                 noise_std=None,
                 regularizer_weighting=0.01,
                 name='BinaryOutputLayer'):
        super(BinaryOutputLayer, self).__init__(input_layer, (1,),
                                                session=session,
                                                bias=bias,
                                                weights=weights,
                                                back_bias=back_bias,
                                                freeze=freeze,
                                                weight_extender_func=weight_extender_func,
                                                noise_std=noise_std,
                                                regularizer_weighting=regularizer_weighting,
                                                name=name)

    @lazyprop
    def accuracy_op(self):
        correct_prediction = tf.equal(
            tf.round(tf.abs(self.activation_predict - self.target_placeholder)), 0)

        return tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))
