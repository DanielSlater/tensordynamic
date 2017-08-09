import logging
import math
import random

import tensorflow as tf

from tensor_dynamic.data.data_set import DataSet
from tensor_dynamic.layers.hidden_layer import HiddenLayer
from tensor_dynamic.lazyprop import lazyprop
from tensor_dynamic.utils import get_tf_optimizer_variables, train_till_convergence

logger = logging.getLogger(__name__)


def bayesian_model_comparison_evaluation(model, data_set):
    """Use bayesian model comparison to evaluate a trained model

    Args:
        model (OutputLayer): Trained model to evaluate
        data_set (DataSet): data set this model was trained on, tends to be test set, but can be train if set up so

    Returns:
        float : log_probability_og_model_generating_data - log(number_of_parameters)
    """
    log_prob, _, _ = model.last_layer.evaluation_stats(data_set)
    param = model.get_parameters_all_layers()
    score = log_prob - math.log(param)
    print (model.get_resizable_dimension_size(), score, log_prob, param)
    return score


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
                 layer_noise_std=None,
                 drop_out_prob=None,
                 batch_normalize_input=None,
                 batch_norm_transform=None,
                 batch_norm_scale=None,
                 regularizer_weighting=0.01,
                 regularizer_op=tf.nn.l2_loss,
                 save_checkpoints=0,
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
                                          layer_noise_std=layer_noise_std,
                                          drop_out_prob=drop_out_prob,
                                          batch_normalize_input=batch_normalize_input,
                                          batch_norm_transform=batch_norm_transform,
                                          batch_norm_scale=batch_norm_scale,
                                          name=name)
        self._regularizer_weighting = regularizer_weighting
        self._regularizer_op = regularizer_op
        self._save_checkpoints = save_checkpoints


        with self.name_scope():
            self._target_placeholder = tf.placeholder('float', shape=(None,) + self.output_nodes, name='target')

    @property
    def target_placeholder(self):
        return self._target_placeholder

    def _target_loss_op(self, input_tensor):
        return tf.reduce_mean(tf.reduce_sum(tf.square(input_tensor - self._target_placeholder), 1))

    @lazyprop
    def target_loss_op_train(self):
        with self.name_scope(is_train=True):
            return self._target_loss_op(self.activation_train)

    @lazyprop
    def target_loss_op_predict(self):
        with self.name_scope(is_predict=True):
            return self._target_loss_op(self.activation_predict)

    @lazyprop
    def loss_op_train(self):
        if self._regularizer_weighting > 0.:
            return self.target_loss_op_train * (1. - self._regularizer_weighting) + \
                   self.regularizer_loss_op * self._regularizer_weighting
        else:
            return self.target_loss_op_train

    @lazyprop
    def loss_op_predict(self):
        if self._regularizer_weighting > 0.:
            return self.target_loss_op_predict * (1. - self._regularizer_weighting) + \
                   self.regularizer_loss_op * self._regularizer_weighting
        else:
            return self.target_loss_op_train

    @lazyprop
    def regularizer_loss_op(self):
        with self.name_scope():
            weights_squared = [self._regularizer_op(variable) for variable in self.regularizable_variables_all_layers]

            # TODO improve
            chain_weights_squared = weights_squared[0]
            for x in weights_squared[1:]:
                chain_weights_squared = chain_weights_squared + x

            return tf.reduce_mean(chain_weights_squared)

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

        self._session.run(tf.variables_initializer(list(get_tf_optimizer_variables(optimizer_instance))))
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
                print(train_error, test_error, acc / parts)

            if on_iteration_complete_func is not None:
                on_iteration_complete_func(self, iterations[0], train_error=train_error, test_error=test_error)

            return test_error or train_error

        error = train_till_convergence(train, log=False, continue_epochs=continue_epochs)

        logger.info("iterations = %s error = %s", iterations[0], error)

        return error, iterations[0]

    def evaluation_stats(self, dataset):
        """Returns stats related to run

        Args:
            dataset (DataSet):

        Returns:
            (float, float, float): log_probability of the targets given the data, accuracy, target_loss
        """
        log_prob, accuracy, target_loss = self.session.run([self.log_probability_of_targets_op,
                                                            self.accuracy_op,
                                                            self.target_loss_op_predict],
                                                           feed_dict={self.input_placeholder: dataset.features,
                                                                      self._target_placeholder: dataset.labels})

        return log_prob, accuracy, target_loss

    @property
    def kwargs(self):
        kwargs = super(OutputLayer, self).kwargs

        kwargs['regularizer_weighting'] = self._regularizer_weighting
        kwargs['regularizer_op'] = self._regularizer_op
        kwargs['save_checkpoints'] = self._save_checkpoints

        return kwargs

    def learn_structure_layer_by_layer(self, data_set_train, data_set_validation, start_learn_rate=0.001,
                                       continue_learn_rate=0.0001,
                                       model_evaluation_function=bayesian_model_comparison_evaluation,
                                       add_layers=False,
                                       save_checkpoint_path=None):
        self.train_till_convergence(data_set_train, data_set_validation, learning_rate=start_learn_rate)
        best_score = model_evaluation_function(self, data_set_validation)

        if save_checkpoint_path:
            self.save_checkpoints(save_checkpoint_path)

        while True:
            best_score = self._best_sizes_for_current_layer_number(best_score, continue_learn_rate, data_set_train,
                                                                   data_set_validation, model_evaluation_function,
                                                                   save_checkpoint_path)

            if add_layers:
                state = self.get_network_state()
                self.input_layer.add_intermediate_cloned_layer()
                self.last_layer.train_till_convergence(data_set_train, data_set_validation,
                                                       learning_rate=continue_learn_rate)
                result = model_evaluation_function(self, data_set_validation)
                if result > best_score:
                    best_score = result

                    if save_checkpoint_path:
                        self.save_checkpoints(save_checkpoint_path)
                else:
                    # adding a layer didn't help, so reset
                    self.set_network_state(state)
                    return
            else:
                return

    def _best_sizes_for_current_layer_number(self, best_score, continue_learn_rate, data_set_train, data_set_validation,
                                             model_evaluation_function,
                                             save_checkpoint_path):
        layers = list(self.get_all_resizable_layers())
        index = 0
        attempts_with_out_resize = 0
        while attempts_with_out_resize < len(layers):
            resized, best_score = layers[index].find_best_size(data_set_train, data_set_validation,
                                                               model_evaluation_function=model_evaluation_function,
                                                               best_score=best_score,
                                                               tuning_learning_rate=continue_learn_rate)
            if resized:
                attempts_with_out_resize = 1
                if save_checkpoint_path:
                    self.save_checkpoints(save_checkpoint_path)
            else:
                attempts_with_out_resize += 1
            index += 1
            if index == len(layers):
                index = 0
        return best_score

    def save_checkpoints(self, checkpoint_path):
        self._save_checkpoints += 1
        with open(checkpoint_path + "_" + str(self._save_checkpoints) + ".tdc", "w") as f:
            pkl = self.get_network_pickle()
            f.write(pkl)

    def learn_structure_random(self, data_set_train, data_set_validate, start_learn_rate=0.01,
                               continue_learn_rate=0.0001,
                               evaluation_method=bayesian_model_comparison_evaluation,
                               save_checkpoint_path=None):
        rejected_changes = 0
        self.train_till_convergence(data_set_train, data_set_validate, learning_rate=start_learn_rate)

        if save_checkpoint_path:
            self.save_checkpoints(save_checkpoint_path)


        number_of_convergences = 1

        best_model_weight = evaluation_method(self, data_set_validate)
        last_change_was_success = False
        layer_to_resize = None
        node_change = None

        # make random change
        while rejected_changes <= 4:
            network_start_state = self.get_network_state()

            if not last_change_was_success:
                # Only make a new random choice if the last choice was a failure, and if so choose a different layer
                layer_to_resize = random.choice(
                    list(x for x in self.get_all_resizable_layers() if x != layer_to_resize))
                node_change = random.choice([self.GROWTH_MULTIPLYER, self.SHRINK_MULTIPLYER])

            start_size = layer_to_resize.get_resizable_dimension_size()
            new_node_count = layer_to_resize._get_new_node_count(node_change)
            layer_to_resize.resize(new_node_count)

            self.train_till_convergence(data_set_train, data_set_validate, learning_rate=continue_learn_rate)

            number_of_convergences += 1

            # did it work?
            new_model_weight = evaluation_method(self, data_set_validate)

            if new_model_weight <= best_model_weight:
                rejected_changes += 1
                print("REJECTED change of layer %s" % (layer_to_resize.layer_number,))
                print("from size:%s param:%s score:%s" % (start_size, best_param,
                                                          best_model_weight))
                print("To size:%s param:%s score:%s score change" % (new_node_count,
                                                                     self.get_parameters_all_layers(),
                                                                     new_model_weight))
                self.set_network_state(network_start_state)
                last_change_was_success = False
            else:
                rejected_changes = 0
                print("ACCEPTED change of layer %s" % (layer_to_resize.layer_number,))
                print("from size:%s param:%s score:%s" % (start_size, best_param,
                                                          best_model_weight))
                print("To size:%s param:%s score:%s score change" % (new_node_count,
                                                                     self.get_parameters_all_layers(),
                                                                     new_model_weight))
                best_param = self.get_parameters_all_layers()
                best_model_weight = new_model_weight

                last_change_was_success = True

                if save_checkpoint_path:
                    self.save_checkpoints(save_checkpoint_path)


