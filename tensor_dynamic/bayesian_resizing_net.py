import logging
import sys
from math import log

import tensorflow as tf
from enum import Enum

from tensor_dynamic.abstract_resizable_net import AbstractResizableNet
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.layers.output_layer import CategoricalOutputLayer
from tensor_dynamic.utils import train_till_convergence, get_tf_optimizer_variables

logger = logging.getLogger(__name__)


class EDataType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


class BasicResizableNetWrapper(AbstractResizableNet):
    def __init__(self, initial_size, session, alpha=0.0001, beta=.9999, learning_rate=.1,
                 model_selection_data_type=EDataType.TEST):
        last_layer = InputLayer(initial_size[0])

        for hidden_nodes in initial_size[1:-1]:
            last_layer = Layer(last_layer, hidden_nodes, session, non_liniarity=tf.sigmoid)

        output = CategoricalOutputLayer(last_layer, initial_size[-1], session,
                                        regularizer_weighting=alpha, target_weighting=beta)

        self._net = output
        self._learn_rate_placeholder = tf.placeholder("float", shape=[], name="learn_rate")
        self._start_learning_rate = learning_rate
        self._learning_rate = learning_rate
        self.optimizer_dict = {}
        self._current_optimizer = None
        self.model_selection_data_type = model_selection_data_type
        # self._train_op = tf.train.GradientDescentOptimizer(self._learn_rate_placeholder).minimize(self._net.loss)

    def get_dimensions(self):
        return [layer.output_nodes for layer in self._net.all_layers]

    def predict(self, inputs):
        return self._net.activate_predict(inputs)

    def accuracy(self, inputs, labels):
        return self._net.accuracy(inputs, labels)

    def resize_layer(self, layer_index, new_size, data_set):
        if new_size <= 0:
            raise ValueError("new_size must all be greater than 0 was %s" % (new_size,))
        # TODO improve
        list(self._net.all_layers)[layer_index].resize(new_output_nodes=new_size)

    def train_till_convergence(self, data_set):
        # optimizer_key = tuple(self.get_dimensions())
        # if optimizer_key not in self.optimizer_dict:
        #     optimizer = tf.train.RMSPropOptimizer(0.001, name="prop_for_%s" % (self.get_dimensions()[1],))
        #     train_op = optimizer.minimize(self._net.loss)
        #     self._net.session.run(tf.initialize_variables(list(get_tf_rmsprop_optimizer_variables(optimizer))))
        #     self.optimizer_dict[optimizer_key] = train_op
        #     self._current_optimizer = optimizer
        #     print(optimizer._name)
        #
        # train_op = self.optimizer_dict[optimizer_key]

        optimizer = tf.train.RMSPropOptimizer(0.001, name="prop_for_%s" % (self.get_dimensions()[1],))
        train_op = optimizer.minimize(self._net.loss)

        self._net.session.run(tf.initialize_variables(list(get_tf_optimizer_variables(optimizer))))
        print(optimizer._name)

        iterations = [0]

        def train():
            iterations[0] += 1  # ehhhh
            error = 0.
            current_epoch = data_set.train.epochs_completed
            while data_set.train.epochs_completed == current_epoch:
                images, labels = data_set.train.next_batch(100)
                _, batch_error = self._net.session.run([train_op, self._net.loss],
                                                       feed_dict={self._net.input_placeholder: images,
                                                                  self._net.target_placeholder: labels,
                                                                  self._learn_rate_placeholder: self._learning_rate})

                error += batch_error

            self._learning_rate *= .99
            return error

        error = train_till_convergence(train, log=False, continue_epochs=6)

        logger.info("iterations = %s error = %s", iterations[0], error)

        return error

    def add_layer(self, layer_index_to_add_after, hidden_nodes):
        self._net.all_layers()[layer_index_to_add_after].add_intermediate_layer(
            lambda input_layer: Layer(input_layer, hidden_nodes, self.session, non_liniarity=tf.sigmoid))


class BayesianResizingNet(object):
    GROWTH_MULTIPLYER = 1.1
    SHRINK_MULTIPLYER = 1. / GROWTH_MULTIPLYER
    MINIMUM_GROW_AMOUNT = 3

    def __init__(self, resizable_net):
        if not isinstance(resizable_net, AbstractResizableNet):
            raise TypeError("resizable_net must implement AbstractResizableNet")
        self._resizable_net = resizable_net

    def run(self, data_set):
        # DataSet must be multimodel for now
        self._resizable_net.train_till_convergence(data_set)
        best_score = self.model_weight_score(data_set)
        best_dimensions = self._resizable_net.get_dimensions()

        logger.info("starting dim %s score %s", best_score, best_dimensions)

        current_layer_index = 1
        layers_unsuccessfully_resized = 0

        while True:
            resized, new_best_score = self.try_resize_layer(data_set, current_layer_index, best_score)
            if resized:
                best_score = new_best_score
                layers_unsuccessfully_resized = 0
                if self._resizable_net.get_hidden_layers_count() == 1:
                    break
            else:
                layers_unsuccessfully_resized += 1
                if layers_unsuccessfully_resized >= self._resizable_net.get_hidden_layers_count():
                    # we are done resizing
                    break

            current_layer_index += 1
            if current_layer_index > len(self._resizable_net.get_dimensions()) - 1:
                current_layer_index = 1

        # TOOD adding layers
        logger.info("Finished with best:%s dims:%s", best_score, self._resizable_net.get_dimensions())

    def try_resize_layer(self, data_set, layer_index, best_score):
        start_size = self._resizable_net.get_dimensions()
        best_layer_size = start_size[layer_index]
        resized = False

        # try bigger
        new_score = self._layer_resize_converge(data_set, layer_index,
                                                self.GROWTH_MULTIPLYER)

        # keep getting bigger until we stop improving
        while new_score > best_score:
            resized = True
            best_score = new_score
            best_layer_size = self._resizable_net.get_layer_size(layer_index)
            new_score = self._layer_resize_converge(data_set, layer_index,
                                                    self.GROWTH_MULTIPLYER)
        if not resized:
            logger.info("From start_size %s Bigger failed, trying smaller", start_size)
            # try smaller, doing this twice feels wrong...
            self._layer_resize_converge(data_set, layer_index,
                                        self.SHRINK_MULTIPLYER)

            new_score = self._layer_resize_converge(data_set, layer_index,
                                                    self.SHRINK_MULTIPLYER)

            while new_score > best_score:
                resized = True
                best_score = new_score
                best_layer_size = self._resizable_net.get_layer_size(layer_index)
                new_score = self._layer_resize_converge(data_set, layer_index,
                                                        self.SHRINK_MULTIPLYER)

        logger.info("From start_size %s Found best was %s", start_size, best_layer_size)

        # return to the best size we found
        self._resizable_net.resize_layer(layer_index,
                                         best_layer_size,
                                         data_set)

        self._resizable_net.train_till_convergence(data_set)

        return resized, best_score

    def _layer_resize_converge(self, data_set, layer_index, size_multiplier):
        new_size = int(self._resizable_net.get_layer_size(layer_index) * size_multiplier)
        # in case the multiplier is too small to changes values
        if abs(new_size - self._resizable_net.get_layer_size(layer_index)) < self.MINIMUM_GROW_AMOUNT:
            if size_multiplier > 1.:
                new_size = self._resizable_net.get_layer_size(layer_index) + self.MINIMUM_GROW_AMOUNT
            else:
                new_size = self._resizable_net.get_layer_size(layer_index) - self.MINIMUM_GROW_AMOUNT

        if new_size <= 0:
            logger.info("layer too small stopping downsize")
            return -sys.float_info.max

        self._resizable_net.resize_layer(layer_index,
                                         new_size,
                                         data_set)

        self._resizable_net.train_till_convergence(data_set)
        result = self.model_weight_score(data_set)
        logger.info("layer resize converge for dim: %s result: %s", self._resizable_net.get_dimensions(), result)
        return result

    def model_weight_score(self, data_set):
        if self.model_selection == EDataType.TRAIN:
            evaluation_features = data_set.train.features
            evaluation_labels = data_set.train.labels
        elif self.model_selection == EDataType.TEST:
            evaluation_features = data_set.test.features
            evaluation_labels = data_set.test.labels
        elif self.model_selection == EDataType.VALIDATION:
            evaluation_features = data_set.validation.features
            evaluation_labels = data_set.validation.labels
        else:
            raise Exception("unknown model_selection_data_type %s", self.model_selection)

        log_liklihood = log_probability_of_targets_given_weights_multimodal(lambda x: self._resizable_net.predict(x),
                                                                            evaluation_features,
                                                                            evaluation_labels)
        model_parameters = BayesianResizingNet.get_model_parameters(self._resizable_net.get_dimensions())
        return bayesian_model_selection(log_liklihood, model_parameters, len(data_set.train.features))

    @staticmethod
    def get_model_parameters(dimensions):
        # returns the number of parameters
        parameters = 0
        for i in xrange(len(dimensions) - 1):
            in_dim = dimensions[i]
            out_dim = dimensions[i + 1]

            # weights
            parameters += in_dim * out_dim

            # biases
            parameters += out_dim

        return parameters


def log_probability_of_targets_given_weights_multimodal(network_prediction_function, inputs, targets):
    predictions = network_prediction_function(inputs)

    result = 0.
    for i in range(len(predictions)):
        result += log(sum(predictions[i] * targets[i]))

    return result


def bayesian_model_selection(log_liklihood, number_of_parameters, number_of_data_points):
    score = log_liklihood - log(number_of_parameters)
    logger.info("log_liklihood %s number_of_parameters %s score %s", log_liklihood, number_of_parameters, score)
    return score


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    import tensor_dynamic.data.mnist_data as mnist

    data_set = mnist.read_data_sets("data/MNIST_data", one_hot=True, limit_train_size=1000)

    with tf.Session() as session:
        resizable_net = BasicResizableNetWrapper([784, 5, 10], session)
        brn = BayesianResizingNet(resizable_net)
        brn.run(data_set)

        # when running with 10 starting nodes
        # INFO:tensor_dynamic.utils:finished with best error 2630.17977381
        # INFO:__main__:new dim -9335.37930272 score [784, 60, 10]
