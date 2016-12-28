import logging
import sys
from math import log

import tensorflow as tf

from tensor_dynamic.abstract_resizable_net import AbstractResizableNet
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.layers.output_layer import CatigoricalOutputLayer
from tensor_dynamic.utils import train_till_convergence, get_tf_adam_optimizer_variables

logger = logging.getLogger(__name__)


class BasicResizableNetWrapper(AbstractResizableNet):
    def __init__(self, initial_size, session, alpha=0.0001, beta=.9999, learning_rate=.1):
        last_layer = InputLayer(initial_size[0])

        for hidden_nodes in initial_size[1:-1]:
            last_layer = Layer(last_layer, hidden_nodes, session, non_liniarity=tf.sigmoid)

        output = CatigoricalOutputLayer(last_layer, initial_size[-1], session,
                                        regularizer_weighting=alpha, target_weighting=beta)

        self._net = output
        self._learn_rate_placeholder = tf.placeholder("float", shape=[], name="learn_rate")
        self._start_learning_rate = learning_rate
        self._learning_rate = learning_rate
        #self._train_op = tf.train.GradientDescentOptimizer(self._learn_rate_placeholder).minimize(self._net.loss)

    def get_dimensions(self):
        return [layer.output_nodes for layer in self._net.all_layers]

    def predict(self, inputs):
        return self._net.activate_predict(inputs)

    def accuracy(self, inputs, labels):
        return self._net.accuracy(inputs, labels)

    def resize_layer(self, layer_index, new_size, data_set):
        # TODO improve
        list(self._net.all_layers)[layer_index].resize(new_output_nodes=new_size)

    def train_till_convergence(self, data_set):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(self._net.loss)
        self._net.session.run(tf.initialize_variables(list(get_tf_adam_optimizer_variables(optimizer))))
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

        def on_no_improvement():
            self._learning_rate *= .9

        train_till_convergence(train, log=False, continue_epochs=5,
                               on_no_improvement_func=None)

        logger.info("iterations = %s", iterations[0])

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

        self._resizable_net.resize_layer(layer_index,
                                         new_size,
                                         data_set)

        self._resizable_net.train_till_convergence(data_set)
        result = self.model_weight_score(data_set)
        logger.info("trying size %s result %s", self._resizable_net.get_dimensions(), result)
        return result

    def model_weight_score(self, data_set):
        log_probability = log_probability_of_targets_given_weights_multimodal(lambda x: self._resizable_net.predict(x),
                                                                              data_set.train.images,
                                                                              data_set.train.labels)
        model_complexity = BayesianResizingNet.get_model_complexity(self._resizable_net.get_dimensions())

        return log_probability - log(model_complexity)

    @staticmethod
    def get_model_complexity(dimensions):
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
