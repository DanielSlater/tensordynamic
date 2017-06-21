import logging
import sys
from math import log

import tensorflow as tf
from enum import Enum

from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.layers.output_layer import CategoricalOutputLayer, OutputLayer

logger = logging.getLogger(__name__)


class EDataType(Enum):
    TRAIN = 0
    TEST = 1
    VALIDATION = 2


def create_network(initial_size, session, regularizer_coeff=0.0001, beta=.9999):
    last_layer = InputLayer(initial_size[0])

    for hidden_nodes in initial_size[1:-1]:
        last_layer = Layer(last_layer, hidden_nodes, session, non_liniarity=tf.sigmoid)

    output = CategoricalOutputLayer(last_layer, initial_size[-1], session,
                                    regularizer_weighting=regularizer_coeff, target_weighting=beta)
    return output


class BayesianResizingNet(object):
    GROWTH_MULTIPLYER = 1.1
    SHRINK_MULTIPLYER = 1. / GROWTH_MULTIPLYER
    MINIMUM_GROW_AMOUNT = 3

    def __init__(self, output_layer, model_selection_data_type = EDataType.TEST):
        if not isinstance(output_layer, OutputLayer):
            raise TypeError("resizable_net must implement AbstractResizableNet")
        self._output_layer = output_layer
        self.model_selection_data_type = model_selection_data_type

    def run(self, data_set, initial_learning_rate=0.01):
        """Train the network to find the best size

        Args:
            initial_learning_rate (float):
            data_set (tensor_dynamic.data.data_set_collection.DataSetCollection):
        """
        # DataSet must be multi-model for now
        self._output_layer.train_till_convergence(data_set.train, self.get_evaluation_data_set(data_set),
                                                  learning_rate=initial_learning_rate)
        best_score = self.model_weight_score(self._output_layer, self.get_evaluation_data_set(data_set))
        best_dimensions = self._output_layer.get_resizable_dimension_size_all_layers()

        logger.info("starting dim %s score %s", best_score, best_dimensions)

        unresized_layers = list(self._output_layer.get_all_resizable_layers())

        if len(unresized_layers) == 0:
            # no layers to resize
            print("Found no layers to resize")
            return

        current_resize_target = unresized_layers[0]

        while True:
            resized, new_best_score = current_resize_target.find_best_size(data_set.train,
                                                                           self.get_evaluation_data_set(data_set),
                                                                           self.model_weight_score,
                                                                           best_score=best_score)
            if resized:
                best_score = new_best_score
                layers_unsuccessfully_resized = 0
                if len(unresized_layers) == 1:
                    break
                else:
                    layers_unsuccessfully_resized += 1
                    if layers_unsuccessfully_resized >= len(unresized_layers):
                        # we are done resizing
                        break

            index = unresized_layers.index(current_resize_target) + 1
            if index >= len(unresized_layers):
                unresized_layers[0]
            else:
                unresized_layers[index]

        # TOOD adding layers
        logger.info("Finished with best:%s dims:%s", best_score, self._output_layer.get_resizable_dimensions())

    def get_evaluation_data_set(self, data_set):
        if self.model_selection_data_type == EDataType.TRAIN:
            return data_set.train
        elif self.model_selection_data_type == EDataType.TEST:
            return data_set.test
        elif self.model_selection_data_type == EDataType.VALIDATION:
            return data_set.validation
        else:
            raise Exception("unknown model_selection_data_type %s", self._output_layer.model_selection_data_type)

    @staticmethod
    def model_weight_score(layer, evaluation_data_set):
        evaluation_features = evaluation_data_set.features
        evaluation_labels = evaluation_data_set.labels

        log_liklihood = log_probability_of_targets_given_weights_multimodal(lambda x: layer.last_layer.activate_predict(x),
                                                                            evaluation_features,
                                                                            evaluation_labels)
        model_parameters = layer.get_parameters_all_layers()
        return bayesian_model_selection(log_liklihood, model_parameters)


def log_probability_of_targets_given_weights_multimodal(network_prediction_function, inputs, targets):
    predictions = network_prediction_function(inputs)

    result = 0.
    for i in range(len(predictions)):
        result += log(sum(predictions[i] * targets[i]))

    return result


def bayesian_model_selection(log_liklihood, number_of_parameters):
    logger.info("log_liklihood %s number_of_parameters %s", log_liklihood, number_of_parameters)
    return log_liklihood - log(number_of_parameters)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    import tensor_dynamic.data.mnist_data as mnist

    data_set = mnist.read_data_sets("data/MNIST_data", one_hot=True, limit_train_size=1000)

    with tf.Session() as session:
        brn = BayesianResizingNet(create_network((784, 5, 10), session))
        brn.run(data_set)

        # when running with 10 starting nodes
        # INFO:tensor_dynamic.utils:finished with best error 2630.17977381
        # INFO:__main__:new dim -9335.37930272 score [784, 60, 10]
