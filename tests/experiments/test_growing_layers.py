"""
This script tests
* creating a network with 1 hidden layer
* training it until convergence
* then repeating
    * adding a layer
    * train till convergence

The results are, you can add a layer and it seems to always improve test error

TODO:   compare to highway layer
        compare to residule layer
        compare to network of same size trained from scratch
"""
import sys
import tensorflow as tf

from tensor_dynamic.categorical_trainer import CategoricalTrainer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tests.base_tf_testcase import BaseTfTestCase

HIDDEN_NOES = 100
MAX_EPOCHS = 1000


def train_until_no_improvement_for_epochs(train_data_set, net, max_epochs_without_improvement, validation_data_set=None):
    trainer = CategoricalTrainer(net, 0.1)
    best_error = sys.float_info.max
    epochs_since_best_error = 0

    for x in range(MAX_EPOCHS):
        error = trainer.train_one_epoch(train_data_set, 100)
        print("iteration {0} error {1}".format(x, error))

        if validation_data_set is not None:
            error = trainer.ac

        trainer.learn_rate *= 0.995

        if error < best_error:
            best_error = error
            epochs_since_best_error = 0
        else:
            if epochs_since_best_error > max_epochs_without_improvement:
                break
            epochs_since_best_error += 1

    return best_error


class TestGrowingLayers(BaseTfTestCase):
    def test_increase_layers_until_stop_decreasing_test_error(self):
        data = self.mnist_data
        input = InputLayer(784)
        hidden = Layer(input, HIDDEN_NOES, self.session, non_liniarity=tf.sigmoid, bactivate=False)
        output = Layer(hidden, 10, self.session, non_liniarity=tf.sigmoid, bactivate=False, supervised_cost=1.)

        best_score = train_until_no_improvement_for_epochs(data, output, 3)

        for hidden_layer_count in range(1, 10):
            print("hidden_layers {0} best_score {1}".format(hidden_layer_count, best_score))

            candidate = output.clone()
            last_hidden_layer = candidate.last_layer.input_layer
            last_hidden_layer.add_intermediate_layer(
                lambda input_layer: Layer(input_layer, HIDDEN_NOES, self.session, non_liniarity=tf.sigmoid,
                                          bactivate=False))

            new_best_score = train_until_no_improvement_for_epochs(data, candidate, 3)
            if new_best_score > best_score:
                # failed to get improvement
                print("failed to get improvement with layer {0}".format(hidden_layer_count))
                break
            else:
                best_score = new_best_score
                output = candidate