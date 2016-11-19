import tensorflow as tf

from tensor_dynamic.categorical_trainer import CategoricalTrainer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tests.base_tf_testcase import BaseTfTestCase

HIDDEN_NOES = 100


class TestGrowingLayers(BaseTfTestCase):
    def test_increase_layers_until_stop_decreasing_test_error(self):
        data = self.mnist_data
        input = InputLayer(784)
        hidden = Layer(input, HIDDEN_NOES, self.session, non_liniarity=tf.sigmoid, bactivate=False)
        output = Layer(hidden, 10, self.session, non_liniarity=tf.sigmoid, bactivate=False, supervised_cost=1.)

        cost = output.supervised_cost_predict()
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

        trainer = CategoricalTrainer(input, 0.1)

        best_score = train_until_no_improvement_for_epochs(data, cost, optimizer, 4)

        for hidden_layer_count in range(1, 10):
            print("hidden_layers {0} best_score {1}".format(hidden_layer_count, best_score))

            candidate = input.clone()
            last_hidden_layer = candidate.last_layer().input_layer()
            last_hidden_layer.add_intermediate_layer(
                lambda input_layer: Layer(input_layer, HIDDEN_NOES, self.session, non_liniarity=tf.sigmoid,
                                          bactivate=False))

            cost = candidate.last_layer.supervised_cost_predict()
            optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

            new_best_score = train_until_no_improvement_for_epochs(data, cost, optimizer, 4)

            if new_best_score > best_score:
                # failed to get improvement
                print("failed to get improvement with layer {0}".format(hidden_layer_count))
                break
            else:
                best_score = new_best_score
                input = candidate