import numpy as np

from tensor_dynamic.categorical_trainer import CategoricalTrainer
from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.duel_state_relu_layer import DuelStateReluLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.train_policy import DuelStateReluTrainPolicy
from tests.layers.base_layer_testcase import BaseLayerWrapper


class TestDuelStateReluLayer(BaseLayerWrapper.BaseLayerTestCase):
    def _create_layer_for_test(self):
        return DuelStateReluLayer(self._input_layer, self.OUTPUT_NODES, session=self.session)

    def test_mnist_start_large(self):
        data = self.mnist_data

        input_layer = InputLayer(784)
        hidden_1 = DuelStateReluLayer(input_layer, 200, session=self.session, inactive_nodes_to_leave=200)
        output = Layer(hidden_1, self.MNIST_OUTPUT_NODES, session=self.session)
        trainer = CategoricalTrainer(output, 0.1)

        end_epoch = data.train.epochs_completed + 5

        print(trainer.accuracy(data.test.images, data.test.labels))

        while data.train.epochs_completed <= end_epoch:
            train_x, train_y = data.train.next_batch(100)
            trainer.train(train_x, train_y)

        accuracy, cost = trainer.accuracy(data.test.images, data.test.labels)
        print(accuracy, cost)
        # print(output.active_nodes())
        print(hidden_1.active_nodes())

        self.assertGreater(accuracy, 90.)
        # self.assertEqual(output.active_nodes(), self.MNIST_OUTPUT_NODES, msg='expect all output nodes to be active')
        self.assertLess(hidden_1.active_nodes(), hidden_1.output_nodes, msg='expect not all hidden nodes to be active')

    def test_prune_layer(self):
        # create layer and active in such a way that all but 1 output node is useless
        self.INPUT_NODES = 3
        self.OUTPUT_NODES = 1
        x = np.zeros((self.INPUT_NODES, self.OUTPUT_NODES), np.float32)
        for i in range(self.OUTPUT_NODES):
            x[0, i - 1] = 1.0
        y = np.zeros((self.OUTPUT_NODES, self.OUTPUT_NODES), np.float32)
        np.fill_diagonal(y, 1.)
        layer_1 = DuelStateReluLayer(InputLayer(self.INPUT_NODES), self.INPUT_NODES, session=self.session, weights=x,
                                     width_regularizer_constant=1e-2)
        layer_2 = Layer(layer_1, self.OUTPUT_NODES, weights=y, freeze=True)
        trainer = CategoricalTrainer(layer_2, 0.1)

        data_1 = [1.0] * self.INPUT_NODES
        data_2 = [0.0] * self.INPUT_NODES
        label_1 = [1.0] + [0.0] * (self.OUTPUT_NODES - 1)  # only the first node is correlated with the input
        label_2 = [0.0] * self.OUTPUT_NODES

        inputs = [data_1, data_2]
        labels = [label_1, label_2]

        for i in range(500):
            self.session.run([trainer._train],
                             feed_dict={layer_2.input_placeholder: inputs[:1],
                                        trainer._target_placeholder: labels[:1],
                                        trainer._learn_rate_placeholder: 0.05})
            self.session.run([trainer._train],
                             feed_dict={layer_2.input_placeholder: inputs[1:],
                                        trainer._target_placeholder: labels[1:],
                                        trainer._learn_rate_placeholder: 0.05})

        # layer should only have 1 active node
        self.assertGreater(layer_1.width()[0], DuelStateReluLayer.ACTIVE_THRESHOLD)
        self.assertEqual(layer_1.active_nodes(), 1)

        activation_pre_prune = self.session.run([layer_2.activation_predict],
                                                feed_dict={layer_1.input_placeholder: inputs})

        # after pruning layer should have 2 nodes
        layer_1.prune(inactive_nodes_to_leave=1)

        self.assertEqual(layer_1.output_nodes, 2)

        activation_post_prune = self.session.run([layer_2.activation_predict],
                                                 feed_dict={layer_1.input_placeholder: inputs})

        np.testing.assert_array_almost_equal(activation_pre_prune, activation_post_prune, decimal=2)

    def test_mnist(self):
        data = self.mnist_data
        input = InputLayer(self.MNIST_INPUT_NODES)
        d_1 = DuelStateReluLayer(input, 3, width_regularizer_constant=1e-7, width_binarizer_constant=1e-10,
                                 session=self.session)
        # when we add in batch norm layers we find that no active nodes are created, width is always less than 0.5?
        # bn_1 = BatchNormLayer(d_1)
        d_2 = DuelStateReluLayer(d_1, 3, width_regularizer_constant=1e-7, width_binarizer_constant=1e-10, )
        # bn_2 = BatchNormLayer(d_2)
        output = Layer(d_2, self.MNIST_OUTPUT_NODES)

        trainer = CategoricalTrainer(output, 0.1)
        end_epoch = data.train.epochs_completed + 20

        while data.train.epochs_completed <= end_epoch:
            train_x, train_y = data.train.next_batch(100)
            trainer.train(train_x, train_y)

        accuracy, cost = trainer.accuracy(data.test.images, data.test.labels)
        print(accuracy, cost)
        print("active nodes ", d_1.active_nodes())
        self.assertGreater(accuracy, 70.)

    def test_with_train_policy(self):
        data = self.mnist_data
        input = InputLayer(self.MNIST_INPUT_NODES)
        d_1 = DuelStateReluLayer(input, 1, width_regularizer_constant=1e-7, width_binarizer_constant=1e-9,
                                 session=self.session)
        bn_1 = BatchNormLayer(d_1)
        d_2 = DuelStateReluLayer(bn_1, 1, width_regularizer_constant=1e-7, width_binarizer_constant=1e-9, )
        bn_2 = BatchNormLayer(d_2)
        output = Layer(bn_2, self.MNIST_OUTPUT_NODES)
        trainer = CategoricalTrainer(output, 0.1)
        trainer = DuelStateReluTrainPolicy(trainer, data, 100, max_iterations=300, stop_accuracy=90.,
                                           grow_after_turns_without_improvement=1,)
        trainer.run_full(True)