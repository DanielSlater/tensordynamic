import numpy as np
import tensorflow as tf

from tensor_dynamic.layers.back_weight_layer import BackWeightLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tests.layers.base_layer_testcase import BaseLayerWrapper


class TestBackWeightLayer(BaseLayerWrapper.BaseLayerTestCase):
    def _create_layer_for_test(self):
        return BackWeightLayer(self._input_layer, self.OUTPUT_NODES, session=self.session)

    def test_more_nodes_improves_reconstruction_loss(self):
        recon_1 = self.reconstruction_loss_for(1)
        recon_2 = self.reconstruction_loss_for(2)
        recon_5 = self.reconstruction_loss_for(5)
        recon_20 = self.reconstruction_loss_for(20)
        self.assertLess(recon_2, recon_1)
        self.assertLess(recon_5, recon_2)
        self.assertLess(recon_20, recon_5)

    def reconstruction_loss_for(self, output_nodes):
        data = self.mnist_data
        bw_layer1 = BackWeightLayer(InputLayer(784), output_nodes, non_liniarity=tf.nn.sigmoid, session=self.session,
                                    noise_std=0.3)

        cost = bw_layer1.unsupervised_cost_train()
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        self.session.run(tf.initialize_all_variables())

        end_epoch = data.train.epochs_completed + 5

        while data.train.epochs_completed <= end_epoch:
            train_x, train_y = data.train.next_batch(100)
            self.session.run(optimizer, feed_dict={bw_layer1.input_placeholder: train_x})

        result = self.session.run(bw_layer1.unsupervised_cost_predict(),
                                  feed_dict={bw_layer1.input_placeholder: data.train.images})
        print("denoising with %s hidden nodes had cost %s" % (output_nodes, result))
        return result

    def test_reconstruction_of_single_input(self):
        input_layer = InputLayer(1)
        layer = BackWeightLayer(input_layer, 1, non_liniarity=tf.nn.sigmoid, session=self.session, noise_std=0.3)

        cost = layer.unsupervised_cost_train()
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        self.session.run(tf.initialize_all_variables())

        data = np.random.normal(0.5, 0.5, size=[200, 1])

        for x in range(100):
            self.session.run([optimizer], feed_dict={input_layer.input_placeholder: data})

        result = self.session.run([cost], feed_dict={input_layer.input_placeholder: data})
        print result