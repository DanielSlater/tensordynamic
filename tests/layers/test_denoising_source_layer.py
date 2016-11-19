import tensorflow as tf

from tensor_dynamic.layers.back_weight_candidate_layer import BackWeightCandidateLayer
from tensor_dynamic.layers.denoising_source_layer import DenoisingSourceLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tests import BaseLayerWrapper


class TestBackWeightCandidateLayer(BaseLayerWrapper.BaseLayerTestCase):
    def _create_layer_for_test(self):
        return DenoisingSourceLayer(self._input_layer, self.OUTPUT_NODES, session=self.session)

    def test_more_nodes_improves_reconstruction_loss_mnist(self):
        data = self.mnist_data.train.images
        recon_1 = self.reconstruction_loss_for(1, data)
        recon_2 = self.reconstruction_loss_for(2, data)
        recon_5 = self.reconstruction_loss_for(5, data)
        recon_20 = self.reconstruction_loss_for(20, data)
        recon_100 = self.reconstruction_loss_for(100, data)
        self.assertLess(recon_2, recon_1)
        self.assertLess(recon_5, recon_2)
        self.assertLess(recon_20, recon_5)
        self.assertLess(recon_100, recon_20)

    def test_more_nodes_improves_reconstruction_loss_gauss(self):
        data = self.data_sum_of_gaussians(5, 40, 500)
        recon_1 = self.reconstruction_loss_for(1, data)
        recon_2 = self.reconstruction_loss_for(2, data)
        recon_5 = self.reconstruction_loss_for(5, data)
        recon_20 = self.reconstruction_loss_for(20, data)
        recon_100 = self.reconstruction_loss_for(100, data)
        self.assertLess(recon_2, recon_1)
        self.assertLess(recon_5, recon_2)
        self.assertLess(recon_20, recon_5)
        self.assertLess(recon_100, recon_20)

    def reconstruction_loss_for(self, output_nodes, data):
        bw_layer1 = BackWeightCandidateLayer(InputLayer(len(data[0])), output_nodes, non_liniarity=tf.nn.sigmoid,
                                             session=self.session,
                                             noise_std=0.3)

        cost = bw_layer1.unsupervised_cost_train()
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

        self.session.run(tf.initialize_all_variables())

        for i in range(5):
            for j in range(0, len(data) - 100, 100):
                self.session.run(optimizer, feed_dict={bw_layer1.input_placeholder: data[j:j + 100]})

        result = self.session.run(bw_layer1.unsupervised_cost_predict(),
                                  feed_dict={bw_layer1.input_placeholder: data})
        print("denoising with %s hidden layer had cost %s" % (output_nodes, result))
        return result
