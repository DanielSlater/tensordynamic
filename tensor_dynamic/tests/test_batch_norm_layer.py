import numpy as np

from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.tests.base_layer_testcase import BaseLayerWrapper
from tensor_dynamic.tests.base_tf_testcase import BaseTfTestCase


class TestBatchNormLayer(BaseLayerWrapper.BaseLayerTestCase):
    def _create_layer_for_test(self):
        return BatchNormLayer(self._input_layer, self.session)

    def test_normalize(self):
        samples = 20
        input_nodes = 20
        input = InputLayer(input_nodes)
        batchLayer = BatchNormLayer(input, self.session)

        data = np.random.normal(200., 100., size=(samples, input_nodes))

        result = self.session.run(batchLayer.activation_train,
                                  feed_dict={batchLayer.input_placeholder:
                                                 data})

        self.assertAlmostEqual(result.mean(), 0., 3)
        self.assertAlmostEqual(result.var(), 1., 3)

    def test_predict_after_training(self):
        samples = 200
        input_nodes = 2
        input = InputLayer(input_nodes)
        batchLayer = BatchNormLayer(input, self.session)

        for i in range(200):
            data = np.random.normal(200., 10., size=(samples, input_nodes))
            self.session.run(batchLayer.activation_train,
                             feed_dict={batchLayer.input_placeholder:
                                            data})

        data2 = np.random.normal(-200., 10., size=(samples, input_nodes))

        result = self.session.run(batchLayer.activation_predict,
                                  feed_dict={batchLayer.input_placeholder:
                                                 data2})

        self.assertAlmostEqual(result.mean(), -40., delta=10.)
        self.assertAlmostEqual(result.var(), 1., delta=1.)

    def test_resize(self):
        # batch norm layer is resized based only on it's input layer
        input_nodes = 2
        input = InputLayer(input_nodes)
        layer = Layer(input, 2, self.session)
        batchLayer = BatchNormLayer(layer, self.session)

        RESIZE_NODES = 3
        layer.resize(RESIZE_NODES)

        self.assertEqual(batchLayer.output_nodes, RESIZE_NODES)

        self.session.run(batchLayer.activation_predict, feed_dict={batchLayer.input_placeholder: [np.ones(2)]})
