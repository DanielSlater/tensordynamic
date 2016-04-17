import tensorflow as tf
import numpy as np

from tensor_dynamic.batch_norm_layer import BatchNormLayer
from tensor_dynamic.input_layer import InputLayer
from tensor_dynamic.tests.base_tf_testcase import BaseTfTestCase


class TestBatchNormLayer(BaseTfTestCase):
    def test_normalize(self):
        samples = 20
        input_nodes = 20
        input = InputLayer(input_nodes)
        batchLayer = BatchNormLayer(input, self.session)

        data = np.random.normal(200., 100., size=(samples, input_nodes))

        result = self.session.run(batchLayer.activation,
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
            self.session.run(batchLayer.activation,
                                      feed_dict={batchLayer.input_placeholder:
                                          data})

        batchLayer.predict = True

        data2 = np.random.normal(-200., 10., size=(samples, input_nodes))

        result = self.session.run(batchLayer.activation,
                                  feed_dict={batchLayer.input_placeholder:
                                      data2})

        self.assertAlmostEqual(result.mean(), -40., delta=10.)
        self.assertAlmostEqual(result.var(), 1., delta=1.)