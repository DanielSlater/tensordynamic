# from unittest import TestCase
# import tensorflow as tf
# import numpy as np
#
# from tensor_dynamic.data_functions import XOR_INPUTS, XOR_TARGETS
# from tensor_dynamic.ladder_layer import LadderLayer
# from tensor_dynamic.layer import Layer
# from tensor_dynamic.net import Net
#
#
# class TestNet(TestCase):
#     INPUT_NODES = 3
#     OUTPUT_NODES = 2
#
#     def setUp(self):
#         self.session = tf.Session()
#         self.session.__enter__()
#         self.net = Net(self.session, self.INPUT_NODES, self.OUTPUT_NODES)
#
#     def tearDown(self):
#         self.session.__exit__(None, None, None)
#
#     def test_hidden_nodes(self):
#         self.net = self.net.add_hidden_layer(self.session, 5, -1)
#
#         self.assertEquals(self.net.hidden_nodes, [5])
#
#     def test_add_outer_layer(self):
#         self.net = self.net.add_hidden_layer(self.session, 5, -1)
#         self.net = self.net.add_hidden_layer(self.session, 3, -1)
#
#         self.assertEquals(self.net.hidden_nodes, [5, 3])
#
#     def test_add_node_to_layer(self):
#         self.net = self.net.add_hidden_layer(self.session, 3, -1)
#         self.net = self.net.add_hidden_layer(self.session, 2, -1)
#
#         new_net = self.net.add_node_to_hidden_layer(self.session, 0)
#
#         self.assertEquals(new_net.hidden_nodes, [4, 2])
#
#     def test_output_layer(self):
#         self.net = self.net.add_hidden_layer(self.session, 1, -1)
#         self.assertEquals(self.net.hidden_nodes, [1])
#         self.assertEquals(self.net.next_layer.output_nodes, self.OUTPUT_NODES)
#
#         new_net = self.net.add_node_to_hidden_layer(self.session, 0)
#         self.assertEquals(new_net.next_layer.output_nodes, self.OUTPUT_NODES)
#
#     def test_train_xor(self):
#         train_x = XOR_INPUTS
#         train_y = XOR_TARGETS
#         net = Net(self.session, 2, 1)
#
#         loss_1 = net.train(train_x, train_y, batch_size=1)
#         loss_2 = net.train(train_x, train_y, batch_size=1)
#
#         self.assertGreater(loss_1, loss_2, msg="Expected loss to reduce")
#
#     def test_train_xor_multi_layer(self):
#         train_x = XOR_INPUTS
#         train_y = XOR_TARGETS
#         net = Net(self.session, 2, 1).add_hidden_layer(self.session, 2, -1)
#
#         loss_1 = net.train(train_x, train_y, batch_size=1)
#         loss_2 = net.train(train_x, train_y, batch_size=1)
#
#         self.assertGreater(loss_1, loss_2, msg="Expected loss to reduce")
#
#     def test_bactivate(self):
#         train_x = XOR_INPUTS
#         train_y = XOR_TARGETS
#         net = Net(self.session, 2, 1).add_hidden_layer(self.session, 3, bactivate=True)
#
#         loss_1 = net.train(train_x, train_y, batch_size=1)
#         loss_2 = net.train(train_x, train_y, batch_size=1)
#
#         self.assertGreater(loss_1, loss_2, msg="Expected loss to reduce")
#
#     def test_bactivate_multi_layer(self):
#         train_x = XOR_INPUTS
#         train_y = XOR_TARGETS
#         net = Net(self.session, 2, 1)\
#             .add_hidden_layer(self.session, 2, bactivate=True)\
#             .add_hidden_layer(self.session, 1, bactivate=True)
#
#         loss_1 = net.train(train_x, train_y, batch_size=1)
#         loss_2 = net.train(train_x, train_y, batch_size=1)
#
#         self.assertGreater(loss_1, loss_2, msg="Expected loss to reduce")
#
#     def test_backtivate_add_hidden_nodes(self):
#         train_x = XOR_INPUTS
#         train_y = XOR_TARGETS
#
#         net = Net(self.session, 2, 1)
#         net = net.add_hidden_layer(self.session, 1, bactivate=True)
#         net = net.add_hidden_layer(self.session, 1, bactivate=True)
#
#         net.train(train_x, train_y, batch_size=1)
#
#         net = net.add_node_to_hidden_layer(self.session, 0)
#
#         loss_1 = net.train(train_x, train_y, batch_size=1)
#         loss_2 = net.train(train_x, train_y, batch_size=1)
#
#         self.assertGreater(loss_1, loss_2, msg="Expected loss to reduce")
#
#     def test_ladder_layer_vs_layer(self):
#         net_layer = Net(self.session, 100, 10, layer_class=Layer)
#         net_layer = net_layer.add_hidden_layer(self.session, 80)
#         net_layer = net_layer.add_hidden_layer(self.session, 50)
#
#         net_ladder_layer = Net(self.session, 100, 10, layer_class=LadderLayer)
#         net_ladder_layer = net_ladder_layer.add_hidden_layer(self.session, 80)
#         net_ladder_layer = net_ladder_layer.add_hidden_layer(self.session, 50)
#
#         self.assertEqual(net_layer.next_layer.output_nodes, net_ladder_layer.next_layer.output_nodes)
#         self.assertSequenceEqual(net_layer.hidden_nodes, net_ladder_layer.hidden_nodes)