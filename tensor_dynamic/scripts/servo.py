import sys

import itertools

from tensor_dynamic.data.servo import get_data
from tensor_dynamic.data_functions import shuffle
from tensor_dynamic.net import Net
import tensorflow as tf
import numpy as np


train_x, train_y = get_data('../data/')
INPUT_DIM, OUTPUT_DIM = len(train_x[0]), len(train_y[0])
# train_x = tf.constant(train_x)
# train_y = tf.constant(train_y)
max_iterations = 5000
no_back_iterations = 0
batch_size = 5

def withGrowth():
    global train_x, train_y, max_iterations, batch_size
    with tf.Session() as session:
        net = Net(session, INPUT_DIM, OUTPUT_DIM)
        net = net.add_hidden_layer(session, 1, bactivate=True, non_liniarity=tf.nn.sigmoid)
        net = net.add_hidden_layer(session, 1, bactivate=True, non_liniarity=tf.nn.sigmoid)
        last_loss = 1000000000.0
        loss_counts = 0
        for i in range(max_iterations - no_back_iterations):
            train_x, train_y = shuffle(train_x, train_y)
            loss = net.train(train_x, train_y, batch_size=batch_size)
            print(i, loss)
            if loss > last_loss:
                if loss_counts > 9:
                    print "adding new nodes"
                    back_loss = net.get_reconstruction_error_per_hidden_layer(train_x, train_y)
                    print "Back loss %s" % (back_loss,)
                    layer_with_greatest_back_loss = back_loss.index(max(back_loss))
                    net = net.add_node_to_hidden_layer(session, layer_with_greatest_back_loss)
                    print net.hidden_nodes
                    last_loss = 10000000000.0
                    loss_counts = 0
                else:
                    last_loss = loss
                    loss_counts += 1
            else:
                last_loss = loss

        net.use_bactivate = False
        for j in range(no_back_iterations):
            i = j + max_iterations - no_back_iterations
            train_x, train_y = shuffle(train_x, train_y)
            loss = net.train(train_x, train_y, batch_size=batch_size)
            print(i, loss)


    print "final loss %s, %s" % (i, loss)
    print "nodes %s" % (net.hidden_nodes, )

def noGrowth(layers, bactivate=False):
    global train_x, train_y, max_iterations, batch_size
    with tf.Session() as session:
        net = Net(session, INPUT_DIM, OUTPUT_DIM)

        for l in layers:
            net = net.add_hidden_layer(session, l, bactivate=bactivate)

        for i in range(max_iterations):
            train_x, train_y = shuffle(train_x, train_y)
            loss = net.train(train_x, train_y, batch_size=batch_size)
            print(i, loss)

    print "final loss %s, %s" % (i, loss)
    return loss


MAX_LAYERS = 3
MAX_NODES_PER_LAYER = 14

noGrowth([10], bactivate=True)

# methods = [lambda x:noGrowth(x, bactivate=False), lambda x:noGrowth(x, bactivate=True)]
#
# with open('results.csv', 'a') as f:
#     for m in range(len(methods)):
#         for layers in range(1, MAX_LAYERS+1):
#             x = []
#             for arrangement in itertools.product(*[range(1, MAX_NODES_PER_LAYER+1)]*layers):
#                 res1 = methods[m](arrangement)
#
#                 f.write("%s, %s, %s\n" % (m, arrangement, res1))


# 10x10 no backivate = 0.18
# 10x10 backivate = 0.11
# growth 1x1 backivate = 0.11 fin 7 x 7, 0.12 fin 7 x 5, 0.14 fin 7 x 6