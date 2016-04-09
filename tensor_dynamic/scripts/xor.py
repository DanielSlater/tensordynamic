import tensorflow as tf

from tensor_dynamic.data_functions import XOR_INPUTS, XOR_TARGETS
from tensor_dynamic.net import Net

train_x = XOR_INPUTS
train_y = XOR_TARGETS
max_iterations = 2000
batch_size = 1

with tf.Session() as session:
    net = Net(session, 2, 1)
    net = net.add_hidden_layer(session, 40)
    net = net.add_hidden_layer(session, 10)
    net = net.add_hidden_layer(session, 4)

    for i in range(max_iterations):
        #train_x, train_y = shuffle(train_x, train_y)
        loss = net.train(train_x, train_y, batch_size=batch_size)
        print(loss)