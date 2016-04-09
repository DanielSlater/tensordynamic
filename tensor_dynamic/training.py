import tensorflow as tf

from tensor_dynamic.data_functions import shuffle
from tensor_dynamic.ladder_layer import LadderLayer
from tensor_dynamic.net import Net


def train_no_growth(train_x, train_y, layers, bactivate=False, iterations=1000, batch_size=5, layer_class=None):
    with tf.Session() as session:
        net = Net(session, len(train_x[0]), len(train_y[0]), layer_class=layer_class)

        for l in layers:
            net = net.add_hidden_layer(session, l, bactivate=bactivate)

        for i in range(iterations):
            train_x, train_y = shuffle(train_x, train_y)
            loss = net.train(train_x, train_y, batch_size=batch_size)
            print(i, loss)

            if i % 50 == 0:
                print "acc" + str(net.catagorical_accurasy(train_x, train_y))

        acc = net.catagorical_accurasy(train_x, train_y)
    print "final loss %s, %s, %s" % (i, loss, acc)
    return loss, acc

def train_growth(train_x, train_y, layers, iterations=1000, no_back_iterations=0, batch_size=5):
    with tf.Session() as session:
        net = Net(session, len(train_x[0]), len(train_y[0]))
        for l in layers:
            net = net.add_hidden_layer(session, l, bactivate=True, non_liniarity=tf.nn.sigmoid)

        last_loss = 1000000000.0
        loss_counts = 0
        for i in range(iterations - no_back_iterations):
            train_x, train_y = shuffle(train_x, train_y)
            loss = net.train(train_x, train_y, batch_size=batch_size)
            print(i, loss)
            if loss > last_loss:
                if loss_counts > 8:
                    print "adding new nodes"
                    back_loss = net.get_reconstruction_error_per_hidden_layer(train_x, train_y)
                    print "Back loss %s" % (back_loss,)
                    layer_with_greatest_back_loss = back_loss.index(max(back_loss))
                    net = net.add_node_to_hidden_layer(session, layer_with_greatest_back_loss)
                    print net.hidden_nodes
                    last_loss = 10000000000.0
                    loss_counts = 0
                else:
                    loss_counts += 1
            else:
                last_loss = loss
                loss_counts = 0

        net.use_bactivate = False
        for j in range(no_back_iterations):
            i = j + iterations - no_back_iterations
            train_x, train_y = shuffle(train_x, train_y)
            loss = net.train(train_x, train_y, batch_size=batch_size)
            print(i, loss)


    acc = net.catagorical_accurasy(train_x, train_y)
    print "final loss %s, %s, %s" % (i, loss, acc)
    return loss, acc