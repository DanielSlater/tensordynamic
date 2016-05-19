import tensorflow as tf

import tensor_dynamic.data.input_data as mnist
from tensor_dynamic.layers.input_layer import NoisyInputLayer
from tensor_dynamic.layers.ladder_layer import LadderLayer, LadderGammaLayer
from tensor_dynamic.layers.ladder_output_layer import LadderOutputLayer

num_labeled = 100
data = mnist.read_data_sets("../data/MNIST_data", n_labeled=num_labeled, one_hot=True)

NOISE_STD = 0.3
batch_size = 100
num_epochs = 1
num_examples = 60000
num_iter = (num_examples/batch_size) * num_epochs
learning_rate = 0.1
inputs = tf.placeholder(tf.float32, shape=(None, 784))
targets = tf.placeholder(tf.float32)

with tf.Session() as s:
    s.as_default()
    i = NoisyInputLayer(inputs, NOISE_STD, s)
    l1 = LadderLayer(i, 500, 1000.0, s)
    l2 = LadderGammaLayer(l1, 10, 10.0, s)
    ladder = LadderOutputLayer(l2, 0.1, s)
    l3 = ladder

    assert int(i.z.get_shape()[-1]) == 784
    assert int(l1.z_corrupted.get_shape()[-1]) == 500
    assert int(l2.z_corrupted.get_shape()[-1]) == 10

    assert int(l3.z_est.get_shape()[-1]) == 10
    assert int(l2.z_est.get_shape()[-1]) == 500
    assert int(l1.z_est.get_shape()[-1]) == 784

    assert int(l1.mean_corrupted_unlabeled.get_shape()[0]) == 500
    assert int(l2.mean_corrupted_unlabeled.get_shape()[0]) == 10

    loss = ladder.cost_all_layers_train(targets)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    pred_cost = -tf.reduce_mean(tf.reduce_sum(targets * tf.log(ladder.activation), 1))  # cost used for prediction

    correct_prediction = tf.equal(tf.argmax(ladder.activation, 1), tf.argmax(targets, 1))  # no of correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

    s.run(tf.initialize_all_variables())

    ladder.set_all_deterministic(True)

    print "acc", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})

    ladder.set_all_deterministic(False)

    for i in range(num_iter):
        images, labels = data.train.next_batch(batch_size)
        _, loss_val = s.run([train_step, loss], feed_dict={inputs: images, targets: labels})

        print(i, loss_val)

        # if i % 50 == 0:
        #     print "acc" + str(net.catagorical_accurasy(train_x, train_y))

        print "acc", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})

    ladder.set_all_deterministic(True)

    print "acc", s.run([accuracy], feed_dict={inputs: data.test.images, targets: data.test.labels})
