import logging
import sys
from math import log, exp

import tensor_dynamic.data.mnist_data as mnist
import tensorflow as tf

from tensor_dynamic.utils import train_till_convergence

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

checkpoint_path = "bayesian_neural_network_hidden_40"

restore = False

data = mnist.get_mnist_data_set_collection("../data/MNIST_data", one_hot=True)

input_nodes = 784
output_nodes = 10
hidden_nodes = 40
num_weights = input_nodes * hidden_nodes + hidden_nodes + hidden_nodes * output_nodes + output_nodes
alpha = 300. / num_weights
beta = 1.

input_placeholder = tf.placeholder(tf.float32, shape=(None, input_nodes))
target_placeholder = tf.placeholder(tf.float32, shape=(None, output_nodes))

weights_hidden = tf.Variable(tf.random_normal(shape=(input_nodes, hidden_nodes), mean=0., stddev=1. / input_nodes))
bias_hidden = tf.Variable(tf.zeros((hidden_nodes,)))

hidden_layer = tf.sigmoid(tf.matmul(input_placeholder, weights_hidden) + bias_hidden)

weights_output = tf.Variable(tf.random_normal(shape=(hidden_nodes, output_nodes), mean=0.,
                                              stddev=1. / hidden_nodes))
bias_output = tf.Variable(tf.zeros((output_nodes,)))

output_layer = tf.nn.softmax(tf.matmul(hidden_layer, weights_output) + bias_output)

cross_entropy = -tf.reduce_sum(
    tf.reduce_mean(target_placeholder * tf.log(output_layer) + (1. - target_placeholder) * tf.log(1. - output_layer),
                   1))
target_loss = beta * cross_entropy
weights_squared = tf.reduce_sum(tf.square(weights_hidden)) + tf.reduce_sum(tf.square(bias_hidden)) + \
                  tf.reduce_sum(tf.square(weights_output)) + tf.reduce_sum(tf.square(bias_output))
regularization_loss = weights_squared * .5 * alpha

loss_op = target_loss + regularization_loss

correct_op = tf.nn.in_top_k(output_layer, tf.argmax(target_placeholder, 1), 1)


def log_probability_of_targets_given_weights(network_prediction, data, targets):
    predictions = network_prediction(data)

    result = 0.
    for i in range(len(predictions)):
        result += log(sum(predictions[i] * targets[i]))

    return result


def log_prior_probability_of_weights(weights_squared, alpha, min_weight_squared, max_weight_squared):
    numerator = exp(-alpha * .5 * weights_squared)

    integral = lambda x: -2 * exp(-.5 * alpha * x) / alpha
    return log(numerator / (integral(max_weight_squared) - integral(min_weight_squared)))


with tf.Session() as session:
    train_op = tf.train.AdamOptimizer().minimize(loss_op)

    session.run(tf.initialize_all_variables())


    def train():
        error = 0.
        current_epoch = data.train.epochs_completed
        while data.train.epochs_completed == current_epoch:
            images, labels = data.train.next_batch(100)
            _, batch_error = session.run([train_op, loss_op],
                                         feed_dict={input_placeholder: images, target_placeholder: labels})
            error += batch_error
        return error


    saver = tf.train.Saver()

    if restore:
        saver.restore(session, checkpoint_path)
    else:
        final_error = train_till_convergence(train, log=True)
        saver.save(session, checkpoint_path)

    accuracy = session.run(tf.reduce_mean(tf.cast(correct_op, tf.float32)),
                           feed_dict={input_placeholder: data.train.features, target_placeholder: data.train.labels})

    print("accuracy = %s " % accuracy)

    ln_prob_weights = log_prior_probability_of_weights(session.run(weights_squared), alpha, 0., 16. * num_weights)

    ln_prob_labels_given_weights = log_probability_of_targets_given_weights(
        lambda x: session.run(output_layer, feed_dict={input_placeholder: x}), data.train.features, data.train.labels)

    # ln p(y) = sum(log(p(y_i))
    prob_of_labels = len(data.train.labels) * log(0.1)

    ln_prob_weights_given_labels_given_data = ln_prob_labels_given_weights + ln_prob_weights - prob_of_labels

    print(ln_prob_weights_given_labels_given_data, ln_prob_labels_given_weights, ln_prob_weights, prob_of_labels)