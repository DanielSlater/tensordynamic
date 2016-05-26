import tensorflow as tf


def squared_loss(x, y):
    return tf.reduce_mean(tf.square(x - y))


def cross_entropy_loss(x, y):
    return -tf.reduce_mean((x * tf.log(y) + (1 - x) * tf.log(1 - y)))
