import tensorflow as tf

import tensor_dynamic.data.input_data as mnist
from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.train_policy import TrainPolicy
from tensor_dynamic.categorical_trainer import CategoricalTrainer

batch_size = 100

data = mnist.read_data_sets("../data/MNIST_data", one_hot=True, validation_size=5000)

with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, shape=(None, 784))

    bactivate = True
    noise_std = 0.3
    non_lin = tf.sigmoid
    input_layer = InputLayer(inputs)
    bn1 = BatchNormLayer(input_layer, sess)
    net1 = Layer(bn1, 220, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1, noise_std=noise_std)
    bn2 = BatchNormLayer(net1, sess)
    net2 = Layer(bn2, 71, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1, noise_std=noise_std)
    bn3 = BatchNormLayer(net2, sess)
    net3 = Layer(bn3, 9, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1, noise_std=noise_std)
    bn4 = BatchNormLayer(net3, sess)
    net4 = Layer(bn4, 5, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1, noise_std=noise_std)
    bn5 = BatchNormLayer(net4, sess)
    net5 = Layer(bn5, 5, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1, noise_std=noise_std)
    bn6 = BatchNormLayer(net5, sess)
    net6 = Layer(bn6, 5, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1, noise_std=noise_std)
    bn7 = BatchNormLayer(net6, sess)
    outputNet = Layer(bn7, 10, sess, non_liniarity=tf.sigmoid, bactivate=False, supervised_cost=1.)

    trainer = CategoricalTrainer(outputNet, 0.15)
    trainPolicy = TrainPolicy(trainer, data, batch_size, 5,
                              grow_after_turns_without_improvement=5,
                              start_grow_epoch=20,
                              learn_rate_decay=0.99,
                              learn_rate_boost=0.01)

    trainPolicy.run_full()

    print trainer.accuracy(data.test.images, data.test.labels)
