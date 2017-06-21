import tensorflow as tf

import tensor_dynamic.data.mnist_data as mnist
from tensor_dynamic.layers.back_weight_layer import BackWeightLayer
from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.train_policy import TrainPolicy
from tensor_dynamic.categorical_trainer import CategoricalTrainer

batch_size = 100

data = mnist.get_mnist_data_set_collection("../data/MNIST_data", one_hot=True, validation_size=5000)

with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, shape=(None, 784))

    bactivate = True
    noise_std = 0.3
    beta = 0.5
    gamma = 0.5
    non_lin = tf.nn.sigmoid
    input_layer = InputLayer(inputs)
    bn1 = BatchNormLayer(input_layer, sess, beta=beta, gamma=gamma)
    net1 = Layer(bn1, 1, sess, non_liniarity=non_lin, bactivate=bactivate, unsupervised_cost=.001, noise_std=noise_std)
    bn2 = BatchNormLayer(net1, sess, beta=beta, gamma=gamma)
    net2 = Layer(bn2, 1, sess, non_liniarity=non_lin, bactivate=bactivate, unsupervised_cost=.001, noise_std=noise_std)
    bn3 = BatchNormLayer(net2, sess, beta=beta, gamma=gamma)
    net3 = Layer(bn3, 1, sess, non_liniarity=non_lin, bactivate=bactivate, unsupervised_cost=.001, noise_std=noise_std)
    bn4 = BatchNormLayer(net3, sess, beta=beta, gamma=gamma)
    net4 = Layer(bn4, 1, sess, non_liniarity=non_lin, bactivate=bactivate, unsupervised_cost=.001, noise_std=noise_std)
    bn5 = BatchNormLayer(net4, sess, beta=beta, gamma=gamma)
    outputNet = Layer(bn5, 10, sess, non_liniarity=tf.sigmoid, bactivate=False, supervised_cost=1.)

    trainer = CategoricalTrainer(outputNet, 0.15)
    trainPolicy = TrainPolicy(trainer, data, batch_size, max_iterations=3000,
                              grow_after_turns_without_improvement=2,
                              start_grow_epoch=1,
                              learn_rate_decay=0.99,
                              learn_rate_boost=0.01,
                              back_loss_on_misclassified_only=True)

    trainPolicy.run_full()

    print trainer.accuracy(data.test.features, data.test.labels)
