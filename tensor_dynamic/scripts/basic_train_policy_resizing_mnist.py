import os

import tensorflow as tf

import tensor_dynamic.data.mnist_data as mnist
from tensor_dynamic.layers.batch_norm_layer import BatchNormLayer
from tensor_dynamic.layers.input_layer import InputLayer
from tensor_dynamic.layers.layer import Layer
from tensor_dynamic.train_policy import TrainPolicy
from tensor_dynamic.categorical_trainer import CategoricalTrainer

# load data
batch_size = 100
initail_learning_rate = 0.15
resize_learning_rate = 0.05
minimal_model_training_epochs = 50
learn_rate_decay = 0.96
hidden_layers = [200, 100, 50, 10]
checkpoint_path = 'resizeing_results'
SAVE = True

data = mnist.get_mnist_data_set_collection("../data/MNIST_data", one_hot=True, validation_size=5000)


def create_network(sess, hidden_layers):
    inputs = tf.placeholder(tf.float32, shape=(None, 784))
    bactivate = True
    noise_std = 0.3
    non_lin = tf.nn.relu
    input_layer = InputLayer(inputs)
    last = BatchNormLayer(input_layer, sess)
    for hidden_nodes in hidden_layers:
        last = Layer(last, hidden_nodes, sess, bactivate=bactivate, non_liniarity=non_lin, unsupervised_cost=.1,
                     noise_std=noise_std)
        last = BatchNormLayer(last, sess)

    outputs = Layer(last, 10, sess, non_liniarity=tf.sigmoid, bactivate=False, supervised_cost=1.)

    trainer = CategoricalTrainer(outputs, initail_learning_rate)

    return outputs, trainer


with tf.Session() as sess:
    net, trainer = create_network(sess, hidden_layers)

    # train minimal model on mnist/load checkpoints
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    saver = tf.train.Saver()
    checkpoints = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoints:
        saver.restore(sess, checkpoints.model_checkpoint_path)
        print("Loaded checkpoints %s" % checkpoints.model_checkpoint_path)
    else:
        print("retraining network")
        tp = TrainPolicy(trainer, data, batch_size, learn_rate_decay=learn_rate_decay)
        tp.train_till_convergence()

        if SAVE:
            saver.save(sess, checkpoint_path + "/network")

    # get train error
    print("train error ", trainer.accuracy(data.validation.features, data.validation.labels))

    # get reconstruction errors
    print trainer.back_losses_per_layer(data.train.features)

    # get error just on miss-classifications
    print trainer.back_losses_per_layer(data.train.features, misclassification_only=True, labels=data.train.labels)

    results = {}

    # try each different resize, see how it does
    for x in range(len(hidden_layers)):
        print("resizing layer ", x)
        cloned = net.clone()
        hidden_layers = [layer for layer in cloned.all_connected_layers if type(layer) == Layer]
        hidden_layers[x].resize()  # add 1 node
        new_trainer = CategoricalTrainer(net, resize_learning_rate)
        new_tp = TrainPolicy(new_trainer, data, batch_size, learn_rate_decay=learn_rate_decay)
        new_tp.train_till_convergence()
        acc, cost = trainer.accuracy(data.validation.features, data.validation.labels)
        print("train error ", acc, cost)
        results[x] = (acc, cost)

print results