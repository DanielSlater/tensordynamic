import tensorflow as tf


class CategoricalTrainer(object):
    def __init__(self, net, learn_rate):
        """
        Sets up an optimizer and various helper methods against a network

        Parameters
        ----------
        net : tensorflow_dynamic.layers.BaseLayer
            Net that we will be training against
        learn_rate : float
        """
        self._net = net
        self._target_placeholder = tf.placeholder(tf.float32, shape=net.output_shape)
        self._learn_rate_placeholder = tf.placeholder("float", shape=[], name="learn_rate")
        self._cost = self._net.cost_all_layers_train(self._target_placeholder)
        self._prediction = tf.argmax(self._net.activation_predict, 1)
        self._correct_prediction = tf.equal(self._prediction, tf.argmax(self._target_placeholder, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, "float")) * tf.constant(100.0)
        optimizer = tf.train.GradientDescentOptimizer(self._learn_rate_placeholder)
        self._train = optimizer.minimize(self._cost)
        self.learn_rate = learn_rate

    @property
    def net(self):
        return self._net

    def predict(self, input_data):
        return self._net.session.run(self._prediction, feed_dict={self._net.input_placeholder: input_data})

    def accuracy(self, input_data, labels):
        return self._net.session.run([self._accuracy, self._cost],
                                     feed_dict={self._net.input_placeholder: input_data,
                                                self._target_placeholder: labels})

    def train(self, input_data, labels):
        _, cost = self._net.session.run([self._train, self._cost],
                                        feed_dict={self._net.input_placeholder: input_data,
                                                   self._target_placeholder: labels,
                                                   self._learn_rate_placeholder: self.learn_rate})
        return cost

    def back_losses_per_layer(self, input_data):
        """
        The loss per bactivating layer

        Parameters
        ----------
        input_data : np.array

        Returns
        -------
        {tensorflow_dynamic.layers.BaseLayer, float}

        Back loss per layer
        """
        back_losses = [(layer, layer.unsupervised_cost_predict()) for layer in self._net.all_connected_layers if
                       layer.bactivate]
        results = self._net.session.run([a[1] for a in back_losses],
                                        feed_dict={self._net.input_placeholder: input_data})
        return dict((a[0][0], a[1]) for a in zip(back_losses, results))

    def back_losses_per_layer_misclassified_only(self, input_data, labels):
        """
        The loss per bactivating layer only for examples that were misclassified

        Parameters
        ----------
        input_data : np.array
        labels : np.array

        Returns
        -------
        {tensorflow_dynamic.layers.BaseLayer, float}

        Back loss per layer
        """
        predictions = self._net.session.run(self._correct_prediction,
                                            feed_dict={self._net.input_placeholder: input_data,
                                                       self._target_placeholder: labels})

        misclassified = [item for item, prediction in zip(input_data, predictions) if prediction <= 0.0]

        back_losses = [(layer, layer.unsupervised_cost_predict()) for layer in self._net.all_connected_layers if
                       layer.bactivate]
        results = self._net.session.run([a[1] for a in back_losses],
                                        feed_dict={self._net.input_placeholder: misclassified})
        return dict((a[0][0], a[1]) for a in zip(back_losses, results))
