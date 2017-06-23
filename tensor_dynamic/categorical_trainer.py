import itertools
import tensorflow as tf

# I think this can be depricated
class CategoricalTrainer(object):
    def __init__(self, net, learn_rate):
        """Sets up an optimizer and various helper methods against a network

        Args:
            net (tensorflow_dynamic.layers.BaseLayer): Net that we will be training against
            learn_rate (float):
        """
        self._net = net
        self._target_placeholder = tf.placeholder(tf.float32, shape=net.output_shape)
        self._learn_rate_placeholder = tf.placeholder("float", shape=[], name="learn_rate")
        self._cost = self._net.cost_all_layers_train(self._target_placeholder)
        self._prediction = tf.argmax(self._net.activation_predict, 1)
        self._correct_prediction = tf.equal(self._prediction, tf.argmax(self._target_placeholder, 1))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, "float")) * tf.constant(100.0)
        optimizer = tf.train.GradientDescentOptimizer(self._learn_rate_placeholder).minimize(self._cost)

        # temp = set(tf.all_variables())
        # optimizer = tf.train.AdamOptimizer()
        # self.net.session.run(tf.initialize_variables(set(tf.all_variables()) - temp))

        assigns = [x.assign_op for x in net.all_layers if x.assign_op is not None]
        if assigns:
            assigns_group = tf.group(*assigns)
            with tf.control_dependencies([optimizer]):
                optimizer = tf.group(assigns_group)

        self._train = optimizer
        self.learn_rate = learn_rate

        # adam_optimizer_variables = itertools.chain(*[x.values() for x in optimizer._slots.values()])
        # self.net.session.run(tf.initialize_variables(adam_optimizer_variables))
        # self.net.session.run(tf.initialize_variables([optimizer._beta1_t, optimizer._beta2_t, optimizer._epsilon_t,
        #                                               optimizer._lr_t]))

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

    def train_one_epoch(self, data_set, batch_size):
        start_epoch = data_set.train.epochs_completed
        cost = 0.
        while start_epoch == data_set.train.epochs_completed:
            train_x, train_y = data_set.train.next_batch(batch_size)
            cost += self.train(train_x, train_y)

        return cost

    def back_losses_per_layer(self, input_data, misclassification_only=False, labels=None):
        """The loss per bactivating layer

        Args:
            input_data (np.array):
            misclassification_only (bool): If True back loss is only checked on data that has been misclassified
            labels (np.array): Labels for the input data, required if using misclassification_only

        Returns:
            {tensorflow_dynamic.layers.BaseLayer, float}
        """
        back_losses = [(layer, layer.unsupervised_cost_predict()) for layer in self._net.all_connected_layers if
                       layer.bactivate]

        if not back_losses:
            return None

        if misclassification_only:
            if not labels:
                raise Exception('must supply labels when using misclassification_only')

            predictions = self._net.session.run(self._correct_prediction,
                                                feed_dict={self._net.input_placeholder: input_data,
                                                           self._target_placeholder: labels})

            input_data = [item for item, prediction in zip(input_data, predictions) if prediction <= 0.0]

        results = self._net.session.run([a[1] for a in back_losses],
                                        feed_dict={self._net.input_placeholder: input_data})
        return dict((a[0][0], a[1] / a[0][0].input_nodes) for a in zip(back_losses, results))
