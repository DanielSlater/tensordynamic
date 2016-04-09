import tensorflow as tf

from tensor_dynamic.input_layer import InputLayer
from tensor_dynamic.layer import Layer
from tensor_dynamic.lazyprop import lazyprop, clear_all_lazyprops


class Net(object):
    def __init__(self, session, input_nodes, output_nodes):
        self._use_bactivate = True
        self._session = session
        self._input_p = tf.placeholder("float", (None, input_nodes), name="Input")
        self._target_p = tf.placeholder("float", (None, output_nodes), name="Target")
        self._first_layer = InputLayer(self._input_p)
        self._back_loss_list = None

    @property
    def output_nodes(self):
        return int(self._target_p.get_shape()[-1])

    @property
    def input_nodes(self):
        return int(self._input_p.get_shape()[-1])

    @lazyprop
    def loss_op(self):
        costs = [x.get_cost(self._target_p) for x in self.all_layers]
        return tf.add_n(costs)

    @lazyprop
    def train_op(self):
        return tf.train.GradientDescentOptimizer(0.05).minimize(self.loss_op)

    @lazyprop
    def cost_op(self):
        return tf.reduce_mean(self.loss_op)

    @lazyprop
    def accuracy_op(self):
        equals = tf.equal(tf.round(self.output_layer.activation), tf.round(self._target_p))
        return tf.reduce_mean(tf.cast(equals, "float"))

    @lazyprop
    def catagorical_accurasy_op(self):
        correct_predictions = tf.equal(tf.argmax(self.output_layer.activation, 1),
                                                 tf.argmax(self._target_p, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, "float")) * tf.constant(100.0)

    def catagorical_accurasy(self, test_x, test_y):
        return self._session.run([self.catagorical_accurasy_op],
                                 feed_dict={self._input_p: test_x,
                                            self._target_p: test_y})

    def add_hidden_layer(self, new_session, hidden_nodes, hidden_layer=-1, bactivate=False,
                         non_liniarity=tf.nn.sigmoid, layer_class=Layer):
        if isinstance(hidden_layer, Layer):
            hidden_layer = self.hidden_layers.index(hidden_layer)

        new_net = Net(new_session, self.input_nodes, self.output_nodes)
        last = new_net._first_layer
        for i, x in enumerate(self.hidden_layers):
            if i == hidden_layer:
                new_layer = layer_class(last, hidden_nodes, session=new_session, bactivate=bactivate,
                                  non_liniarity=non_liniarity)
                hidden_layer = None
                last = new_layer

            cloned = x.clone(last, self._session, new_session)
            last = cloned

        if hidden_layer == -1 or hidden_layer > i:
            layer_class(last, hidden_nodes, session=new_session, bactivate=bactivate,
                                  non_liniarity=non_liniarity)

        return new_net

    @property
    def output_layer(self):
        return self._first_layer.last_layer

    @property
    def hidden_nodes(self):
        return [x.output_nodes for x in self._first_layer.downstream_layers]

    @property
    def activation(self):
        if self._first_layer is None:
            return self._input_p
        return self._first_layer.output_layer_activation

    @property
    def hidden_layers(self):
        # TODO: make this less obtuse
        return list(self._first_layer.downstream_layers)

    def add_node_to_hidden_layer(self, new_session, hidden_layer):
        if isinstance(hidden_layer, int):
            hidden_layer = self.hidden_layers[hidden_layer]

        new_net = Net(new_session, self.input_nodes, self.output_nodes)
        last = new_net._input_p
        for x in self.hidden_layers:
            if x == hidden_layer:
                cloned = x.add_output(last, self._session, new_session)
            else:
                cloned = x.clone(last, self._session, new_session)
            if new_net._first_layer is None:
                new_net._first_layer = cloned
            last = cloned

        return new_net

    @property
    def all_layers(self):
        return self.hidden_layers + [self.output_layer]

    @property
    def use_bactivate(self):
        return self._use_bactivate

    @use_bactivate.setter
    def use_bactivate(self, value):
        self._use_bactivate = not self._use_bactivate
        clear_all_lazyprops(self)

    def train(self, inputs, targets, batch_size=20):
        loss_result = 0.0
        for start, end in zip(range(0, len(inputs), batch_size), range(batch_size, len(inputs), batch_size)):
            _, loss_value = self._session.run([self.train_op, self.cost_op],
                                              feed_dict={self._input_p: inputs[start:end],
                                                         self._target_p: targets[
                                                                         start:end]})

            loss_result += loss_value
        return loss_result

    def get_reconstruction_error_per_hidden_layer(self, inputs, targets):
        total_back_loss = [tf.reduce_mean(l) for l in self._back_loss_list]

        back_losses = self._session.run(total_back_loss, feed_dict={self._input_p: inputs,
                                                                    self._target_p: targets})

        return back_losses
