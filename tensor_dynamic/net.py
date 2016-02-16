import tensorflow as tf
from tensor_dynamic.layer import Layer


class Net(object):
    def __init__(self, input_p, target_p):
        self.target_p = target_p
        self.input_p = input_p
        self.first_layer = None
        self.train_op = None
        self.cost_op = None
        self.accuracy_op = None

    def add_outer_layer(self, output_nodes, session=None):
        if not self.first_layer:
            self.first_layer = Layer(self.input_p, output_nodes, session)
        else:
            Layer(self.first_layer.get_layers_list()[-1], output_nodes, session)

    @property
    def output_layer(self):
        return self.first_layer.get_layers_list()[-1]

    @property
    def hidden_nodes(self):
        return [x.output_nodes for x in self.first_layer.get_layers_list()]

    @property
    def activation(self):
        if self.first_layer is None:
            return self.input_p
        return self.first_layer.get_output_layer_activation()

    @property
    def hidden_layers(self):
        return self.first_layer.get_layers_list()

    def add_node_to_layer(self, session, hidden_layer):
        if isinstance(hidden_layer, int):
            hidden_layer = self.hidden_layers[hidden_layer]

        new_net = Net(self.input_p, self.target_p)
        last = self.input_p
        for i, x in enumerate(self.hidden_layers):
            print i
            if x == hidden_layer:
                cloned = x.add_output(session)
            else:
                cloned = x.clone(session, last)
            if new_net.first_layer is None:
                new_net.first_layer = cloned
            last = cloned

        new_net.generate_methods()

        return new_net

    def generate_methods(self):
        forward_loss = tf.square(self.first_layer.get_output_layer_activation() - self.target_p)
        #back_loss = tf.add_n([tf.square(l.bactivation - l.input_layer) for l in dynamic_layer.get_layers_list() if l.bactivate])
        loss = forward_loss
        self.train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        self.cost_op = tf.reduce_mean(forward_loss)
        # mean_back_loss = tf.reduce_mean(back_loss)
        equals = tf.equal(tf.round(self.first_layer.get_output_layer_activation()), tf.round(self.target_p))
        self.accuracy_op = tf.reduce_mean(tf.cast(equals, "float"))

    def train(self, session, input, targets, batch_size=20):
        loss_result = 0.0
        for start, end in zip(range(0, len(input), batch_size), range(batch_size, len(input), batch_size)):
            _, loss_value = session.run([self.train_op, self.cost_op], feed_dict={self.input_p: input[start:end],
                                                                                  self.target_p: targets[start:end]})
            loss_result += loss_value
        return loss_result