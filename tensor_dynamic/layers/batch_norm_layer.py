import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer


class BatchNormLayer(BaseLayer):
    def __init__(self, input_layer,
                 session=None,
                 name='BatchNormLayer',
                 running_mean=None,
                 running_var=None,
                 ewma_running_mean=None,
                 ewma_running_var=None):
        super(BatchNormLayer, self).__init__(input_layer,
                                             input_layer.output_nodes,
                                             session,
                                             name=name)
        self._running_mean = self._create_variable(
            "running_mean",
            (BaseLayer.INPUT_BOUND_VALUE,),
            running_mean if running_mean is not None else tf.zeros((self.input_nodes,)))
        self._running_var = self._create_variable(
            "running_var",
            (BaseLayer.INPUT_BOUND_VALUE,),
            running_var if running_var is not None else tf.ones((self.input_nodes,)))

        self._ewma = tf.train.ExponentialMovingAverage(decay=.99)

        self._activation_train = self.update_batch_norm(ewma_running_var=ewma_running_var,
                                                        ewma_running_mean=ewma_running_mean)
        self._activation_predict = self.evaluate()

    @property
    def activation_predict(self):
        return self._activation_predict

    @property
    def activation_train(self):
        return self._activation_train

    def clone(self, session=None):
        return self.__class__(self.input_layer.clone(session or self._session),
                              session=session or self._session,
                              name=self._name)

    def update_batch_norm(self, ewma_running_mean=None, ewma_running_var=None):
        "batch normalize + update average mean and variance of layer"
        mean, var = tf.nn.moments(self._input_layer.activation_train, axes=[0])
        assign_mean = self._running_mean.assign(mean)
        assign_var = self._running_var.assign(var)
        # TODO: add this back in?
        assign_op = self._ewma.apply([self._running_mean, self._running_var])

        # need to init the variables the ewma creates
        self._session.run(tf.initialize_variables(self._ewma._averages.values()))

        ewma_running_mean_variable = self._ewma.average(self._running_mean)
        if ewma_running_mean is not None:
            assign = tf.assign(ewma_running_mean_variable, ewma_running_mean)
            self.session.run(assign)
        self._register_variable('ewma_running_mean', (self.OUTPUT_BOUND_VALUE,), ewma_running_mean_variable)

        ewma_running_var_variable = self._ewma.average(self._running_var)
        if ewma_running_var is not None:
            assign = tf.assign(ewma_running_var_variable, ewma_running_var)
            self.session.run(assign)
        self._register_variable('ewma_running_var', (self.OUTPUT_BOUND_VALUE,), ewma_running_var_variable)

        with tf.control_dependencies([assign_mean, assign_var, assign_op]):
            return self.batch_normalize(self._input_layer.activation_train, mean, var)

    def evaluate(self):
        mean = self._ewma.average(self._running_mean)
        var = self._ewma.average(self._running_var)
        return self.batch_normalize(self._input_layer.activation_predict, mean, var)

    @staticmethod
    def batch_normalize(batch, mean, var):
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    def resize(self, new_output_nodes=None):
        if new_output_nodes is None:
            super(BatchNormLayer, self).resize(new_output_nodes=self.input_layer.output_nodes)
        else:
            super(BatchNormLayer, self).resize(new_output_nodes=new_output_nodes)
