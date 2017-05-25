import tensorflow as tf

from tensor_dynamic.layers.base_layer import BaseLayer


class BatchNormLayer(BaseLayer):
    def __init__(self, input_layer,
                 session=None,
                 name='BatchNormLayer',
                 running_mean=None,
                 running_var=None,
                 ewma_running_mean=None,
                 ewma_running_var=None,
                 beta=None,
                 gamma=None):
        super(BatchNormLayer, self).__init__(input_layer,
                                             input_layer.output_nodes,
                                             session,
                                             name=name)
        # self._running_mean = self._create_variable(
        #     "running_mean",
        #     (BaseLayer.INPUT_BOUND_VALUE,),
        #     running_mean if running_mean is not None else tf.zeros((self.input_nodes,)))
        # self._running_var = self._create_variable(
        #     "running_var",
        #     (BaseLayer.INPUT_BOUND_VALUE,),
        #     running_var if running_var is not None else tf.ones((self.input_nodes,)))
        #
        # self._ewma = tf.train.ExponentialMovingAverage(decay=.99)
        #
        # self._batch_norm_beta = self._create_variable("beta", (self.INPUT_BOUND_VALUE,), tf.zeros((self.input_nodes,)),
        #                                               is_kwarg=False, is_trainable=False)
        # self._batch_norm_gamma = self._create_variable("gamma", (self.INPUT_BOUND_VALUE,),
        #                                                tf.ones((self.input_nodes,)),
        #                                                is_kwarg=False,
        #                                                is_trainable=False)

        # self._activation_train = self._update_batch_norm(ewma_running_var=ewma_running_var,
        #                                                 ewma_running_mean=ewma_running_mean)
        # self._activation_predict = self._evaluate()

        self._beta = beta
        self._gamma = gamma

        mean, var = tf.nn.moments(self._input_layer.activation_train, axes=[0])
        self._register_variable("mean", (self.INPUT_BOUND_VALUE,), mean)
        self._register_variable("var", (self.INPUT_BOUND_VALUE,), var)

        self._activation_train = self._batch_normalize(self.input_layer.activation_train, mean, var)
        self._activation_predict = self._batch_normalize(self.input_layer.activation_predict, mean,
                                                         var)
        #self._register_variable()

    @property
    def activation_predict(self):
        return self._activation_predict

    # @property
    # def assign_op(self):
    #     return self._assign_op

    @property
    def activation_train(self):
        return self._activation_train

    def clone(self, session=None):
        return self.__class__(self.input_layer.clone(session or self._session),
                              session=session or self._session,
                              name=self._name)

    def _update_batch_norm(self, ewma_running_mean=None, ewma_running_var=None):
        "batch normalize + update average mean and variance of layer"
        mean, var = tf.nn.moments(self._input_layer.activation_train, axes=[0])
        assign_mean = self._running_mean.assign(mean)
        assign_var = self._running_var.assign(var)
        self._assign_op = self._ewma.apply([self._running_mean, self._running_var])

        # need to init the variables the ewma creates
        self._session.run(tf.initialize_variables(self._ewma._averages.values()))

        ewma_running_mean_variable = self._ewma.average(self._running_mean)
        if ewma_running_mean is not None:
            self.session.run(tf.assign(ewma_running_mean_variable, ewma_running_mean))

        self._register_variable('ewma_running_mean', (self.OUTPUT_BOUND_VALUE,), ewma_running_mean_variable)

        ewma_running_var_variable = self._ewma.average(self._running_var)
        if ewma_running_var is not None:
            self.session.run(tf.assign(ewma_running_var_variable, ewma_running_var))

        self._register_variable('ewma_running_var', (self.OUTPUT_BOUND_VALUE,), ewma_running_var_variable)

        with tf.control_dependencies([assign_mean, assign_var]):
            return self._batch_normalize_no_resize(self._input_layer.activation_train, mean, var)

    def _evaluate(self):
        mean = self._ewma.average(self._running_mean)
        var = self._ewma.average(self._running_var)
        return self._batch_normalize_no_resize(self._input_layer.activation_predict, mean, var)

    def _batch_normalize_no_resize(self, batch, mean, var):
        # the tf.nn.batch_norm_with_global_normalization only supports convolutional networks so for now we have to
        # reshape to a convolution normalize then shape back... but can't be resized then... :(

        reshape_to_conv = tf.reshape(batch, [-1, 1, 1, self.input_nodes], name="reshape_to_conv")
        self._r1 = reshape_to_conv
        self._register_variable("reshape_to_conv", (-1, 1, 1, self.INPUT_BOUND_VALUE), reshape_to_conv, is_constructor_variable=False)
        batch_normalized = tf.nn.batch_norm_with_global_normalization(reshape_to_conv, mean, var,
                                                                      self._batch_norm_beta,
                                                                      self._batch_norm_gamma,
                                                                      0.00001,
                                                                      False)
        reshape_from_conv = tf.reshape(batch_normalized, [-1, self.input_nodes], name="reshape_from_conv")
        self._r2 = reshape_from_conv
        self._register_variable("reshape_from_conv", (-1, self.INPUT_BOUND_VALUE), reshape_from_conv, is_constructor_variable=False)

        return reshape_from_conv

    def _batch_normalize(self, batch, mean, var):
        # this version doesn't produce correct numbers when running predict after training
        normalized = ((batch - mean) / tf.sqrt(var + tf.constant(1e-10)))
        if self._gamma:
            normalized = normalized * self._gamma
        if self._beta:
            normalized = normalized + self._beta
        return normalized

    def resize(self, new_output_nodes=None, input_nodes_to_prune=None, output_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None,
               split_nodes_noise_std=None):
        new_output_nodes = new_output_nodes or self.input_layer.output_nodes

        super(BatchNormLayer, self).resize(new_output_nodes=new_output_nodes,
                                           input_nodes_to_prune=input_nodes_to_prune,
                                           output_nodes_to_prune=output_nodes_to_prune,
                                           split_output_nodes=split_output_nodes,
                                           split_input_nodes=split_input_nodes,
                                           split_nodes_noise_std=split_nodes_noise_std)

    @property
    def kwargs(self):
        kwargs = super(BatchNormLayer, self).kwargs
        kwargs['beta'] = self._beta
        kwargs['gamma'] = self._gamma

        return kwargs
