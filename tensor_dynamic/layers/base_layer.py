import functools
import logging
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
import operator

import sys
import tensorflow as tf

from tensor_dynamic.lazyprop import clear_all_lazyprops, lazyprop, clear_lazyprop_on_lazyprop_cleared, has_lazyprop
from tensor_dynamic.utils import tf_resize, bias_init, weight_init
from tensor_dynamic.weight_functions import noise_weight_extender, array_extend

logger = logging.getLogger(__name__)


class BaseLayer(object):
    __metaclass__ = ABCMeta

    GROWTH_MULTIPLYER = 1.1
    SHRINK_MULTIPLYER = 1. / GROWTH_MULTIPLYER
    MINIMUM_GROW_AMOUNT = 3

    OUTPUT_BOUND_VALUE = 'output'
    INPUT_BOUND_VALUE = 'input'
    INPUT_DIM_3_BOUND_VALUE = 'input_3'
    OUTPUT_DIM_3_BOUND_VALUE = 'output_3'

    _BoundVariable = namedtuple('_BoundVariable', ['name', 'dimensions', 'variable', 'is_kwarg'])

    def __init__(self,
                 input_layer,
                 output_nodes,
                 session=None,
                 weight_extender_func=None,
                 weight_initializer_func=None,
                 bias_initializer_func=None,
                 layer_noise_std=None,
                 drop_out_prob=None,
                 batch_normalize_input=False,
                 batch_norm_transform=None,
                 batch_norm_scale=None,
                 name=None,
                 freeze=False):
        """Base class from which all layers will inherit. This is an abstract class

        Args:
            weight_initializer_func ((int)->weights): function that creates initial values for weights for this layer
            bias_initializer_func (int->weights): function that creates initial values for weights for this layer
            layer_noise_std (float): If not None gaussian noise with mean 0 and this std is applied to the input of this
                layer
            input_layer (tensor_dynamic.base_layer.BaseLayer): This layer will work on the activation of the input_layer
            output_nodes (int | tuple of ints): Number of output nodes for this layer, can be a tuple of multi dimensional output, e.g. convolutional network
            session (tensorflow.Session): The session within which all these variables should be created
            weight_extender_func (func): Method that extends the size of matrix or vectors
            name (str): Used for identifying the layer and when initializing tensorflow variables
            freeze (bool):If True then weights in this layer are not trainable
        """
        if not isinstance(input_layer, BaseLayer):
            raise TypeError("input_layer must be of type %s" % BaseLayer)

        assert isinstance(output_nodes, (int, tuple))
        assert isinstance(input_layer, BaseLayer)
        self._bound_variable_assign_data = {}
        self._input_layer = input_layer
        self._layer_noise_std = layer_noise_std
        self._drop_out_prob = drop_out_prob
        self._batch_normalize_input = batch_normalize_input
        self._name = name
        self._output_nodes = (output_nodes,) if type(output_nodes) == int else output_nodes
        self._input_nodes = self._input_layer._output_nodes
        self._next_layer = None

        self._session = self._get_property_or_default(session, '_session',
                                                      None)
        self._weight_extender_func = self._get_property_or_default(weight_extender_func, '_weight_extender_func',
                                                                   noise_weight_extender)

        self._weight_initializer_func = self._get_property_or_default(weight_initializer_func,
                                                                      '_weight_initializer_func',
                                                                      weight_init)
        self._bias_initializer_func = self._get_property_or_default(bias_initializer_func,
                                                                    '_bias_initializer_func',
                                                                    bias_init)
        self._freeze = freeze
        self._bound_variables = {}
        input_layer._attach_next_layer(self)

        if self._batch_normalize_input:
            self._batch_norm_mean_train, self._batch_norm_var_train = (None, None)
            self._batch_norm_mean_predict, self._batch_norm_var_predict = (None, None)

            with self.name_scope():
                self._batch_norm_scale = self._create_variable("batch_norm_scale", (self.INPUT_BOUND_VALUE,),
                                                               batch_norm_scale if batch_norm_scale is not None else tf.ones(
                                                                   self.input_nodes), is_kwarg=True)
                self._batch_norm_transform = self._create_variable("batch_norm_transform", (self.INPUT_BOUND_VALUE,),
                                                                   batch_norm_transform if batch_norm_transform is not None else tf.zeros(
                                                                       self.input_nodes), is_kwarg=True)
                self._normalized_train = None
                self._normalized_predict = None

    def _get_property_or_default(self, init_value, property_name, default_value):
        if init_value is not None:
            return init_value
        if self.input_layer is not None:
            if hasattr(self.input_layer, property_name) and getattr(self.input_layer, property_name) is not None:
                return getattr(self.input_layer, property_name)
            else:
                earlier_in_stream_result = self.input_layer._get_property_or_default(init_value, property_name,
                                                                                     default_value)
                if earlier_in_stream_result is not None:
                    return earlier_in_stream_result

        return default_value

    def name_scope(self, is_train=False, is_predict=False):
        """Used for naming variables associated with this layer in TensorFlow in a consistent way

        Format = "{layer_number}_{layer_name}"

        Examples:
            with self.name_scope():
                my_new_variable = tf.Variable(default_val, name="name")

        Args:
            is_train (bool): Set for parts of tensorflow graph just for training
            is_predict (bool): Set for parts of tensorflow graph just for predicting

        Returns:
            A context manager that installs `name` as a new name scope in the
            default graph.
        """
        name = str(self.layer_number) + "_" + self._name

        if is_train and not is_predict:
            name += "_train"
        elif is_predict:
            name += "_predict"

        return tf.name_scope(name)

    @lazyprop
    def activation_train(self):
        """The activation used for training this layer, this will often be the same as prediction except with dropout or
        random noise applied.

        Returns:
            tensorflow.Tensor
        """
        clear_lazyprop_on_lazyprop_cleared(self, 'activation_train', self.input_layer)
        input_tensor = self.input_layer.activation_train

        with self.name_scope(is_train=True):
            input_tensor = self._process_input_activation_train(input_tensor)

            return self._layer_activation(input_tensor, True)

    def _process_input_activation_train(self, input_tensor):
        if self._batch_normalize_input:
            self._batch_norm_mean_train, self._batch_norm_var_train = tf.nn.moments(self._input_layer.activation_train,
                                                                                    axes=range(len(self.input_nodes)))
            self._normalized_train = (
                (input_tensor - self._batch_norm_mean_train) / tf.sqrt(self._batch_norm_var_train + tf.constant(1e-10)))
            input_tensor = (self._normalized_train + self._batch_norm_transform) * self._batch_norm_scale

        if self._drop_out_prob:
            input_tensor = tf.nn.dropout(input_tensor, self._drop_out_prob)

        if self._layer_noise_std is not None:
            input_tensor = input_tensor + tf.random_normal(tf.shape(self.input_layer.activation_train),
                                                           stddev=self._layer_noise_std)

        return input_tensor

    @lazyprop
    def activation_predict(self):
        """The activation used for predictions from this layer, this will often be the same as training except without
        dropout or random noise applied.

        Returns:
            tensorflow.Tensor
        """
        clear_lazyprop_on_lazyprop_cleared(self, 'activation_predict', self.input_layer)
        input_tensor = self.input_layer.activation_predict

        with self.name_scope(is_predict=True):
            input_tensor = self._process_input_activation_predict(input_tensor)
            return self._layer_activation(input_tensor, False)

    def _process_input_activation_predict(self, input_tensor):
        if self._batch_normalize_input:
            self._batch_norm_mean_predict, self._batch_norm_var_predict = tf.nn.moments(
                self._input_layer.activation_predict,
                axes=range(len(self.input_nodes)))

            # TODO: Note this is the WRONG way to apply this, will result in bad results for prediction sizes
            # that do not equal the batch_size...
            self._normalized_predict = (
                (input_tensor - self._batch_norm_mean_predict) / tf.sqrt(
                    self._batch_norm_var_predict + tf.constant(1e-10)))
            input_tensor = (self._normalized_predict + self._batch_norm_transform) * self._batch_norm_scale
        return input_tensor

    @abstractmethod
    def _layer_activation(self, input_tensor, is_train):
        """The activation for this layer

        Args:
            input_tensor (tensorflow.Tensor):
            is_train (bool): If true this is activation for training, if false for prediction

        Returns:
            tensorflow.Tensor
        """
        raise NotImplementedError()

    @property
    def bactivate(self):
        """All layers have output activation, some unsupervised layer activate backwards as well.

        e.g. Layers in ladder networks, Denoising Autoencoders

        Returns:
            bool
        """
        return False

    @property
    def is_input_layer(self):
        """Are we the input layer

        Returns:
            bool
        """
        return False

    @property
    def is_output_layer(self):
        """Is this the final layer of the network

        Returns:
            bool
        """
        return self._next_layer is None

    @property
    def output_nodes(self):
        """The number of output nodes

        Returns:
            tuple of ints
        """
        return self._output_nodes

    @property
    def input_nodes(self):
        """The number of input nodes to this layer

        Returns:
            tuple of ints
        """
        return self._input_nodes

    @property
    def session(self):
        """Session used to create the variables in this layer

        Returns:
            tensorflow.Session
        """
        return self._session

    @lazyprop
    def bactivation_train(self):
        """The activation used for training this layer, this will often be the same as prediction except with dropout or
        random noise applied.

        Returns:
            tensorflow.Tensor
        """
        clear_lazyprop_on_lazyprop_cleared(self, 'bactivation_train', self, 'activation_train')
        return self._layer_bactivation(self.activation_train, True)

    @lazyprop
    def bactivation_predict(self):
        """The activation used for predictions from this layer, this will often be the same as training except without
        dropout or random noise applied.

        Returns:
            tensorflow.Tensor
        """
        clear_lazyprop_on_lazyprop_cleared(self, 'bactivation_predict', self, 'activation_predict')
        return self._layer_bactivation(self.activation_predict, False)

    def _layer_bactivation(self, input_tensor, is_train):
        """The bactivation for this layer

        Args:
            input_tensor (tensorflow.Tensor):
            is_train (bool): If true this is activation for training, if false for prediction

        Returns:
            tensorflow.Tensor
        """
        raise NotImplementedError()

    @property
    def has_bactivation(self):
        return False

    @property
    def output_shape(self):
        return self.activation_predict.get_shape().as_list()

    @property
    def input_shape(self):
        return self._input_layer.output_shape

    @property
    def next_layer(self):
        return self._next_layer

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def has_next_layer(self):
        return self.next_layer

    def _attach_next_layer(self, layer):
        if self.has_next_layer:
            raise Exception("Can not attach_next_layer to Layer: %s which already has a next layer" % self._name)
        if not isinstance(layer, BaseLayer):
            raise TypeError("Attached layer must be of type %s" % BaseLayer)

        self._next_layer = layer

    @property
    def last_layer(self):
        if self._next_layer is not None:
            return self._next_layer.last_layer
        return self

    @property
    def first_layer(self):
        return self.input_layer.first_layer

    @property
    def input_placeholder(self):
        return self.first_layer.input_placeholder

    @property
    def target_placeholder(self):
        return self.last_layer.target_placeholder

    @property
    def downstream_layers(self):
        if self._next_layer:
            yield self._next_layer
            for d in self._next_layer.downstream_layers:
                yield d

    @property
    def upstream_layers(self):
        if self._input_layer:
            yield self._input_layer
            for d in self._input_layer.upstream_layers:
                yield d

    @property
    def all_layers(self):
        return self.all_connected_layers

    @property
    def all_connected_layers(self):
        for u in list(reversed(list(self.upstream_layers))):
            yield u
        yield self
        for d in self.downstream_layers:
            yield d

    def activate_predict(self, data_set):
        """Get the prediction activation of this network given the data_set as input

        Args:
            data_set (np.array): np.array or Array matching the dimensions of the input placeholder

        Returns:
            np.array: prediction activation of the network
        """
        return self.session.run(self.activation_predict, feed_dict={self.input_placeholder: data_set})

    def supervised_cost_train(self, targets):
        return None

    def unsupervised_cost_train(self):
        return None

    def cost_train(self, targets):
        supervised_cost = self.supervised_cost_train(targets)
        unsupervised_cost = self.unsupervised_cost_train()

        if supervised_cost is not None:
            if unsupervised_cost is not None:
                return supervised_cost + unsupervised_cost
            return supervised_cost
        if unsupervised_cost is not None:
            return unsupervised_cost

        return None

    def cost_all_layers_train(self, targets):
        all_costs = [x.cost_train(targets) for x in self.all_connected_layers]
        all_costs = filter(lambda v: v is not None, all_costs)
        return tf.add_n(all_costs, name="cost_all_layers")

    @property
    def layer_number(self):
        return len(list(self.upstream_layers))

    @property
    def kwargs(self):
        kwargs = {
            'weight_extender_func': self._weight_extender_func,
            'layer_noise_std': self._layer_noise_std,
            'drop_out_prob': self._drop_out_prob,
            'batch_normalize_input': self._batch_normalize_input,
            'freeze': self._freeze,
            'name': self._name}
        kwargs.update(self._bound_variables_as_kwargs())
        return kwargs

    def _bound_variables_as_kwargs(self):
        kwarg_dict = {}
        for name, bound_variable in self._bound_variables.iteritems():
            if bound_variable.is_kwarg:
                kwarg_dict[name] = self.session.run(bound_variable.variable)

        return kwarg_dict

    def clone(self, session=None):
        """Produce a clone of this layer AND all connected upstream layers

        Args:
            session (tensorflow.Session): If passed in the clone will be created with all variables initialised in this session
                                          If None then the current session of this layer is used

        Returns:
            tensorflow_dynamic.BaseLayer: A copy of this layer and all upstream layers
        """
        new_self = self.__class__(self.input_layer.clone(session or self.session),
                                  self.output_nodes,
                                  session=session or self._session,
                                  **self.kwargs)

        return new_self

    def resize_needed(self):
        """ If there is a mismatch between the input size of this layer and the output size of it's previous layer will
        return True

        Returns:
            bool
        """
        if self._input_layer.output_nodes != self.input_nodes:
            return True
        return False

    def resize(self, new_output_nodes=None,
               output_nodes_to_prune=None,
               input_nodes_to_prune=None,
               split_output_nodes=None,
               split_input_nodes=None,
               data_set_train=None,
               data_set_validation=None,
               no_splitting_or_pruning=False,
               split_nodes_noise_std=.1):
        """Resize this layer by changing the number of output nodes. Will also resize any downstream layers

        Args:
            data_set_validation (DataSet):Data set used for validating this network
            data_set_train (DataSet): Data set used for training this network
            no_splitting_or_pruning (bool): If set to true then noise is just added randomly rather than splitting nodes
            new_output_nodes (int | tuple of ints): If passed we change the number of output nodes of this layer to be new_output_nodes
            output_nodes_to_prune ([int]): list of indexes of the output nodes we want pruned e.g. [1, 3] would remove
                the 1st and 3rd output node from this layer
            input_nodes_to_prune ([int]): list of indexes of the input nodes we want pruned e.g. [1, 3] would remove the
                1st and 3rd input node from this layer
            split_output_nodes ([int]): list of indexes of nodes to split. This is for growing the layer
            split_input_nodes: ([int]): list of indexes of nodes that where split in the previous layer.
            split_nodes_noise_std (float): standard deviation of noise to add when splitting a node
        """
        if isinstance(new_output_nodes, tuple):
            new_output_nodes = new_output_nodes[self.get_resizable_dimension()]
        elif new_output_nodes is not None and not isinstance(new_output_nodes, int):
            raise ValueError("new_output_nodes must be tuple of int %s" % (new_output_nodes,))

        if not no_splitting_or_pruning:
            # choose nodes to split or prune
            if new_output_nodes is not None:
                if output_nodes_to_prune is None and split_output_nodes is None:
                    if new_output_nodes < self.get_resizable_dimension_size():
                        output_nodes_to_prune = self._choose_nodes_to_prune(new_output_nodes, data_set_train,
                                                                            data_set_validation)
                    elif new_output_nodes > self.get_resizable_dimension_size():
                        split_output_nodes = self._choose_nodes_to_split(new_output_nodes, data_set_train,
                                                                         data_set_validation)
            elif self.has_resizable_dimension():
                new_output_nodes = self.get_resizable_dimension_size()
                if output_nodes_to_prune:
                    new_output_nodes -= len(output_nodes_to_prune)
                if split_output_nodes:
                    new_output_nodes += len(split_output_nodes)

        new_input_nodes = self.input_layer.output_nodes
        input_nodes_changed = new_input_nodes != self._input_nodes

        if self.has_resizable_dimension():
            output_nodes_changed = new_output_nodes != self.get_resizable_dimension_size()
            temp_output_nodes = list(self._output_nodes)
            temp_output_nodes[self.get_resizable_dimension()] = new_output_nodes

            self._output_nodes = tuple(temp_output_nodes)
        else:
            output_nodes_changed = False

        self._input_nodes = new_input_nodes

        for name, bound_variable in self._bound_variables.iteritems():
            if input_nodes_changed and self._bound_dimensions_contains_input(bound_variable.dimensions) or \
                            output_nodes_changed and self._bound_dimensions_contains_output(bound_variable.dimensions):

                self._forget_assign_op(name)

                int_dims = self._bound_dimensions_to_ints(bound_variable.dimensions)

                if isinstance(bound_variable.variable, tf.Variable):
                    old_values = self._session.run(bound_variable.variable)
                    if output_nodes_to_prune or split_output_nodes:
                        output_bound_axis = bound_variable.dimensions.index(self.OUTPUT_BOUND_VALUE)
                        if output_nodes_to_prune:
                            old_values = np.delete(old_values, output_nodes_to_prune, output_bound_axis)
                        else:  # split
                            old_values = array_extend(old_values, {output_bound_axis: split_output_nodes},
                                                      noise_std=split_nodes_noise_std)
                    if input_nodes_to_prune or split_input_nodes:
                        input_bound_axis = bound_variable.dimensions.index(self.INPUT_BOUND_VALUE)
                        if input_nodes_to_prune:
                            old_values = np.delete(old_values, input_nodes_to_prune, input_bound_axis)
                        else:  # split
                            old_values = array_extend(old_values, {input_bound_axis: split_input_nodes},
                                                      halve_extended_vectors=True)
                    if no_splitting_or_pruning:
                        new_values = self._weight_extender_func(old_values, int_dims)
                    else:
                        new_values = old_values

                    tf_resize(self._session, bound_variable.variable, int_dims,
                              new_values, self._get_assign_function(name))
                else:
                    # this is a tensor, not a variable so has no weights
                    tf_resize(self._session, bound_variable.variable, int_dims)

        if input_nodes_changed and self._batch_normalize_input:
            tf_resize(self._session, self._batch_norm_mean_train, self._input_nodes)
            tf_resize(self._session, self._batch_norm_var_train, self._input_nodes)
            tf_resize(self._session, self._batch_norm_mean_predict, self._input_nodes)
            tf_resize(self._session, self._batch_norm_var_predict, self._input_nodes)
            if self._normalized_train is not None:
                tf_resize(self._session, self._normalized_train, (None,) + self._input_nodes)
            if self._normalized_predict is not None:
                tf_resize(self._session, self._normalized_predict, (None,) + self._input_nodes)

            # This line fixed the issue, this is all very hacky...
            # self._mat_mul.op.inputs[0]._shape = TensorShape((None,) + self._input_nodes)
            from tensorflow.python.framework.tensor_shape import TensorShape

            if '_mat_mul_is_train_equal_' + str(True) in self.__dict__:
                self.__dict__['_mat_mul_is_train_equal_' + str(True)].op.inputs[0]._shape = TensorShape(
                    (None,) + self._input_nodes)
                self.__dict__['_mat_mul_is_train_equal_' + str(True)].op.inputs[0].op.inputs[0]._shape = TensorShape(
                    (None,) + self._input_nodes)
                self.__dict__['_mat_mul_is_train_equal_' + str(True)].op.inputs[0].op.inputs[0].op.inputs[
                    0]._shape = TensorShape((None,) + self._input_nodes)
                self.__dict__['_mat_mul_is_train_equal_' + str(True)].op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[
                    0]._shape = TensorShape((None,) + self._input_nodes)
                # tf_resize(self._session, self.__dict__['_mat_mul_is_train_equal_' + str(True)], (None,) + self._input_nodes)
            if '_mat_mul_is_train_equal_' + str(False) in self.__dict__:
                self.__dict__['_mat_mul_is_train_equal_' + str(False)].op.inputs[0]._shape = TensorShape(
                    (None,) + self._input_nodes)
                self.__dict__['_mat_mul_is_train_equal_' + str(False)].op.inputs[0].op.inputs[0]._shape = TensorShape(
                    (None,) + self._input_nodes)
                self.__dict__['_mat_mul_is_train_equal_' + str(False)].op.inputs[0].op.inputs[0].op.inputs[
                    0]._shape = TensorShape((None,) + self._input_nodes)
                self.__dict__['_mat_mul_is_train_equal_' + str(False)].op.inputs[0].op.inputs[0].op.inputs[0].op.inputs[
                    0]._shape = TensorShape((None,) + self._input_nodes)
                # tf_resize(self._session, self.__dict__['_mat_mul_is_train_equal_' + str(False)], (None,) + self._input_nodes)

        if output_nodes_changed:
            if has_lazyprop(self, 'activation_predict'):
                tf_resize(self._session, self.activation_predict, (None,) + self._output_nodes)
            if has_lazyprop(self, 'activation_train'):
                tf_resize(self._session, self.activation_train, (None,) + self._output_nodes)

        if input_nodes_changed and self.bactivate:
            if has_lazyprop(self, 'bactivation_train'):
                tf_resize(self._session, self.bactivation_train, (None,) + self._input_nodes)
            if has_lazyprop(self, 'bactivation_predict'):
                tf_resize(self._session, self.bactivation_predict, (None,) + self._input_nodes)

        if self._next_layer and self._next_layer.resize_needed():
            self._next_layer.resize(input_nodes_to_prune=output_nodes_to_prune, split_input_nodes=split_output_nodes,
                                    no_splitting_or_pruning=no_splitting_or_pruning)

    def _forget_assign_op(self, name):
        if name in self._bound_variable_assign_data:
            del self._bound_variable_assign_data[name]

    def _bound_dimensions_to_ints(self, bound_dims):
        int_dims = ()
        for x in bound_dims:
            if isinstance(x, int):
                if x == -1:
                    int_dims += (None,)
                else:
                    int_dims += (x,)
            elif x == self.OUTPUT_BOUND_VALUE:
                int_dims += (self._output_nodes[0],)
            elif x == self.INPUT_BOUND_VALUE:
                int_dims += (self._input_nodes[0],)
            elif x == self.INPUT_DIM_3_BOUND_VALUE:
                assert len(self._input_nodes) == 3, "must have 3 input dimensions"
                int_dims += (self._input_nodes[2],)
            elif x == self.OUTPUT_DIM_3_BOUND_VALUE:
                assert len(self._input_nodes) == 3, "must have 3 output dimensions"
                int_dims += (self._output_nodes[2],)
            elif x is None:
                int_dims += (None,)
            else:
                raise Exception("bound dimension must be either int or 'input' or 'output' or None found %s" % (x,))
        return int_dims

    def _create_variable(self, name, bound_dimensions, default_val, is_kwarg=True, is_trainable=True):
        int_dims = self._bound_dimensions_to_ints(bound_dimensions)

        with self.name_scope():
            if isinstance(default_val, np.ndarray):
                default_val = self._weight_extender_func(default_val, int_dims)
            elif default_val is None:
                if len(int_dims) == 1:
                    default_val = self._bias_initializer_func(int_dims[0])
                else:
                    default_val = self._weight_initializer_func(int_dims)

            var = tf.Variable(default_val, trainable=(not self._freeze) and is_trainable, name=name)

            self._session.run(tf.initialize_variables([var]))
            self._bound_variables[name] = self._BoundVariable(name, bound_dimensions, var, is_kwarg)
            return var

    def _register_tensor(self, name, bound_dimensions, variable, is_constructor_variable=True):
        """Register a variable that will need to be resized with this layer

        Args:
            name (str): Name used for displaying errors and debuging
            bound_dimensions (tuple of (self.OUTPUT_BOUND_VALUE or self.INPUT_BOUND_VALUE or int)):
            variable (tf.Tensor): The variable to bind
            is_constructor_variable (bool): If true this variable is passed as an arg to the constructor of this class if it is cloned
        """
        int_dims = self._bound_dimensions_to_ints(bound_dimensions)
        assert tuple(variable.get_shape().as_list()) == tuple(int_dims)
        self._bound_variables[name] = self._BoundVariable(name, bound_dimensions, variable, is_constructor_variable)

    def _bound_dimensions_contains_input(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.INPUT_BOUND_VALUE or x == self.INPUT_DIM_3_BOUND_VALUE)

    def _bound_dimensions_contains_output(self, bound_dimensions):
        return any(x for x in bound_dimensions if x == self.OUTPUT_BOUND_VALUE or x == self.OUTPUT_DIM_3_BOUND_VALUE)

    def _get_assign_function(self, name):
        bound_variable = self._bound_variables[name]

        if name not in self._bound_variable_assign_data:
            with self.name_scope():
                placeholder = tf.placeholder(bound_variable.variable.dtype.base_dtype,
                                             shape=self._bound_dimensions_to_ints(bound_variable.dimensions))
                assign_op = tf.assign(bound_variable.variable, placeholder, validate_shape=False)
                self._bound_variable_assign_data[name] = (assign_op, placeholder)

        assign_op, placeholder = self._bound_variable_assign_data[name]

        return lambda value: self.session.run(assign_op, feed_dict={placeholder: value})

    def remove_layer_from_network(self):
        """Attempt to remove this layer from the network, may resize the input layer of the next, when
        removing
        """
        input_layer = self.input_layer
        next_layer = self.next_layer
        if next_layer.input_nodes != input_layer.output_nodes:
            # need to resize layer so there is a match when we cut
            self.resize(input_layer.output_nodes,
                        no_splitting_or_pruning=True)

        self.detach_output()
        input_layer.detach_output()

        input_layer._next_layer = next_layer
        next_layer._input_layer = input_layer

    def detach_output(self):
        """Detaches the connect between this layer and the next layer

        Returns:
            BaseLayer : The next layer, now detached from this layer
        """
        if self._next_layer is None:
            raise ValueError("Cannot detach_output if there is no next layer")
        next_layer = self._next_layer

        next_layer._input_layer = None
        clear_all_lazyprops(next_layer)

        self._next_layer = None
        clear_all_lazyprops(self)

        return next_layer

    def _get_deeper_net_kwargs(self):
        raise NotImplemented()

    def add_intermediate_cloned_layer(self):
        """Add a layer after the current one, that is an exact clone of this layer, but with net 2 deeper net weight
        initlialization"""
        kwargs = self._get_deeper_net_kwargs()
        if self._batch_normalize_input:
            kwargs['batch_norm_transform'] = np.zeros(shape=kwargs['batch_norm_transform'].shape,
                                                      dtype=kwargs['batch_norm_transform'].dtype)
            kwargs['batch_norm_scale'] = np.ones(shape=kwargs['batch_norm_scale'].shape,
                                                 dtype=kwargs['batch_norm_scale'].dtype)

        self.add_intermediate_layer(lambda x: self.__class__(self, self.output_nodes, session=self.session, **kwargs))

    def add_intermediate_layer(self, layer_creation_func, *args, **kwargs):
        """Adds a layer to the network between this layer and the next one.

        Args:
            layer_creation_func (BaseLayer->BaseLayer): Method that creates the intermediate layer, takes this layer as a
                parameter. Any args or kwargs get passed in after passing in this layer
        """
        old_next_layer = self.detach_output()
        new_next_layer = layer_creation_func(self, *args, **kwargs)

        new_next_layer._next_layer = old_next_layer
        old_next_layer._input_layer = new_next_layer
        print("SFASF")

    @property
    def assign_op(self):
        """Optional tensor flow op that will be set as a dependency of the train step. Useful for things like clamping
        variables, or setting mean/var in batch normalization layer

        Returns:
            tensorflow.Operation or None
        """
        return None

    @property
    def variables(self):
        """Get all the tensorflow variables used in this layer, useful for weight regularization

        Returns:
            Iterable of tf.Variable:
        """
        for bound_variable in self._bound_variables.values():
            yield bound_variable.variable

    @property
    def regularizable_variables(self):
        """variables that can be regularized on this layer"""
        raise NotImplementedError()

    @property
    def variables_all_layers(self):
        """Get all the tensorflow variables used in all connected layers, useful for weight regularization

        Returns:
            Iterable of tf.Variable:
        """
        for layer in self.all_layers:
            for variable in layer.variables:
                yield variable

    @property
    def regularizable_variables_all_layers(self):
        """variables that can be regularized on all connected layers"""
        for layer in self.all_layers:
            for variable in layer.regularizable_variables:
                yield variable

    def get_parameters_all_layers(self):
        """The number of parameters in this layer

        Returns:
            int
        """
        total = 0

        for layer in self.all_layers:
            total += layer.get_parameters()

        return total

    def get_parameters(self):
        """The number of parameters in this layer

        Returns:
            int
        """
        total = 0

        for bound_variable in self._bound_variables.values():
            total += int(functools.reduce(operator.mul, bound_variable.variable.get_shape()))

        return total

    def has_resizable_dimension(self):
        """True if this layer can be resized, otherwise false

        Returns:
            bool
        """
        return False

    def get_resizable_dimension_size(self):
        """Get the size of the dimension that is resized by the resize method.
        In the future may support multiple of these for conv layers

        Returns:
            int
        """
        return None

    def get_all_resizable_layers(self):
        """Yields all layers connected to this one that are resiable, orders them by how close
        to the input layer they are

        Returns:
            Generator of BaseLayer
        """
        for layer in self.all_connected_layers:
            if layer.has_resizable_dimension():
                yield layer

    def get_resizable_dimension(self):
        return 0

    def get_resizable_dimension_size_all_layers(self):
        """

        Returns:
            (int,): Tuple of each layer size for each layer that is resizable in the network
        """
        return tuple(layer.get_resizable_dimension_size() for layer in self.get_all_resizable_layers())

    def _get_new_node_count(self, size_multiplier, from_size=None):
        if not self.has_resizable_dimension():
            raise Exception("Can not resize this dimension")

        from_size = from_size or self.get_resizable_dimension_size()

        new_size = int(from_size * size_multiplier)
        # in case the multiplier is too small to changes values
        if abs(new_size - from_size) < self.MINIMUM_GROW_AMOUNT:
            if size_multiplier > 1.:
                new_size = from_size + self.MINIMUM_GROW_AMOUNT
            else:
                new_size = from_size - self.MINIMUM_GROW_AMOUNT

        return new_size

    def _layer_resize_converge(self, data_set_train, data_set_validation,
                               model_evaluation_function,
                               new_size,
                               learning_rate):
        if new_size <= 0:
            logger.info("layer too small stopping downsize")
            return -sys.float_info.max

        self.resize(new_output_nodes=new_size,
                    data_set_train=data_set_train,
                    data_set_validation=data_set_validation)

        self.last_layer.train_till_convergence(data_set_train, data_set_validation,
                                               learning_rate=learning_rate)
        result = model_evaluation_function(self, data_set_validation)
        logger.info("layer resize converge for dim: %s result: %s", self.get_resizable_dimension_size_all_layers(),
                    result)
        return result

    def find_best_size(self, data_set_train, data_set_validation,
                       model_evaluation_function, best_score=None,
                       initial_learning_rate=0.001, tuning_learning_rate=0.0001):
        """Attempts to resize this layer to minimize the loss against the validation dataset by resizing this layer

        Args:
            data_set_train (tensor_dynamic.data.data_set.DataSet):
            data_set_validation (tensor_dynamic.data.data_set.DataSet):
            model_evaluation_function (BaseLayer, tensor_dynamic.data.data_set.DataSet -> float): Method for judging
                success of training. We try to maximize this
            best_score (float): Best score achieved so far, this is purly for optimization. If it is not passed this is
                calculated in the method
            initial_learning_rate (float): Learning rate to use for first run
            tuning_learning_rate (float): Learning rate to use for subsequent runs, normally smaller than
                initial_learning_rate

        Returns:
            (bool, float) : if we resized, the best score we achieved from the evaluation function
        """
        if not self.has_resizable_dimension():
            raise Exception("Can not resize unresizable layer %s" % (self,))

        if best_score is None:
            self.last_layer.train_till_convergence(data_set_train, data_set_validation,
                                                   learning_rate=initial_learning_rate)
            best_score = model_evaluation_function(self, data_set_validation)

        start_size = self.get_resizable_dimension_size_all_layers()
        best_state = self.get_network_state()

        resized = False

        # try bigger
        new_score = self._layer_resize_converge(data_set_train, data_set_validation,
                                                model_evaluation_function,
                                                self._get_new_node_count(self.GROWTH_MULTIPLYER),
                                                tuning_learning_rate)

        # keep getting bigger until we stop improving
        while new_score > best_score:
            resized = True
            best_score = new_score
            best_state = self.get_network_state()

            new_score = self._layer_resize_converge(data_set_train, data_set_validation,
                                                    model_evaluation_function,
                                                    self._get_new_node_count(self.GROWTH_MULTIPLYER),
                                                    tuning_learning_rate)
        if not resized:
            logger.info("From start_size %s Bigger failed, trying smaller", start_size)
            self.set_network_state(best_state)

            new_score = self._layer_resize_converge(data_set_train, data_set_validation,
                                                    model_evaluation_function,
                                                    self._get_new_node_count(self.SHRINK_MULTIPLYER),
                                                    tuning_learning_rate)

            while new_score > best_score:
                resized = True
                best_score = new_score
                best_state = self.get_network_state()
                new_score = self._layer_resize_converge(data_set_train, data_set_validation,
                                                        model_evaluation_function,
                                                        self._get_new_node_count(self.SHRINK_MULTIPLYER),
                                                        tuning_learning_rate)

        # return to the best size we found
        self.set_network_state(best_state)

        logger.info("From start_size %s Found best was %s", start_size, self.get_resizable_dimension_size())

        return resized, best_score

    def _choose_nodes_to_split(self, desired_size, data_set_train, data_set_validation):
        assert isinstance(desired_size, int)

        current_size = self.get_resizable_dimension_size()

        if desired_size <= current_size:
            raise ValueError("Can't split to get smaller than we are")

        importance = self._get_node_importance(data_set_train, data_set_validation)

        to_split = set()

        while desired_size > current_size + len(to_split):
            max_node = np.argmax(importance)
            importance[max_node] = -sys.float_info.max
            to_split.add(max_node)

        return list(to_split)

    def _choose_nodes_to_prune(self, desired_size, data_set_train, data_set_validation):
        assert isinstance(desired_size, int)

        current_size = self.get_resizable_dimension_size()

        if desired_size >= current_size:
            raise ValueError("Can't prune to size larger than we are")

        importance = self._get_node_importance(data_set_train, data_set_validation)

        to_prune = set()

        while desired_size < current_size - len(to_prune):
            min_node = np.argmin(importance)
            importance[min_node] = sys.float_info.max
            to_prune.add(min_node)

        return list(to_prune)

    def _get_layer_state(self):
        """Returns an object that can be used to set this layer to it's current state and size

        Returns:
            object
        """
        return self.__class__, self.get_resizable_dimension_size(), self.kwargs

    def _set_layer_state(self, state):
        """Set this to the state passed, may cause resizing

        Args:
            state ((type, int, dict)): Object create by self.get_layer_state
        """
        class_type, size, kwargs = state
        assert class_type == type(self)

        if not hasattr(self, '_bound_variables'):
            return

        if self.get_resizable_dimension_size() != size:
            self.resize(size, no_splitting_or_pruning=True)

        for name, value in kwargs.iteritems():
            assert hasattr(self, '_' + name), 'expected to have property with name %s' % ('_' + name,)

            attribute = getattr(self, '_' + name)

            if isinstance(attribute, tf.Variable):
                self._get_assign_function(name)(value)
            elif type(attribute) == type(value) or isinstance(attribute, type(value)) or isinstance(value,
                                                                                                    type(attribute)):
                setattr(self, '_' + name, value)
            else:
                raise Exception("Mismatch variable type for %s, existing type was %s new type was %s" %
                                (name, type(attribute), type(value)))

    def get_network_state(self):
        return [layer._get_layer_state() for layer in self.all_connected_layers]

    def set_network_state(self, state):
        all_current_layers = list(self.all_connected_layers)

        current_layers_iter = iter(all_current_layers)
        state_iter = iter(state)

        while True:
            try:
                next_state = state_iter.next()
            except StopIteration:
                return

            next_current_layer = current_layers_iter.next()
            class_type, size, kwargs = next_state

            if class_type == type(next_current_layer):
                next_current_layer._set_layer_state(next_state)
            else:  # next layer needs to be removed
                layer_after = current_layers_iter.next()
                assert class_type == type(layer_after)

                next_current_layer.remove_layer_from_network()
                layer_after._set_layer_state(next_state)
