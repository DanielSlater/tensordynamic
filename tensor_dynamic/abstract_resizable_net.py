from abc import ABCMeta, abstractmethod


class AbstractResizableNet(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_dimensions(self):
        raise NotImplementedError()

    @abstractmethod
    def add_layer(self, layer_index_to_add_after, hidden_nodes):
        raise NotImplementedError()

    @abstractmethod
    def resize_layer(self, layer_index, new_size, data_set):
        raise NotImplementedError()

    def get_layer_size(self, layer_index):
        if not isinstance(layer_index, int):
            raise TypeError("layer_index must be an int was %s" % type(layer_index))
        if layer_index < 0:
            raise ValueError("layer index must be 0 or greater was %s" % (layer_index, ))

        dimensions = self.get_dimensions()

        if layer_index > len(dimensions):
            raise ValueError("layer index must be less than number of layers %s" % (layer_index, ))

        return dimensions[layer_index]

    @abstractmethod
    def train_till_convergence(self, data_set):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, inputs):
        raise NotImplementedError()

    def get_hidden_layers_count(self):
        return len(self.get_dimensions()) - 2