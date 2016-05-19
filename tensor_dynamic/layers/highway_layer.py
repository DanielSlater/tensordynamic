from tensor_dynamic.layers.base_layer import BaseLayer
from tensor_dynamic.lazyprop import lazyprop

# TODO...
class HighwayLayer(BaseLayer):
    def __init__(self, first_layer, second_layer=None, session=None, name=None):
        assert first_layer.output_nodes == second_layer.output_nodes
        super(HighwayLayer, self).__init__(second_layer, first_layer.output_nodes, session=session, name=name)
        self._carry_layer = first_layer
        self._gate = # this should be a layer itself with strong negative bias self._create_variable((BaseLayer.INPUT_BOUND_VALUE,), self)

    @lazyprop
    def activation(self):
        return self._gate * self._carry_layer.activation + ((1 - self._gate) * self.input_layer.activation)

    def resize(self):
        raise NotImplementedError()