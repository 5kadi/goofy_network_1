from __future__ import annotations
from copy import copy
from matrix import Matrix
from layer import BaseLayer, InputLayer, HiddenLayer, OutputLayer, LayerSequence

class BaseNetwork:
    def __init__(self):
        self.seq = LayerSequence()
    
    def auto_create(self):
        cls_vars = vars(self)
        hidden_layers: list[HiddenLayer] = []
        for var_key in cls_vars.keys():
            if type(cls_vars[var_key]) == HiddenLayer:
                hidden_layers.append(cls_vars[var_key])
        input_layer = InputLayer(hidden_layers[0].inputs, hidden_layers[0].inputs)
        output_layer = OutputLayer(hidden_layers[-1].outputs)
        self.seq = LayerSequence(input_layer, hidden_layers, output_layer)

    def __call__(self, input_values: Matrix) -> Matrix:
        output: Matrix = self.seq.input_values(input_values)
        return output

    def backpropagate(self, error: Matrix):
        self.seq.backpropagate(error)


class Network4x1(BaseNetwork):
    def __init__(self):
        self.l1 = HiddenLayer(4, 2)
        self.l2 = HiddenLayer(2, 1)
        self.auto_create()

class Network2x1(BaseNetwork):
    def __init__(self):
        self.l1 = HiddenLayer(2, 1)
        self.auto_create()
                
