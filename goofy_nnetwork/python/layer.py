from __future__ import annotations
from copy import copy
from numpy import sqrt
from matrix import Matrix
from loss import Loss
from random import uniform


class BaseLayer:
    def __init__(self, inputs: int, outputs: int):
        self.inputs = inputs
        self.outputs = outputs
        self.links: int = self.inputs
        self.create_weights()
        self.signal: Matrix = None
        self.error: Matrix = None

    def create_weights(self):
        new_weights: list = []
        for k in range(self.outputs):
            vector: list = []
            for j in range(self.inputs):
                vector.append(uniform(0.0, 1.0/sqrt(self.links)))
            new_weights.append(vector)
        self.weights = Matrix.matrix_init(new_weights)

    def __call__(self, input_values: Matrix) -> Matrix:
        if input_values.shape[0] * input_values.shape[1] == self.inputs:
            input_vector: Matrix = copy(input_values)
            input_vector.reshape([self.inputs, 1])
            self.signal = input_vector
            output: Matrix = self.weights @ self.signal
            return output
        raise Exception("Matrix input_values size should be equal to Layer inputs")

    def backpropagate(self, error: Matrix):
        weights_ratio: list = []
        for k in range(self.outputs):
            vector: list = []
            new_sum: float = 0
            for j in range(self.inputs):
                new_sum += self.weights.matrix[k][j]
            for x in self.weights.matrix[k]:
                vector.append(x/new_sum)
            weights_ratio.append(vector)
        weights_ratio: Matrix = Matrix.matrix_init(weights_ratio)
        weights_ratio.transpose()
        self.error = weights_ratio @ error

    def __str__(self) -> str:
        return f"{self.inputs}x{self.outputs}"

class InputLayer(BaseLayer):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(inputs, outputs)

class HiddenLayer(BaseLayer):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(inputs, outputs)

class OutputLayer(BaseLayer):
    def __init__(self, outputs: int):
        self.outputs = outputs
        self.signal: Matrix = None
        self.error: Matrix = None
    
    def __call__(self, input_values: Matrix):
        input_vector: Matrix = copy(input_values)
        input_vector.reshape([self.outputs, 1]) 
        self.signal = input_vector
        return self.signal
    
    def backpropagate(self, error: Matrix):
        self.error = error
    
class LayerSequence:
    def __init__(self, input_layer: InputLayer , hidden_layers: list[HiddenLayer], output_layer: OutputLayer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.active_layers = [input_layer] + hidden_layers
        self.num_layers = 2 + len(self.hidden_layers)
        self.num_active_layers = self.num_layers - 1
        self.update()
        self.set_links()

    def set_links(self):
        self.input_layer.links = self.input_layer.inputs
        for layer_idx in range(1, self.num_active_layers):
            self.active_layers[layer_idx].links = self.active_layers[layer_idx - 1].inputs

    def update(self):
        self.weights: list = [self.input_layer.weights] + [layer.weights for layer in self.hidden_layers]
        self.signals: list = [self.input_layer.signal] + [layer.signal for layer in self.hidden_layers] + [self.output_layer.signal]
        self.errors: list = [self.input_layer.error] + [layer.error for layer in self.hidden_layers] + [self.output_layer.error]

    def info(self):
        for layer_idx in range(self.num_active_layers):
            print(f"{self.active_layers[layer_idx]}:{self.errors[layer_idx]}\n")

