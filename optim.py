from __future__ import annotations
from copy import copy
from numpy import exp
from network import BaseNetwork
from layer import LayerSequence, BaseLayer
from matrix import Matrix




class Optim:
    def __init__(self, network: BaseNetwork, lr: float):
        self.seq = network.seq
        self.lr = lr

    def step(self) -> Matrix: 
        for layer_idx in range(self.seq.num_active_layers):
            current_layer: BaseLayer = self.seq.active_layers[layer_idx]
            next_layer_error: Matrix = self.seq.errors[layer_idx + 1]
            new_weights: Matrix = current_layer.weights - self.calculate_derivative(current_layer, next_layer_error)
            current_layer.weights = new_weights

    def calculate_derivative(self, layer: BaseLayer, next_layer_error: Matrix) -> Matrix:
        derivative_matrix: list = []
        weights: Matrix = layer.weights
        signal: Matrix = layer.signal
        for k in range(layer.outputs): 
            derivative_vector: list = []
            new_sum: float = 0
            for j in range(layer.inputs):
                new_sum += weights[k][j] * signal[j][0]
            new_sigmoid: float = self.sigmoid(new_sum)
            for j in range(layer.inputs):
                new_derivative: float = -next_layer_error[k][0]*new_sigmoid*(1 - new_sigmoid)*signal[j][0]
                derivative_vector.append(new_derivative)
            derivative_matrix.append(derivative_vector)
        derivative_matrix: Matrix = Matrix.matrix_init(derivative_matrix)
        return derivative_matrix

    @staticmethod
    def sigmoid(x: float) -> float:
        res = float(1 / (1 + exp(-x)))
        return res
        
#надо найти dE/dwjk