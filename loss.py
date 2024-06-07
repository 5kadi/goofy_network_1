from copy import copy
from matrix import Matrix

class Loss:
    def __init__(self, network):
        self.network = network

    def __call__(self, output: Matrix, reference: Matrix) -> Matrix:
        if (reference.shape == output.shape):
            output = copy(output)
            reference = copy(reference)
            self.error = reference - output
    
    def backpropagate(self):
        self.network.backpropagate(self.error)
        
    
        
