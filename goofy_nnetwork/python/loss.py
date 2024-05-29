from copy import copy
from matrix import Matrix

class Loss:
    def __init__(self, network):
        self.network = network

    def __call__(self, output: Matrix, reference: Matrix) -> Matrix:
        output = copy(output)
        reference = copy(reference)
        matrix_p1: Matrix = reference - output
        matrix_p2: list = []
        for y in matrix_p1.matrix:
            vector: list = []
            for x in y:
                vector.append(x)
            matrix_p2.append(vector)
        self.error = Matrix.matrix_init(matrix_p2)
    
    def backward(self):
        self.network.backpropagate(self.error)
        
    
        
