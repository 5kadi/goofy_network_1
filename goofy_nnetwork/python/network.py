from __future__ import annotations
from copy import copy
from matrix import Matrix
from layer import InputLayer, HiddenLayer, OutputLayer, LayerSequence

class BaseNetwork:
    def __init__(self):
        self.seq = LayerSequence()
        return


class Network4x1:
    def __init__(self):
        self.il = InputLayer(4, 4)
        self.hl1 = HiddenLayer(4, 2)
        self.hl2 = HiddenLayer(2, 1)
        self.ol = OutputLayer(1)
        self.seq = LayerSequence(self.il, [self.hl1, self.hl2], self.ol)

    def __call__(self, input_values: Matrix) -> Matrix:
        input_values = copy(input_values)
        iout: Matrix = self.il(input_values)
        h1out: Matrix = self.hl1(iout)
        h2out: Matrix = self.hl2(h1out)
        oout: Matrix = self.ol(h2out)
        self.seq.update()
        return oout

    def backpropagate(self, error: Matrix):
        self.ol.backpropagate(error)
        self.hl2.backpropagate(self.ol.error)
        self.hl1.backpropagate(self.hl2.error)
        self.il.backpropagate(self.hl1.error)
        self.seq.update()

    def info(self):
        self.seq.info()

class Network2x1:
    def __init__(self):
        self.il = InputLayer(2, 2)
        self.hl1 = HiddenLayer(2, 1)
        self.ol = OutputLayer(1)
        self.seq = LayerSequence(self.il, [self.hl1], self.ol)

    def __call__(self, input_values: Matrix) -> Matrix:
        input_values = copy(input_values)
        iout: Matrix = self.il(input_values)
        h1out: Matrix = self.hl1(iout)
        oout: Matrix = self.ol(h1out)
        self.seq.update()
        return oout

    def backpropagate(self, error: Matrix):
        self.ol.backpropagate(error)
        self.hl1.backpropagate(self.ol.error)
        self.il.backpropagate(self.hl1.error)
        self.seq.update()

    def info(self):
        self.seq.info()
                
