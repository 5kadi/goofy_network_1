from __future__ import annotations
from random import uniform
from copy import copy
from typing import Any

class Matrix:
    def __init__(self, data: list, shape: list, matrix: list = []):
        self.data = data
        self.shape = shape
        self.matrix = matrix
        if len(matrix) == 0:
            self.reshape(self.shape)
    
    @classmethod
    def matrix_init(сls, matrix: list):
        data: list = []
        shape: list = [len(matrix), len(matrix[0])]
        matrix: Matrix = matrix
        for y in matrix:
            for x in y:
                data.append(x)
        Matrix_obj: Matrix = сls(data, shape, matrix=matrix)
        return Matrix_obj


    def reshape(self, new_shape: list):
        if new_shape[0] * new_shape[1] == len(self.data):
            new_matrix: list = []
            start_x: int = 0
            end_x: int = new_shape[1]
            for y in range(new_shape[0]):
                new_matrix.append(self.data[start_x:end_x])
                start_x = end_x 
                end_x += new_shape[1]
            self.matrix = new_matrix
            self.shape = new_shape
            return
        raise Exception("Matrix shape and data size are different")

    def transpose(self):
        new_matrix: list =[]
        for x in range(self.shape[1]):
            vector: list= []
            for y in range(self.shape[0]):
                vector.append(self.matrix[y][x])
            new_matrix.append(vector)
        self.matrix = new_matrix
        self.shape = [self.shape[1], self.shape[0]]
         
    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        if self.shape[1] == other.shape[0]:
            temp: Matrix = copy(other)
            new_matrix: list = []
            temp.transpose()
            for self_y in self.matrix:
                vector: list = []
                for other_y in temp.matrix:
                    new_element = 0
                    for x in range(len(other_y)):
                        new_element += self_y[x] * other_y[x]
                    vector.append(new_element)
                new_matrix.append(vector)
            return Matrix.matrix_init(new_matrix)
        raise Exception("Matrix_1 x should be equal to Matrix_2 y")
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        if self.shape == other.shape:
            new_matrix: list = []
            for y in range(self.shape[0]):
                vector: list = []
                for x in range(self.shape[1]):
                    new_element = self[y][x] - other[y][x]
                    vector.append(new_element)
                new_matrix.append(vector)
            return Matrix.matrix_init(new_matrix)
        raise Exception("Matrix_1 shape should be equal to Matrix_2 shape")
    
    def __getitem__(self, index: int) -> Any:
        return self.matrix[index]
 
    def __iter__(self) -> list:
        return iter(self.matrix)
    
    def __str__(self) -> str:
        return str(self.matrix)
    

def matrix_uniform(shape: list) -> Matrix:
    data = [uniform(0.0, 1.0) for d in range(shape[0]*shape[1])]
    matrix: Matrix = Matrix(data, shape)
    return matrix


#FAQ:
#to multiply Matrix_1 x should be equal to Matrix_2 y
#after multiplication shape of a new matrix: [Matrix_1 y, Matrix_2 x]