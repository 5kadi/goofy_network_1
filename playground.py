from matrix import Matrix, matrix_uniform
from loss import Loss
from network import Network2x1, Network4x1
from optim import Optim


inp = matrix_uniform([100, 4])
ref = Matrix.matrix_init([[a+b+c+d] for a, b, c, d in inp])
nn = Network4x1()
loss = Loss(nn)
optim = Optim(nn, 0.1)
test = Matrix([1.0, 2.0, 3.0, 4.0], [4, 1])

"""
inp = matrix_uniform([100, 2])
ref = Matrix.matrix_init([[a + b] for a, b in inp.matrix])
nn = Network2x1()
loss = Loss(nn)
optim = Optim(nn, 0.11)
test = Matrix([2.0, 2.0], [2, 1])
"""

for epoch in range(5):
    for input_vals, reference in zip(inp.matrix, ref.matrix):
        out = nn(Matrix.matrix_init([input_vals]))
        loss(out, Matrix.matrix_init([[reference]]))
        loss.backpropagate()
        optim.step()
        print(f"{epoch}:{loss.error}")

print(nn(test)[0])













