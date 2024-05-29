from matrix import Matrix
from loss import Loss
from network import Network2x1, Network4x1
from optim import Optim

"""
inp = Matrix([1.0, 1.0, 1.0, 1.0], [4, 1])
ref = Matrix([4.0], [1, 1])
nn = Network4x1()
loss = Loss(nn)
optim = Optim(nn, 0.25)
test = Matrix([1.0, 2.0, 3.0, 4.0], [4, 1])
"""

inp = Matrix([1.0, 1.0], [2, 1])
ref = Matrix([2.0], [1, 1])
nn = Network2x1()
loss = Loss(nn)
optim = Optim(nn, 0.1)
test = Matrix([2.0, 2.0], [2, 1])

for i in range(15):
    out = nn(inp)
    loss(out, ref)
    loss.backward()
    optim.step()
    print(f"{loss.error}")

test = nn(test)
print(test)










