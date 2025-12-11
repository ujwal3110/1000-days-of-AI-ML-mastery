# layers.py
import random
from .tensor import Tensor
from .ops import add, mul
from .ops import matmul

# --------------------------------------------------------
# Activation functions
# --------------------------------------------------------
def relu(x):
    out = Tensor([[max(0, v) for v in row] for row in x.data], requires_grad=x.requires_grad)

    def _backward():
        if out.grad is None:
            return
        grad = [[out.grad[i][j] * (1 if x.data[i][j] > 0 else 0)
                for j in range(x.cols)] for i in range(x.rows)]
        if x.requires_grad:
            x.grad = grad

    out._backward = _backward
    out._parents = [x]
    return out

def sigmoid(x):
    import math
    sig = [[1/(1+math.exp(-v)) for v in row] for row in x.data]
    out = Tensor(sig, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            grad = [[out.grad[i][j] * sig[i][j] * (1 - sig[i][j])
                    for j in range(x.cols)] for i in range(x.rows)]
            x.grad = grad

    out._backward = _backward
    out._parents = [x]
    return out

def tanh(x):
    import math
    t = [[math.tanh(v) for v in row] for row in x.data]
    out = Tensor(t, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad:
            grad = [[out.grad[i][j] * (1 - t[i][j]**2)
                    for j in range(x.cols)] for i in range(x.rows)]
            x.grad = grad

    out._backward = _backward
    out._parents = [x]
    return out

# --------------------------------------------------------
# Dense Layer
# --------------------------------------------------------
class Dense:
    def __init__(self, in_features, out_features):
        self.W = Tensor([[random.uniform(-0.1, 0.1) for _ in range(out_features)]
                         for _ in range(in_features)], requires_grad=True)
        self.b = Tensor([[0 for _ in range(out_features)]], requires_grad=True)

    def forward(self, x):
        return matmul(x, self.W) + self.b
