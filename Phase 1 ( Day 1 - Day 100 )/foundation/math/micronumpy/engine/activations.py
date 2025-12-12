# engine/activations.py
import math
from .tensor import Tensor

def relu(X: Tensor):
    out_data = [[max(0.0, x) for x in row] for row in X.data]
    out = Tensor(out_data, requires_grad=X.requires_grad)
    out._parents = [X]
    def _backward():
        if out.grad is None:
            return
        gradX = [[out.grad[i][j] * (1.0 if X.data[i][j] > 0 else 0.0)
                  for j in range(X.shape[1])] for i in range(X.shape[0])]
        if X.requires_grad:
            X._acc_grad(gradX)
    out._backward = _backward
    return out

def sigmoid(X: Tensor):
    s = [[1.0 / (1.0 + math.exp(-x)) for x in row] for row in X.data]
    out = Tensor(s, requires_grad=X.requires_grad)
    out._parents = [X]
    def _backward():
        if out.grad is None:
            return
        gradX = [[out.grad[i][j] * s[i][j] * (1.0 - s[i][j])
                  for j in range(X.shape[1])] for i in range(X.shape[0])]
        if X.requires_grad:
            X._acc_grad(gradX)
    out._backward = _backward
    return out

def tanh(X: Tensor):
    t = [[math.tanh(x) for x in row] for row in X.data]
    out = Tensor(t, requires_grad=X.requires_grad)
    out._parents = [X]
    def _backward():
        if out.grad is None:
            return
        gradX = [[out.grad[i][j] * (1.0 - t[i][j] * t[i][j])
                  for j in range(X.shape[1])] for i in range(X.shape[0])]
        if X.requires_grad:
            X._acc_grad(gradX)
    out._backward = _backward
    return out
