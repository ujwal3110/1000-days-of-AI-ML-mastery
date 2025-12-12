# engine/tensor.py
from typing import List, Callable, Any

def _zeros_like(data):
    return [[0.0 for _ in row] for row in data]

def _shape(data):
    return (len(data), len(data[0]) if data else 0)

def _transpose(data):
    return [list(row) for row in zip(*data)]

class Tensor:
    """
    Minimal 2D Tensor with autograd.
    data: list of list floats (rows x cols)
    """
    def __init__(self, data: List[List[float]], requires_grad=False):
        # normalize scalar -> 1x1 matrix not supported here (use [[x]])
        self.data = [[float(x) for x in row] for row in data]
        self.shape = _shape(self.data)
        self.requires_grad = requires_grad
        self.grad = None  # same shape as data after backward
        self._parents = []  # list of parent tensors
        self._backward: Callable[[], None] = lambda: None
        self._op = None  # operation name (optional)

    def zero_grad(self):
        self.grad = _zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    # Utility: accumulate gradient (elementwise add)
    def _acc_grad(self, g):
        if g is None:
            return
        if self.grad is None:
            self.grad = [[float(x) for x in row] for row in g]
        else:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.grad[i][j] += g[i][j]

    # Topological sort for backward
    def backward(self, grad: List[List[float]] = None):
        # seed gradient
        if grad is None:
            grad = [[1.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
        self._acc_grad(grad)

        topo = []
        visited = set()
        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for parent in v._parents:
                build(parent)
            topo.append(v)
        build(self)

        for node in reversed(topo):
            node._backward()

    # basic printing values
    def numpy(self):
        return self.data

    # convenience arithmetic operators (call ops to maintain single implementation)
    def __add__(self, other):
        from engine.ops import add
        return add(self, other)

    def __mul__(self, other):
        from engine.ops import mul
        return mul(self, other)

    def matmul(self, other):
        from engine.ops import matmul
        return matmul(self, other)
