# tensor.py
import math
import random

class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, (int, float)):
            data = [[data]]
        
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
        
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._parents = []

    def shape(self):
        return (self.rows, self.cols)

    def zero_grad(self):
        self.grad = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def __repr__(self):
        return f"Tensor(shape={self.shape()}, data={self.data})"

    # --------------------------------------------------------
    # Elementwise ops via broadcasting (imported lazily)
    # --------------------------------------------------------
    def __add__(self, other):
        from .ops import add
        return add(self, other)

    def __sub__(self, other):
        from .ops import sub
        return sub(self, other)

    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)

    # matrix multiplication
    def matmul(self, other):
        from .ops import matmul
        return matmul(self, other)
