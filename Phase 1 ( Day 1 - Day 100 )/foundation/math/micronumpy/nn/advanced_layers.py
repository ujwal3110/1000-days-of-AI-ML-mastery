import random
from engine.tensor import Tensor

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def __call__(self, x):
        self.mask = [(random.random() > self.p) for _ in x.data]
        return Tensor([xi * mi for xi, mi in zip(x.data, self.mask)], requires_grad=True)

class Flatten:
    def __call__(self, x):
        return x  # placeholder for future multidimensional tensors

class Softmax:
    def __call__(self, x):
        exps = [pow(2.71828, xi) for xi in x.data]
        s = sum(exps)
        return Tensor([e / s for e in exps], requires_grad=True)
