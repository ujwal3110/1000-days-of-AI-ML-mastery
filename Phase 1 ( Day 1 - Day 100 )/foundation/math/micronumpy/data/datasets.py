import random
from engine.tensor import Tensor

def xor_dataset(n=100):
    X, y = [], []
    for _ in range(n):
        a, b = random.randint(0, 1), random.randint(0, 1)
        X.append(Tensor([a, b], requires_grad=False))
        y.append(Tensor([a ^ b], requires_grad=False))
    return X, y

def spiral_dataset(n=100, classes=2):
    X, y = [], []
    for i in range(n):
        r = i / n
        t = i * 4
        X.append(Tensor([r * t, r * t], requires_grad=False))
        y.append(Tensor([i % classes], requires_grad=False))
    return X, y
