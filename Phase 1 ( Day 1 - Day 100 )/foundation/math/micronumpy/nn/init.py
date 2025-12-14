import math
import random

def zeros(shape):
    return [0.0 for _ in range(shape)]

def ones(shape):
    return [1.0 for _ in range(shape)]

def random_uniform(shape, scale=0.01):
    return [(random.random() - 0.5) * scale for _ in range(shape)]

def xavier(shape):
    limit = math.sqrt(6 / shape)
    return [(random.random() * 2 - 1) * limit for _ in range(shape)]

def he(shape):
    std = math.sqrt(2 / shape)
    return [random.gauss(0, std) for _ in range(shape)]
