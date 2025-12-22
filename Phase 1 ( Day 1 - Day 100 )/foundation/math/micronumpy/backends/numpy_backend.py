import numpy as np

def matmul(A, B):
    return (np.array(A) @ np.array(B)).tolist()

def add_bias(mat, bias):
    return (np.array(mat) + np.array(bias)).tolist()

def conv1d(x, kernel):
    x = np.array(x)
    k = np.array(kernel)
    out = []
    for i in range(len(x) - len(k) + 1):
        out.append(np.sum(x[i:i+len(k)] * k))
    return out
