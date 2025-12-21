from engine.tensor import Tensor
from backends.cpu import matmul

def im2col_2d(x, kernel_size):
    H, W = len(x), len(x[0])
    k = kernel_size
    cols = []

    for i in range(H - k + 1):
        for j in range(W - k + 1):
            col = []
            for ki in range(k):
                for kj in range(k):
                    col.append(x[i+ki][j+kj])
            cols.append(col)

    return cols


class Conv2D:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.kernel = Tensor(
            [[0.01]*(kernel_size*kernel_size)],
            requires_grad=True
        )

    def __call__(self, x):
        cols = im2col_2d(x.data, self.kernel_size)
        out = matmul(cols, list(zip(*self.kernel.data)))
        return Tensor([o[0] for o in out], requires_grad=True)
