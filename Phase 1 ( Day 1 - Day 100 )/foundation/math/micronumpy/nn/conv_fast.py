from engine.tensor import Tensor
from nn.im2col import im2col_1d

class Conv1DFast:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.kernel = Tensor(
            [0.01] * kernel_size,
            requires_grad=True
        )

    def __call__(self, x):
        cols = im2col_1d(x.data, self.kernel_size)

        out = []
        for col in cols:
            s = 0.0
            for a, b in zip(col, self.kernel.data):
                s += a * b
            out.append(s)

        return Tensor(out, requires_grad=True)
