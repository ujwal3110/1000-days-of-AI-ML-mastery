from engine.tensor import Tensor

class Conv1D:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.kernel = Tensor(
            [0.01] * kernel_size, requires_grad=True
        )

    def __call__(self, x):
        out = []
        for i in range(len(x.data) - self.kernel_size + 1):
            s = 0.0
            for k in range(self.kernel_size):
                s += x.data[i + k] * self.kernel.data[k]
            out.append(s)
        return Tensor(out, requires_grad=True)


class Conv2D:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.kernel = Tensor(
            [[0.01]*kernel_size for _ in range(kernel_size)],
            requires_grad=True
        )

    def __call__(self, x):
        h = len(x)
        w = len(x[0])
        k = self.kernel_size
        out = []

        for i in range(h - k + 1):
            row = []
            for j in range(w - k + 1):
                s = 0.0
                for ki in range(k):
                    for kj in range(k):
                        s += x[i+ki][j+kj] * self.kernel.data[ki][kj]
                row.append(s)
            out.append(row)

        return Tensor(out, requires_grad=True)
