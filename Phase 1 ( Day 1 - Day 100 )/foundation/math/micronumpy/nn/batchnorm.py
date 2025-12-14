from engine.tensor import Tensor

class BatchNorm1D:
    def __init__(self, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(1.0, requires_grad=True)
        self.beta = Tensor(0.0, requires_grad=True)

    def __call__(self, x):
        mean = sum(x.data) / len(x.data)
        var = sum((xi - mean)**2 for xi in x.data) / len(x.data)

        norm = [(xi - mean) / ((var + self.eps) ** 0.5) for xi in x.data]
        out = [self.gamma.data * ni + self.beta.data for ni in norm]

        return Tensor(out, requires_grad=True)
