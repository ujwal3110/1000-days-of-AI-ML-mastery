# training/optim.py
class SGD:
    def __init__(self, params, lr=0.1, momentum=0.0):
        """
        params: list of Tensors (with requires_grad=True)
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [None for _ in params]

    def step(self):
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            # gradient descent: p = p - lr * grad
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    g = p.grad[i][j]
                    if self.momentum:
                        if self.velocities[idx] is None:
                            self.velocities[idx] = [[0.0]*p.shape[1] for _ in range(p.shape[0])]
                        self.velocities[idx][i][j] = self.momentum * self.velocities[idx][i][j] - self.lr * g
                        p.data[i][j] += self.velocities[idx][i][j]
                    else:
                        p.data[i][j] -= self.lr * g

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
