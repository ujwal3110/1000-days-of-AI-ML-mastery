class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = {id(p): 0.0 for p in params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            v = self.velocities[id(p)]
            v = self.momentum * v - self.lr * p.grad
            p.data += v
            self.velocities[id(p)] = v
            p.grad = 0.0
