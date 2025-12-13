class RMSProp:
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.cache = {id(p): 0.0 for p in params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            c = self.cache[id(p)]
            c = self.beta * c + (1 - self.beta) * (p.grad ** 2)
            p.data -= self.lr * p.grad / ((c ** 0.5) + self.eps)
            self.cache[id(p)] = c
            p.grad = 0.0
