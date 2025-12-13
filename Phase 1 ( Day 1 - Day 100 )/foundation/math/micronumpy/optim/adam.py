class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {id(p): 0.0 for p in params}
        self.v = {id(p): 0.0 for p in params}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            m = self.m[id(p)]
            v = self.v[id(p)]

            m = self.beta1 * m + (1 - self.beta1) * p.grad
            v = self.beta2 * v + (1 - self.beta2) * (p.grad ** 2)

            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / ((v_hat ** 0.5) + self.eps)

            self.m[id(p)] = m
            self.v[id(p)] = v
            p.grad = 0.0
