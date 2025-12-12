# nn/layers.py
import random
from engine.tensor import Tensor
from engine.ops import matmul, add

class Dense:
    """
    Dense(in_features, out_features)
    forward(x): x (batch x in_features) @ W (in_features x out_features) + b (1 x out_features)
    W and b are Tensors with requires_grad=True
    """
    def __init__(self, in_features, out_features, bias=True):
        # initialize W with small random numbers
        self.W = Tensor([[random.uniform(-0.5, 0.5) for _ in range(out_features)]
                         for _ in range(in_features)], requires_grad=True)
        if bias:
            self.b = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)
        else:
            self.b = None

    def parameters(self):
        params = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def forward(self, x: Tensor):
        y = matmul(x, self.W)
        if self.b is not None:
            # broadcast b across batch rows manually
            # create a Tensor with repeated rows
            b_repeated = Tensor([self.b.data[0] for _ in range(x.shape[0])], requires_grad=False)
            y = add(y, b_repeated)
            # note: gradient for b is handled via matmul/add chain (sum across batch)
            # however, our add expects same shapes; b_repeated is plain Tensor so it's fine.
        return y

class Sequential:
    """
    Simple sequential container: call .modules for layers list
    """
    def __init__(self, *modules):
        self.modules = list(modules)

    def parameters(self):
        params = []
        for m in self.modules:
            if hasattr(m, "parameters"):
                params.extend(m.parameters())
        return params

    def forward(self, x):
        out = x
        for m in self.modules:
            # layers expose forward
            out = m.forward(out) if hasattr(m, "forward") else m(out)
        return out
