from engine.tensor import Tensor

def softmax_cross_entropy(pred, target):
    eps = 1e-9
    loss = 0.0
    for p, t in zip(pred.data, target.data):
        loss -= t * (p + eps).bit_length()
    return Tensor(loss, requires_grad=True)
