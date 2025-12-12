# training/losses.py
from engine.ops import mean_all, sum_all, add, mul
from engine.tensor import Tensor

def mse_loss(pred: Tensor, target: Tensor):
    # ensure shapes match
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape for MSE")
    # compute (pred - target)^2 elementwise, then mean
    diff = add(pred, Tensor([[-v for v in row] for row in target.data], requires_grad=False))
    sq = mul(diff, diff)
    return mean_all(sq)

def bce_loss(pred: Tensor, target: Tensor, eps=1e-12):
    # pred assumed to be probabilities in (0,1)
    import math
    if pred.shape != target.shape:
        raise ValueError("pred and target must have same shape for BCE")
    # compute elementwise: -[t*log(p) + (1-t)*log(1-p)], then mean
    loss_sum = 0.0
    rows, cols = pred.shape
    for i in range(rows):
        for j in range(cols):
            p = min(max(pred.data[i][j], eps), 1.0 - eps)
            t = target.data[i][j]
            loss_sum += -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))
    out = Tensor([[loss_sum / (rows*cols)]], requires_grad=False)
    # Manual backward not built here; for training use mse/sigmoid or implement proper bce with autograd support later.
    return out
