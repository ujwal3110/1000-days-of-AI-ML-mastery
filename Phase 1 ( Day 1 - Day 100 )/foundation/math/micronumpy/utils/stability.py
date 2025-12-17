import math

def test_softmax_stability(x):
    max_x = max(x)
    exp = [math.exp(i - max_x) for i in x]
    return [e / sum(exp) for e in exp]


def test_log_stability(x, eps=1e-12):
    return [math.log(i + eps) for i in x]
