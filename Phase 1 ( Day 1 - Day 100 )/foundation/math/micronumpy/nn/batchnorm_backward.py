def batchnorm_backward(x, grad_out, mean, var, gamma, eps=1e-5):
    """
    Manual BatchNorm backward (1D)
    """

    N = len(x)
    std_inv = 1.0 / ((var + eps) ** 0.5)

    dx_norm = [g * gamma for g in grad_out]

    dvar = sum(dx_norm[i] * (x[i] - mean) for i in range(N)) * -0.5 * std_inv**3
    dmean = sum(dx_norm) * -std_inv + dvar * sum(-2*(x[i] - mean) for i in range(N)) / N

    dx = []
    for i in range(N):
        dx.append(
            dx_norm[i] * std_inv +
            dvar * 2 * (x[i] - mean) / N +
            dmean / N
        )

    return dx
