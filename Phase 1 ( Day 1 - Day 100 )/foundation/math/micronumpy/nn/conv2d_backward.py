def conv2d_backward(cols, kernel, grad_out):
    """
    cols: im2col output
    kernel: flattened kernel
    grad_out: gradient from output
    """

    grad_k = [0.0] * len(kernel)
    grad_cols = [[0.0]*len(kernel) for _ in cols]

    for i, g in enumerate(grad_out):
        for j in range(len(kernel)):
            grad_k[j] += cols[i][j] * g
            grad_cols[i][j] += kernel[j] * g

    return grad_k, grad_cols
