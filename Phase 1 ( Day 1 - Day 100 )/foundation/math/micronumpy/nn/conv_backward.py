from engine.tensor import Tensor

def conv1d_backward(x, kernel, grad_out):
    """
    x        : input Tensor
    kernel   : kernel Tensor
    grad_out : gradient wrt output
    """

    grad_x = [0.0] * len(x.data)
    grad_k = [0.0] * len(kernel.data)

    k = len(kernel.data)

    for i in range(len(grad_out.data)):
        for j in range(k):
            grad_x[i + j] += grad_out.data[i] * kernel.data[j]
            grad_k[j] += grad_out.data[i] * x.data[i + j]

    kernel.grad = grad_k
    x.grad = grad_x
