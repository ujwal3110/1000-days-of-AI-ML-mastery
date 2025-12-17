from nn.conv import Conv1D
from utils.grad_check import grad_check

def f(x):
    conv = Conv1D(kernel_size=3)
    return sum(conv(x).data)

x = [1.0, 2.0, 3.0, 4.0, 5.0]
analytic_grad = [1, 1, 1, 1, 1]  # simplified sanity

grad_check(f, x, analytic_grad)
