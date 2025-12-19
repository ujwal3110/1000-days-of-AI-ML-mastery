from engine.tensor import Tensor
from nn.conv import Conv1D
from nn.conv_fast import Conv1DFast
from utils.benchmark import benchmark

x = Tensor([i for i in range(100)])
conv_naive = Conv1D(kernel_size=5)
conv_fast = Conv1DFast(kernel_size=5)

t1 = benchmark(lambda: conv_naive(x))
t2 = benchmark(lambda: conv_fast(x))

print(f"Naive Conv1D: {t1:.6f}s")
print(f"Fast  Conv1D: {t2:.6f}s")
print(f"Speedup: {t1 / t2:.2f}x")
