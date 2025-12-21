from nn.conv2d_im2col import Conv2D
from engine.tensor import Tensor

x = Tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

conv = Conv2D(kernel_size=2)
y = conv(x)

print("Conv2D output:", y)
