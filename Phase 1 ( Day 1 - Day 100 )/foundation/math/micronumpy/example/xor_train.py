# examples/xor_train.py
from engine.tensor import Tensor
from engine.activations import sigmoid, relu
from nn.layers import Dense, Sequential
from training.losses import mse_loss
from training.optim import SGD

# XOR dataset (4 samples)
# inputs as 4x2 matrix, targets as 4x1 matrix
inputs = Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], requires_grad=False)

targets = Tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], requires_grad=False)

# model: 2 -> 4 -> 1
l1 = Dense(2, 4)
l2 = Dense(4, 1)
model = Sequential(l1, lambda x: sigmoid(l1.forward(x)), l2)  # simpler: we'll compute properly below

# We'll do manual forward with activations to keep chain clear:
def forward(x):
    h = l1.forward(x)
    h = sigmoid(h)
    out = l2.forward(h)
    out = sigmoid(out)
    return out

params = l1.parameters() + l2.parameters()
opt = SGD(params, lr=0.7)

# training loop
for epoch in range(2000):
    # forward
    pred = forward(inputs)  # Tensor shape (4,1)
    loss = mse_loss(pred, targets)  # 1x1 Tensor

    # backward
    # zero grads first
    opt.zero_grad()
    # seed backward from scalar loss
    loss.backward()

    # update parameters
    opt.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}, loss = {loss.data[0][0]:.6f}")

# Evaluate final predictions
pred = forward(inputs)
print("Final predictions (rounded):")
for i in range(pred.shape[0]):
    print(f"input={inputs.data[i]}, pred={pred.data[i][0]:.4f}, target={targets.data[i][0]}")
