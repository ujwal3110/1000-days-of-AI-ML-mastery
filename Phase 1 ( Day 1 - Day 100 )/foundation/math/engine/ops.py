# ops.py
from .tensor import Tensor

def broadcast(A, B):
    if isinstance(B, (int, float)):
        B = Tensor([[B] * A.cols for _ in range(A.rows)])
    return A.data, B.data

# ----------------------------------------------
# Elementwise ops
# ----------------------------------------------
def add(A, B):
    Adata, Bdata = broadcast(A, B)
    out = Tensor([[Adata[i][j] + Bdata[i][j] for j in range(A.cols)] for i in range(A.rows)],
                 requires_grad=A.requires_grad or getattr(B, 'requires_grad', False))

    def _backward():
        if out.grad is None:
            return
        if A.requires_grad:
            A.grad = add_grads(A.grad, out.grad)
        if isinstance(B, Tensor) and B.requires_grad:
            B.grad = add_grads(B.grad, out.grad)

    out._backward = _backward
    out._parents = [A, B] if isinstance(B, Tensor) else [A]
    return out

def sub(A, B):
    return add(A, mul(B, -1))

def mul(A, B):
    B_tensor = B if isinstance(B, Tensor) else Tensor(B)
    Adata, Bdata = broadcast(A, B_tensor)

    out = Tensor([[Adata[i][j] * Bdata[i][j] for j in range(A.cols)] for i in range(A.rows)],
                 requires_grad=A.requires_grad or B_tensor.requires_grad)

    def _backward():
        if out.grad is None:
            return
        if A.requires_grad:
            A.grad = add_grads(A.grad, mul_mats(out.grad, Bdata))
        if B_tensor.requires_grad:
            B_tensor.grad = add_grads(B_tensor.grad, mul_mats(out.grad, Adata))

    out._backward = _backward
    out._parents = [A, B_tensor]
    return out

def add_grads(a, b):
    if a is None:
        return [row[:] for row in b]
    return [[a[i][j] + b[i][j] for j in range(len(b[0]))] for i in range(len(b))]

def mul_mats(A, B):
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

# ----------------------------------------------
# Matrix multiplication
# ----------------------------------------------
def matmul(A, B):
    assert A.cols == B.rows, "Shape mismatch"

    data = [[sum(A.data[i][k] * B.data[k][j] for k in range(A.cols))
             for j in range(B.cols)]
            for i in range(A.rows)]

    out = Tensor(data, requires_grad=A.requires_grad or B.requires_grad)

    def _backward():
        if out.grad is None:
            return

        if A.requires_grad:
            A.grad = add_grads(
                A.grad,
                [[sum(out.grad[i][j] * B.data[k][j] for j in range(B.cols))
                  for k in range(A.cols)]
                 for i in range(A.rows)]
            )

        if B.requires_grad:
            B.grad = add_grads(
                B.grad,
                [[sum(A.data[i][k] * out.grad[i][j] for i in range(A.rows))
                  for j in range(B.cols)]
                 for k in range(A.cols)]
            )

    out._backward = _backward
    out._parents = [A, B]
    return out
