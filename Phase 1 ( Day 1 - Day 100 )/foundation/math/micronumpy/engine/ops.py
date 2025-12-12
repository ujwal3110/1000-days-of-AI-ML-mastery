# engine/ops.py
from typing import List
from .tensor import Tensor, _zeros_like, _transpose

# helpers
def _ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

def _check_shape_equal(a: Tensor, b: Tensor):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

# -------------------------
# Elementwise add
# -------------------------
def add(A, B):
    A = _ensure_tensor(A)
    B = _ensure_tensor(B)
    # broadcasting not implemented widely: require equal shape for simplicity
    _check_shape_equal(A, B)
    out_data = [[A.data[i][j] + B.data[i][j] for j in range(A.shape[1])]
                for i in range(A.shape[0])]
    out = Tensor(out_data, requires_grad=(A.requires_grad or B.requires_grad))
    out._parents = [A, B]

    def _backward():
        if out.grad is None:
            return
        if A.requires_grad:
            A._acc_grad(out.grad)
        if B.requires_grad:
            B._acc_grad(out.grad)
    out._backward = _backward
    return out

# -------------------------
# Elementwise multiply
# -------------------------
def mul(A, B):
    A = _ensure_tensor(A)
    B = _ensure_tensor(B)
    _check_shape_equal(A, B)
    out_data = [[A.data[i][j] * B.data[i][j] for j in range(A.shape[1])]
                for i in range(A.shape[0])]
    out = Tensor(out_data, requires_grad=(A.requires_grad or B.requires_grad))
    out._parents = [A, B]
    def _backward():
        if out.grad is None:
            return
        if A.requires_grad:
            # dA = out.grad * B
            gradA = [[out.grad[i][j] * B.data[i][j] for j in range(A.shape[1])]
                     for i in range(A.shape[0])]
            A._acc_grad(gradA)
        if B.requires_grad:
            gradB = [[out.grad[i][j] * A.data[i][j] for j in range(A.shape[1])]
                     for i in range(A.shape[0])]
            B._acc_grad(gradB)
    out._backward = _backward
    return out

# -------------------------
# Matrix multiplication
# -------------------------
def matmul(A: Tensor, B: Tensor):
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes for matmul: {A.shape} @ {B.shape}")
    out_rows = A.shape[0]
    out_cols = B.shape[1]
    # compute result
    out_data = []
    for i in range(out_rows):
        new_row = []
        for j in range(out_cols):
            s = 0.0
            for k in range(A.shape[1]):
                s += A.data[i][k] * B.data[k][j]
            new_row.append(s)
        out_data.append(new_row)
    out = Tensor(out_data, requires_grad=(A.requires_grad or B.requires_grad))
    out._parents = [A, B]

    def _backward():
        if out.grad is None:
            return
        # dA = out_grad @ B^T
        if A.requires_grad:
            Bt = _transpose(B.data)
            gradA = []
            for i in range(A.shape[0]):
                row = []
                for k in range(A.shape[1]):
                    s = 0.0
                    for j in range(out.shape[1]):
                        s += out.grad[i][j] * B.data[k][j]
                    row.append(s)
                gradA.append(row)
            A._acc_grad(gradA)
        # dB = A^T @ out_grad
        if B.requires_grad:
            At = _transpose(A.data)
            gradB = []
            for k in range(B.shape[0]):
                row = []
                for j in range(B.shape[1]):
                    s = 0.0
                    for i in range(A.shape[0]):
                        s += A.data[i][k] * out.grad[i][j]
                    row.append(s)
                gradB.append(row)
            B._acc_grad(gradB)
    out._backward = _backward
    return out

# -------------------------
# Sum reduction (all elements => 1x1 tensor)
# -------------------------
def sum_all(A: Tensor):
    total = 0.0
    for row in A.data:
        for x in row:
            total += x
    out = Tensor([[total]], requires_grad=A.requires_grad)
    out._parents = [A]
    def _backward():
        if out.grad is None:
            return
        # out.grad is 1x1 matrix; distribute its single value to all entries
        gval = out.grad[0][0]
        gradA = [[gval for _ in range(A.shape[1])] for _ in range(A.shape[0])]
        if A.requires_grad:
            A._acc_grad(gradA)
    out._backward = _backward
    return out

# -------------------------
# Mean reduction
# -------------------------
def mean_all(A: Tensor):
    n = A.shape[0] * A.shape[1]
    s = sum_all(A)
    # s is 1x1 tensor; scale by 1/n
    out = Tensor([[s.data[0][0] / n]], requires_grad=A.requires_grad)
    out._parents = [A]
    def _backward():
        if out.grad is None:
            return
        gval = out.grad[0][0] / n
        gradA = [[gval for _ in range(A.shape[1])] for _ in range(A.shape[0])]
        if A.requires_grad:
            A._acc_grad(gradA)
    out._backward = _backward
    return out
