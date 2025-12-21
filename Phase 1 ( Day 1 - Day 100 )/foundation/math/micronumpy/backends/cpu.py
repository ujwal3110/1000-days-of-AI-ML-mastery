def matmul(A, B):
    """
    Matrix multiplication (pure Python backend)
    A: [m x n]
    B: [n x p]
    """
    m, n = len(A), len(A[0])
    p = len(B[0])

    out = [[0.0]*p for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                out[i][j] += A[i][k] * B[k][j]
    return out


def add_bias(mat, bias):
    return [[v + bias[j] for j, v in enumerate(row)] for row in mat]
