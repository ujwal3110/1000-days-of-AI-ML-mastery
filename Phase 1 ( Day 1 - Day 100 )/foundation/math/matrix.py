class Matrix:
    """
    A simple 2D matrix class supporting:
        - shape
        - add, sub, scale
        - hadamard
        - transpose
        - matvec, matmul
        - row/column slicing
        - Frobenius norm
        - outer product
        - broadcasting for +,-,*,/
    """

    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Matrix must be list of lists.")

        row_lens = {len(row) for row in data}
        if len(row_lens) != 1:
            raise ValueError("All rows must have equal length.")

        self.data = [[float(x) for x in row] for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])

    def __repr__(self):
        return f"Matrix({self.data})"

    def shape(self):
        return (self.rows, self.cols)

    def __getitem__(self, idx):
        return self.data[idx]

    # ------------------------------------------------------------
    # Basic Ops
    # ------------------------------------------------------------
    def add(self, other):
        A, B = broadcast(self, other)
        return Matrix([[a + b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)])

    def sub(self, other):
        A, B = broadcast(self, other)
        return Matrix([[a - b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)])

    def scale(self, alpha):
        return Matrix([[alpha * x for x in row] for row in self.data])

    def hadamard(self, other):
        A, B = broadcast(self, other)
        return Matrix([[a * b for a, b in zip(rowA, rowB)] for rowA, rowB in zip(A, B)])

    # ------------------------------------------------------------
    # Norm
    # ------------------------------------------------------------
    def norm(self):
        return (sum(x * x for row in self.data for x in row)) ** 0.5

    # ------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------
    def transpose(self):
        return Matrix(list(map(list, zip(*self.data))))

    # ------------------------------------------------------------
    # Row/Column slicing
    # ------------------------------------------------------------
    def row(self, i):
        return Matrix([self.data[i]])

    def col(self, j):
        return Matrix([[self.data[i][j]] for i in range(self.rows)])

    # ------------------------------------------------------------
    # matvec & matmul
    # ------------------------------------------------------------
    def matvec(self, v):
        from .vector import Vector
        if len(v) != self.cols:
            raise ValueError("Vector length mismatch.")
        return Vector([sum(a * b for a, b in zip(row, v.data)) for row in self.data])

    def matmul(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("matmul expects a Matrix.")

        if self.cols != other.rows:
            raise ValueError("Shape mismatch for matrix multiplication.")

        other_t = other.transpose()
        return Matrix([
            [sum(a * b for a, b in zip(row, col)) for col in other_t.data]
            for row in self.data
        ])

    # ------------------------------------------------------------
    # Outer Product
    # ------------------------------------------------------------
    @staticmethod
    def outer(u, v):
        from .vector import Vector
        if not isinstance(u, Vector) or not isinstance(v, Vector):
            raise TypeError("outer() requires two Vectors.")
        return Matrix([[a * b for b in v.data] for a in u.data])


# ------------------------------------------------------------
# Broadcasting Engine
# ------------------------------------------------------------
def broadcast(A, B):
    """Return expanded (A, B) lists of lists ready for elementwise ops."""
    # scalar
    if isinstance(B, (int, float)):
        return (A.data, [[B] * A.cols for _ in range(A.rows)])
    if isinstance(A, (int, float)):
        return ([[A] * B.cols for _ in range(B.rows)], B.data)

    # convert matrices to raw lists
    A_raw, B_raw = A.data, B.data

    if (A.rows == B.rows) and (A.cols == B.cols):
        return (A_raw, B_raw)

    # row vector broadcasting
    if B.rows == 1 and B.cols == A.cols:
        return (A_raw, [B_raw[0] for _ in range(A.rows)])

    # column vector broadcasting
    if B.cols == 1 and B.rows == A.rows:
        return (A_raw, [[B_raw[i][0] for _ in range(A.cols)] for i in range(A.rows)])

    raise ValueError(f"Cannot broadcast shapes {A.shape()} and {B.shape()}")
