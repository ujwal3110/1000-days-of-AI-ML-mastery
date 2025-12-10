class Matrix:
    """
    A simple 2D matrix class to build the foundation of a mini-NumPy library.
    Supports:
        - shape (rows, cols)
        - addition, subtraction
        - scalar multiplication
        - elementwise (Hadamard) product
        - matrix-vector multiplication
        - matrix-matrix multiplication
        - transpose
    """

    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Matrix data must be a list of lists.")

        row_lengths = {len(row) for row in data}
        if len(row_lengths) != 1:
            raise ValueError("All rows in the matrix must have the same length.")

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
    # Basic ops
    # ------------------------------------------------------------
    def add(self, other):
        _check_same_shape(self, other)
        return Matrix([
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        ])

    def sub(self, other):
        _check_same_shape(self, other)
        return Matrix([
            [a - b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        ])

    def scale(self, alpha):
        if not isinstance(alpha, (int, float)):
            raise TypeError("Scale factor must be a number.")
        return Matrix([[alpha * x for x in row] for row in self.data])

    # ------------------------------------------------------------
    # Elementwise Hadamard product
    # ------------------------------------------------------------
    def hadamard(self, other):
        _check_same_shape(self, other)
        return Matrix([
            [a * b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, other.data)
        ])

    # ------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------
    def transpose(self):
        return Matrix(list(map(list, zip(*self.data))))

    # ------------------------------------------------------------
    # Matrix-vector multiplication
    # ------------------------------------------------------------
    def matvec(self, vector):
        from .vector import Vector

        if vector.__class__.__name__ != "Vector":
            raise TypeError("matvec expects a Vector.")

        if self.cols != len(vector):
            raise ValueError(
                f"Matrix columns ({self.cols}) must match vector length ({len(vector)})."
            )

        result = []
        for row in self.data:
            dot_val = sum(a * b for a, b in zip(row, vector.data))
            result.append(dot_val)

        return Vector(result)

    # ------------------------------------------------------------
    # Matrix-matrix multiplication
    # ------------------------------------------------------------
    def matmul(self, other):
        if other.__class__.__name__ != "Matrix":
            raise TypeError("matmul expects a Matrix.")

        if self.cols != other.rows:
            raise ValueError(
                f"Incompatible shapes: ({self.rows},{self.cols}) cannot "
                f"multiply with ({other.rows},{other.cols})"
            )

        other_t = other.transpose()
        result = []

        for row in self.data:
            new_row = []
            for col in other_t.data:
                dot_val = sum(a * b for a, b in zip(row, col))
                new_row.append(dot_val)
            result.append(new_row)

        return Matrix(result)


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _check_same_shape(a, b):
    if (a.rows != b.rows) or (a.cols != b.cols):
        raise ValueError("Matrices must have the same shape!")
