# foundation/math/vector.py

class Vector:
    """
    A simple vector class to build the foundation of a mini-NumPy library.
    Supports:
        - addition
        - subtraction
        - scalar multiplication
        - dot product
        - magnitude, normalization
        - distance
        - hadamard product
    """

    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Vector data must be a list or tuple.")

        # Convert all elements to float
        self.data = [float(x) for x in data]

    def __repr__(self):
        return f"Vector({self.data})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # ------------------------------------------------------------
    # Basic Ops
    # ------------------------------------------------------------
    def add(self, other):
        _check_same_length(self, other)
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def sub(self, other):
        _check_same_length(self, other)
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def scale(self, alpha):
        if not isinstance(alpha, (int, float)):
            raise TypeError("Scale factor must be a number.")
        return Vector([alpha * x for x in self.data])

    # ------------------------------------------------------------
    # Dot product
    # ------------------------------------------------------------
    def dot(self, other):
        _check_same_length(self, other)
        return sum(a * b for a, b in zip(self.data, other.data))

    # ------------------------------------------------------------
    # Magnitude
    # ------------------------------------------------------------
    def magnitude(self):
        return (sum(x * x for x in self.data)) ** 0.5

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------
    def normalize(self, eps=1e-12):
        mag = self.magnitude()
        if mag < eps:
            raise ZeroDivisionError("Cannot normalize a zero or near-zero vector.")
        return self.scale(1.0 / mag)

    # ------------------------------------------------------------
    # Distance between vectors
    # ------------------------------------------------------------
    def distance(self, other):
        _check_same_length(self, other)
        return self.sub(other).magnitude()

    # ------------------------------------------------------------
    # Hadamard (elementwise) product
    # ------------------------------------------------------------
    def hadamard(self, other):
        _check_same_length(self, other)
        return Vector([a * b for a, b in zip(self.data, other.data)])


# ------------------------------------------------------------
# Utility function
# ------------------------------------------------------------

def _check_same_length(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length!")
