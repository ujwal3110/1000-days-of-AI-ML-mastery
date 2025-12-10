#foundation/math/vector.py

class Vector:
    """
    A simple vector class for building a mini-NumPy engine.
    Supports:
        - addition, subtraction
        - scalar multiplication
        - dot product
        - magnitude, normalization
        - distance
        - hadamard product
        - projection
        - cosine similarity
        - angle between vectors
    """

    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Vector data must be a list or tuple.")
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
            raise TypeError("Scale factor must be numeric.")
        return Vector([alpha * x for x in self.data])

    # ------------------------------------------------------------
    # Dot product & magnitude
    # ------------------------------------------------------------
    def dot(self, other):
        _check_same_length(self, other)
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return (sum(x * x for x in self.data)) ** 0.5

    def normalize(self, eps=1e-12):
        mag = self.magnitude()
        if mag < eps:
            raise ZeroDivisionError("Cannot normalize near-zero vector.")
        return self.scale(1.0 / mag)

    # ------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------
    def distance(self, other):
        _check_same_length(self, other)
        return self.sub(other).magnitude()

    # ------------------------------------------------------------
    # Hadamard product
    # ------------------------------------------------------------
    def hadamard(self, other):
        _check_same_length(self, other)
        return Vector([a * b for a, b in zip(self.data, other.data)])

    # ------------------------------------------------------------
    # Day 3 features: projection, cosine similarity, angle
    # ------------------------------------------------------------
    def projection(self, base):
        """Project this vector onto another vector."""
        denom = base.dot(base)
        if denom == 0:
            raise ZeroDivisionError("Cannot project onto zero vector.")
        scalar = self.dot(base) / denom
        return base.scale(scalar)

    def cosine_similarity(self, other, eps=1e-12):
        den = self.magnitude() * other.magnitude()
        if den < eps:
            raise ZeroDivisionError("Cosine similarity undefined for zero vectors.")
        return self.dot(other) / den

    def angle_with(self, other):
        """Angle in radians."""
        import math
        cos = self.cosine_similarity(other)
        cos = max(-1.0, min(1.0, cos))
        return math.acos(cos)


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def _check_same_length(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have same length.")
