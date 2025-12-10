# foundation/math/vector.py

import math


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
        - projection
        - cosine similarity
        - angle between vectors
    """

    def __init__(self, data):
        if not isinstance(data, (list, tuple)):
            raise TypeError("Vector data must be a list or tuple.")
        self.data = [float(x) for x in data]  # internal float storage

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
        return math.sqrt(sum(x * x for x in self.data))

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
    # Projection of one vector onto another
    # ------------------------------------------------------------
    def project_onto(self, other, eps=1e-12):
        """
        Projection of vector a onto vector b:
        proj = (a·b / ||b||²) * b
        """
        _check_same_length(self, other)
        denom = other.magnitude() ** 2
        if denom < eps:
            raise ZeroDivisionError("Cannot project onto a zero vector.")
        scale_factor = self.dot(other) / denom
        return other.scale(scale_factor)

    # ------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------
    def cosine_similarity(self, other, eps=1e-12):
        """
        cos(theta) = (a·b) / (||a|| * ||b||)
        """
        _check_same_length(self, other)
        mag1 = self.magnitude()
        mag2 = other.magnitude()
        if mag1 < eps or mag2 < eps:
            raise ZeroDivisionError("Cannot compute cosine similarity with zero vector.")
        return self.dot(other) / (mag1 * mag2)

    # ------------------------------------------------------------
    # Angle between vectors (in radians)
    # ------------------------------------------------------------
    def angle_with(self, other, eps=1e-12):
        """
        Returns angle in radians between vectors.
        """
        cos_theta = self.cosine_similarity(other)
        # clamp for numerical stability
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)


# ------------------------------------------------------------
# Utility function
# ------------------------------------------------------------
def _check_same_length(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length!")
