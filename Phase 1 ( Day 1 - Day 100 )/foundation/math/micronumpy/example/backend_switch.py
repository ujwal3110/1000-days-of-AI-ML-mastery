from backends.backend import set_backend
from backends.numpy_backend import matmul

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

set_backend("numpy")
print("NumPy backend matmul:", matmul(A, B))
