def im2col_1d(x, kernel_size, stride=1):
    """
    Converts 1D input into column matrix
    Example:
      x = [1,2,3,4]
      k = 2
      → [[1,2],
         [2,3],
         [3,4]]
    """
    cols = []
    for i in range(0, len(x) - kernel_size + 1, stride):
        cols.append(x[i:i + kernel_size])
    return cols
