# autograd.py

def backward(tensor):
    # seed gradient
    tensor.grad = [[1 for _ in range(tensor.cols)] for _ in range(tensor.rows)]

    stack = [tensor]
    visited = set()

    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))

        node._backward()

        for parent in getattr(node, "_parents", []):
            if isinstance(parent, list):
                for p in parent:
                    stack.append(p)
            else:
                stack.append(parent)
