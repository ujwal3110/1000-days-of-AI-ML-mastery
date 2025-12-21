class BufferPool:
    def __init__(self):
        self.pool = {}

    def get(self, size):
        if size not in self.pool:
            self.pool[size] = [0.0] * size
        return self.pool[size]
