import time

class Profiler:
    def __init__(self):
        self.times = {}

    def start(self, name):
        self.times[name] = time.time()

    def stop(self, name):
        elapsed = time.time() - self.times[name]
        print(f"[PROFILE] {name}: {elapsed:.6f}s")
