import time

def benchmark(fn, runs=100):
    start = time.time()
    for _ in range(runs):
        fn()
    end = time.time()
    return (end - start) / runs
