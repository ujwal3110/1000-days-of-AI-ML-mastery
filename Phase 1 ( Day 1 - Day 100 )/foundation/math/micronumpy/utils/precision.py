import numpy as np

def to_fp16(x):
    return np.array(x, dtype=np.float16).astype(np.float32).tolist()

def to_fp32(x):
    return np.array(x, dtype=np.float32).tolist()

def loss_scale(loss, scale=1024.0):
    return loss * scale
