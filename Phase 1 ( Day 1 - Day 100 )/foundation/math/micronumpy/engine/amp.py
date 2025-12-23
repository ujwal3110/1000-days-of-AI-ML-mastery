import numpy as np
from .loss_scaler import LossScaler

class AMPContext:
    def __init__(self, enabled=True, dtype=np.float16):
        self.enabled = enabled
        self.dtype = dtype
        self.scaler = LossScaler()

    def cast(self, tensor):
        if not self.enabled:
            return tensor
        return tensor.astype(self.dtype)
