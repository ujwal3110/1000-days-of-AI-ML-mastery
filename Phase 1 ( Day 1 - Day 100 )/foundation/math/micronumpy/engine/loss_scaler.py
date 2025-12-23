class LossScaler:
    def __init__(self, scale=1024.0):
        self.scale = scale

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_grads(self, parameters):
        for p in parameters:
            if p.grad is not None:
                p.grad /= self.scale
