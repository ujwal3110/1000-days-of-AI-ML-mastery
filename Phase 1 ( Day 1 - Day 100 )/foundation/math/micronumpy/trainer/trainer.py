class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                pred = self.model(xi)
                loss = self.loss_fn(pred, yi)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.data
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
