def accuracy(preds, targets):
    correct = 0
    for p, t in zip(preds, targets):
        if p.data.index(max(p.data)) == t.data[0]:
            correct += 1
    return correct / len(preds)

def mse(preds, targets):
    total = 0.0
    for p, t in zip(preds, targets):
        total += (p.data[0] - t.data[0]) ** 2
    return total / len(preds)
