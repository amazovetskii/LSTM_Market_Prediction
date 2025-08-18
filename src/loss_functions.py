import torch

def directional_loss(pred, target):
    mse = torch.mean((pred - target) ** 2)
    sign_acc = torch.mean((torch.sign(pred[1:] - pred[:-1]) == torch.sign(target[1:] - target[:-1])).float())
    return mse * (1 - sign_acc)