import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path="model_weights.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # save best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True