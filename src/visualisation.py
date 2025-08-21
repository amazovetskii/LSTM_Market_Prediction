import matplotlib.pyplot as plt
import os
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_next_filename(base_name, extension, directory="."):
        if not extension.startswith("."):
                extension = "." + extension
        i = 1
        while True:
                filename = f"{base_name}_{i}{extension}"
                filepath = os.path.join(directory, filename)
                if not os.path.exists(filepath):
                        return filepath
                i += 1

def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Test Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0)
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

def plot_predictions(visualisation_set, predictions, save_to_examples = False):
        assert len(predictions) <= len(visualisation_set), \
                "Predictions cannot be longer than visualisation_set"

        window_size = len(visualisation_set)
        lookback_days = window_size - len(predictions)
        future_days = window_size - lookback_days

        x = range(window_size)
        y1 = visualisation_set.tolist()
        y2 = [y1[lookback_days-1]] + predictions.tolist()

        fig, ax = plt.subplots(figsize=(8, 5))

        # Truth and predictions
        ax.plot(x, y1, 'b-', label="Truth")
        ax.plot(x[-future_days-1:], y2, 'r-', label="Prediction")

        # Vertical line marking prediction start
        ax.axvline(x=lookback_days, color='orange', linestyle='--', linewidth=1.5, label="Prediction start")

        # Labels, legend, grid
        ax.set_xlabel("Days")
        ax.set_ylabel("Close Price")
        ax.set_title("Close Price Prediction")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        if save_to_examples:
                name = f"predictions_{lookback_days}-{future_days}"
                path = get_next_filename(name, ".png", "examples")
                plt.savefig(path)
        else:
                plt.show()
