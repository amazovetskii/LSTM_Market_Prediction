import matplotlib.pyplot as plt
import os


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
