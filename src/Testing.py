import torch
from src import visualisation as vis
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Testing:
    def __init__(self, model, device):
        self.device = device
        self.model = model

    def get_n_last_prediction_windows(self, data_np, n_images, lookback_days, future_days, scalers, column_name):
        window_size = lookback_days + future_days
        for n in range(1, n_images + 1):
            end_of_window = -window_size * (n - 1)
            visualisation_set = data_np[-window_size * n:None if end_of_window == 0 else end_of_window]

            context = torch.tensor(visualisation_set[:lookback_days], dtype=torch.float32).to(self.device).unsqueeze(0)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(context).cpu().numpy()[0, :].reshape(-1, 1)

            vis_column_scaled = visualisation_set[:, list(scalers.keys()).index(column_name)].reshape(-1, 1)
            vis_close = scalers[column_name].inverse_transform(vis_column_scaled).squeeze()

            pred_close = scalers[column_name].inverse_transform(outputs).squeeze()

            vis.plot_predictions(vis_close, pred_close, save_to_examples=False)