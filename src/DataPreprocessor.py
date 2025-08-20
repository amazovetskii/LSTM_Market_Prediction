import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


class DataPreprocessor:
    def __init__(self, path):
        self.df = pd.read_csv(path, parse_dates=["Date"])
        self.df = self.df.sort_values("Date", ascending=True).reset_index(drop=True)

        self.scalers = {}
        self.df_scaled = None
        self.data_np = None

    def z_score(self, cols_to_standardize, train_split_part):
        split_idx = int(train_split_part * len(self.df))
        df_train, df_test = self.df.iloc[:split_idx].copy(), self.df.iloc[split_idx:].copy()

        for col in cols_to_standardize:
            scaler = StandardScaler()
            df_train[col] = scaler.fit_transform(df_train[[col]])
            df_test[col] = scaler.transform(df_test[[col]])
            self.scalers[col] = scaler

        self.df_scaled = pd.concat([df_train, df_test], axis=0)
        return self.scalers

    def split_data(self, context_size:int, future_size:int, train_features:list, features_to_predict:list):
        assert type(self.df_scaled) == pd.DataFrame, "data should be scaled first"
        self.data_np = self.df_scaled[train_features].values.astype(np.float32)
        features_to_predict_idx = [train_features.index(x) for x in features_to_predict]

        x, y = [], []
        for i in range(len(self.data_np) - context_size - future_size + 1):
            x.append(self.data_np[i:i + context_size])
            y.append(self.data_np[i + context_size:i + context_size + future_size, features_to_predict_idx])
        x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

        return x, y