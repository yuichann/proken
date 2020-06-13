import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset


class Wine(Dataset):
    def __init__(self, csv_path):
        # csv ファイルを読み込む。
        df = pd.read_csv(csv_path)
        data = df.iloc[:, 0]  # データ (2 ~ 14列目)
        labels = df.iloc[:, 1]  # ラベル (1列目)
        print(labels)
        for label in labels:
            if label == 'spring':
                label=0
            elif label == 'summer':
                label=1
            elif label == 'autumn':
                label=2
            elif label == 'winter':
                label=3
        # データを標準化する。
        # data = normalize(data)
        # クラス ID を 0 始まりにする。[1, 2, 3] -> [0, 1, 2]
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        """csv の行数を返す。
        """
        return len(self.data)


# Dataset を作成する。
dataset = Wine('new_csv.csv')
# DataLoader を作成する。
dataloader = DataLoader(dataset, batch_size=64)

for X_batch, y_batch in dataloader:
    print(X_batch.shape, y_batch.shape)