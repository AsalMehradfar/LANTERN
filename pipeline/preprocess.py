import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, labels, features, idxs, scaler=None):
        self.scaler = scaler or MinMaxScaler(feature_range=(-1, 1))
        self.labels = np.array(labels)[:, None]
        self.features = np.array(features)
        self.idxs = np.array(idxs)[:, None]

    def transform(self):
        data = np.hstack((np.copy(self.features), np.copy(self.labels)))
        self.scaler.fit(data)
        scaled = self.scaler.transform(data)

        feature_dim = self.features.shape[1]
        return scaled[:, :feature_dim], scaled[:, feature_dim:], self.idxs, self.scaler

    def split(self, split_type='random', dataset='AGILE'):
        X, y, idxs, _ = self.transform()
        X_size, y_size = X.shape[1], y.shape[1]
        combined = np.hstack((X, y, idxs))

        train_idx, val_idx, test_idx = np.load(f'data/splits/{dataset}/{split_type}.npy', allow_pickle=True)

        get = lambda idx: combined[idx, :]
        train, val, test = get(train_idx), get(val_idx), get(test_idx)

        tmp = [train, val, test]
        true_labels = [self.scaler.inverse_transform(x[:, :-1]) for x in tmp]
        true_labels = [x[:, X_size:X_size+y_size] for x in true_labels]
        
        return {
            'percentage': len(train_idx) / (len(train_idx) + len(val_idx) + len(test_idx)),
            'X': [train[:, :X_size], val[:, :X_size], test[:, :X_size]],
            'y': [train[:, X_size:X_size+y_size], val[:, X_size:X_size+y_size], test[:, X_size:X_size+y_size]],
            'true': true_labels,
            'idxs': [train[:, -1:], val[:, -1:], test[:, -1:]]
        }