import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class SklearnModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, train_X, train_y, val_X, val_y):
        self.model.fit(train_X, train_y)
        return {}

    def predict(self, X):
        return self.model.predict(X)

    def reset(self):
        print('Resetting sklearn model...')
        self.model = self.model.__class__()

    def save(self, path):
        joblib.dump(self.model, path)


class PytorchModelWrapper:
    def __init__(self, model, train_config):
        self.model = model
        self.train_config = train_config
        self.device = train_config.get("device", "cpu")

    def reset(self):
        print('Resetting PyTorch model...')
        for layer in self.model.children():
            if isinstance(layer, nn.Sequential):
                for sub in layer:
                    if hasattr(sub, 'reset_parameters'):
                        sub.reset_parameters()
            elif hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def fit(self, train_X, train_y, val_X, val_y):
        train_loader = DataLoader(BasePytorchModelDataset(train_X, train_y), batch_size=100)
        val_loader   = DataLoader(BasePytorchModelDataset(val_X, val_y), batch_size=100)
        return self._train_model(train_loader, val_loader)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            # X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            if isinstance(X, torch.Tensor):
                X_tensor = X.detach().clone().float().to(self.device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_tensor)
        return preds.cpu().numpy()

    def _train_model(self, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.train_config["lr"])
        self.model.to(self.device)

        train_losses, val_losses = [], []
        best_val_loss = None
        best_model = None

        for epoch in range(self.train_config["epochs"]):
            self.model.train()
            epoch_train_loss, n_train = 0.0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).float()
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss = torch.clamp(loss, -5e5, 5e5)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * X_batch.size(0)
                n_train += X_batch.size(0)

            avg_train_loss = epoch_train_loss / n_train

            # Evaluate
            self.model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).float()
                    preds = self.model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss = torch.clamp(loss, -5e5, 5e5)
                    val_loss += loss.item() * X_batch.size(0)
                    n_val += X_batch.size(0)

            avg_val_loss = val_loss / n_val

            if best_val_loss is None or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = deepcopy(self.model)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            if self.train_config.get("loss_per_epoch", False):
                print(f"Epoch {epoch:<3} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch:<3}")

        print(f"\nBest model — Val Loss: {best_val_loss:.4f}\n")
        self.model = best_model
        return {"Train": train_losses, "Validation": val_losses}
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)


class BasePytorchModelDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(np.array(features), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]