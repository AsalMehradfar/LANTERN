import os
import joblib

from utils.data_utils import inverse_transform


class ModelTrainer:
    def __init__(self, model, config, scaler, train_data, val_data, test_data):
        self.model = model                    # Already wrapped
        self.config = config
        self.scaler = scaler

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train(self):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        return self.model.fit(X_train, y_train, X_val, y_val)

    def evaluate(self):
        return [self.predict(*self.train_data),
                self.predict(*self.val_data),
                self.predict(*self.test_data)]

    def predict(self, X, _):
        preds = self.model.predict(X)
        inv_preds, _ = inverse_transform(preds, X, self.scaler)
        return inv_preds

    def reset(self):
        self.model.reset()

    def save_best_model(self):
        # Determine checkpoint directory
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Choose extension based on model type
        filename = f"{self.config['model_save_name']}.pth" if self.config["model_type"] else f"{self.config['model_save_name']}.pkl"
        checkpoint_path = os.path.join(checkpoint_dir, filename)

        # Call wrapper's save method
        self.model.save(checkpoint_path)
        print(f"✅ Best model saved to: {checkpoint_path}")

        # Save scaler
        scaler_path = os.path.join(checkpoint_dir, f"{self.config['model_save_name']}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"📊 Scaler saved to: {scaler_path}")