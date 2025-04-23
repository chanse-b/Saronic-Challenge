import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import os
import joblib 
import torch
import torch.nn as nn

AHRS = pd.read_csv('ahrs.csv')
CONTROL = pd.read_csv('vehicle_control.csv')
AHRS_COLS = list(AHRS.columns.drop('ts'))
NEXT_AHRS_COLS = [col+"_next" for col in AHRS_COLS]
CONTROL_COLS = list(CONTROL.columns.drop('ts'))

## This is just a utility function file that contains the main models for each predictor in their respective
## ipynb files. Copied here for convience when importing into driver.py


# for the Control Sequence Predictor
class MixedRandomForest:
    def __init__(self, metric:str):
        os.makedirs("Control Models",exist_ok=True)
        self.metric = metric.lower().strip()
        if metric == 'throttle' or metric == 'turn':
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1) # throttle, turn
        elif metric == 'trim' or metric == 'gear':
            self.model = RandomForestClassifier(n_estimators=100,n_jobs=-1) # trim, gear
        else:
            raise Exception("The metric to train is not valid. Must be throttle,turn,gear, or trim")
        self.model_path = f"Control Models/{self.metric}_model.pkl"
        self.is_fitted = os.path.exists(self.model_path)

    def fit(self, X, y):
        # fits or loads the prevously trained model
        if os.path.exists(self.model_path):
            print("loading instead of training...")
            self.model = joblib.load(self.model_path)
            self.is_fitted = True
        else:
            print(f"Training new model for '{self.metric}'...")
            self.model.fit(X, y)
            self.is_fitted = True
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to '{self.model_path}'")
            return self
        
    def predict(self,X):
        # given X, a row vector of inputs outputs raw control
        self.model = joblib.load(self.model_path)
        return self.model.predict(X)

    def predictfromDF(self, row_df, decimal_places=4):
        # given a row of the merged dataframe, outputs a tuple of atual,predicted
        if not self.is_fitted:
            raise Exception("Model must be fitted first.")

        # Ensure DataFrame shape
        if isinstance(row_df, pd.Series):
            row_df = row_df.to_frame().T
        elif not isinstance(row_df, pd.DataFrame):
            raise TypeError("Expected a pandas Series or DataFrame.")

       
        s_t = row_df[AHRS_COLS].values
        s_tp1 = row_df[NEXT_AHRS_COLS].values
        x_input = np.hstack([s_t, s_tp1])

        preds = np.array(self.model.predict(x_input)).reshape(-1) # shape issue?
        actuals = np.array(row_df[self.metric].values).reshape(-1)
        # print(preds.shape)
        #print(preds)

        # Format predictions
        if self.metric in {'gear', 'trim'}:
            preds = preds.astype(int)
        else:
            preds = np.round(preds, decimals=decimal_places)

        if self.metric in {'gear', 'trim'}:
            actuals = actuals.astype(int)
        else:
            actuals = np.round(actuals, decimals=decimal_places)

        # Return (pred, actual) pairs
        return np.stack((preds, actuals), axis=-1)



# for the state predictor
class MultiHeadModel(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.combo = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.head_vel = nn.Linear(128, 3)
        self.head_acc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.head_ang = nn.Linear(128, 3)
        self.head_omega = nn.Linear(128, 3)

        self.path = "AHRS Models/state_predictor.pt"
        self.X_scaler_path = "AHRS Models/X_scaler.pkl"
        self.y_scaler_path = "AHRS Models/y_scaler.pkl"
        self.X_scaler = None
        self.y_scaler = None  
        self.gear_idx = 12
        self.trim_idx = 14

    def forward(self, x):
        combo = self.combo(x)
        return torch.cat([
            self.head_vel(combo),
            self.head_acc(combo),
            self.head_ang(combo),
            self.head_omega(combo),
        ], dim=1)

    def one_hot_encode_np(self, X):
        assert X.shape[1] == 16
        gear = X[:, self.gear_idx].astype(int)
        trim = X[:, self.trim_idx].astype(int)

        gear_encoded = np.eye(3)[gear]  # shape (N, 3)
        trim_encoded = np.eye(3)[trim]  # shape (N, 3)

        # Remove gear and trim columns
        X_base = np.delete(X, [self.gear_idx, self.trim_idx], axis=1)

        # Concatenate all together
        X_encoded = np.hstack([X_base, gear_encoded, trim_encoded])
        return X_encoded

    def fit(self, X_train, y_train, epochs=20, batch_size=64, lr=1e-3):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # One-hot encode gear and trim
        X_train = self.one_hot_encode_np(X_train)
        if os.path.exists(self.X_scaler_path) and os.path.exists(self.y_scaler_path): 
            print("Loading previously saved scalers...")
            self.X_scaler = joblib.load(self.X_scaler_path)
            self.y_scaler = joblib.load(self.y_scaler_path)
        else:
            print("Fitting and saving new scalers...")
            self.X_scaler = StandardScaler().fit(X_train)
            self.y_scaler = StandardScaler().fit(y_train)
            joblib.dump(self.X_scaler, self.X_scaler_path)
            joblib.dump(self.y_scaler, self.y_scaler_path)

        # Scale input and output
        X_scaled = self.X_scaler.transform(X_train)
        y_scaled = self.y_scaler.transform(y_train)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train or load model
        if not os.path.exists(self.path):
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            self.history = []

            for epoch in tqdm(range(epochs), "Training..."):
                total_loss = 0
                for batch_inputs, batch_targets in loader:
                    optimizer.zero_grad()  # Reset gradients
                    predictions = self(batch_inputs)  
                    loss = loss_fn(predictions, batch_targets) 
                    loss.backward()  
                    optimizer.step()  
                    total_loss += loss.item()  

                avg_loss = total_loss / len(loader)
                self.history.append(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            torch.save(self.state_dict(), self.path)
        else:
            print("Loading previously saved model. Delete AHRS Models to start fresh.")
            self.load_state_dict(torch.load(self.path))

    def predict(self, X):
        self.load_state_dict(torch.load(self.path))
        self.eval()

        if self.X_scaler is None or self.y_scaler is None:
            self.X_scaler = joblib.load(self.X_scaler_path)
            self.y_scaler = joblib.load(self.y_scaler_path)

        # One-hot encode gear and trim
        X = self.one_hot_encode_np(X)
        # Scale
        X_scaled = self.X_scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            preds_scaled = self(X_tensor).numpy()

        preds = self.y_scaler.inverse_transform(preds_scaled)
        return preds