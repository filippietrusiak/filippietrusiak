"""
Anomaly detection for industrial sensor streams and inspection data.
Used at Hexagon AB for predictive maintenance and quality control.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


@dataclass
class AnomalyResult:
    scores: np.ndarray       # per-sample anomaly scores
    labels: np.ndarray       # 0 = normal, 1 = anomaly
    threshold: float
    anomaly_rate: float


class Autoencoder(nn.Module):
    """Reconstruction-based anomaly detector using a symmetric autoencoder."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)


class IsolationForestAnomalyDetector:
    """Isolation Forest wrapper for unsupervised anomaly scoring."""

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True

    def predict(self, X: np.ndarray) -> AnomalyResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        X_scaled = self.scaler.transform(X)
        # IsolationForest: -1 = anomaly, 1 = normal
        raw_scores = -self.model.score_samples(X_scaled)  # higher = more anomalous
        labels = (self.model.predict(X_scaled) == -1).astype(int)
        threshold = float(np.percentile(raw_scores, 95))
        return AnomalyResult(
            scores=raw_scores,
            labels=labels,
            threshold=threshold,
            anomaly_rate=float(labels.mean()),
        )


class AutoencoderAnomalyDetector:
    """Deep autoencoder for anomaly detection on high-dimensional sensor data."""

    def __init__(self, input_dim: int, latent_dim: int = 16, threshold_percentile: float = 95.0):
        self.model = Autoencoder(input_dim, latent_dim)
        self.scaler = StandardScaler()
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None

    def fit(self, X: np.ndarray, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64):
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        tensor = torch.FloatTensor(X_scaled)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(tensor))
            for i in range(0, len(tensor), batch_size):
                batch = tensor[perm[i:i + batch_size]]
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Set threshold from training reconstruction errors
        self.model.eval()
        train_errors = self.model.reconstruction_error(tensor).numpy()
        self.threshold = float(np.percentile(train_errors, self.threshold_percentile))

    def predict(self, X: np.ndarray) -> AnomalyResult:
        X_scaled = self.scaler.transform(X).astype(np.float32)
        tensor = torch.FloatTensor(X_scaled)
        self.model.eval()
        scores = self.model.reconstruction_error(tensor).numpy()
        labels = (scores > self.threshold).astype(int)
        return AnomalyResult(
            scores=scores,
            labels=labels,
            threshold=self.threshold,
            anomaly_rate=float(labels.mean()),
        )
