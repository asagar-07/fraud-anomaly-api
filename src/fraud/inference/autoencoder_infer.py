from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from scipy import stats
import torch
import numpy as np
import joblib
from fraud.models.autoencoder import FraudAutoencoder

''' Suppressing Warning explicitly for
    Pytest Run on autoencoder_infer.py, test_autoencoder_smoke.py   

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)
    
'''

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AutoencoderArtifacts:
    model_path: Path = Path("artifacts/autoencoder/model.pt")
    scaler_path: Path = Path("artifacts/shared/scaler.joblib")
    preprocessor_config_path: Path = Path("artifacts/shared/preprocessor_config.json")
    model_config_path: Path = Path("artifacts/autoencoder/model_config.json")
    recon_stats_path: Path = Path("artifacts/autoencoder/recon_stats.json")

class AutoencoderPredictor:
    def __init__(self, artifacts: AutoencoderArtifacts):
        self.artifacts = artifacts

        # Load preprocessing config and scaler
        self.pre_cfg = json.loads(artifacts.preprocessor_config_path.read_text())
        self.feature_order = self.pre_cfg["feature_order"]
        self.amount_transform = self.pre_cfg["amount_transform"]
        self.scaler = joblib.load(artifacts.scaler_path)
        
        # Load model config
        self.model_cfg = json.loads(artifacts.model_config_path.read_text())
        self.input_dim = self.model_cfg["input_dim"]

        # Instantiate and load model weights
        self.device = torch.device("cpu")
        self.model = FraudAutoencoder(input_dim=self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(artifacts.model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Autoencoder model loaded successfully from {artifacts.model_path}.")

        # Canonical index of Amount feature in Feature Vector
        try:
            self.amount_index = self.feature_order.index("Amount")
        except ValueError as e:
            logger.error("Amount feature not found in feature order.")

    def _preprocess_one(self, features: list[float]) -> np.ndarray:
        # Validate input length
        if len(features) != len(self.feature_order):
            raise ValueError(f"Expected {len(self.feature_order)} features, got {len(features)}")
        
        # Reorder features according to feature_order
        feature_array = np.array(features, dtype=np.float64)

        # Apply same transformations as during training
        # Amount has to be non-negative for log1p
        if self.amount_transform == "log1p":
            amount_value = feature_array[self.amount_index]
            if amount_value < 0:
                raise ValueError("Amount feature must be non-negative for log1p transformation")
            feature_array[self.amount_index] = np.log1p(amount_value)

        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))
        return feature_array_scaled.astype(np.float32)

    @torch.no_grad()
    def reconstruct(self, features: list[float]) -> np.ndarray:
        features_preprocessed = self._preprocess_one(features)
        input_tensor = torch.tensor(features_preprocessed, dtype=torch.float32, device=self.device)
        reconstructed_tensor = self.model(input_tensor)
        reconstructed_array = reconstructed_tensor.cpu().numpy().reshape(-1)
        return reconstructed_array
    
    def predict_score(self, features: list[float]) -> dict:
        reconstructed = self.reconstruct(features)
        features_preprocessed = self._preprocess_one(features).reshape(-1)

        # is_anomalous_p99 (boolean)
        is_anomalous_p99 = float(np.mean(np.abs(features_preprocessed - reconstructed))) > float(np.percentile(reconstructed, 99.0))

        predicted_scores = {
            "anomaly_score": float(np.mean(np.abs(features_preprocessed - reconstructed))),
            "p95": float(np.percentile(reconstructed, 95.0)),
            "p99": float(np.percentile(reconstructed, 99.0)),
            "p995": float(np.percentile(reconstructed, 99.5)),
            "is_anomalous_p99": is_anomalous_p99,
            "model_type": "autoencoder",
            "model_version": "1.0.0",
            }

        return predicted_scores
