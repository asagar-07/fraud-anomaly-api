# Inference script for the fraud detection classifier model.
# module exposes a function predict_proba(features: List[float]) -> float
# features: List[float] - list of 30 numeric features in canonical order
# returns: float - predicted probability of fraud (between 0 and 1)
# Apply same preprocessing as used during training before calling this function.
# Load model weights from artifacts/classifier/model.pth
# Load output probability (sigmoid) 

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import torch
import numpy as np
import joblib
from fraud.models.classifier_mlp import FraudMLP

''' Suppressing Warning explicitly for
    Pytest Run on classifier_infer.py, test_classifier_smoke.py 

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
class ClassifierArtifacts:
    model_path: Path = Path("artifacts/classifier/model.pt")
    scaler_path: Path = Path("artifacts/shared/scaler.joblib")
    preprocessor_config_path: Path = Path("artifacts/shared/preprocessor_config.json")
    model_config_path: Path = Path("artifacts/classifier/model_config.json")

class ClassifierPredictor:
    def __init__(self, artifacts: ClassifierArtifacts):
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
        self.model = FraudMLP(input_dim=self.input_dim).to(self.device)
        self.model.load_state_dict(torch.load(artifacts.model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {artifacts.model_path}")

        # Canonical index for Amount in feature Vector
        try:
            self.amount_index = self.feature_order.index("Amount")
        except ValueError as e:
            raise ValueError("Amount feature not found in feature_order") from e
        
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

        # Scale features using loaded scaler
        feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))
        return feature_array_scaled.astype (np.float32)

    @torch.no_grad()
    def predict_proba(self, features: list[float]) -> float:
        features_preprocessed = self._preprocess_one(features)
        input_tensor = torch.tensor(features_preprocessed, dtype=torch.float32, device=self.device)
        logits = self.model(input_tensor).reshape(-1)
        prob = torch.sigmoid(logits).item()
        return float(prob)


