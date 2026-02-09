from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from fraud.inference.classifier_infer import ClassifierPredictor, ClassifierArtifacts
from fraud.inference.autoencoder_infer import AutoencoderPredictor, AutoencoderArtifacts

# creating one fraud inference service module
# used by API to get fraud predictions
# function signature: score_transaction(input_data: dict) -> dict
# score_transaction validates input schema (all 30 keys must be present, values are numeric, no extra keys: reject if more or less than 30 keys)
# score_transaction loads preprocessor_config.json for feature_order
# score_transaction runs both classifier_predictor and autoencoder's predict_score 
""" apply below decision logic 
	1.	classifier_threshold = 0.5 (hardcoded for now, tuned later)
	2.	is_fraud_by_classifier = fraud_probability >= classifier_threshold
	3.	is_anomalous_p99 = ae_out["is_anomalous_p99"]
	4.	decision = is_fraud_by_classifier or is_anomalous_p99
	5.	decision_reasons = []
	6.	add "CLASSIFIER_THRESHOLD" if classifier fired
	7.	add "ANOMALY_P99" if AE fired
"""
# return response dict with keys: fraud_probability (float between 0 and 1), is_fraud (boolean), decision_reasons (list of strings)# Example response:
"""{
    "fraud_probability": 0.8123,
    "anomaly_score ": 0.2891,
    "thresholds": {
        "classifier_threshold": 0.5,
        "anomaly_p99_threshold": 1.4575
    },
    "flags": {
        "is_fraud_by_classifier": true,
        "is_anomalous_p99": false,
        "decision": true
    },
    "decision_reasons": ["CLASSIFIER_THRESHOLD"],
    "model_info": {
        "preprocessor_version": "shared_v1",
        "classifier_version": "v1.0.0",
        "autoencoder_version": "v1.0.0"
    },
    "timing_ms: {
        "preprocessing_time": 1.2,
        "classifier_inference_time": 0.4,
        "autoencoder_inference_time": 0.6,
        "total_inference_time": 2.2
        }
    }

"""    

@dataclass(frozen=True)
class FraudServiceConfig:
    classifier_threshold: float = 0.5
    preprocessor_version: str = "shared_v1"

class FraudScoringService:
    ''' Unified fraud scoring service combining classifier and autoencoder predictions
        strict schema validation (missing/extra keys => ValueError)
        canonical feature ordering from preprocessor config
        decision = classifier_flag OR anomaly_p99_flag
    '''
    def __init__(self, repo_root: Path | None = None, cfg: FraudServiceConfig | None = None):
        self.cfg = cfg or FraudServiceConfig()
        # resolve repo root
        # src/fraud/inference/fraud_service.py => repo_root is 2 levels up
        self.repo_root = repo_root or Path(__file__).resolve().parents[3].resolve()

        # Build artifacts with path objects
        cls_artifacts = ClassifierArtifacts(
            model_path=self.repo_root/"artifacts/classifier/model.pt",
            scaler_path=self.repo_root/"artifacts/shared/scaler.joblib",
            preprocessor_config_path=self.repo_root/"artifacts/shared/preprocessor_config.json",
            model_config_path=self.repo_root/"artifacts/classifier/model_config.json",
        )

        ae_artifacts = AutoencoderArtifacts(
            model_path=self.repo_root/"artifacts/autoencoder/model.pt",
            scaler_path=self.repo_root/"artifacts/shared/scaler.joblib",
            preprocessor_config_path=self.repo_root/"artifacts/shared/preprocessor_config.json",
            model_config_path=self.repo_root/"artifacts/autoencoder/model_config.json",
            recon_stats_path=self.repo_root/"artifacts/autoencoder/recon_stats.json",
        )

        # Load feature order from preprocessor config for canonical ordering
        pre_cfg = json.loads(Path(cls_artifacts.preprocessor_config_path).read_text())
        self.feature_order: List[str] = list(pre_cfg["feature_order"])

        # Epected keys set
        self.expected_keys = set(self.feature_order)

        # Load model configs only once
        self.classifier_model_config = json.loads(Path(cls_artifacts.model_config_path).read_text())
        self.autoencoder_model_config = json.loads(Path(ae_artifacts.model_config_path).read_text())

        # Initialize classifier and autoencoder predictors
        self.classifier_predictor = ClassifierPredictor(cls_artifacts)
        self.autoencoder_predictor = AutoencoderPredictor(ae_artifacts)

    def _validate_features(self, features: Dict[str, Any]) -> None: 
        if not isinstance(features, dict):
            raise ValueError("Input features must be a dictionary of named numeric fields")
        
        actual = set(features.keys())
        missing = sorted(self.expected_keys - actual)
        extra = sorted(actual - self.expected_keys)

        if missing:
            raise ValueError(f"Missing required feature keys: {missing}")
        if extra:
            raise ValueError(f"Unexpected extra feature keys: {extra}")
        
        # Validate numeric values all-around ( accept int/float, reject bool)
        for k in self.feature_order:
            v = features[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"Non-numeric value found for '{k}': {v!r} (type={type(v).__name__})")

    def score_transaction(self, features: Dict[str, Any]) -> Dict[str,Any]:
        t0 = time.perf_counter()
        
        # Validate
        self._validate_features(features)

        # Canonical ordering (caller provides dict; we enforce order)
        t_pre0 = time.perf_counter()
        feature_list = [float(features[k]) for k in self.feature_order]
        t_pre1 = time.perf_counter()

        # Classifier
        t_c0 = time.perf_counter()
        fraud_probability = float(self.classifier_predictor.predict_proba(feature_list))
        t_c1 = time.perf_counter()

        # Autoencoder
        t_ae0 = time.perf_counter()
        ae_out = self.autoencoder_predictor.predict_score(feature_list)
        t_ae1 = time.perf_counter()

        anomaly_score = float(ae_out["anomaly_score"])
        is_anomalous_p99 = bool(ae_out["is_anomalous_p99"])
        ae_p99 = float(ae_out["p99"])

        # Decision policy
        t_cls = float(self.cfg.classifier_threshold)
        is_fraud_by_classifier = fraud_probability >= t_cls
        decision = bool(is_fraud_by_classifier or is_anomalous_p99)

        decision_reasons: List[str] = []
        if is_fraud_by_classifier:
            decision_reasons.append("CLASSIFIER_THRESHOLD")
        if is_anomalous_p99:
            decision_reasons.append("ANOMALY_P99")

        t1 = time.perf_counter()    

        # Versions (prefer config, fall back to predictor output is present)
        cls_ver = (
            self.classifier_model_config.get("model_version")
            or self.classifier_model_config.get("version")
            or "1.0.0"
        )

        ae_ver = (
            self.autoencoder_model_config.get("model_version")
            or self.autoencoder_model_config.get("version")
            or ae_out.get("model_version")
            or "1.0.0"
        )

        # Construct response
        response = {
            "fraud_probability": fraud_probability,
            "anomaly_score": anomaly_score,
            "thresholds": {
                "classifier_threshold": t_cls,
                "anomaly_p99_threshold": ae_p99,
            },
            "flags": {
                "is_fraud_by_classifier": bool(is_fraud_by_classifier),
                "is_anomalous_p99": bool(is_anomalous_p99),
                "decision": bool(decision),
            },
            "decision_reasons": decision_reasons,
            "model_info": {
                "preprocessor_version": self.cfg.preprocessor_version,
                "classifier_version": str(cls_ver),
                "autoencoder_version": str(ae_ver),
            },
            "timing_ms": {
                "preprocessing_time": round((t_pre1 - t_pre0) * 1000.0, 3),
                "classifier_inference_time": round((t_c1 - t_c0) * 1000.0, 3),
                "autoencoder_inference_time": round((t_ae1 - t_ae0) * 1000.0, 3),
                "total_inference_time": round((t1 - t0) * 1000.0, 3),
            },
        }   

        return response
    
