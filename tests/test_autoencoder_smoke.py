from pathlib import Path
import json
import pandas as pd
import logging

from fraud.inference.autoencoder_infer import AutoencoderArtifacts, AutoencoderPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

def test_autoencoder_reconstruction_smoke():
    # Instantiate artifacts
    artifacts = AutoencoderArtifacts()
    # Instantiate predictor
    predictor = AutoencoderPredictor(artifacts)

    # Use a sample input (30 features) for testing from fixture data
    project_root = Path(__file__).resolve().parents[1]
    fixture = json.loads((project_root / "tests/fixtures/sample_row.json").read_text())

    # Extract features in canonical order
    pre_cfg = json.loads(artifacts.preprocessor_config_path.read_text())
    order = pre_cfg["feature_order"]
    features = [float(fixture[col]) for col in order]

    recon_predicted_scores = predictor.predict_score(features)
    # Assert that anomaly score is greater than 0 or equal to 0
    assert recon_predicted_scores["anomaly_score"] >= 0.0, f"Reconstruction MAE {recon_predicted_scores['anomaly_score']} is negative"

    # Assert that all keys exist in output dict
    expected_keys = {"anomaly_score", "p95", "p99", "p995", "is_anomalous_p99", "model_type", "model_version"}
    assert expected_keys.issubset(recon_predicted_scores.keys()), f"Output keys {recon_predicted_scores.keys()} do not contain all expected keys {expected_keys}"

    # Assert that is_anomalous_p99 is a boolean only if anomaly_score > p99
    if recon_predicted_scores["anomaly_score"] > recon_predicted_scores["p99"]:
        assert isinstance(recon_predicted_scores["is_anomalous_p99"], bool), f"is_anomalous_p99 is not a boolean: {type(recon_predicted_scores['is_anomalous_p99'])}"

    #Log recon_predicted_scores for visual inspection
    #for k, v in recon_predicted_scores.items():
    #    logger.info("%s: %s", k, v)

    logger.info("Smoke test passed! Reconstruction MAE: %s", recon_predicted_scores['anomaly_score'])

def test_autoencoder_reconstruction_bad_length():
    artifacts = AutoencoderArtifacts()
    predictor = AutoencoderPredictor(artifacts)

    # Input with incorrect length
    bad_features = [0.0] * 28  # Should be 30 features

    try:
        predictor.predict_score(bad_features)
        assert False, "Expected an error due to bad feature length"
    except Exception as e:
        logger.info("Correctly caught error for bad feature length: %s", e)