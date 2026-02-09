from pathlib import Path
import json
from fraud.inference.classifier_infer import ClassifierArtifacts, ClassifierPredictor
import pytest
pytestmark = pytest.mark.integration

def test_predict_proba_smoke():
    # Instantiate artifacts
    artifacts = ClassifierArtifacts()
    # Instantiate predictor
    predictor = ClassifierPredictor(artifacts)

    # Use a sample input (30 features) for testing from fixture data
    project_root = Path(__file__).resolve().parents[1]
    fixture = json.loads((project_root / "tests/fixtures/sample_row.json").read_text())

    # Extract features in canonical order
    pre_cfg = json.loads(artifacts.preprocessor_config_path.read_text())
    order = pre_cfg["feature_order"]
    features = [float(fixture[col]) for col in order]

    p = predictor.predict_proba(features)
    # Assert that output is a float
    assert isinstance(p, float), f"Predicted probability is not a float: {type(p)}"
    
    # Assert that probability is between 0 and 1
    assert 0.0 <= p <= 1.0, f"Predicted probability {p} is out of bounds"

    print(f"Smoke test passed! Predicted probability: {p}")

def test_predict_proba_bad_length():
    artifacts = ClassifierArtifacts()
    predictor = ClassifierPredictor(artifacts)

    # Input with incorrect length
    bad_features = [0.0] * 25  # Should be 30 features

    try:
        predictor.predict_proba(bad_features)
        assert False, "Expected an error due to bad feature length"
    except Exception as e:
        print(f"Correctly caught error for bad feature length: {e}")