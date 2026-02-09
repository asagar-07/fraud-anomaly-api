from pathlib import Path
import json
import logging
from fraud.inference.fraud_service import FraudScoringService
import pytest
pytestmark = pytest.mark.integration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_fraud_scoring_service_smoke():
    # Instantiate the fraud scoring service
    fraud_service = FraudScoringService()

    # Use a sample input (30 features) for testing from fixture data
    project_root = Path(__file__).resolve().parents[1]
    fixture = json.loads((project_root / "tests/fixtures/sample_row.json").read_text())

    # Score the transaction
    response = fraud_service.score_transaction(fixture)

    # Log response for visual inspection
    for k, v in response.items():
        logger.info("%s: %s", k, v)

    # Assert response returns floats, bools, and dicts and keys as expected
    assert isinstance(response, dict), f"Response is not a dict: {type(response)}"
    assert isinstance(response.get("fraud_probability"), float), f"fraud_probability is not a float: {type(response.get('fraud_probability'))}"
    assert isinstance(response.get("anomaly_score"), float), f"anomaly_score is not a float: {type(response.get('anomaly_score'))}"
    
    assert isinstance(response.get("thresholds").get("classifier_threshold"), float), f"classifier_threshold is not a float: {type(response.get('thresholds').get('classifier_threshold'))}"
    assert isinstance(response.get("thresholds").get("anomaly_p99_threshold"), float), f"anomaly_p99_threshold is not a float: {type(response.get('thresholds').get('anomaly_p99_threshold'))}"
    
    assert isinstance(response.get("flags").get("is_fraud_by_classifier"), bool), f"is_fraud_by_classifier is not a bool: {type(response.get('flags').get('is_fraud_by_classifier'))}"
    assert isinstance(response.get("flags").get("is_anomalous_p99"), bool), f"is_anomalous_p99 is not a bool: {type(response.get('flags').get('is_anomalous_p99'))}"
    assert isinstance(response.get("flags").get("decision"), bool), f"decision is not a bool: {type(response.get('flags').get('decision'))}"
    
    assert isinstance(response.get("decision_reasons"), list), f"decision_reasons is not a list: {type(response.get('decision_reasons'))}"
    assert isinstance(response.get("model_info"), dict), f"model_info is not a dict: {type(response.get('model_info'))}"
    assert isinstance(response.get("timing_ms"), dict), f"timing_ms is not a dict: {type(response.get('timing_ms'))}"

    # Assert that response contains expected keys
    expected_keys = {
        "fraud_probability",
        "anomaly_score",
        "thresholds",
        "flags",
        "decision_reasons",
        "model_info",
        "timing_ms",
    }
    assert expected_keys.issubset(response.keys()), f"Response keys {response.keys()} do not contain all expected keys {expected_keys}"

    # Assert that fraud_probability is between 0 and 1
    assert 0.0 <= response["fraud_probability"] <= 1.0, f"Fraud probability {response['fraud_probability']} is out of bounds"

    # Assert that anomaly_score is non-negative
    assert response["anomaly_score"] >= 0.0, f"Anomaly score {response['anomaly_score']} is negative"

    # Raise ValueError that missing features are handled gracefully
    bad_fixture_missing = fixture.copy()
    bad_fixture_missing.pop("V28")  # Remove one feature
    with pytest.raises(ValueError) as e:
        fraud_service.score_transaction(bad_fixture_missing)
    assert "missing" in str(e.value).lower()

    # Raise ValueError for extra features
    bad_fixture_extra = fixture.copy()
    bad_fixture_extra["foo"] = 1  # Add an extra feature
    with pytest.raises(ValueError) as e:
        fraud_service.score_transaction(bad_fixture_extra)
    assert "unexpected" in str(e.value).lower() or "extra" in str(e.value).lower()

    # Raise ValueError for non-numeric feature values
    bad_fixture_nonnumeric = fixture.copy()
    bad_fixture_nonnumeric["Amount"] = "abc"  # Set one feature to non-numeric
    with pytest.raises(ValueError) as e:
        fraud_service.score_transaction(bad_fixture_nonnumeric)
    assert "non-numeric" in str(e.value).lower()

    logger.info("Smoke test passed! Fraud Probability: %s", response["fraud_probability"])
    