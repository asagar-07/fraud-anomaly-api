import pytest
from pydantic import ValidationError

from fraud.api.schemas import (
    SinglePredictRequest,
    BatchItem,
    BatchPredictRequest,
)


def test_single_predict_request_forbids_extra_root_keys():
    payload = {
        "request_id": "r1",
        "features": {"Time": 1.0, "Amount": 2.0},
        "unexpected": "nope",
    }
    with pytest.raises(ValidationError):
        SinglePredictRequest.model_validate(payload)


def test_single_predict_request_allows_missing_request_id():
    payload = {"features": {"Time": 1.0, "Amount": 2.0}}
    req = SinglePredictRequest.model_validate(payload)
    assert req.request_id is None
    assert isinstance(req.features, dict)


def test_features_coerce_int_to_float():
    # Pydantic will typically coerce ints to floats for Dict[str, float]
    payload = {"request_id": "r2", "features": {"Time": 1, "Amount": 2}}
    req = SinglePredictRequest.model_validate(payload)
    assert isinstance(req.features["Time"], float)
    assert isinstance(req.features["Amount"], float)


def test_batch_item_forbids_extra_keys():
    payload = {"item_id": "i1", "features": {"Time": 1.0, "Amount": 2.0}, "extra": 123}
    with pytest.raises(ValidationError):
        BatchItem.model_validate(payload)


def test_batch_item_allows_missing_item_id():
    payload = {"features": {"Time": 1.0, "Amount": 2.0}}
    item = BatchItem.model_validate(payload)
    assert item.item_id is None
    assert isinstance(item.features, dict)


def test_batch_predict_request_forbids_extra_root_keys():
    payload = {
        "request_id": "b1",
        "items": [{"item_id": "i1", "features": {"Time": 1.0, "Amount": 2.0}}],
        "junk": True,
    }
    with pytest.raises(ValidationError):
        BatchPredictRequest.model_validate(payload)