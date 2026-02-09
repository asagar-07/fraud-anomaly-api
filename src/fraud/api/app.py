from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, status, HTTPException, Depends
from fraud.api.deps import get_service
from fraud.api.errors import value_error_handler, unhandled_exception_handler
from contextlib import asynccontextmanager
from fraud.inference.fraud_service import FraudScoringService, FraudServiceConfig
from fraud.api.schemas import SinglePredictRequest, SinglePredictResponse, BatchPredictRequest, BatchPredictResponse, ItemResult, ErrorDetail

@asynccontextmanager
async def lifespan(app: FastAPI):
    #---Startup---
    try:
        repo_root = Path(__file__).resolve().parents[3] 
        cfg = FraudServiceConfig()
        service = FraudScoringService(repo_root=repo_root, cfg=cfg) # construct the Fraud Servce
        app.state.service = service # store service in app.state.service
        app.state.ready = True # set ready state = True
        app.state.startup_error = None
    except Exception as e:
        # Handle initialization failure
        app.state.service = None
        app.state.ready = False
        app.state.startup_error = str(e)
    yield
    app.state.service = None

app = FastAPI(lifespan=lifespan)
app.add_exception_handler(ValueError, value_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

@app.get("/health") # return ok if artifacts are loaded successfully, 200 when ready and 503 when not ready (with the error message)
async def health_check():
    """Readiness + Liveness check: 200 only when artifacts loaded successfully, else 503"""
    if getattr(app.state, "ready", False):
        return {"ready": True,
                "startup_error": None
                }
    err = getattr(app.state, "startup_error", "Unknown startup error")
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Fraud-Scoring Service Application failed to load. Error: {err}")
        
@app.get("/version") # return versions pulled all pre-saved JSON configs
async def get_version(service: FraudScoringService = Depends(get_service)):
    """ Report versions from artifact metadata loaded by the service at startup"""
    preprocessor_ver = getattr(service.cfg, "preprocessor_version", "unknown")
    classifier_cfg = getattr(service, "classifier_model_config", None) or getattr(service, "classifier_model_cfg", {})
    autoencoder_cfg = getattr(service, "autoencoder_model_config", None) or getattr(service, "autencoder_model_cfg", {})
    cls_ver = (classifier_cfg.get("model_version") or classifier_cfg.get("version") or "unknown")
    ae_ver = (autoencoder_cfg.get("model_version") or autoencoder_cfg.get("version") or "unknown")
    cls_type = classifier_cfg.get("model_type") or "classifier"
    ae_type = autoencoder_cfg.get("model_type") or "autoencoder"
    return {
        "preprocessor_version": preprocessor_ver,
        "classifier": {
            "model_type": cls_type,
            "model_version": cls_ver
        },
        "autoencoder": {
            "model_type": ae_type,
            "model_version": ae_ver
        }
    }

@app.post("/predict") # Sinlge request: unified response for singular named feature dict
async def single_predict(input: SinglePredictRequest, service: FraudScoringService = Depends(get_service)) -> SinglePredictResponse:
    # Input features are validated internally and raised exceptions bubbles up and FASTAPI catches and runs value_error_handler
    return {
        "request_id": input.request_id,
        "result": service.score_transaction(input.features)
    }
    
@app.post("/predict:batch") 
# Batch request: list of unified responses (per row), with per-item erros if any captured for list of featured dicts
# Batch guardrails, max_batch_size = 256, reject with 413 or 422 if larger than 256, each item should have features dict
# Missing, extra, non-numeric feature keys are already rejected by my serivce ( ValueError)
async def batch_predict(input: BatchPredictRequest, service: FraudScoringService = Depends(get_service)) -> BatchPredictResponse:
    maximum = len(input.items)
    if maximum > 256:
        raise ValueError(f"Batch size cannot exceed 256, found {maximum} instead!")

    succeeded = 0
    failed = 0
    item_results = []

    for item in input.items:
        try:
            scoring_data = service.score_transaction(item.features)
            item_results.append(ItemResult(
                item_id = item.item_id,
                result = scoring_data, 
                error = None
                )
                )
            succeeded += 1
        except ValueError as e:
            failed += 1
            item_results.append(ItemResult(
                item_id = item.item_id,
                result = None, 
                error=ErrorDetail(type="ValidationError", message=str(e))
                )
                )
            #errors[key] = {"error": { "type": "ValidationError", "message": str(e)}}
    
    return {
        "request_id": input.request_id if input.request_id is not None else None,
        "results": item_results,
        "summary": {"total": len(input.items), "succeeded": succeeded, "failed": failed}
    }
        
