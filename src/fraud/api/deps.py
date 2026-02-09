# get_service() dependency to test
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from fastapi import Request
from fraud.inference.fraud_service import FraudScoringService, FraudServiceConfig

@lru_cache(maxsize=1)
def _build_service() -> FraudScoringService:
    """
    Build and cache the FraudScoringService exactly once per process at startup.
    Why - avoid reloading artifacts/models, not per request
    """
    repo_root = Path(__file__).resolve().parents[3] 
    cfg = FraudServiceConfig()
    return FraudScoringService(repo_root=repo_root, cfg=cfg)


def get_service(request: Request) -> FraudScoringService:
    """
    Dependency to fetch the singleton service.
    If app at startup stores a service in app.state.service, prefer that.
    Else fallback to the lru_cache singleton.
    """
    service = getattr(request.app.state, "service", None)
    if service is not None:
        return service
    return _build_service()

def teardown_service():
    _build_service.cache_clear()