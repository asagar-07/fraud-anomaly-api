# centralize exception mapping ( ValueError -> 422)

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette import status

logger = logging.getLogger(__name__)


def _error_payload(
    error_type: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"error": {"type": error_type, "message": message}}
    if request_id is not None:
        payload["request_id"] = request_id
    if details:
        payload["error"]["details"] = details
    return payload


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """
    Map any business/contract validation errors into HTTP 422.

    FraudScoringService raises ValueError for:
      1. missing keys
      2. extra keys
      3.  non-numeric values
      4. incorrect feature length (if ever used)
    """
    logger.info("Validation error: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_payload("ValidationError", str(exc)),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Map unexpected exceptions into HTTP 500.
    Keep the response generic; log details server-side.
    """
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload("InternalServerError", "Unexpected server error"),
    )