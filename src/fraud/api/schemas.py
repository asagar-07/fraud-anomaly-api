# holds pydantic request and response models
# Single Predict
# Request
#   request_id : Optional[str]
#   features: Dict[str,float]
# 
# Response
#   request_id : (echo)
#   result: my fraud_service.py output
# 
# Batch Predict
# Request
#   request_id : Optional[str]
#   items: List[{item_id?:str, features: Dict[str,float]}]
# 
# Batch Response
#   request_id: (echo)
#   results: List[{ item_id?: str, results?: dict, error?:{type, message}}]
#   summary: {total, succeeded, failed}

from pydantic import BaseModel, ConfigDict
from typing import Union, Dict, List, Optional, Any

#--------------------------------------------
# ValueTypes = Union[float, Dict[str, float], Dict[str, bool], List[str], Dict[str, str]]

class SinglePredictRequest(BaseModel):
    request_id: Optional[str] = None
    features: Dict[str, float]
    model_config = ConfigDict(extra='forbid')


class SinglePredictResponse(BaseModel):
    request_id: Optional[str] = None
    result: Dict[str, Any]
#-----------------------------------------------

class BatchItem(BaseModel):
    item_id: Optional[str] = None
    features: Dict[str, float]
    model_config = ConfigDict(extra='forbid')


class ErrorDetail(BaseModel):
    type: str
    message: str

class ItemResult(BaseModel):
    item_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail]= None

class Summary(BaseModel):
    total: int
    succeeded: int
    failed: int

class BatchPredictRequest(BaseModel):
    request_id: Optional[str] = None
    items: List[BatchItem]
    model_config = ConfigDict(extra='forbid')

class BatchPredictResponse(BaseModel):
    request_id: Optional[str] = None
    results: List[ItemResult]
    summary: Summary
