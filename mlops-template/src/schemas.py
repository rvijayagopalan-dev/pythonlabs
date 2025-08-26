from pydantic import BaseModel, Field
from typing import List, Optional


class IrisRecord(BaseModel):
    sepal_length: float = Field(..., description="sepal length (cm)")
    sepal_width: float = Field(..., description="sepal width (cm)")
    petal_length: float = Field(..., description="petal length (cm)")
    petal_width: float = Field(..., description="petal width (cm)")


class PredictRequest(BaseModel):
    records: List[IrisRecord]


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
