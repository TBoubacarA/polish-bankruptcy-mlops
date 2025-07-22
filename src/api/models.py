"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class RiskLevel(str, Enum):
    """Risk level enumeration"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionRequest(BaseModel):
    """Request model for bankruptcy prediction"""

    # Financial attributes (attr1 to attr64)
    attr1: float = Field(..., description="Financial attribute 1")
    attr2: float = Field(..., description="Financial attribute 2")
    attr3: float = Field(..., description="Financial attribute 3")
    attr4: float = Field(..., description="Financial attribute 4")
    attr5: float = Field(..., description="Financial attribute 5")
    attr6: float = Field(..., description="Financial attribute 6")
    attr7: float = Field(..., description="Financial attribute 7")
    attr8: float = Field(..., description="Financial attribute 8")
    attr9: float = Field(..., description="Financial attribute 9")
    attr10: float = Field(..., description="Financial attribute 10")
    attr11: float = Field(..., description="Financial attribute 11")
    attr12: float = Field(..., description="Financial attribute 12")
    attr13: float = Field(..., description="Financial attribute 13")
    attr14: float = Field(..., description="Financial attribute 14")
    attr15: float = Field(..., description="Financial attribute 15")
    attr16: float = Field(..., description="Financial attribute 16")
    attr17: float = Field(..., description="Financial attribute 17")
    attr18: float = Field(..., description="Financial attribute 18")
    attr19: float = Field(..., description="Financial attribute 19")
    attr20: float = Field(..., description="Financial attribute 20")
    attr21: float = Field(..., description="Financial attribute 21")
    attr22: float = Field(..., description="Financial attribute 22")
    attr23: float = Field(..., description="Financial attribute 23")
    attr24: float = Field(..., description="Financial attribute 24")
    attr25: float = Field(..., description="Financial attribute 25")
    attr26: float = Field(..., description="Financial attribute 26")
    attr27: float = Field(..., description="Financial attribute 27")
    attr28: float = Field(..., description="Financial attribute 28")
    attr29: float = Field(..., description="Financial attribute 29")
    attr30: float = Field(..., description="Financial attribute 30")
    attr31: float = Field(..., description="Financial attribute 31")
    attr32: float = Field(..., description="Financial attribute 32")
    attr33: float = Field(..., description="Financial attribute 33")
    attr34: float = Field(..., description="Financial attribute 34")
    attr35: float = Field(..., description="Financial attribute 35")
    attr36: float = Field(..., description="Financial attribute 36")
    attr37: float = Field(..., description="Financial attribute 37")
    attr38: float = Field(..., description="Financial attribute 38")
    attr39: float = Field(..., description="Financial attribute 39")
    attr40: float = Field(..., description="Financial attribute 40")
    attr41: float = Field(..., description="Financial attribute 41")
    attr42: float = Field(..., description="Financial attribute 42")
    attr43: float = Field(..., description="Financial attribute 43")
    attr44: float = Field(..., description="Financial attribute 44")
    attr45: float = Field(..., description="Financial attribute 45")
    attr46: float = Field(..., description="Financial attribute 46")
    attr47: float = Field(..., description="Financial attribute 47")
    attr48: float = Field(..., description="Financial attribute 48")
    attr49: float = Field(..., description="Financial attribute 49")
    attr50: float = Field(..., description="Financial attribute 50")
    attr51: float = Field(..., description="Financial attribute 51")
    attr52: float = Field(..., description="Financial attribute 52")
    attr53: float = Field(..., description="Financial attribute 53")
    attr54: float = Field(..., description="Financial attribute 54")
    attr55: float = Field(..., description="Financial attribute 55")
    attr56: float = Field(..., description="Financial attribute 56")
    attr57: float = Field(..., description="Financial attribute 57")
    attr58: float = Field(..., description="Financial attribute 58")
    attr59: float = Field(..., description="Financial attribute 59")
    attr60: float = Field(..., description="Financial attribute 60")
    attr61: float = Field(..., description="Financial attribute 61")
    attr62: float = Field(..., description="Financial attribute 62")
    attr63: float = Field(..., description="Financial attribute 63")
    attr64: float = Field(..., description="Financial attribute 64")

    # Temporal attribute
    years_before_bankruptcy: int = Field(
        ..., description="Years before potential bankruptcy (1-5)", ge=1, le=5
    )

    # Optional company identifier
    company_id: Optional[str] = Field(None, description="Company identifier")

    @validator(
        "attr1",
        "attr2",
        "attr3",
        "attr4",
        "attr5",
        "attr6",
        "attr7",
        "attr8",
        "attr9",
        "attr10",
        "attr11",
        "attr12",
        "attr13",
        "attr14",
        "attr15",
        "attr16",
        "attr17",
        "attr18",
        "attr19",
        "attr20",
        "attr21",
        "attr22",
        "attr23",
        "attr24",
        "attr25",
        "attr26",
        "attr27",
        "attr28",
        "attr29",
        "attr30",
        "attr31",
        "attr32",
        "attr33",
        "attr34",
        "attr35",
        "attr36",
        "attr37",
        "attr38",
        "attr39",
        "attr40",
        "attr41",
        "attr42",
        "attr43",
        "attr44",
        "attr45",
        "attr46",
        "attr47",
        "attr48",
        "attr49",
        "attr50",
        "attr51",
        "attr52",
        "attr53",
        "attr54",
        "attr55",
        "attr56",
        "attr57",
        "attr58",
        "attr59",
        "attr60",
        "attr61",
        "attr62",
        "attr63",
        "attr64",
    )
    def validate_financial_attributes(cls, v):
        """Validate financial attributes are reasonable"""
        if v is None:
            raise ValueError("Financial attribute cannot be None")
        # Replace infinity with large finite values
        if v == float("inf"):
            return 1e10
        if v == float("-inf"):
            return -1e10
        # Check for reasonable bounds
        if abs(v) > 1e12:
            raise ValueError("Financial attribute value too extreme")
        return v

    class Config:
        schema_extra = {
            "example": {
                "attr1": 0.56,
                "attr2": 0.12,
                "attr3": 0.89,
                "attr4": 0.23,
                "attr5": 1.45,
                "attr6": 0.67,
                "attr7": 0.34,
                "attr8": 0.78,
                "attr9": 0.45,
                "attr10": 0.91,
                "attr11": 0.56,
                "attr12": 0.34,
                "attr13": 0.78,
                "attr14": 0.23,
                "attr15": 0.45,
                "attr16": 0.67,
                "attr17": 0.89,
                "attr18": 0.12,
                "attr19": 0.56,
                "attr20": 0.34,
                "attr21": 0.78,
                "attr22": 0.45,
                "attr23": 0.67,
                "attr24": 0.23,
                "attr25": 0.89,
                "attr26": 0.56,
                "attr27": 0.34,
                "attr28": 0.78,
                "attr29": 0.45,
                "attr30": 0.67,
                "attr31": 0.23,
                "attr32": 0.89,
                "attr33": 0.56,
                "attr34": 0.34,
                "attr35": 0.78,
                "attr36": 0.45,
                "attr37": 0.67,
                "attr38": 0.23,
                "attr39": 0.89,
                "attr40": 0.56,
                "attr41": 0.34,
                "attr42": 0.78,
                "attr43": 0.45,
                "attr44": 0.67,
                "attr45": 0.23,
                "attr46": 0.89,
                "attr47": 0.56,
                "attr48": 0.34,
                "attr49": 0.78,
                "attr50": 0.45,
                "attr51": 0.67,
                "attr52": 0.23,
                "attr53": 0.89,
                "attr54": 0.56,
                "attr55": 0.34,
                "attr56": 0.78,
                "attr57": 0.45,
                "attr58": 0.67,
                "attr59": 0.23,
                "attr60": 0.89,
                "attr61": 0.56,
                "attr62": 0.34,
                "attr63": 0.78,
                "attr64": 0.45,
                "years_before_bankruptcy": 2,
                "company_id": "COMPANY_001",
            }
        }


class PredictionResponse(BaseModel):
    """Response model for bankruptcy prediction"""

    prediction: int = Field(..., description="Predicted class (0=healthy, 1=bankrupt)")
    probability: float = Field(..., description="Probability of bankruptcy (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    features_used: int = Field(..., description="Number of features used in prediction")
    company_id: Optional[str] = Field(
        None, description="Company identifier if provided"
    )

    @validator("probability")
    def validate_probability(cls, v):
        """Validate probability is between 0 and 1"""
        if not (0 <= v <= 1):
            raise ValueError("Probability must be between 0 and 1")
        return v

    @validator("prediction")
    def validate_prediction(cls, v):
        """Validate prediction is 0 or 1"""
        if v not in [0, 1]:
            raise ValueError("Prediction must be 0 (healthy) or 1 (bankrupt)")
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch bankruptcy predictions"""

    samples: List[PredictionRequest] = Field(
        ..., description="List of prediction requests"
    )

    @validator("samples")
    def validate_samples(cls, v):
        """Validate batch size"""
        if len(v) == 0:
            raise ValueError("Batch must contain at least one sample")
        if len(v) > 1000:  # Limit batch size for performance
            raise ValueError("Batch size cannot exceed 1000 samples")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch bankruptcy predictions"""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of prediction responses"
    )
    total_samples: int = Field(..., description="Total number of samples processed")
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(..., description="Batch processing timestamp")


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Service status (healthy/degraded/unhealthy)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    mlflow_connection: bool = Field(..., description="MLflow connection status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information"""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Model stage (None/Staging/Production)")
    description: Optional[str] = Field(None, description="Model description")
    creation_timestamp: Optional[int] = Field(
        None, description="Model creation timestamp"
    )
    metrics: Optional[dict] = Field(None, description="Model performance metrics")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
