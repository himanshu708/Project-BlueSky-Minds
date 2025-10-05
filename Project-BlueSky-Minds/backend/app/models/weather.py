from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime


class LocationModel(BaseModel):
    latitude: float = Field(
        ..., ge=-90, le=90, description="Latitude in decimal degrees"
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Longitude in decimal degrees"
    )


class WeatherVariables(BaseModel):
    temperature: bool = True
    precipitation: bool = False
    wind_speed: bool = False
    humidity: bool = False
    pressure: bool = False


class WeatherProbabilityRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    date: date
    variables: List[str] = ["temperature"]
    years_back: Optional[int] = Field(30, ge=5, le=50)
    prediction_mode: str = Field(
        "live_training", description="Prediction mode: 'live_training' or 'pretrained'"
    )


class ProbabilityResult(BaseModel):
    variable: str
    mean: float
    std: float
    percentiles: Dict[str, float]
    extreme_probabilities: Dict[str, float]
    trend_analysis: Dict[str, Any]


class WeatherProbabilityResponse(BaseModel):
    location: LocationModel
    date: date
    results: List[ProbabilityResult]
    metadata: Dict[str, Any]
    generated_at: datetime


class ExtendedForecastRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    months_ahead: int = Field(1, ge=1, le=6)
    variables: List[str] = ["temperature"]
    prediction_mode: str = Field("live_training", description="Prediction mode: 'live_training' or 'pretrained'")


class ForecastResult(BaseModel):
    month: int
    year: int
    variable: str
    predicted_value: float
    confidence_interval: Dict[str, float]
    probability_extremes: Dict[str, float]


class ExtendedForecastResponse(BaseModel):
    location: LocationModel
    forecast_results: List[ForecastResult]
    methodology: str
    confidence_score: float
    generated_at: datetime
