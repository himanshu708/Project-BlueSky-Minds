from fastapi import APIRouter
from app.api.endpoints import weather, location, export

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(weather.router, prefix="/weather", tags=["weather"])

api_router.include_router(location.router, prefix="/location", tags=["location"])

api_router.include_router(export.router, prefix="/export", tags=["export"])


# Health check endpoint
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "WeatherOdds Pro API", "version": "1.0.0"}
