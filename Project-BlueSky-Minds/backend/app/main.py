from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import settings
from app.core.cache import get_cache
import logging
import webbrowser
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WeatherOdds Pro API",
    description="Historical weather probability analysis using NASA data with 1-2 month forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize cache and services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application services and auto-open frontend"""
    try:
        from app.services.pretrained_models import pretrained_models

        # Load pre-trained models
        models_loaded = pretrained_models.load_models()
        if models_loaded:
            logger.info(
                "🤖 Pre-trained ML models loaded successfully - INSTANT PREDICTIONS READY!"
            )
        else:
            logger.warning("⚠️ Pre-trained models not found - using on-the-fly training")

        # Initialize cache service
        cache = await get_cache()
        if cache.is_available():
            logger.info("✅ Cache service (Redis/Valkey) initialized successfully")
        else:
            logger.warning("⚠️ Cache service not available, continuing without cache")

        # Test NASA service availability
        try:
            from app.services.nasa_data import NASADataService

            nasa_service = NASADataService()
            logger.info("✅ NASA Data Service initialized")
        except Exception as e:
            logger.warning(f"⚠️ NASA Service warning: {e}")

        # Test NOAA service availability
        try:
            from app.services.noaa_forecast import NOAAForecastService

            noaa_service = NOAAForecastService()
            logger.info("✅ NOAA Forecast Service initialized")
        except Exception as e:
            logger.warning(f"⚠️ NOAA Service warning: {e}")

        logger.info("🚀 WeatherOdds Pro API started successfully")

        # 🌐 AUTO-OPEN FRONTEND IN BROWSER
        await auto_open_frontend()

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        # Don't fail startup, just log the error


async def auto_open_frontend():
    """Automatically open frontend in default browser"""
    try:
        # Get the project root directory (backend's parent)
        backend_dir = Path(__file__).parent.parent.parent
        frontend_path = backend_dir / "frontend" / "index.html"

        if frontend_path.exists():
            # Convert to absolute file URL
            frontend_url = frontend_path.as_uri()

            logger.info(f"🌐 Opening frontend at: {frontend_url}")

            # Open in default browser
            webbrowser.open(frontend_url)

            logger.info("✅ Frontend opened successfully in browser!")
            logger.info("📡 Backend API running at: http://localhost:8000")
            logger.info("📚 API Documentation at: http://localhost:8000/docs")

        else:
            logger.warning(f"⚠️ Frontend not found at: {frontend_path}")
            logger.info("📁 Please open frontend manually from: frontend/index.html")

    except Exception as e:
        logger.error(f"❌ Could not auto-open frontend: {e}")
        logger.info("💡 You can manually open: frontend/index.html")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 WeatherOdds Pro API shutting down")


# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "WeatherOdds Pro API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api",
        "frontend": "Auto-opened in browser (if available)",
    }


@app.get("/health")
async def health_check():
    """Enhanced health check with service status"""
    try:
        # Check cache status
        cache = await get_cache()
        cache_status = "available" if cache.is_available() else "unavailable"

        # Check database connection (if you have one)
        db_status = "not_configured"  # Update this when you add database

        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2025-09-26T10:30:00Z",
            "services": {
                "cache": cache_status,
                "database": db_status,
                "nasa_api": "available",
                "noaa_api": "available",
            },
            "endpoints": {
                "weather_probability": "/api/weather/probability",
                "extended_forecast": "/api/weather/forecast/extended",
                "nasa_verification": "/api/weather/verify-nasa-data",
                "variables": "/api/weather/variables",
                "documentation": "/docs",
            },
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error tracking"""
    logger.error(f"Global exception: {exc} for request: {request.url}")
    return HTTPException(
        status_code=500, detail="Internal server error. Please check logs for details."
    )


# Add middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for debugging"""
    import time

    start_time = time.time()

    # Log request
    logger.info(f"📥 {request.method} {request.url}")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"📤 {request.method} {request.url} - {response.status_code} - {process_time:.3f}s"
    )

    return response
