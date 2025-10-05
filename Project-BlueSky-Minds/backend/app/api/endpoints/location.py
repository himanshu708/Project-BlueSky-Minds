from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import requests
import logging
from app.models.weather import LocationModel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/geocode")
async def geocode_location(
    place: str = Query(..., description="Place name to geocode"),
    limit: int = Query(5, ge=1, le=10, description="Maximum number of results"),
) -> List[Dict[str, Any]]:
    """
    Geocode a place name to coordinates using free geocoding service
    """
    try:
        # Using Nominatim (OpenStreetMap) - completely free
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place, "format": "json", "limit": limit, "addressdetails": 1}

        headers = {"User-Agent": "WeatherOdds-Pro/1.0"}  # Required by Nominatim

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        results = response.json()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "display_name": result.get("display_name", ""),
                    "latitude": float(result.get("lat", 0)),
                    "longitude": float(result.get("lon", 0)),
                    "type": result.get("type", ""),
                    "importance": float(result.get("importance", 0)),
                    "country": result.get("address", {}).get("country", ""),
                    "state": result.get("address", {}).get("state", ""),
                    "city": result.get("address", {}).get("city", ""),
                }
            )

        return formatted_results

    except requests.RequestException as e:
        logger.error(f"Geocoding request failed: {e}")
        raise HTTPException(status_code=503, detail="Geocoding service unavailable")
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        raise HTTPException(status_code=500, detail="Geocoding failed")


@router.get("/reverse")
async def reverse_geocode(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> Dict[str, Any]:
    """
    Reverse geocode coordinates to place name
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": latitude,
            "lon": longitude,
            "format": "json",
            "addressdetails": 1,
        }

        headers = {"User-Agent": "WeatherOdds-Pro/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        result = response.json()

        if "error" in result:
            raise HTTPException(status_code=404, detail="Location not found")

        address = result.get("address", {})

        return {
            "display_name": result.get("display_name", ""),
            "latitude": latitude,
            "longitude": longitude,
            "country": address.get("country", ""),
            "state": address.get("state", ""),
            "city": address.get("city", ""),
            "county": address.get("county", ""),
            "postcode": address.get("postcode", ""),
        }

    except requests.RequestException as e:
        logger.error(f"Reverse geocoding failed: {e}")
        raise HTTPException(
            status_code=503, detail="Reverse geocoding service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reverse geocoding error: {e}")
        raise HTTPException(status_code=500, detail="Reverse geocoding failed")


@router.get("/validate")
async def validate_coordinates(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> Dict[str, Any]:
    """
    Validate if coordinates have available weather data
    """
    try:
        from app.services.nasa_data import NASADataService

        nasa_service = NASADataService()
        is_valid = nasa_service.validate_location(latitude, longitude)

        # Get location info
        location_info = await reverse_geocode(latitude, longitude)

        return {
            "valid": is_valid,
            "latitude": latitude,
            "longitude": longitude,
            "location_info": location_info if is_valid else None,
            "reason": (
                "Valid location"
                if is_valid
                else "No weather data available for this location"
            ),
        }

    except Exception as e:
        logger.error(f"Coordinate validation error: {e}")
        return {
            "valid": False,
            "latitude": latitude,
            "longitude": longitude,
            "location_info": None,
            "reason": "Validation failed",
        }
