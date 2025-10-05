import requests
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from app.core.config import settings
import aiohttp  # Add this line after your existing imports

logger = logging.getLogger(__name__)


class NOAAForecastService:
    def __init__(self):
        self.base_url = "https://www.cpc.ncep.noaa.gov/products/predictions/"
        self.api_endpoints = {
            "seasonal": "long_range/",
            "monthly": "monthly/",
            "temperature": "temperature/",
            "precipitation": "precipitation/",
        }

    # ADD THIS METHOD after line 25 in your NOAAForecastService class

    async def _fetch_real_noaa_data(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch real NOAA Climate Prediction Center data
        """
        try:
            # Real NOAA CPC API endpoints
            headers = {"User-Agent": "WeatherOdds-Pro/1.0"}

            # Temperature outlook URL
            temp_url = f"https://www.cpc.ncep.noaa.gov/products/predictions/long_range/tools/data/temp_lead1.txt"

            # Precipitation outlook URL
            precip_url = f"https://www.cpc.ncep.noaa.gov/products/predictions/long_range/tools/data/prcp_lead1.txt"

            async with aiohttp.ClientSession() as session:
                # Fetch temperature data
                async with session.get(temp_url, headers=headers) as temp_response:
                    if temp_response.status == 200:
                        temp_data = await temp_response.text()
                        temp_outlook = self._parse_noaa_text_data(
                            temp_data, lat, lon, "temperature"
                        )
                    else:
                        logger.warning(f"NOAA temp API returned {temp_response.status}")
                        temp_outlook = None

                # Fetch precipitation data
                async with session.get(precip_url, headers=headers) as precip_response:
                    if precip_response.status == 200:
                        precip_data = await precip_response.text()
                        precip_outlook = self._parse_noaa_text_data(
                            precip_data, lat, lon, "precipitation"
                        )
                    else:
                        logger.warning(f"NOAA precip API returned {precip_response.status}")
                        precip_outlook = None

            if temp_outlook and precip_outlook:
                logger.info(f"Successfully fetched real NOAA data for {lat}, {lon}")
                return {
                    "temperature_outlook": temp_outlook,
                    "precipitation_outlook": precip_outlook,
                    "data_source": "NOAA CPC Real Data",
                    "confidence": 0.78,
                }
            else:
                logger.warning("Failed to fetch real NOAA data, falling back to simulation")
                return None

        except Exception as e:
            logger.error(f"Real NOAA fetch error: {e}")
            return None

    # ADD THIS METHOD after the above method
    def _parse_noaa_text_data(
        self, data_text: str, lat: float, lon: float, variable: str
    ) -> Dict[str, float]:
        """
        Parse NOAA text data format and extract probabilities for location
        """
        try:
            lines = data_text.strip().split("\n")

            # Find closest grid point (simplified - NOAA uses 2.5° grid)
            grid_lat = round(lat / 2.5) * 2.5
            grid_lon = round(lon / 2.5) * 2.5

            # Parse NOAA probability format
            # Format: LAT LON ABOVE_NORMAL NORMAL BELOW_NORMAL
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    try:
                        file_lat = float(parts[0])
                        file_lon = float(parts[1])

                        # Check if this is our grid point (within tolerance)
                        if (
                            abs(file_lat - grid_lat) <= 1.25
                            and abs(file_lon - grid_lon) <= 1.25
                        ):
                            above_normal = float(parts[2])
                            normal = float(parts[3])
                            below_normal = float(parts[4])

                            logger.info(f"Found NOAA data for grid {grid_lat}, {grid_lon}")
                            return {
                                "above_normal": above_normal,
                                "normal": normal,
                                "below_normal": below_normal,
                            }
                    except (ValueError, IndexError):
                        continue

            # If no exact match, use regional average
            logger.warning(
                f"No exact NOAA grid match for {lat}, {lon}, using regional average"
            )
            return self._get_regional_noaa_average(data_text, lat, lon)

        except Exception as e:
            logger.error(f"NOAA data parsing error: {e}")
            return None

    # ADD THIS METHOD after the above method
    def _get_regional_noaa_average(
        self, data_text: str, lat: float, lon: float
    ) -> Dict[str, float]:
        """
        Get regional average when exact grid point not found
        """
        try:
            lines = data_text.strip().split("\n")
            regional_data = []

            # Collect data from nearby grid points (within 10 degrees)
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    try:
                        file_lat = float(parts[0])
                        file_lon = float(parts[1])

                        if abs(file_lat - lat) <= 10 and abs(file_lon - lon) <= 10:
                            above_normal = float(parts[2])
                            normal = float(parts[3])
                            below_normal = float(parts[4])

                            regional_data.append(
                                {
                                    "above_normal": above_normal,
                                    "normal": normal,
                                    "below_normal": below_normal,
                                }
                            )
                    except (ValueError, IndexError):
                        continue

            if regional_data:
                # Calculate regional average
                avg_above = sum(d["above_normal"] for d in regional_data) / len(
                    regional_data
                )
                avg_normal = sum(d["normal"] for d in regional_data) / len(regional_data)
                avg_below = sum(d["below_normal"] for d in regional_data) / len(
                    regional_data
                )

                logger.info(f"Using regional average from {len(regional_data)} grid points")
                return {
                    "above_normal": avg_above,
                    "normal": avg_normal,
                    "below_normal": avg_below,
                }
            else:
                logger.warning("No regional NOAA data found")
                return None

        except Exception as e:
            logger.error(f"Regional NOAA average error: {e}")
            return None

    # ADD this method after line 140 (after _get_regional_noaa_average method)


    async def _validate_noaa_response(self, outlook_data: Dict[str, float]) -> bool:
        """Validate NOAA response data quality"""
        try:
            # Check if probabilities sum to ~100%
            total = (
                outlook_data.get("above_normal", 0)
                + outlook_data.get("normal", 0)
                + outlook_data.get("below_normal", 0)
            )

            if 95 <= total <= 105:  # Allow 5% tolerance
                logger.info(f"NOAA data validation passed: total={total}%")
                return True
            else:
                logger.warning(
                    f"NOAA data validation failed: probabilities sum to {total}%"
                )
                return False

        except Exception as e:
            logger.error(f"NOAA validation error: {e}")
            return False


    async def get_seasonal_outlook(
        self, lat: float, lon: float, months_ahead: int = 3
    ) -> Dict[str, Any]:
        """
        Get NOAA seasonal climate outlook for extended forecasting
        """
        try:
            # Try to get real NOAA data first
            real_noaa_data = await self._fetch_real_noaa_data(lat, lon)

            if real_noaa_data:
                logger.info("Using real NOAA CPC data")
                return real_noaa_data
            else:
                logger.warning("Real NOAA data unavailable, using simulation")
                # Fall back to your existing simulation
                outlook = self._simulate_seasonal_outlook(lat, lon, months_ahead)

                return {
                    "temperature_outlook": outlook["temperature"],
                    "precipitation_outlook": outlook["precipitation"],
                    "confidence": outlook["confidence"] * 0.7,  # Lower confidence for simulated
                    "valid_period": outlook["valid_period"],
                    "source": "Simulated (NOAA unavailable)",
                }

        except Exception as e:
            logger.error(f"NOAA forecast error: {e}")
            return self._get_default_outlook()

    def _simulate_seasonal_outlook(
        self, lat: float, lon: float, months_ahead: int
    ) -> Dict[str, Any]:
        """
        Simulate NOAA seasonal outlook based on location and time
        """
        # Determine season and regional patterns
        current_month = datetime.now().month
        target_months = [
            (current_month + i - 1) % 12 + 1 for i in range(1, months_ahead + 1)
        ]

        # Regional climate patterns (simplified)
        climate_regions = self._get_climate_region(lat, lon)

        # Generate outlook based on typical patterns
        temperature_outlook = self._generate_temperature_outlook(
            lat, target_months, climate_regions
        )
        precipitation_outlook = self._generate_precipitation_outlook(
            lat, target_months, climate_regions
        )

        return {
            "temperature": temperature_outlook,
            "precipitation": precipitation_outlook,
            "confidence": self._calculate_outlook_confidence(lat, months_ahead),
            "valid_period": {
                "start_month": target_months[0],
                "end_month": target_months[-1],
                "year": datetime.now().year,
            },
        }

    def _get_climate_region(self, lat: float, lon: float) -> str:
        """
        Determine climate region based on coordinates
        """
        if lat > 50:
            return "arctic"
        elif lat > 35:
            if lon < -100:
                return "continental"
            else:
                return "temperate"
        elif lat > 23.5:
            if lon < -80:
                return "subtropical"
            else:
                return "mediterranean"
        else:
            return "tropical"

    def _generate_temperature_outlook(
        self, lat: float, target_months: List[int], climate_region: str
    ) -> Dict[str, float]:
        """
        Generate temperature outlook probabilities
        """
        # Base probabilities (equal chances)
        base_prob = 33.33

        # Seasonal adjustments
        seasonal_bias = self._get_seasonal_temperature_bias(lat, target_months)

        # Climate change adjustment (warming bias)
        warming_bias = 5.0 if abs(lat) > 60 else 3.0

        above_normal = base_prob + seasonal_bias + warming_bias
        below_normal = base_prob - seasonal_bias - warming_bias
        normal = 100 - above_normal - below_normal

        # Ensure probabilities are valid
        above_normal = max(20, min(60, above_normal))
        below_normal = max(20, min(60, below_normal))
        normal = 100 - above_normal - below_normal

        return {
            "above_normal": round(above_normal, 1),
            "normal": round(normal, 1),
            "below_normal": round(below_normal, 1),
        }

    def _generate_precipitation_outlook(
        self, lat: float, target_months: List[int], climate_region: str
    ) -> Dict[str, float]:
        """
        Generate precipitation outlook probabilities
        """
        base_prob = 33.33

        # Seasonal precipitation patterns
        seasonal_bias = self._get_seasonal_precipitation_bias(lat, target_months)

        above_normal = base_prob + seasonal_bias
        below_normal = base_prob - seasonal_bias
        normal = 100 - above_normal - below_normal

        # Ensure valid probabilities
        above_normal = max(25, min(50, above_normal))
        below_normal = max(25, min(50, below_normal))
        normal = 100 - above_normal - below_normal

        return {
            "above_normal": round(above_normal, 1),
            "normal": round(normal, 1),
            "below_normal": round(below_normal, 1),
        }

    def _get_seasonal_temperature_bias(
        self, lat: float, target_months: List[int]
    ) -> float:
        """
        Calculate seasonal temperature bias
        """
        # Northern hemisphere seasonal patterns
        if lat > 0:
            winter_months = [12, 1, 2]
            summer_months = [6, 7, 8]

            if any(month in winter_months for month in target_months):
                return -2.0  # Slight cold bias in winter
            elif any(month in summer_months for month in target_months):
                return 3.0  # Warm bias in summer
        else:
            # Southern hemisphere (opposite seasons)
            winter_months = [6, 7, 8]
            summer_months = [12, 1, 2]

            if any(month in winter_months for month in target_months):
                return -2.0
            elif any(month in summer_months for month in target_months):
                return 3.0

        return 0.0

    def _get_seasonal_precipitation_bias(
        self, lat: float, target_months: List[int]
    ) -> float:
        """
        Calculate seasonal precipitation bias
        """
        # Simplified seasonal precipitation patterns
        if 30 <= abs(lat) <= 60:  # Mid-latitudes
            wet_months = [10, 11, 12, 1, 2, 3]  # Winter wet season
            if any(month in wet_months for month in target_months):
                return 5.0
            else:
                return -3.0
        elif abs(lat) < 30:  # Tropics/subtropics
            # Monsoon patterns (simplified)
            monsoon_months = [6, 7, 8, 9]
            if any(month in monsoon_months for month in target_months):
                return 8.0
            else:
                return -5.0

        return 0.0

    def _calculate_outlook_confidence(self, lat: float, months_ahead: int) -> float:
        """
        Calculate confidence score for the outlook
        """
        # Confidence decreases with forecast lead time
        base_confidence = 0.7
        time_decay = 0.1 * (months_ahead - 1)

        # Confidence varies by latitude (more predictable in tropics)
        if abs(lat) < 23.5:  # Tropics
            latitude_factor = 0.1
        elif abs(lat) < 60:  # Mid-latitudes
            latitude_factor = 0.0
        else:  # Polar regions
            latitude_factor = -0.1

        confidence = base_confidence - time_decay + latitude_factor
        return max(0.3, min(0.9, confidence))

    def _get_default_outlook(self) -> Dict[str, Any]:
        """
        Return default outlook when data is unavailable
        """
        return {
            'temperature_outlook': {
                'above_normal': 33.3,
                'normal': 33.3,
                'below_normal': 33.3
            },
            'precipitation_outlook': {
                'above_normal': 33.3,
                'normal': 33.3,
                'below_normal': 33.3
            },
            'confidence': 0.5,
            'valid_period': {
                'start_month': datetime.now().month,
                'end_month': (datetime.now().month + 2) % 12 + 1,
                'year': datetime.now().year
            },
            'source': 'Default climatology'
        }

    async def get_enso_status(self) -> Dict[str, Any]:
        """
        Get current El Niño/La Niña status for improved forecasting
        """
        try:
            # In production, fetch from NOAA ENSO data
            # For demo, simulate current ENSO conditions

            return {
                'current_phase': 'neutral',  # neutral, el_nino, la_nina
                'strength': 'weak',          # weak, moderate, strong
                'oni_index': 0.2,           # Oceanic Niño Index
                'forecast': 'continuing',    # continuing, developing, weakening
                'impact_regions': ['pacific_northwest', 'southeast', 'southwest']
            }

        except Exception as e:
            logger.error(f"ENSO status error: {e}")
            return {'current_phase': 'neutral', 'strength': 'weak'}
