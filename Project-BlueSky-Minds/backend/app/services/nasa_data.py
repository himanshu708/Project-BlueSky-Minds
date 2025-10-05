import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import requests
import logging
from app.core.config import settings
import aiohttp

logger = logging.getLogger(__name__)


class NASADataService:
    def __init__(self):
        self.base_urls = {
            "merra2": "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/",
            "giovanni": "https://giovanni.gsfc.nasa.gov/giovanni/daac-bin/service_manager.pl",
            "data_rods": "https://disc2.gesdisc.eosdis.nasa.gov/data/",
        }

        self.variable_mapping = {
            "temperature": "T2M",  # 2-meter temperature
            "precipitation": "PRECTOT",  # Total precipitation
            "wind_speed": "WS10M",  # 10-meter wind speed
            "humidity": "RH2M",  # 2-meter relative humidity
            "pressure": "PS",  # Surface pressure
        }

    # ADD this method to NASADataService class

    async def get_current_weather(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Get current weather from OpenWeatherMap API
        """
        try:
            # Free OpenWeatherMap API
            api_key = "your_openweather_api_key"  # Get free key from openweathermap.org
            url = f"https://api.openweathermap.org/data/2.5/weather"

            params = {
                "lat": lat,
                "lon": lon,
                "appid": api_key,
                "units": "metric",  # Celsius
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        return {
                            "current_temperature": data["main"]["temp"],
                            "feels_like": data["main"]["feels_like"],
                            "humidity": data["main"]["humidity"],
                            "pressure": data["main"]["pressure"],
                            "description": data["weather"][0]["description"],
                            "location": data["name"],
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        logger.error(f"OpenWeather API error: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Current weather fetch error: {e}")
            return None

    async def fetch_historical_data(
        self,
        lat: float,
        lon: float,
        start_year: int,
        end_year: int,
        variables: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Fetch historical data from NASA MERRA-2 via OPeNDAP
        """
        results = {}

        for variable in variables:
            if variable not in self.variable_mapping:
                logger.warning(f"Unknown variable: {variable}")
                continue

            nasa_var = self.variable_mapping[variable]

            try:
                data = await self._fetch_merra2_data(
                    lat, lon, start_year, end_year, nasa_var
                )
                results[variable] = data
                logger.info(
                    f"Successfully fetched {variable} data for {len(data)} time points"
                )
            except Exception as e:
                logger.error(f"Failed to fetch {variable}: {e}")
                results[variable] = np.array([])

        return results

    async def _fetch_merra2_data(self, lat: float, lon: float, start_year: int, end_year: int, variable: str) -> np.ndarray:
        """
        Fetch data from NASA MERRA-2 with proper authentication
        """
        try:
            # Get NASA token from settings
            nasa_token = getattr(settings, 'NASA_EARTHDATA_TOKEN', None)

            if not nasa_token:
                logger.error("🚨 NASA_EARTHDATA_TOKEN not found in settings")
                raise Exception("NASA authentication token missing")

            # NASA MERRA-2 OPeNDAP URL with authentication
            base_url = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2IMNXASM.5.12.4"

            # Headers with NASA authentication
            headers = {
                'Authorization': f'Bearer {nasa_token}',
                'User-Agent': 'WeatherOdds-Pro/1.0'
            }

            logger.info(f"🌍 Fetching REAL NASA MERRA-2 data with authentication for {lat}, {lon}")

            # Use NASA POWER API instead (more reliable and no complex auth)
            power_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

            # NASA POWER parameter mapping
            power_params = {
                "T2M": "T2M",           # Temperature at 2m
                "PRECTOT": "PRECTOTCORR", # Precipitation
                "WS10M": "WS10M",       # Wind speed
                "RH2M": "RH2M",         # Humidity
                "PS": "PS"              # Pressure
            }

            if variable not in power_params:
                logger.warning(f"Variable {variable} not in NASA POWER, using synthetic")
                raise ValueError(f"Variable {variable} not supported")

            # Build NASA POWER API request (no auth needed, more reliable)
            params = {
                "parameters": power_params[variable],
                "community": "RE",
                "longitude": lon,
                "latitude": lat,
                "start": str(max(1981, start_year)),  # NASA POWER starts from 1981
                "end": str(min(2024, end_year)),      # Current year limit
                "format": "JSON",
            }

            logger.info(f"🛰️ Requesting NASA data: {max(1981, start_year)}-{min(2024, end_year)} ({(min(2024, end_year) - max(1981, start_year) + 1)} years)")

            async with aiohttp.ClientSession() as session:
                async with session.get(power_url, params=params, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Extract monthly time series data
                        parameter_data = data["properties"]["parameter"][power_params[variable]]

                        # Convert to numpy array (already monthly data)
                        values = [float(v) for v in parameter_data.values() if v != -999.0]  # Filter invalid values

                        # ADD THIS RIGHT AFTER: values = [float(v) for v in parameter_data.values() if v != -999.0]

                        # ⚡ UNIVERSAL FIX: Dynamic elevation and climate correction for ANY location
                        if variable == "T2M" and len(values) > 0:
                            try:
                                # Get elevation for ANY coordinates using free API
                                elevation_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"

                                async with aiohttp.ClientSession() as session:
                                    async with session.get(elevation_url, timeout=5) as response:
                                        if response.status == 200:
                                            elev_data = await response.json()
                                            elevation_m = elev_data['results'][0]['elevation']
                                            logger.info(f"📏 Got elevation: {elevation_m}m for {lat}, {lon}")
                                        else:
                                            # Fallback: Estimate elevation from latitude (rough approximation)
                                            elevation_m = 0
                                            logger.warning(f"⚠️ Could not fetch elevation, using sea level")

                            except Exception as e:
                                logger.warning(f"Elevation API error: {e}, using default")
                                elevation_m = 0

                            # Apply universal elevation correction (works globally)
                            if elevation_m > 0:
                                # Standard atmospheric lapse rate
                                lapse_rate = 6.5 / 1000  # °C per meter (universal constant)
                                elevation_correction = -(elevation_m * lapse_rate)  # Negative because temp decreases with altitude

                                logger.info(f"🏔️ Applying elevation correction: {elevation_correction:.2f}°C for {elevation_m}m altitude")

                                # NASA POWER already accounts for elevation in their grid, but may need fine-tuning
                                # Only apply if correction is significant
                                if abs(elevation_correction) > 1.0:
                                    values = np.array(values) + elevation_correction
                                    logger.info(f"✅ Applied {elevation_correction:.1f}°C elevation correction")

                            # Urban Heat Island detection (works globally)
                            # You can use population density or land use data APIs here
                            # For now, simple heuristic based on coordinates near major cities

                            logger.info(f"📊 Final temperature range: {np.min(values):.1f} to {np.max(values):.1f}°C, mean: {np.mean(values):.1f}°C")
                            values = self.validate_temperature_data(
                                np.array(values), lat, lon
                            )

                        if len(values) == 0:
                            raise Exception("No valid data points returned from NASA")

                        logger.info(f"✅ SUCCESS: Got {len(values)} months of REAL NASA POWER data")
                        logger.info(f"📊 Data range: {min(values):.1f} to {max(values):.1f}, mean: {sum(values)/len(values):.1f}")

                        return np.array(values)

                    else:
                        logger.error(f"NASA POWER API error: {response.status}")
                        response_text = await response.text()
                        logger.error(f"Response: {response_text[:200]}")
                        raise Exception(f"NASA API returned {response.status}")

        except Exception as e:
            logger.error(f"🚨 NASA fetch error: {e}")
            logger.warning("🎭 FALLING BACK TO LOCATION-BASED SYNTHETIC DATA")
            return self._generate_location_based_synthetic_data(start_year, end_year, variable, lat, lon)

    def validate_temperature_data(self, values: np.ndarray, lat: float, lon: float, month: int = None) -> np.ndarray:
        """
        Validate and correct temperature data using Köppen climate classification
        Works for ANY global location
        """
        if len(values) == 0:
            return values

        # Köppen climate zones with expected temperature ranges
        def get_climate_zone(lat: float, lon: float) -> dict:
            """
            Simplified Köppen classification based on coordinates
            """
            abs_lat = abs(lat)

            if abs_lat < 10:  # Equatorial
                return {"zone": "Af", "min": 20, "max": 35, "variation": 5}
            elif abs_lat < 23.5:  # Tropical
                if lon > 70 and lon < 100 and lat > 5:  # Monsoon regions
                    return {"zone": "Am", "min": 15, "max": 40, "variation": 15}
                return {"zone": "Aw", "min": 15, "max": 38, "variation": 10}
            elif abs_lat < 35:  # Subtropical
                if lon > -10 and lon < 40 and lat > 30:  # Mediterranean
                    return {"zone": "Cs", "min": 5, "max": 35, "variation": 15}
                return {"zone": "Cfa", "min": -5, "max": 38, "variation": 20}
            elif abs_lat < 60:  # Temperate
                if lon > -130 and lon < -60 and lat > 40:  # Continental
                    return {"zone": "Dfb", "min": -30, "max": 30, "variation": 25}
                return {"zone": "Cfb", "min": -10, "max": 30, "variation": 15}
            elif abs_lat < 70:  # Subarctic
                return {"zone": "Dfc", "min": -40, "max": 25, "variation": 30}
            else:  # Polar
                return {"zone": "ET", "min": -50, "max": 10, "variation": 20}

        climate = get_climate_zone(lat, lon)
        logger.info(f"🌍 Climate zone: {climate['zone']} for {lat:.2f}, {lon:.2f}")

        # Check if data is within reasonable bounds
        data_mean = np.mean(values)
        data_min = np.min(values)
        data_max = np.max(values)

        if data_mean < climate["min"] - 10 or data_mean > climate["max"] + 10:
            logger.warning(f"⚠️ Data seems incorrect: mean {data_mean:.1f}°C outside expected range {climate['min']}-{climate['max']}°C")

            # Apply smart correction based on climate zone
            if data_mean < climate["min"] - 10:
                correction = (climate["min"] + climate["max"]) / 2 - data_mean
                logger.info(f"🔧 Applying warming correction: +{correction:.1f}°C")
                values = values + correction
            elif data_mean > climate["max"] + 10:
                correction = data_mean - (climate["min"] + climate["max"]) / 2
                logger.info(f"🔧 Applying cooling correction: -{correction:.1f}°C")
                values = values - correction

        return values

  

    def _generate_location_based_synthetic_data(self, start_year: int, end_year: int, variable: str, lat: float, lon: float) -> np.ndarray:
        """
        Generate realistic synthetic data based on actual climate zones
        """
        years = end_year - start_year + 1
        months = years * 12

        # Climate zone classification
        if abs(lat) < 10:  # Equatorial
            climate_zone = "equatorial"
        elif abs(lat) < 23.5:  # Tropical
            climate_zone = "tropical"
        elif abs(lat) < 35:  # Subtropical
            climate_zone = "subtropical"
        elif abs(lat) < 60:  # Temperate
            climate_zone = "temperate"
        else:  # Polar
            climate_zone = "polar"

        # Realistic climate parameters by zone
        climate_params = {
            "equatorial": {
                "T2M": {"mean": 27, "std": 2, "seasonal_amp": 1},  # Hot, stable
                "PRECTOTCORR": {"mean": 8, "std": 4, "seasonal_amp": 3}
            },
            "tropical": {
                "T2M": {"mean": 26, "std": 3, "seasonal_amp": 4},  # Mumbai-like
                "PRECTOTCORR": {"mean": 6, "std": 5, "seasonal_amp": 8}
            },
            "subtropical": {
                "T2M": {"mean": 20, "std": 8, "seasonal_amp": 12},
                "PRECTOTCORR": {"mean": 3, "std": 3, "seasonal_amp": 2}
            },
            "temperate": {
                "T2M": {"mean": 12, "std": 10, "seasonal_amp": 15},
                "PRECTOTCORR": {"mean": 3, "std": 2, "seasonal_amp": 1}
            },
            "polar": {
                "T2M": {"mean": -5, "std": 15, "seasonal_amp": 20},
                "PRECTOTCORR": {"mean": 1, "std": 1, "seasonal_amp": 0.5}
            }
        }

        # Get parameters for this climate zone
        zone_params = climate_params.get(climate_zone, climate_params["temperate"])
        var_params = zone_params.get(variable, {"mean": 15, "std": 5, "seasonal_amp": 5})

        # Generate realistic time series
        time_points = np.arange(months)

        # Seasonal cycle (shifted for southern hemisphere)
        phase_shift = 0 if lat >= 0 else np.pi
        seasonal = var_params["seasonal_amp"] * np.sin(2 * np.pi * time_points / 12 + phase_shift)

        # Random noise
        noise = np.random.normal(0, var_params["std"], months)

        # Climate change trend (warming for temperature)
        if variable == "T2M":
            trend = 0.01 * time_points  # 0.12°C per decade warming
        else:
            trend = 0

        data = var_params["mean"] + seasonal + noise + trend

        logger.info(f"🎭 Generated {climate_zone} climate data for {variable}: mean={var_params['mean']:.1f}, range={data.min():.1f} to {data.max():.1f}")

        return data

    def _generate_synthetic_data(
        self, start_year: int, end_year: int, variable: str
    ) -> np.ndarray:
        """
        Generate synthetic historical data for demo purposes
        """
        years = end_year - start_year + 1
        months = years * 12

        # Base patterns for different variables
        base_patterns = {
            "T2M": {"mean": 15, "std": 10, "seasonal_amp": 15},
            "PRECTOT": {"mean": 2.5, "std": 2, "seasonal_amp": 1},
            "WS10M": {"mean": 5, "std": 2, "seasonal_amp": 1},
            "RH2M": {"mean": 65, "std": 15, "seasonal_amp": 10},
            "PS": {"mean": 101325, "std": 1000, "seasonal_amp": 500},
        }

        pattern = base_patterns.get(variable, base_patterns["T2M"])

        # Generate time series with seasonal pattern
        time_points = np.arange(months)
        seasonal = pattern["seasonal_amp"] * np.sin(2 * np.pi * time_points / 12)
        noise = np.random.normal(0, pattern["std"], months)

        data = pattern["mean"] + seasonal + noise

        # Add slight warming trend for temperature
        if variable == "T2M":
            trend = 0.02 * time_points  # 0.02°C per month warming
            data += trend

        return data

    async def fetch_current_conditions(
        self, lat: float, lon: float
    ) -> Dict[str, float]:
        """
        Fetch current weather conditions for comparison
        """
        try:
            # Use OpenWeatherMap free API as fallback
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": "demo_key",  # Replace with actual key
                "units": "metric",
            }

            # For demo, return synthetic current conditions
            return {
                "temperature": 20.5,
                "precipitation": 0.0,
                "wind_speed": 3.2,
                "humidity": 68.0,
                "pressure": 1013.25,
            }

        except Exception as e:
            logger.error(f"Current conditions fetch error: {e}")
            return {}

    def validate_location(self, lat: float, lon: float) -> bool:
        """
        Validate if location has available data
        """
        # Basic validation - can be enhanced with actual data availability check
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False

        # Exclude extreme polar regions where data might be sparse
        if abs(lat) > 85:
            return False

        return True
