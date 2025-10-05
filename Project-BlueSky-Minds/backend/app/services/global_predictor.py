# Enhanced pretrained models: app/services/global_predictor.py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsRegressor
import logging

logger = logging.getLogger(__name__)


class GlobalClimatePredictor:
    def __init__(self):
        self.training_locations = []
        self.climate_zones = {}
        self.zone_models = {}
        self.interpolation_model = None

    def load_global_models(self, model_dir="trained_models"):
        """Load global models with location interpolation"""
        try:
            # Load your existing pre-trained models
            self.load_base_models(model_dir)

            # Load training location metadata
            self.training_locations = self._load_training_locations(model_dir)

            # Create interpolation model for unknown locations
            self._create_interpolation_model()

            logger.info(
                f"🌍 Global predictor ready for {len(self.training_locations)} locations"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load global models: {e}")
            return False

    def predict_any_location(
        self, lat: float, lon: float, target_month: int, historical_context: list = None
    ) -> dict:
        """Predict for ANY location on Earth"""
        try:
            # Step 1: Check if we have exact training data for this location
            exact_match = self._find_exact_location(lat, lon)
            if exact_match:
                logger.info(f"📍 Exact match found for {lat}, {lon}")
                return self._predict_exact_location(
                    exact_match, target_month, historical_context
                )

            # Step 2: Find nearest trained locations
            nearest_locations = self._find_nearest_locations(lat, lon, k=5)

            # Step 3: Use climate zone similarity
            target_climate_zone = self._classify_climate_zone(lat, lon)
            zone_locations = self._find_zone_locations(target_climate_zone)

            # Step 4: Intelligent interpolation
            prediction = self._interpolate_prediction(
                lat,
                lon,
                target_month,
                nearest_locations,
                zone_locations,
                historical_context,
            )

            logger.info(
                f"🎯 Interpolated prediction for {lat}, {lon} using {len(nearest_locations)} nearby locations"
            )
            return prediction

        except Exception as e:
            logger.error(f"Global prediction error: {e}")
            return self._fallback_prediction(lat, lon, target_month)

    def _find_nearest_locations(self, lat: float, lon: float, k: int = 5):
        """Find k nearest training locations"""
        if not self.training_locations:
            return []

        # Calculate distances to all training locations
        target_point = np.array([[lat, lon]])
        training_points = np.array(
            [[loc["lat"], loc["lon"]] for loc in self.training_locations]
        )

        # Use haversine distance for better accuracy
        distances = self._haversine_distance(target_point, training_points)

        # Get k nearest indices
        nearest_indices = np.argsort(distances.flatten())[:k]

        nearest_locations = []
        for idx in nearest_indices:
            location = self.training_locations[idx].copy()
            location["distance_km"] = distances.flatten()[idx]
            nearest_locations.append(location)

        return nearest_locations

    def _haversine_distance(self, point1, points2):
        """Calculate haversine distance between points"""
        lat1, lon1 = np.radians(point1[0])
        lat2, lon2 = np.radians(points2).T

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth's radius in kilometers
        r = 6371
        return r * c

    def _interpolate_prediction(
        self,
        lat,
        lon,
        target_month,
        nearest_locations,
        zone_locations,
        historical_context,
    ):
        """Intelligent interpolation using multiple methods"""

        predictions = []
        weights = []

        # Method 1: Distance-weighted interpolation
        for location in nearest_locations:
            if location["distance_km"] < 1000:  # Within 1000km
                # Get prediction for this trained location
                pred = self._get_location_prediction(
                    location, target_month, historical_context
                )

                # Weight by inverse distance
                weight = 1.0 / (
                    location["distance_km"] + 1
                )  # +1 to avoid division by zero

                predictions.append(pred)
                weights.append(weight)

        # Method 2: Climate zone average
        if zone_locations:
            zone_predictions = []
            for location in zone_locations[:3]:  # Top 3 from same climate zone
                pred = self._get_location_prediction(
                    location, target_month, historical_context
                )
                zone_predictions.append(pred)

            if zone_predictions:
                zone_avg = np.mean(zone_predictions)
                predictions.append(zone_avg)
                weights.append(0.3)  # Climate zone gets 30% weight

        # Method 3: Latitude-based climatology
        latitude_pred = self._latitude_based_prediction(lat, target_month)
        predictions.append(latitude_pred)
        weights.append(0.2)  # Latitude climatology gets 20% weight

        # Calculate weighted ensemble
        if predictions and weights:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights

            ensemble_prediction = np.average(predictions, weights=weights)

            # Calculate confidence based on agreement
            pred_std = np.std(predictions)
            confidence = (
                max(0.4, 1.0 - (pred_std / abs(ensemble_prediction)))
                if ensemble_prediction != 0
                else 0.4
            )

            return {
                "ensemble_prediction": float(ensemble_prediction),
                "prediction_confidence": float(confidence),
                "interpolation_method": "distance_weighted + climate_zone + latitude",
                "nearest_locations_used": len(nearest_locations),
                "prediction_std": float(pred_std),
                "coverage": "global_interpolation",
            }

        # Fallback
        return self._fallback_prediction(lat, lon, target_month)

    def _latitude_based_prediction(self, lat: float, target_month: int) -> float:
        """Simple latitude-based climatology"""
        # Basic temperature model based on latitude and season
        abs_lat = abs(lat)

        # Base temperature decreases with latitude
        base_temp = 30 - (abs_lat * 0.6)  # Rough approximation

        # Seasonal variation (stronger at higher latitudes)
        seasonal_amplitude = 5 + (abs_lat * 0.3)

        # Northern/Southern hemisphere adjustment
        if lat >= 0:  # Northern hemisphere
            seasonal_adjustment = seasonal_amplitude * np.cos(
                2 * np.pi * (target_month - 7) / 12
            )
        else:  # Southern hemisphere (opposite seasons)
            seasonal_adjustment = seasonal_amplitude * np.cos(
                2 * np.pi * (target_month - 1) / 12
            )

        return base_temp + seasonal_adjustment

    def _fallback_prediction(self, lat: float, lon: float, target_month: int) -> dict:
        """Fallback prediction for any location"""
        # Use latitude-based climatology
        prediction = self._latitude_based_prediction(lat, target_month)

        return {
            "ensemble_prediction": float(prediction),
            "prediction_confidence": 0.5,
            "interpolation_method": "latitude_climatology_fallback",
            "coverage": "global_fallback",
            "note": "Prediction based on latitude climatology - limited accuracy",
        }


# Global instance
global_predictor = GlobalClimatePredictor()
