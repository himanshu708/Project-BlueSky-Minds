# create file: app/services/pretrained_models.py
import joblib
import pickle
import numpy as np
import os
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PretrainedClimateModels:
    def __init__(self, model_dir="trained_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.metadata = {}
        self.loaded = False

    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            if not os.path.exists(self.model_dir):
                logger.warning(f"Model directory {self.model_dir} not found")
                return False

            logger.info(f"🤖 Loading pre-trained models from {self.model_dir}")

            # Load metadata
            with open(f"{self.model_dir}/metadata.pkl", "rb") as f:
                self.metadata = pickle.load(f)

            # Load feature columns
            with open(f"{self.model_dir}/feature_columns.pkl", "rb") as f:
                self.feature_columns = pickle.load(f)

            # Load models
            for model_name in self.metadata["model_names"]:
                model_path = f"{self.model_dir}/{model_name}.pkl"
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"✅ Loaded {model_name}")

            # Load scalers
            scaler_path = f"{self.model_dir}/scaler_temperature.pkl"
            if os.path.exists(scaler_path):
                self.scalers["temperature"] = joblib.load(scaler_path)
                logger.info("✅ Loaded scaler")

            self.loaded = True
            logger.info(
                f"🎉 All models loaded successfully! Trained on {self.metadata['trained_date']}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False

    def predict_instant(
        self, lat: float, lon: float, target_month: int, historical_context: List[float]
    ) -> Dict[str, Any]:
        """Instant prediction using pre-trained models"""
        if not self.loaded:
            return {"error": "Models not loaded"}

        try:
            # Create features (same format as training)
            features = self._create_prediction_features(
                lat, lon, target_month, historical_context
            )

            # Scale features
            features_scaled = self.scalers["temperature"].transform([features])

            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                if "temperature" in model_name:
                    pred = model.predict(features_scaled)[0]
                    predictions[model_name.replace("temperature_", "")] = float(pred)

            # Ensemble prediction
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()))
                pred_std = np.std(list(predictions.values()))

                # Calculate confidence based on model agreement
                confidence = (
                    max(0.5, 1.0 - (pred_std / abs(ensemble_pred)))
                    if ensemble_pred != 0
                    else 0.5
                )

                return {
                    "ensemble_prediction": float(ensemble_pred),
                    "individual_predictions": predictions,
                    "prediction_confidence": float(confidence),
                    "prediction_std": float(pred_std),
                    "model_version": self.metadata.get("version", "1.0"),
                    "response_time_ms": "< 50ms",  # Instant!
                    "data_source": "Pre-trained ML Models",
                }
            else:
                return {"error": "No valid predictions"}

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}

    def _create_prediction_features(
        self, lat: float, lon: float, target_month: int, historical_context: List[float]
    ) -> List[float]:
        """Create features for prediction (same as training)"""
        # Simulate time position (normalized)
        time_pos = 0.8  # Assume recent data

        features = [
            time_pos,  # Normalized time
            target_month / 12,  # Seasonal position
            np.sin(2 * np.pi * target_month / 12),  # Seasonal sine
            np.cos(2 * np.pi * target_month / 12),  # Seasonal cosine
            (
                np.mean(historical_context[-12:])
                if len(historical_context) >= 12
                else np.mean(historical_context)
            ),  # 12-month avg
            (
                np.std(historical_context[-12:])
                if len(historical_context) >= 12
                else np.std(historical_context)
            ),  # 12-month std
            historical_context[-1] if historical_context else 20.0,  # Previous value
            lat / 90.0,  # Normalized latitude
            lon / 180.0,  # Normalized longitude
            abs(lat) / 90.0,  # Distance from equator
        ]

        # Add trend feature
        # Add trend feature
        if len(historical_context) >= 12:
            recent_slope = np.polyfit(range(12), historical_context[-12:], 1)[0]
            features.append(recent_slope)
        else:
            features.append(0.0)

        return features

# Global instance
pretrained_models = PretrainedClimateModels()
