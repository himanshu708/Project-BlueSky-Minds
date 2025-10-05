# create file: train_models.py
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, date
import asyncio
import logging
from app.services.nasa_data import NASADataService

logger = logging.getLogger(__name__)


class ClimateModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.nasa_service = NASADataService()

    async def collect_training_data(self):
        """Collect training data from multiple global locations"""
        print("🌍 Collecting global training data...")

        # Major global cities for diverse climate training
        training_locations = [
            # Tropical
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
            {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
            {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
            # Subtropical
            {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
            {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
            # Temperate
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
            # Continental
            {"name": "Moscow", "lat": 55.7558, "lon": 37.6176},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
            # Other climates
            {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241},
            {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332},
        ]

        all_training_data = []

        for location in training_locations:
            try:
                print(f"📡 Fetching data for {location['name']}...")

                # Get 5 years of data (2019-2023)
                historical_data = await self.nasa_service.fetch_historical_data(
                    lat=location["lat"],
                    lon=location["lon"],
                    start_year=2019,
                    end_year=2023,
                    variables=["temperature", "precipitation"],
                )

                if (
                    "temperature" in historical_data
                    and len(historical_data["temperature"]) > 0
                ):
                    # Create features and targets
                    features, targets = self._create_training_features(
                        historical_data["temperature"], location["lat"], location["lon"]
                    )

                    # Add location info
                    for i, feature_row in enumerate(features):
                        training_row = {
                            "location_name": location["name"],
                            "latitude": location["lat"],
                            "longitude": location["lon"],
                            "target": targets[i] if i < len(targets) else np.nan,
                            **{
                                f"feature_{j}": feature_row[j]
                                for j in range(len(feature_row))
                            },
                        }
                        all_training_data.append(training_row)

                    print(f"✅ {location['name']}: {len(features)} samples collected")
                else:
                    print(f"❌ {location['name']}: No data available")

            except Exception as e:
                print(f"❌ {location['name']}: Error - {e}")
                continue

        print(f"🎯 Total training samples: {len(all_training_data)}")
        return pd.DataFrame(all_training_data)

    def _create_training_features(self, data, lat, lon):
        """Create features for training"""
        features = []
        targets = []

        for i in range(12, len(data)):  # Start from month 12
            # Features (same as your existing feature creation)
            feature_vector = [
                i / len(data),  # Normalized time
                (i % 12) / 12,  # Seasonal position
                np.sin(2 * np.pi * (i % 12) / 12),  # Seasonal sine
                np.cos(2 * np.pi * (i % 12) / 12),  # Seasonal cosine
                np.mean(data[max(0, i - 12) : i]),  # 12-month average
                np.std(data[max(0, i - 12) : i]),  # 12-month std
                data[i - 1] if i > 0 else data[0],  # Previous value
                lat / 90.0,  # Normalized latitude
                lon / 180.0,  # Normalized longitude
                abs(lat) / 90.0,  # Distance from equator
            ]

            # Add trend features
            if i >= 24:
                recent_slope = np.polyfit(range(12), data[i - 12 : i], 1)[0]
                feature_vector.append(recent_slope)
            else:
                feature_vector.append(0.0)

            features.append(feature_vector)
            targets.append(data[i])  # Next month's value

        return features, targets

    def train_models(self, training_df):
        """Train multiple models on collected data"""
        print("🤖 Training climate prediction models...")

        # Prepare features and targets
        feature_cols = [
            col for col in training_df.columns if col.startswith("feature_")
        ]
        self.feature_columns = feature_cols

        X = training_df[feature_cols].values
        y = training_df["target"].values

        # Remove NaN values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"📊 Training on {len(X)} samples with {len(feature_cols)} features")

        # Scale features
        self.scalers["temperature"] = StandardScaler()
        X_scaled = self.scalers["temperature"].fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train multiple models
        models_to_train = {
            "random_forest": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "gradient_boost": GradientBoostingRegressor(
                n_estimators=200, random_state=42
            ),
            "linear": LinearRegression(),
        }

        for name, model in models_to_train.items():
            print(f"🔧 Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)

            print(
                f"✅ {name}: Train R² = {train_score:.3f}, Test R² = {test_score:.3f}"
            )

            # Store model
            self.models[f"temperature_{name}"] = model

        print("🎉 Model training completed!")

    def save_models(self, model_dir="trained_models"):
        """Save trained models to disk"""
        import os

        os.makedirs(model_dir, exist_ok=True)

        print(f"💾 Saving models to {model_dir}/...")

        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{name}.pkl")

        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{model_dir}/scaler_{name}.pkl")

        # Save feature columns
        with open(f"{model_dir}/feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)

        # Save metadata
        metadata = {
            "trained_date": datetime.now().isoformat(),
            "model_names": list(self.models.keys()),
            "feature_count": len(self.feature_columns),
            "version": "1.0",
        }

        with open(f"{model_dir}/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print("✅ Models saved successfully!")


# Training script
async def main():
    """Main training function"""
    trainer = ClimateModelTrainer()

    # Collect training data
    training_data = await trainer.collect_training_data()

    if len(training_data) > 100:  # Need sufficient data
        # Train models
        trainer.train_models(training_data)

        # Save models
        trainer.save_models()

        print("🏆 Training completed! Models ready for deployment.")
    else:
        print("❌ Insufficient training data collected.")


if __name__ == "__main__":
    asyncio.run(main())
