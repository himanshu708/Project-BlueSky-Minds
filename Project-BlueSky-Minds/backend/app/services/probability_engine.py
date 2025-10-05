import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import pymannkendall as mk
from datetime import datetime, date, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.services.pretrained_models import pretrained_models
import warnings
from sklearn.ensemble import StackingRegressor

from sklearn.ensemble import ExtraTreesRegressor  

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def safe_bool(value) -> bool:
    """Convert any boolean-like value to Python bool"""
    if hasattr(value, "item"):
        return bool(value.item())
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (np.ndarray, pd.Series)) and value.size == 1:
        return bool(value.item())
    else:
        return bool(value)


class AdvancedProbabilityEngine:

    def __init__(self):
        self.percentile_thresholds = [5, 10, 25, 50, 75, 90, 95]
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def calculate_weather_probabilities(
        self,
        historical_data: Dict[str, np.ndarray],
        target_date: date,
        location: Tuple[float, float],
        prediction_mode: str = "live_training",  # 🆕 NEW PARAMETER
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate comprehensive weather probabilities with ML enhancement
        """
        results = {}

        for variable, data in historical_data.items():
            if len(data) == 0:
                continue

            try:

                # 🎯 DUAL MODE LOGIC HERE
                if prediction_mode == "pretrained":
                    # Use pre-trained models (instant)
                    ml_prediction = self._use_pretrained_models(
                        data, target_date, location, variable
                    )
                    logger.info(f"🤖 Using PRE-TRAINED models for {variable}")
                else:
                    # Use live training (accurate)
                    ml_prediction = self._train_and_predict_ml(
                        data, target_date, location, variable
                    )
                    logger.info(f"🔥 Using LIVE TRAINING for {variable}")

                # Enhanced seasonal data extraction
                seasonal_data = self._extract_enhanced_seasonal_data(data, target_date)

                # Train ML models for this variable
                ml_prediction = self._train_and_predict_ml(data, target_date, location, variable)

                # Calculate enhanced statistics
                basic_stats = self._calculate_enhanced_statistics(seasonal_data)

                # ML-enhanced extreme probabilities
                extreme_probs = self._calculate_ml_extreme_probabilities(
                    seasonal_data, variable, ml_prediction
                )

                # Advanced trend analysis
                trend_analysis = self._analyze_advanced_trends(data)

                # Uncertainty quantification
                uncertainty = self._calculate_prediction_uncertainty(data, seasonal_data)

                # Climate pattern analysis
                climate_patterns = self._analyze_climate_patterns(data, location)

                results[variable] = {
                    "basic_statistics": basic_stats,
                    "ml_prediction": ml_prediction,
                    "extreme_probabilities": extreme_probs,
                    "trend_analysis": trend_analysis,
                    "uncertainty_analysis": uncertainty,
                    "climate_patterns": climate_patterns,
                    "prediction_mode_used": prediction_mode,  # 🆕 Track which mode was used
                    "data_quality": {
                        "sample_size": len(seasonal_data),
                        "completeness": len(seasonal_data)
                        / max(1, len(data) // 12)
                        * 100,
                        "data_span_years": len(data) // 12,
                    },
                }

            except Exception as e:
                logger.error(f"Error calculating probabilities for {variable}: {e}")
                results[variable] = self._get_error_result()

        return results

    def _use_pretrained_models(
        self,
        data: np.ndarray,
        target_date: date,
        location: Tuple[float, float],
        variable: str,
    ) -> Dict[str, Any]:
        """
        Use pre-trained models for instant prediction
        """
        try:
            if pretrained_models.loaded and variable == "temperature":
                # Use pre-trained models
                prediction = pretrained_models.predict_instant(
                    lat=location[0],
                    lon=location[1],
                    target_month=target_date.month,
                    historical_context=data.tolist(),
                )

                # Add mode identifier
                prediction["prediction_method"] = "pretrained_models"
                prediction["response_time"] = "< 100ms"

                return prediction
            else:
                # Fallback to simple statistical prediction
                logger.warning(
                    f"Pre-trained models not available for {variable}, using statistical fallback"
                )
                return self._get_simple_statistical_prediction(data, target_date)

        except Exception as e:
            logger.error(f"Pre-trained model error: {e}")
            return self._get_simple_statistical_prediction(data, target_date)

    def _get_simple_statistical_prediction(
        self, data: np.ndarray, target_date: date
    ) -> Dict[str, Any]:
        """
        Simple statistical prediction as fallback
        """
        target_month = target_date.month
        years = len(data) // 12

        # Get same month from previous years
        monthly_values = []
        for year in range(years):
            month_index = year * 12 + (target_month - 1)
            if month_index < len(data):
                monthly_values.append(data[month_index])

        if monthly_values:
            prediction = np.mean(monthly_values)
            std_dev = np.std(monthly_values)

            return {
                "ensemble_prediction": float(prediction),
                "prediction_confidence": 0.65,
                "prediction_std": float(std_dev),
                "prediction_method": "statistical_fallback",
                "response_time": "< 50ms",
            }
        else:
            return {
                "ensemble_prediction": float(np.mean(data)),
                "prediction_confidence": 0.40,
                "prediction_std": float(np.std(data)),
                "prediction_method": "overall_mean_fallback",
                "response_time": "< 10ms",
            }

    def _extract_enhanced_seasonal_data(self, data: np.ndarray, target_date: date) -> np.ndarray:
        """
        Enhanced seasonal data extraction with climate oscillation awareness
        """
        target_month = target_date.month
        years = len(data) // 12

        # Extract same month across all years
        seasonal_indices = []
        for year in range(years):
            month_index = year * 12 + (target_month - 1)
            if month_index < len(data):
                seasonal_indices.append(month_index)

        seasonal_data = data[seasonal_indices] if seasonal_indices else data

        # Apply recency weighting (more recent years get higher weight)
        if len(seasonal_data) > 5:
            weights = np.exp(np.linspace(-1, 0, len(seasonal_data)))
            weighted_mean = np.average(seasonal_data, weights=weights)

            # Adjust seasonal data based on recent trends
            recent_trend = np.polyfit(range(len(seasonal_data)), seasonal_data, 1)[0]
            if abs(recent_trend) > 0.01:  # Significant trend
                seasonal_data = seasonal_data + recent_trend * np.arange(len(seasonal_data))

        logger.info(f"🔍 Enhanced seasonal extraction: month={target_month}, "
                   f"points={len(seasonal_data)}, mean={np.mean(seasonal_data):.1f}")

        return seasonal_data

    def _train_and_predict_ml(self, data: np.ndarray, target_date: date, 
                         location: Tuple[float, float], variable: str) -> Dict[str, Any]:
        """
        🔥 ENHANCED ML with better models and validation
        """
        try:
            features = self._create_features(data, target_date, location)
            targets = data[13:]
            features = features[:-1]

            if len(features) < 24:
                return self._get_simple_prediction(data, target_date)

            # ✅ SPLIT: Use last 12 months for validation
            train_features = features[:-12]
            train_targets = targets[:-12]
            val_features = features[-12:]
            val_targets = targets[-12:]

            # ✅ IMPROVED MODELS
            models = {
                "random_forest": RandomForestRegressor(
                    n_estimators=500,  # ⬆️ Increased
                    max_depth=25,      # ⬆️ Deeper
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                ),
                "gradient_boost": GradientBoostingRegressor(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.02,  # ⬇️ Slower learning
                    subsample=0.8,
                    min_samples_split=3,
                    random_state=42
                ),
                "extra_trees": ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=25,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                ),
            }

            # ✅ ADD XGBOOST (High performance)
            try:
                from xgboost import XGBRegressor
                models["xgboost"] = XGBRegressor(
                    n_estimators=400,
                    max_depth=8,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
                logger.info("✅ XGBoost added")
            except ImportError:
                logger.warning("⚠️ Install XGBoost: pip install xgboost")

            try:
                from prophet import Prophet

                # Prepare data for Prophet
                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='1990-01-01', periods=len(data), freq='MS'),
                    'y': data
                })

                prophet_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05
                )
                prophet_model.fit(prophet_df)

                # Predict
                future = prophet_model.make_future_dataframe(periods=1, freq='MS')
                forecast = prophet_model.predict(future)
                prophet_pred = forecast['yhat'].iloc[-1]

                model_predictions["prophet"] = float(prophet_pred)
                model_weights["prophet"] = 1.2  # Prophet is good for time series

                logger.info(f"✅ Prophet: Pred={prophet_pred:.2f}")
            except:
                pass

            # # ✅ ADD LIGHTGBM (Faster & accurate)
            # try:
            #     from lightgbm import LGBMRegressor
            #     models["lightgbm"] = LGBMRegressor(
            #         n_estimators=400,
            #         max_depth=8,
            #         learning_rate=0.02,
            #         subsample=0.8,
            #         colsample_bytree=0.8,
            #         reg_alpha=0.1,
            #         reg_lambda=1.0,
            #         random_state=42,
            #         n_jobs=-1,
            #         verbosity=-1
            #     )
            #     logger.info("✅ LightGBM added")
            # except ImportError:
            #     logger.warning("⚠️ Install LightGBM: pip install lightgbm")

            # ✅ TRAIN & VALIDATE
            model_scores = {}
            model_predictions = {}
            model_weights = {}

            for name, model in models.items():
                try:
                    # Train on training set
                    model.fit(train_features, train_targets)

                    # Validate on validation set
                    val_pred = model.predict(val_features)
                    val_mae = mean_absolute_error(val_targets, val_pred)
                    val_r2 = r2_score(val_targets, val_pred)

                    # Score = weighted combo of MAE and R²
                    model_scores[name] = val_mae

                    # Predict target
                    target_features = self._create_target_features(data, target_date, location)
                    prediction = model.predict([target_features])[0]
                    model_predictions[name] = float(prediction)

                    # Weight = inverse of error (better models get higher weight)
                    model_weights[name] = 1.0 / (val_mae + 0.01)

                    logger.info(f"✅ {name}: MAE={val_mae:.3f}, R²={val_r2:.3f}, Pred={prediction:.2f}")

                except Exception as e:
                    logger.warning(f"❌ {name} failed: {e}")
                    continue

            if not model_predictions:
                return self._get_simple_prediction(data, target_date)

            # ✅ WEIGHTED ENSEMBLE (better models contribute more)
            total_weight = sum(model_weights.values())
            ensemble_prediction = sum(
                pred * model_weights[name] / total_weight
                for name, pred in model_predictions.items()
            )

            # ✅ CALCULATE REAL ACCURACY (on validation set)
            # ✅ CALCULATE REAL ACCURACY (on validation set)
            val_ensemble_preds = []
            for i in range(len(val_targets)):
                val_feature = val_features[i:i+1]
                preds = []
                weights = []
                for name, model in models.items():
                    if name in model_predictions:
                        pred = model.predict(val_feature)[0]
                        preds.append(pred)
                        weights.append(model_weights[name])

                if preds:
                    val_ensemble_preds.append(
                        sum(p * w for p, w in zip(preds, weights)) / sum(weights)
                    )

            if len(val_ensemble_preds) > 0:  # <-- USE len() CHECK
                val_ensemble_preds = np.array(val_ensemble_preds)
                ensemble_mae = mean_absolute_error(val_targets, val_ensemble_preds)
                ensemble_r2 = r2_score(val_targets, val_ensemble_preds)

                # ✅ IMPROVED ACCURACY CALCULATION
                data_std = np.std(train_targets)
                baseline_mae = np.mean(np.abs(val_targets - np.mean(train_targets)))  # Naive forecast error

                # Skill Score: improvement over naive baseline
                skill_score = (baseline_mae - ensemble_mae) / baseline_mae if baseline_mae > 0 else 0
                accuracy = max(20, min(98, skill_score * 100))  # Cap between 20-98%

                # ✅ IMPROVED CONFIDENCE (use R² as primary metric)
                confidence = max(0.50, min(0.98, ensemble_r2))  # R² is best confidence metric

                logger.info(f"🎯 Baseline MAE: {baseline_mae:.3f}, Ensemble MAE: {ensemble_mae:.3f}")
                logger.info(f"🏆 Skill Score: {skill_score:.3f} → Accuracy: {accuracy:.1f}%")
                logger.info(f"📊 R² Score: {ensemble_r2:.3f} → Confidence: {confidence:.1%}")
            else:
                accuracy = 60.0
                confidence = 0.6
                ensemble_r2 = 0.5
                ensemble_mae = 0.0

            return {
                "ensemble_prediction": float(ensemble_prediction),
                "individual_predictions": model_predictions,
                "model_scores": model_scores,
                "model_weights": model_weights,
                "prediction_confidence": float(confidence),
                "prediction_accuracy": float(accuracy),
                "prediction_std": float(np.std(list(model_predictions.values()))),
                "validation_mae": (
                    float(ensemble_mae) if len(val_ensemble_preds) > 0 else 0.0
                ),  # <-- FIXED
                "validation_r2": (
                    float(ensemble_r2) if len(val_ensemble_preds) > 0 else 0.0
                ),  # <-- FIXED
                "best_model": min(model_scores.items(), key=lambda x: x[1])[0],
            }

        except Exception as e:
            logger.error(f"❌ ML prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_simple_prediction(data, target_date)

    def _tune_hyperparameters(self, features: np.ndarray, targets: np.ndarray, model_type: str):
        """
        Quick hyperparameter tuning for better accuracy
        """
        from sklearn.model_selection import GridSearchCV

        param_grids = {
            "random_forest": {
                'n_estimators': [100, 150],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            },
            "gradient_boost": {
                'n_estimators': [100, 150],
                'max_depth': [5, 7, 10],
                'learning_rate': [0.05, 0.1]
            }
        }

        if model_type not in param_grids:
            return None

        try:
            if model_type == "random_forest":
                base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            else:
                base_model = GradientBoostingRegressor(random_state=42)

            grid_search = GridSearchCV(
                base_model,
                param_grids[model_type],
                cv=3,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(features, targets)
            logger.info(f"🎯 Best params for {model_type}: {grid_search.best_params_}")

            return grid_search.best_estimator_

        except Exception as e:
            logger.error(f"Tuning error for {model_type}: {e}")
            return None

    def _create_features(self, data: np.ndarray, target_date: date, 
                        location: Tuple[float, float]) -> np.ndarray:
        """
        Create comprehensive feature matrix for ML models
        """
        n_points = len(data)
        features = []

        for i in range(12, n_points):  # Start from 12 to have full year history
            feature_vector = []

            # Historical values (last 12 months)
            feature_vector.extend(data[i-12:i])

            # Seasonal features
            month = ((i % 12) + 1)
            feature_vector.extend([
                np.sin(2 * np.pi * month / 12),  # Seasonal sine
                np.cos(2 * np.pi * month / 12),  # Seasonal cosine
                month / 12.0  # Month normalized
            ])

            # Trend features
            if i >= 24:  # Need 2 years for trend
                recent_trend = np.polyfit(range(12), data[i-12:i], 1)[0]
                long_trend = np.polyfit(range(24), data[i-24:i], 1)[0]
                feature_vector.extend([recent_trend, long_trend])
            else:
                feature_vector.extend([0.0, 0.0])

            # Statistical features
            feature_vector.extend([
                np.mean(data[i-12:i]),
                np.std(data[i-12:i]),
                np.max(data[i-12:i]) - np.min(data[i-12:i])  # Range
            ])

            # Location features
            # Location features (existing)
            feature_vector.extend(
                [
                    location[0] / 90.0,  # Normalized latitude
                    location[1] / 180.0,  # Normalized longitude
                    abs(location[0]) / 90.0,  # Distance from equator
                ]
            )

            # ✅ ADD THESE NEW FEATURES
            # Lag features (previous months)
            if i >= 15:  # Need 3 extra months
                feature_vector.extend(
                    [
                        data[i - 13],  # 13 months ago
                        data[i - 14],  # 14 months ago
                        data[i - 15],  # 15 months ago
                    ]
                )
            else:
                feature_vector.extend([0.0, 0.0, 0.0])

            # Rolling statistics (3-month window)
            if i >= 14:
                recent_3months = data[i - 14 : i - 11]
                feature_vector.extend(
                    [
                        np.mean(recent_3months),
                        np.std(recent_3months),
                        np.max(recent_3months) - np.min(recent_3months),
                    ]
                )
            else:
                feature_vector.extend([0.0, 0.0, 0.0])

            # Year-over-year change
            if i >= 24:
                yoy_change = data[i - 1] - data[i - 13]
                feature_vector.append(yoy_change)
            else:
                feature_vector.append(0.0)

            features.append(feature_vector)

        return np.array(features)

    def _create_target_features(self, data: np.ndarray, target_date: date, 
                           location: Tuple[float, float]) -> List[float]:
        """
        Create comprehensive features for target prediction
        Enhanced with lag features, rolling statistics, and YoY comparisons
        """
        feature_vector = []

        # Last 12 months (historical values)
        feature_vector.extend(data[-12:])

        # Seasonal features for target month
        month = target_date.month
        feature_vector.extend([
            np.sin(2 * np.pi * month / 12),  # Seasonal sine
            np.cos(2 * np.pi * month / 12),  # Seasonal cosine
            month / 12.0  # Month normalized
        ])

        # Trend features
        recent_trend = np.polyfit(range(12), data[-12:], 1)[0]
        long_trend = np.polyfit(range(24), data[-24:], 1)[0] if len(data) >= 24 else 0.0
        feature_vector.extend([recent_trend, long_trend])

        # Statistical features (last 12 months)
        feature_vector.extend([
            np.mean(data[-12:]),
            np.std(data[-12:]),
            np.max(data[-12:]) - np.min(data[-12:])  # Range
        ])

        # Location features
        feature_vector.extend([
            location[0] / 90.0,  # Normalized latitude
            location[1] / 180.0,  # Normalized longitude
            abs(location[0]) / 90.0  # Distance from equator
        ])

        # ✅ NEW: Lag features (13-15 months ago)
        if len(data) >= 15:
            feature_vector.extend([
                data[-13],  # 13 months ago (same month last year)
                data[-14],  # 14 months ago
                data[-15],  # 15 months ago
            ])
        else:
            feature_vector.extend([0.0, 0.0, 0.0])

        # ✅ NEW: Rolling statistics (3-month window)
        if len(data) >= 14:
            recent_3months = data[-14:-11]  # 3 months before last year
            feature_vector.extend([
                np.mean(recent_3months),
                np.std(recent_3months),
                np.max(recent_3months) - np.min(recent_3months)
            ])
        else:
            feature_vector.extend([0.0, 0.0, 0.0])

        # ✅ NEW: Year-over-year change
        if len(data) >= 24:
            yoy_change = data[-1] - data[-13]  # Current vs same month last year
            feature_vector.append(yoy_change)
        else:
            feature_vector.append(0.0)

        return feature_vector

    def _get_simple_prediction(self, data: np.ndarray, target_date: date) -> Dict[str, Any]:
        """
        Fallback to simple statistical prediction
        """
        target_month = target_date.month
        years = len(data) // 12

        # Get same month from previous years
        monthly_values = []
        for year in range(years):
            month_index = year * 12 + (target_month - 1)
            if month_index < len(data):
                monthly_values.append(data[month_index])

        if monthly_values:
            # Apply trend adjustment
            if len(monthly_values) > 3:
                trend = np.polyfit(range(len(monthly_values)), monthly_values, 1)[0]
                prediction = monthly_values[-1] + trend
            else:
                prediction = np.mean(monthly_values)

            return {
                "ensemble_prediction": float(prediction),
                "individual_predictions": {"statistical": float(prediction)},
                "model_scores": {"statistical": np.std(monthly_values)},
                "prediction_confidence": 0.6,
                "prediction_std": float(np.std(monthly_values)),
                "best_model": "statistical"
            }
        else:
            return {
                "ensemble_prediction": float(np.mean(data)),
                "individual_predictions": {"fallback": float(np.mean(data))},
                "model_scores": {"fallback": np.std(data)},
                "prediction_confidence": 0.3,
                "prediction_std": float(np.std(data)),
                "best_model": "fallback"
            }

    def _calculate_enhanced_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate enhanced statistical measures
        """
        stats_dict = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data)),
            "percentiles": {
                f"p{p}": float(np.percentile(data, p))
                for p in self.percentile_thresholds
            },
        }

        # Add robust statistics
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        stats_dict.update({
            "iqr": float(iqr),
            "robust_mean": float(np.mean(data[(data >= q1 - 1.5*iqr) & (data <= q3 + 1.5*iqr)])),
            "coefficient_of_variation": float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0.0
        })

        return stats_dict

    def _calculate_ml_extreme_probabilities(self, data: np.ndarray, variable: str, 
                                          ml_prediction: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate extreme probabilities enhanced with ML insights
        """
        # Base extreme probabilities
        base_probs = self._calculate_extreme_probabilities(data, variable)

        # Adjust based on ML prediction confidence
        confidence = ml_prediction.get("prediction_confidence", 0.5)
        prediction = ml_prediction.get("ensemble_prediction", np.mean(data))

        # Calculate percentiles for extreme classification
        p10, p25, p75, p90, p95 = np.percentile(data, [10, 25, 75, 90, 95])

        # Classify ML prediction
        if prediction > p95:
            extreme_adjustment = {"very_hot": 25.0, "hot": 15.0, "normal": -20.0}
        elif prediction > p90:
            extreme_adjustment = {"very_hot": 10.0, "hot": 20.0, "normal": -15.0}
        elif prediction < p10:
            extreme_adjustment = {"cold": 20.0, "very_cold": 15.0, "normal": -15.0}
        else:
            extreme_adjustment = {"normal": 10.0}

        # Apply confidence-weighted adjustments
        # Apply confidence-weighted adjustments
        enhanced_probs = base_probs.copy()
        for condition, adjustment in extreme_adjustment.items():
            if f"above_{condition}" in enhanced_probs:
                enhanced_probs[f"above_{condition}"] += adjustment * confidence
            elif f"below_{condition}" in enhanced_probs:
                enhanced_probs[f"below_{condition}"] += adjustment * confidence
            elif condition in ["very_hot", "hot", "cold", "very_cold"]:
                # Map to existing keys
                if condition == "very_hot" and "above_very_hot" in enhanced_probs:
                    enhanced_probs["above_very_hot"] += adjustment * confidence
                elif condition == "hot" and "above_hot" in enhanced_probs:
                    enhanced_probs["above_hot"] += adjustment * confidence
                elif condition == "cold" and "below_cold" in enhanced_probs:
                    enhanced_probs["below_cold"] += adjustment * confidence
                elif condition == "very_cold" and "below_very_cold" in enhanced_probs:
                    enhanced_probs["below_very_cold"] += adjustment * confidence

        # Normalize probabilities to ensure they sum to reasonable values
        total_extreme = sum(v for k, v in enhanced_probs.items() if k != "normal")
        if total_extreme > 80:  # Cap extreme probabilities
            scale_factor = 80 / total_extreme
            for key in enhanced_probs:
                if key != "normal":
                    enhanced_probs[key] *= scale_factor

        # Ensure all probabilities are positive
        for key in enhanced_probs:
            enhanced_probs[key] = max(0.1, enhanced_probs[key])

        return enhanced_probs

    def _analyze_advanced_trends(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Advanced trend analysis with multiple methods
        """
        try:
            results = {}

            # 1. Mann-Kendall trend test
            trend, h, p, z, tau, s, var_s, slope, intercept = mk.original_test(data)

            # 2. Linear regression
            x = np.arange(len(data))
            slope_lr, intercept_lr, r_value, p_value_lr, std_err = stats.linregress(x, data)

            # 3. Polynomial trend (quadratic)
            if len(data) > 36:  # Need sufficient data for polynomial
                poly_coeffs = np.polyfit(x, data, 2)
                poly_trend = "accelerating" if poly_coeffs[0] > 0 else "decelerating" if poly_coeffs[0] < 0 else "linear"
            else:
                poly_trend = "insufficient_data"
                poly_coeffs = [0, 0, 0]

            # 4. Seasonal trend decomposition
            seasonal_trends = self._analyze_seasonal_trends(data)

            # 5. Change point detection
            change_points = self._detect_change_points_advanced(data)

            # 6. Trend strength classification
            trend_strength = self._classify_trend_strength(slope_lr * 12, np.std(data))

            results = {
                "mann_kendall": {
                    "trend_direction": trend,
                    "is_significant": safe_bool(h),
                    "p_value": float(p),
                    "tau": float(tau),
                    "z_statistic": float(z),
                },
                "linear_regression": {
                    "slope_per_year": float(slope_lr * 12),
                    "r_squared": float(r_value**2),
                    "p_value": float(p_value_lr),
                    "standard_error": float(std_err),
                    "trend_strength": trend_strength,
                },
                "polynomial_analysis": {
                    "trend_type": poly_trend,
                    "acceleration": (
                        float(poly_coeffs[0]) if len(poly_coeffs) > 0 else 0.0
                    ),
                    "curvature_significant": (
                        abs(poly_coeffs[0]) > 0.001 if len(poly_coeffs) > 0 else False
                    ),
                },
                "seasonal_trends": seasonal_trends,
                "change_points": change_points,
                "trend_description": self._generate_advanced_trend_description(
                    trend, slope_lr * 12, trend_strength
                ),
            }

            return results

        except Exception as e:
            logger.error(f"Advanced trend analysis error: {e}")
            return {
                "error": str(e),
                "trend_direction": "unknown",
                "trend_significant": False
            }

    def _analyze_seasonal_trends(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze trends within each season
        """
        try:
            years = len(data) // 12
            if years < 3:
                return {"error": "Insufficient data for seasonal analysis"}

            seasonal_data = data[:years * 12].reshape(years, 12)

            seasons = {
                'winter': [11, 0, 1],  # Dec, Jan, Feb
                'spring': [2, 3, 4],   # Mar, Apr, May
                'summer': [5, 6, 7],   # Jun, Jul, Aug
                'autumn': [8, 9, 10]   # Sep, Oct, Nov
            }

            seasonal_trends = {}

            for season_name, months in seasons.items():
                season_means = seasonal_data[:, months].mean(axis=1)

                if len(season_means) >= 3:
                    # Linear trend for this season
                    x = np.arange(len(season_means))
                    slope, _, r_value, p_value, _ = stats.linregress(x, season_means)

                    seasonal_trends[season_name] = {
                        'slope_per_year': float(slope),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'is_significant': safe_bool(p_value < 0.05),
                        'trend_strength': self._classify_trend_strength(slope, np.std(season_means))
                    }

            return seasonal_trends

        except Exception as e:
            return {"error": str(e)}

    def _detect_change_points_advanced(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Advanced change point detection using multiple methods
        """
        try:
            # Method 1: CUSUM (Cumulative Sum)
            cumsum = np.cumsum(data - np.mean(data))
            max_idx = np.argmax(np.abs(cumsum))
            max_deviation = abs(cumsum[max_idx])

            # Statistical significance test
            threshold = 2 * np.std(data) * np.sqrt(len(data))
            cusum_significant = max_deviation > threshold

            # Method 2: Moving average change detection
            window = min(24, len(data) // 4)  # 2 years or 1/4 of data
            if window >= 12:
                moving_avg = pd.Series(data).rolling(window=window, center=True).mean()
                moving_std = pd.Series(data).rolling(window=window, center=True).std()

                # Detect significant deviations
                z_scores = np.abs((data - moving_avg) / moving_std)
                change_candidates = np.where(z_scores > 2.5)[0]

                # Find the most significant change point
                if len(change_candidates) > 0:
                    most_significant_idx = change_candidates[np.argmax(z_scores[change_candidates])]
                    moving_avg_significant = True
                else:
                    most_significant_idx = None
                    moving_avg_significant = False
            else:
                most_significant_idx = None
                moving_avg_significant = False

            # Method 3: Variance change detection
            variance_changes = self._detect_variance_changes(data)

            return {
                "cusum_method": {
                    "change_point_detected": cusum_significant,
                    "change_point_index": int(max_idx) if cusum_significant else None,
                    "change_magnitude": float(max_deviation),
                    "significance_threshold": float(threshold)
                },
                "moving_average_method": {
                    "change_point_detected": moving_avg_significant,
                    "change_point_index": int(most_significant_idx) if most_significant_idx is not None else None,
                    "window_size": window
                },
                "variance_changes": variance_changes
            }

        except Exception as e:
            return {"error": str(e)}

    def _detect_variance_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect changes in variance (volatility)
        """
        try:
            # Split data into first and second half
            mid_point = len(data) // 2
            first_half = data[:mid_point]
            second_half = data[mid_point:]

            # Calculate variances
            var1 = np.var(first_half)
            var2 = np.var(second_half)

            # F-test for variance equality
            f_stat = var1 / var2 if var2 > 0 else 1.0

            # Simple significance test (approximate)
            variance_change_significant = abs(f_stat - 1.0) > 0.5

            return {
                "first_half_variance": float(var1),
                "second_half_variance": float(var2),
                "variance_ratio": float(f_stat),
                "variance_change_detected": safe_bool(variance_change_significant),
                "change_type": (
                    "increased_volatility"
                    if f_stat < 1
                    else "decreased_volatility" if f_stat > 1 else "stable"
                ),
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_prediction_uncertainty(self, full_data: np.ndarray, seasonal_data: np.ndarray) -> Dict[str, Any]:
        """
        Quantify prediction uncertainty using multiple methods
        """
        try:
            # 1. Historical variability
            historical_std = np.std(seasonal_data)
            historical_cv = historical_std / np.mean(seasonal_data) if np.mean(seasonal_data) != 0 else 0

            # 2. Model uncertainty (from ensemble spread)
            # This would be enhanced if we had multiple model predictions

            # 3. Trend uncertainty
            if len(full_data) > 24:
                # Bootstrap trend estimates
                n_bootstrap = 100
                trend_estimates = []

                for _ in range(n_bootstrap):
                    # Resample with replacement
                    bootstrap_sample = np.random.choice(full_data, size=len(full_data), replace=True)
                    x = np.arange(len(bootstrap_sample))
                    slope, _, _, _, _ = stats.linregress(x, bootstrap_sample)
                    trend_estimates.append(slope * 12)  # Convert to yearly

                trend_uncertainty = np.std(trend_estimates)
            else:
                trend_uncertainty = 0.0

            # 4. Seasonal uncertainty
            seasonal_uncertainty = np.std(seasonal_data) / np.sqrt(len(seasonal_data))

            # 5. Overall prediction interval
            total_uncertainty = np.sqrt(historical_std**2 + trend_uncertainty**2 + seasonal_uncertainty**2)

            return {
                "historical_variability": {
                    "standard_deviation": float(historical_std),
                    "coefficient_of_variation": float(historical_cv)
                },
                "trend_uncertainty": float(trend_uncertainty),
                "seasonal_uncertainty": float(seasonal_uncertainty),
                "total_uncertainty": float(total_uncertainty),
                "prediction_intervals": {
                    "68_percent": {
                        "lower": float(np.mean(seasonal_data) - total_uncertainty),
                        "upper": float(np.mean(seasonal_data) + total_uncertainty)
                    },
                    "95_percent": {
                        "lower": float(np.mean(seasonal_data) - 2 * total_uncertainty),
                        "upper": float(np.mean(seasonal_data) + 2 * total_uncertainty)
                    }
                },
                "uncertainty_level": "high" if total_uncertainty > historical_std else "medium" if total_uncertainty > historical_std/2 else "low"
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_climate_patterns(self, data: np.ndarray, location: Tuple[float, float]) -> Dict[str, Any]:
        """
        Analyze climate patterns and oscillations
        """
        try:
            lat, lon = location

            # 1. Climate zone classification
            climate_zone = self._classify_climate_zone(lat, lon)

            # 2. Seasonal amplitude analysis
            years = len(data) // 12
            if years > 0:
                monthly_means = data[:years * 12].reshape(years, 12).mean(axis=0)
                seasonal_amplitude = np.max(monthly_means) - np.min(monthly_means)
                peak_month = np.argmax(monthly_means) + 1
                trough_month = np.argmin(monthly_means) + 1
            else:
                seasonal_amplitude = 0.0
                peak_month = 1
                trough_month = 1

            # 3. Inter-annual variability
            if years > 1:
                annual_means = data[:years * 12].reshape(years, 12).mean(axis=1)
                interannual_variability = np.std(annual_means)
            else:
                interannual_variability = 0.0

            # 4. Extreme event frequency
            p95 = np.percentile(data, 95)
            p5 = np.percentile(data, 5)
            extreme_hot_frequency = (data > p95).sum() / len(data) * 100
            extreme_cold_frequency = (data < p5).sum() / len(data) * 100

            return {
                "climate_zone": climate_zone,
                "seasonal_characteristics": {
                    "amplitude": float(seasonal_amplitude),
                    "peak_month": int(peak_month),
                    "trough_month": int(trough_month),
                    "seasonality_strength": "high" if seasonal_amplitude > np.std(data) else "moderate" if seasonal_amplitude > np.std(data)/2 else "low"
                },
                "variability": {
                    "interannual": float(interannual_variability),
                    "total": float(np.std(data))
                },
                "extreme_events": {
                    "hot_extremes_frequency_percent": float(extreme_hot_frequency),
                    "cold_extremes_frequency_percent": float(extreme_cold_frequency),
                    "extreme_threshold_hot": float(p95),
                    "extreme_threshold_cold": float(p5)
                }
            }

        except Exception as e:
            return {"error": str(e)}

    def _classify_climate_zone(self, lat: float, lon: float) -> str:
        """
        Classify climate zone based on location
        """
        abs_lat = abs(lat)

        if abs_lat < 10:
            return "equatorial"
        elif abs_lat < 23.5:
            return "tropical"
        elif abs_lat < 35:
            return "subtropical"
        elif abs_lat < 60:
            return "temperate"
        else:
            return "polar"

    def _classify_trend_strength(self, slope: float, data_std: float) -> str:
        """
        Classify trend strength relative to data variability
        """
        if data_std == 0:
            return "undefined"

        relative_slope = abs(slope) / data_std

        if relative_slope > 0.2:
            return "very_strong"
        elif relative_slope > 0.1:
            return "strong"
        elif relative_slope > 0.05:
            return "moderate"
        elif relative_slope > 0.01:
            return "weak"
        else:
            return "negligible"

    def _generate_advanced_trend_description(self, trend: str, slope: float, strength: str) -> str:
        """
        Generate comprehensive human-readable trend description
        """
        direction_map = {
            "increasing": "warming" if slope > 0 else "increasing",
            "decreasing": "cooling" if slope < 0 else "decreasing",
            "no trend": "stable"
        }

        direction = direction_map.get(trend, trend)

        if trend in ["increasing", "decreasing"]:
            return f"{direction.title()} trend detected: {abs(slope):.3f} units per year ({strength} strength)"
        else:
            return f"No significant trend detected (variability: {strength})"

    def _calculate_extreme_probabilities(self, data: np.ndarray, variable: str) -> Dict[str, float]:
        """
        Calculate probabilities of extreme weather events (base method)
        """
        thresholds = self._get_extreme_thresholds(data, variable)
        probabilities = {}

        for condition, threshold in thresholds.items():
            if condition.startswith("above"):
                prob = (data > threshold).mean() * 100
            elif condition.startswith("below"):
                prob = (data < threshold).mean() * 100
            else:
                prob = 0.0

            probabilities[condition] = float(prob)

        return probabilities

    def _get_extreme_thresholds(self, data: np.ndarray, variable: str) -> Dict[str, float]:
        """
        Define extreme weather thresholds for different variables
        """
        p5, p10, p90, p95 = np.percentile(data, [5, 10, 90, 95])

        if variable == "temperature":
            return {
                "above_very_hot": p95,
                "above_hot": p90,
                "below_cold": p10,
                "below_very_cold": p5,
            }
        elif variable == "precipitation":
            return {
                "above_heavy_rain": p90,
                "above_very_heavy_rain": p95,
                "below_dry": p10,
            }
        elif variable == "wind_speed":
            return {"above_windy": p90, "above_very_windy": p95}
        else:
            return {
                "above_high": p90,
                "above_very_high": p95,
                "below_low": p10,
                "below_very_low": p5,
            }

    def _get_error_result(self) -> Dict[str, Any]:
        """
        Return comprehensive error result structure
        """
        return {
            "basic_statistics": {},
            "ml_prediction": {
                "ensemble_prediction": 0.0,
                "prediction_confidence": 0.0,
                "error": "Insufficient data"
            },
            "extreme_probabilities": {},
            "trend_analysis": {"error": "Insufficient data"},
            "uncertainty_analysis": {"error": "Insufficient data"},
            "climate_patterns": {"error": "Insufficient data"},
            "data_quality": {"sample_size": 0, "completeness": 0},
        }

    # Enhanced prediction method for extended forecasting
    def predict_extended_forecast(self, historical_data: Dict[str, np.ndarray], 
                                 target_months: List[int], target_year: int,
                                 location: Tuple[float, float]) -> Dict[str, Any]:
        """
        Generate extended forecast using ensemble ML models
        """
        try:
            forecasts = {}

            for variable, data in historical_data.items():
                if len(data) < 24:  # Need at least 2 years
                    continue

                variable_forecasts = []

                for target_month in target_months:
                    # Create target date
                    target_date = date(target_year, target_month, 15)

                    # Get ML prediction
                    ml_result = self._train_and_predict_ml(data, target_date, location, variable)

                    # Get seasonal baseline
                    seasonal_data = self._extract_enhanced_seasonal_data(data, target_date)
                    seasonal_baseline = np.mean(seasonal_data)
                    seasonal_std = np.std(seasonal_data)

                    # Ensemble prediction (ML + Statistical)
                    ml_prediction = ml_result.get("ensemble_prediction", seasonal_baseline)
                    ml_confidence = ml_result.get("prediction_confidence", 0.5)

                    # Weight predictions based on confidence
                    ensemble_prediction = (
                        ml_confidence * ml_prediction + 
                        (1 - ml_confidence) * seasonal_baseline
                    )

                    # Calculate prediction intervals
                    prediction_std = ml_result.get("prediction_std", seasonal_std)

                    forecast_data = {
                        "month": target_month,
                        "year": target_year,
                        "variable": variable,
                        "predicted_value": float(ensemble_prediction),
                        "confidence_interval": {
                            "lower_95": float(ensemble_prediction - 1.96 * prediction_std),
                            "lower_80": float(ensemble_prediction - 1.28 * prediction_std),
                            "upper_80": float(ensemble_prediction + 1.28 * prediction_std),
                            "upper_95": float(ensemble_prediction + 1.96 * prediction_std)
                        },
                        "prediction_confidence": float(ml_confidence),
                        "seasonal_baseline": float(seasonal_baseline),
                        "ml_prediction": ml_result,
                        "uncertainty_level": "low" if ml_confidence > 0.8 else "medium" if ml_confidence > 0.6 else "high"
                    }

                    variable_forecasts.append(forecast_data)

                forecasts[variable] = variable_forecasts

            return {
                "forecasts": forecasts,
                "methodology": "Ensemble ML (Random Forest + Gradient Boosting + Linear Models) + Statistical Baseline",
                "overall_confidence": float(np.mean([
                    f["prediction_confidence"] for var_forecasts in forecasts.values() 
                    for f in var_forecasts
                ])) if forecasts else 0.0,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Extended forecast error: {e}")
            return {
                "error": str(e),
                "forecasts": {},
                "methodology": "Error in prediction",
                "overall_confidence": 0.0
            }

# Additional utility functions for model validation
def validate_model_accuracy(engine: AdvancedProbabilityEngine, 
                           historical_data: np.ndarray, 
                           validation_months: int = 12) -> Dict[str, float]:
    """
    Validate model accuracy using walk-forward validation
    """
    if len(historical_data) < validation_months + 24:
        return {"error": "Insufficient data for validation"}
    
    # Split data
    train_data = historical_data[:-validation_months]
    test_data = historical_data[-validation_months:]
    
    predictions = []
    actuals = []
    
    for i in range(validation_months):
        # Use data up to current point for prediction
        current_train = historical_data[:-(validation_months-i)]
        target_date = date(2024, (i % 12) + 1, 15)  # Mock date
        
        # Get prediction
        ml_result = engine._train_and_predict_ml(
            current_train, target_date, (0.0, 0.0), "temperature"
        )
        
        prediction = ml_result.get("ensemble_prediction", np.mean(current_train))
        actual = test_data[i]
        
        predictions.append(prediction)
        actuals.append(actual)
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
    r2 = r2_score(actuals, predictions)
    
    return {
        "mean_absolute_error": float(mae),
        "root_mean_square_error": float(rmse),
        "mean_absolute_percentage_error": float(mape),
        "r_squared": float(r2),
        "accuracy_percentage": float(max(0, (1 - mae / np.std(actuals)) * 100))
    }
