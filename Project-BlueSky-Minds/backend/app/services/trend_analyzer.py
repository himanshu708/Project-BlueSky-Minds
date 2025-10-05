import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import pymannkendall as mk
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.services.pretrained_models import pretrained_models
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
def safe_bool(value) -> bool:
        """Convert any boolean-like value to Python bool"""
        if hasattr(value, 'item'):
            return bool(value.item())
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (np.ndarray, pd.Series)) and value.size == 1:
            return bool(value.item())
        else:
            return bool(value)

class AdvancedTrendAnalyzer:

    def __init__(self):
        self.min_years_for_trend = 5  # Reduced for more flexibility
        self.significance_level = 0.05
        self.models = {}
        self.scalers = {}

    def analyze_climate_trends(
        self,
        historical_data: Dict[str, np.ndarray],
        location: Tuple[float, float],
        prediction_mode: str = "live_training",  # 🆕 NEW PARAMETER
    ) -> Dict[str, Any]:
        """
        Advanced climate trend analysis with dual mode support
        """
        results = {}

        for variable, data in historical_data.items():
            if len(data) < self.min_years_for_trend * 12:
                results[variable] = self._insufficient_data_result()
                continue

            try:
                logger.info(f"🌡️ Starting ML training for {variable}")
                logger.info(f"📈 Data points available: {len(data)} months ({len(data)//12} years)")

                # 🎯 DUAL MODE LOGIC HERE
                if prediction_mode == "pretrained":
                    # Use faster, simpler trend analysis
                    trend_results = self._fast_trend_analysis(data)
                    logger.info(f"⚡ Using FAST trend analysis for {variable}")
                else:
                    # Use comprehensive trend analysis
                    trend_results = self._comprehensive_trend_analysis(data, location)
                    logger.info(f"🔬 Using COMPREHENSIVE trend analysis for {variable}")

                # Common analysis for both modes
                climate_indicators = self._calculate_enhanced_climate_indicators(
                    data, variable, location
                )
                projections = self._project_future_trends(data, variable)

                results[variable] = {
                    "trend_analysis": trend_results,
                    "climate_indicators": climate_indicators,
                    "future_projections": projections,
                    "analysis_mode_used": prediction_mode,  # 🆕 Track which mode was used
                    "summary": self._generate_enhanced_trend_summary(
                        trend_results, variable
                    ),
                    "confidence_score": self._calculate_trend_confidence(trend_results),
                }

            except Exception as e:
                logger.error(f"Enhanced trend analysis error for {variable}: {e}")
                results[variable] = self._error_result(str(e))

        return results

    def _fast_trend_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Fast trend analysis for pre-trained mode (< 1 second)
        """
        try:
            # Essential trend methods only
            trend_results = {
                "mann_kendall": self._mann_kendall_test(data),
                "linear_regression": self._linear_regression_trend(data),
                "seasonal_trends": self._seasonal_trend_analysis(data),
                "decadal_comparison": self._decadal_comparison(data),
            }

            return trend_results

        except Exception as e:
            return {"error": str(e)}

    def _comprehensive_trend_analysis(
        self, data: np.ndarray, location: Tuple[float, float]
    ) -> Dict[str, Any]:
        """
        Comprehensive trend analysis for live training mode (3-5 seconds)
        """
        try:
            # All advanced trend methods
            trend_results = {
                "mann_kendall": self._mann_kendall_test(data),
                "linear_regression": self._linear_regression_trend(data),
                "polynomial_trends": self._polynomial_trend_analysis(data),
                "ml_trend_detection": self._ml_trend_analysis(data, location),
                "seasonal_trends": self._seasonal_trend_analysis(data),
                "change_point_detection": self._advanced_change_point_detection(data),
                "decadal_comparison": self._decadal_comparison(data),
                "breakpoint_analysis": self._structural_break_analysis(data),
            }

            return trend_results

        except Exception as e:
            return {"error": str(e)}

    def _ml_trend_analysis(self, data: np.ndarray, location: Tuple[float, float]) -> Dict[str, Any]:
        """
        Machine learning-based trend detection
        """
        try:
            # Prepare features for ML trend detection
            features = self._create_trend_features(data, location)

            if len(features) < 24:  # Need sufficient data
                return {'error': 'Insufficient data for ML trend analysis'}

            # Target: rate of change over time
            targets = np.gradient(data)

            # Align features and targets
            min_len = min(len(features), len(targets))
            features = features[:min_len]
            targets = targets[:min_len]

            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'elastic_net': ElasticNet(alpha=0.1, random_state=42)
            }

            model_results = {}

            for name, model in models.items():
                try:
                    # Time series cross-validation
                    tscv = TimeSeriesSplit(n_splits=3)
                    scores = cross_val_score(model, features, targets, cv=tscv, scoring='r2')

                    # Fit model
                    model.fit(features, targets)

                    # Feature importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = np.abs(model.coef_) if hasattr(model, 'coef_') else np.zeros(len(features[0]))

                    # Predict trend direction
                    recent_features = features[-12:]  # Last year
                    trend_predictions = model.predict(recent_features)
                    avg_trend = np.mean(trend_predictions)

                    model_results[name] = {
                        'cv_score': float(np.mean(scores)),
                        'trend_direction': 'increasing' if avg_trend > 0.01 else 'decreasing' if avg_trend < -0.01 else 'stable',
                        'trend_magnitude': float(avg_trend),
                        'feature_importance': importance.tolist(),
                        'model_confidence': float(np.mean(scores)) if np.mean(scores) > 0 else 0.0
                    }

                except Exception as e:
                    logger.warning(f"ML model {name} failed: {e}")
                    continue

            # Ensemble trend prediction
            if model_results:
                # Weight by model performance
                weights = {name: max(0, result['cv_score']) for name, result in model_results.items()}
                total_weight = sum(weights.values()) if sum(weights.values()) > 0 else 1

                ensemble_trend = sum(
                    result['trend_magnitude'] * weights.get(name, 0) / total_weight
                    for name, result in model_results.items()
                )

                ensemble_direction = 'increasing' if ensemble_trend > 0.01 else 'decreasing' if ensemble_trend < -0.01 else 'stable'

                return {
                    'individual_models': model_results,
                    'ensemble_trend': {
                        'direction': ensemble_direction,
                        'magnitude': float(ensemble_trend),
                        'confidence': float(np.mean([r['model_confidence'] for r in model_results.values()]))
                    },
                    'best_model': max(model_results.items(), key=lambda x: x[1]['cv_score'])[0] if model_results else None
                }
            else:
                return {'error': 'All ML models failed'}

        except Exception as e:
            logger.error(f"ML trend analysis error: {e}")
            return {'error': str(e)}

    def _create_trend_features(self, data: np.ndarray, location: Tuple[float, float]) -> np.ndarray:
        """
        Create comprehensive features for trend analysis
        """
        features = []

        for i in range(12, len(data)):  # Start from month 12
            feature_vector = []

            # Time-based features
            feature_vector.extend([
                i / len(data),  # Normalized time
                (i % 12) / 12,  # Seasonal position
                np.sin(2 * np.pi * (i % 12) / 12),  # Seasonal sine
                np.cos(2 * np.pi * (i % 12) / 12),  # Seasonal cosine
            ])

            # Historical context features
            feature_vector.extend([
                np.mean(data[max(0, i-12):i]),  # 12-month average
                np.std(data[max(0, i-12):i]),   # 12-month std
                data[i-1] if i > 0 else data[0],  # Previous value
                np.mean(data[max(0, i-6):i]),   # 6-month average
            ])

            # Trend features
            if i >= 24:
                recent_slope = np.polyfit(range(12), data[i-12:i], 1)[0]
                long_slope = np.polyfit(range(24), data[i-24:i], 1)[0]
                feature_vector.extend([recent_slope, long_slope])
            else:
                feature_vector.extend([0.0, 0.0])

            # Location features
            feature_vector.extend([
                location[0] / 90.0,  # Normalized latitude
                location[1] / 180.0,  # Normalized longitude
                abs(location[0]) / 90.0,  # Distance from equator
            ])

            # Variability features
            if i >= 12:
                variability = np.std(data[i-12:i])
                range_val = np.max(data[i-12:i]) - np.min(data[i-12:i])
                feature_vector.extend([variability, range_val])
            else:
                feature_vector.extend([0.0, 0.0])

            features.append(feature_vector)

        return np.array(features)

    def _polynomial_trend_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Polynomial trend analysis for non-linear patterns
        """
        try:
            x = np.arange(len(data))

            # Fit polynomials of different degrees
            results = {}

            for degree in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(x, data, degree)
                    poly_func = np.poly1d(coeffs)
                    fitted_values = poly_func(x)

                    # Calculate R-squared
                    ss_res = np.sum((data - fitted_values) ** 2)
                    ss_tot = np.sum((data - np.mean(data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                    # Determine trend characteristics
                    if degree == 1:
                        trend_type = 'linear'
                        trend_direction = 'increasing' if coeffs[0] > 0 else 'decreasing'
                    elif degree == 2:
                        trend_type = 'quadratic'
                        trend_direction = 'accelerating' if coeffs[0] > 0 else 'decelerating'
                    else:
                        trend_type = 'cubic'
                        trend_direction = 'complex'

                    results[f'degree_{degree}'] = {
                        'coefficients': coeffs.tolist(),
                        'r_squared': float(r_squared),
                        'trend_type': trend_type,
                        'trend_direction': trend_direction,
                        'aic': self._calculate_aic(data, fitted_values, degree + 1)
                    }

                except Exception as e:
                    results[f'degree_{degree}'] = {'error': str(e)}

            # Select best model based on AIC
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            if valid_results:
                best_model = min(valid_results.items(), key=lambda x: x[1]['aic'])
                results['best_model'] = best_model[0]
                results['best_model_info'] = best_model[1]

            return results

        except Exception as e:
            return {'error': str(e)}

    def _calculate_aic(self, observed: np.ndarray, fitted: np.ndarray, num_params: int) -> float:
        """
        Calculate Akaike Information Criterion
        """
        n = len(observed)
        mse = np.mean((observed - fitted) ** 2)
        aic = n * np.log(mse) + 2 * num_params
        return float(aic)

    def _advanced_change_point_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Advanced change point detection using multiple methods
        """
        try:
            results = {}

            # Method 1: CUSUM with enhanced statistics
            cumsum = np.cumsum(data - np.mean(data))
            max_idx = np.argmax(np.abs(cumsum))
            max_deviation = abs(cumsum[max_idx])

            # Enhanced threshold calculation
            threshold = 2.5 * np.std(data) * np.sqrt(len(data))
            cusum_significant = max_deviation > threshold

            results["cusum"] = {
                "change_point_detected": safe_bool(cusum_significant),
                "change_point_index": int(max_idx) if cusum_significant else None,
                "change_magnitude": float(max_deviation),
                "significance_threshold": float(threshold),
                "confidence": (
                    "high"
                    if max_deviation > 1.5 * threshold
                    else "medium" if cusum_significant else "low"
                ),
            }

            # Method 2: Variance change detection
            results['variance_change'] = self._detect_variance_changes(data)

            # Method 3: Mean shift detection
            results['mean_shift'] = self._detect_mean_shifts(data)

            # Method 4: Trend change detection
            results['trend_change'] = self._detect_trend_changes(data)

            return results

        except Exception as e:
            return {'error': str(e)}

    def _detect_variance_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect changes in variance using sliding window
        """
        try:
            window_size = max(24, len(data) // 4)  # At least 2 years
            variances = []

            for i in range(window_size, len(data) - window_size):
                before_var = np.var(data[i-window_size:i])
                after_var = np.var(data[i:i+window_size])
                variance_ratio = after_var / before_var if before_var > 0 else 1.0
                variances.append((i, variance_ratio))

            if variances:
                # Find most significant variance change
                max_change = max(variances, key=lambda x: abs(np.log(x[1])))
                change_significant = abs(np.log(max_change[1])) > 0.5  # 50% change threshold

                return {
                    "change_detected": safe_bool(change_significant),
                    "change_index": int(max_change[0]) if change_significant else None,
                    "variance_ratio": float(max_change[1]),
                    "change_type": (
                        "increased_volatility"
                        if max_change[1] > 1.5
                        else (
                            "decreased_volatility"
                            if max_change[1] < 0.67
                            else "stable_volatility"
                        )
                    ),
                }
            else:
                return {"change_detected": safe_bool(False)}

        except Exception as e:
            return {'error': str(e)}

    def _detect_mean_shifts(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect significant shifts in mean using t-test
        """
        try:
            window_size = max(24, len(data) // 4)
            shifts = []

            for i in range(window_size, len(data) - window_size):
                before_data = data[i-window_size:i]
                after_data = data[i:i+window_size]

                # Perform t-test
                t_stat, p_value = stats.ttest_ind(before_data, after_data)

                if p_value < 0.05:  # Significant difference
                    mean_diff = np.mean(after_data) - np.mean(before_data)
                    shifts.append((i, abs(mean_diff), p_value, mean_diff))

            if shifts:
                # Find most significant shift
                most_significant = max(shifts, key=lambda x: x[1])

                return {
                    'shift_detected': safe_bool(True),
                    'shift_index': int(most_significant[0]),
                    'shift_magnitude': float(most_significant[3]),
                    'p_value': float(most_significant[2]),
                    'shift_direction': 'upward' if most_significant[3] > 0 else 'downward'
                }
            else:
                return {"shift_detected": safe_bool(False)}

        except Exception as e:
            return {'error': str(e)}

    def _detect_trend_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect changes in trend direction
        """
        try:
            window_size = max(36, len(data) // 3)  # At least 3 years
            trend_changes = []

            for i in range(window_size, len(data) - window_size):
                # Calculate trends before and after
                x_before = np.arange(window_size)
                x_after = np.arange(window_size)

                slope_before, _, _, p_before, _ = stats.linregress(x_before, data[i-window_size:i])
                slope_after, _, _, p_after, _ = stats.linregress(x_after, data[i:i+window_size])

                # Check if both trends are significant and different
                if p_before < 0.1 and p_after < 0.1:
                    slope_diff = abs(slope_after - slope_before)
                    if slope_diff > 0.01:  # Significant slope change
                        trend_changes.append((i, slope_before, slope_after, slope_diff))

            if trend_changes:
                # Find most significant trend change
                max_change = max(trend_changes, key=lambda x: x[3])

                return {
                    "trend_change_detected": safe_bool(True),
                    "change_index": int(max_change[0]),
                    "slope_before": float(max_change[1]),
                    "slope_after": float(max_change[2]),
                    "slope_difference": float(max_change[3]),
                    "change_description": self._describe_trend_change(
                        max_change[1], max_change[2]
                    ),
                }
            else:
                return {"trend_change_detected": safe_bool(False)}

        except Exception as e:
            return {'error': str(e)}

    def _describe_trend_change(self, slope_before: float, slope_after: float) -> str:
        """
        Describe the nature of trend change
        """
        if slope_before > 0 and slope_after > 0:
            if slope_after > slope_before:
                return "accelerating_positive_trend"
            else:
                return "decelerating_positive_trend"
        elif slope_before < 0 and slope_after < 0:
            if slope_after < slope_before:
                return "accelerating_negative_trend"
            else:
                return "decelerating_negative_trend"
        elif slope_before > 0 and slope_after < 0:
            return "trend_reversal_positive_to_negative"
        elif slope_before < 0 and slope_after > 0:
            return "trend_reversal_negative_to_positive"
        else:
            return "trend_stabilization"

    def _structural_break_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect structural breaks using Chow test approach
        """
        try:
            if len(data) < 60:  # Need at least 5 years
                return {'error': 'Insufficient data for structural break analysis'}

            # Test multiple potential break points
            n = len(data)
            min_segment = 24  # Minimum 2 years per segment

            break_tests = []

            for break_point in range(min_segment, n - min_segment, 6):  # Test every 6 months
                # Split data
                data1 = data[:break_point]
                data2 = data[break_point:]

                # Fit separate models
                x1 = np.arange(len(data1))
                x2 = np.arange(len(data2))

                # Linear regression for each segment
                slope1, intercept1, r1, p1, se1 = stats.linregress(x1, data1)
                slope2, intercept2, r2, p2, se2 = stats.linregress(x2, data2)

                # Calculate F-statistic for structural break
                # Simplified Chow test
                rss_restricted = np.sum((data - np.mean(data)) ** 2)
                rss_unrestricted = (np.sum((data1 - (slope1 * x1 + intercept1)) ** 2) + 
                                  np.sum((data2 - (slope2 * x2 + intercept2)) ** 2))

                if rss_unrestricted > 0:
                    f_stat = ((rss_restricted - rss_unrestricted) / 2) / (rss_unrestricted / (n - 4))

                    # Approximate p-value (simplified)
                    p_value = 1 - stats.f.cdf(f_stat, 2, n - 4) if f_stat > 0 else 1.0

                    break_tests.append(
                        {
                            "break_point": break_point,
                            "f_statistic": float(f_stat),
                            "p_value": float(p_value),
                            "slope_before": float(slope1),
                            "slope_after": float(slope2),
                            "significant": safe_bool(p_value < 0.05),
                        }
                    )

            if break_tests:
                # Find most significant break
                significant_breaks = [b for b in break_tests if b['significant']]

                if significant_breaks:
                    most_significant = min(significant_breaks, key=lambda x: x['p_value'])

                    return {
                        "structural_break_detected": safe_bool(True),
                        "most_significant_break": most_significant,
                        "all_significant_breaks": significant_breaks,
                        "total_breaks_tested": len(break_tests),
                    }
                else:
                    return {
                        'structural_break_detected': safe_bool(False),
                        'total_breaks_tested': len(break_tests)
                    }
            else:
                return {'error': 'No break points could be tested'}

        except Exception as e:
            return {'error': str(e)}

    def _calculate_enhanced_climate_indicators(self, data: np.ndarray, variable: str, 
                                             location: Tuple[float, float]) -> Dict[str, Any]:
        """
        Enhanced climate indicators with location-specific analysis
        """
        try:
            base_indicators = self._calculate_climate_indicators(data, variable)

            # Add location-specific enhancements
            lat, lon = location
            climate_zone = self._classify_climate_zone(lat)

            # Enhanced indicators based on climate zone
            if variable == 'temperature':
                enhanced = self._enhanced_temperature_indicators(data, climate_zone)
            elif variable == 'precipitation':
                enhanced = self._enhanced_precipitation_indicators(data, climate_zone)
            else:
                enhanced = {}

            # Combine base and enhanced indicators
            result = {**base_indicators, **enhanced}

            # Add climate zone context
            result['climate_zone_context'] = {
                'climate_zone': climate_zone,
                'expected_variability': self._get_expected_variability(variable, climate_zone),
                'anomaly_threshold': self._get_anomaly_threshold(data, climate_zone)
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def _classify_climate_zone(self, lat: float) -> str:
        """
        Classify climate zone based on latitude
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

    def _enhanced_temperature_indicators(self, data: np.ndarray, climate_zone: str) -> Dict[str, Any]:
        """
        Enhanced temperature indicators based on climate zone
        """
        try:
            # Zone-specific thresholds
            zone_thresholds = {
                'equatorial': {'hot': 32, 'cold': 22},
                'tropical': {'hot': 35, 'cold': 18},
                'subtropical': {'hot': 38, 'cold': 5},
                'temperate': {'hot': 30, 'cold': -5},
                'polar': {'hot': 15, 'cold': -20}
            }

            thresholds = zone_thresholds.get(climate_zone, {'hot': 30, 'cold': 0})

            # Calculate zone-specific indicators
            hot_days = np.sum(data > thresholds['hot'])
            cold_days = np.sum(data < thresholds['cold'])

            # Heat wave detection (3+ consecutive months above threshold)
            heat_waves = self._detect_consecutive_events(data, thresholds['hot'], min_duration=3)
            cold_spells = self._detect_consecutive_events(data, thresholds['cold'], min_duration=3, above=False)

            return {
                'zone_specific_hot_days': int(hot_days),
                'zone_specific_cold_days': int(cold_days),
                'heat_wave_events': heat_waves,
                'cold_spell_events': cold_spells,
                'temperature_extremes_trend': self._calculate_extremes_trend(data, thresholds)
            }

        except Exception as e:
            return {'error': str(e)}

    def _enhanced_precipitation_indicators(self, data: np.ndarray, climate_zone: str) -> Dict[str, Any]:
        """
        Enhanced precipitation indicators based on climate zone
        """
        try:
            # Zone-specific precipitation patterns
            zone_patterns = {
                'equatorial': {'wet_threshold': 200, 'dry_threshold': 50},
                'tropical': {'wet_threshold': 150, 'dry_threshold': 20},
                'subtropical': {'wet_threshold': 100, 'dry_threshold': 10},
                'temperate': {'wet_threshold': 80, 'dry_threshold': 15},
                'polar': {'wet_threshold': 50, 'dry_threshold': 5}
            }

            patterns = zone_patterns.get(climate_zone, {'wet_threshold': 100, 'dry_threshold': 20})

            # Detect wet and dry periods
            wet_months = np.sum(data > patterns['wet_threshold'])
            dry_months = np.sum(data < patterns['dry_threshold'])

            # Drought detection
            droughts = self._detect_consecutive_events(data, patterns['dry_threshold'], min_duration=6, above=False)

            # Flood risk periods
            flood_risk = self._detect_consecutive_events(data, patterns['wet_threshold'], min_duration=2)

            return {
                'zone_specific_wet_months': int(wet_months),
                'zone_specific_dry_months': int(dry_months),
                'drought_events': droughts,
                'flood_risk_periods': flood_risk,
                'precipitation_variability_index': float(np.std(data) / np.mean(data)) if np.mean(data) > 0 else 0.0
            }

        except Exception as e:
            return {'error': str(e)}

    def _detect_consecutive_events(self, data: np.ndarray, threshold: float, 
                                 min_duration: int = 3, above: bool = True) -> List[Dict[str, Any]]:
        """
        Detect consecutive events above or below threshold
        """
        events = []
        current_event = None

        for i, value in enumerate(data):
            condition_met = (value > threshold) if above else (value < threshold)

            if condition_met:
                if current_event is None:
                    current_event = {'start': i, 'values': [value]}
                else:
                    current_event['values'].append(value)
            else:
                if current_event is not None:
                    current_event['end'] = i - 1
                    current_event['duration'] = len(current_event['values'])

                    if current_event['duration'] >= min_duration:
                        current_event['intensity'] = float(np.mean(current_event['values']))
                        current_event['peak'] = float(np.max(current_event['values']) if above else np.min(current_event['values']))
                        events.append(current_event)

                    current_event = None

        # Handle event that continues to end of data
        if current_event is not None:
            current_event['end'] = len(data) - 1
            current_event['duration'] = len(current_event['values'])

            if current_event['duration'] >= min_duration:
                current_event['intensity'] = float(np.mean(current_event['values']))
                current_event['peak'] = float(np.max(current_event['values']) if above else np.min(current_event['values']))
                events.append(current_event)

        return events

    def _calculate_extremes_trend(self, data: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate trends in extreme events
        """
        try:
            # Create binary time series for extremes
            hot_events = (data > thresholds['hot']).astype(int)
            cold_events = (data < thresholds['cold']).astype(int)

            # Calculate trends in extreme frequency
            x = np.arange(len(data))

            hot_slope, _, hot_r, hot_p, _ = stats.linregress(x, hot_events)
            cold_slope, _, cold_r, cold_p, _ = stats.linregress(x, cold_events)

            return {
                "hot_extremes_trend": {
                    "slope_per_year": float(hot_slope * 12),
                    "r_squared": float(hot_r**2),
                    "p_value": float(hot_p),
                    "significant": safe_bool(hot_p < 0.05),
                    "direction": "increasing" if hot_slope > 0 else "decreasing",
                },
                "cold_extremes_trend": {
                    "slope_per_year": float(cold_slope * 12),
                    "r_squared": float(cold_r**2),
                    "p_value": float(cold_p),
                    "significant": safe_bool(cold_p < 0.05),
                    "direction": "increasing" if cold_slope > 0 else "decreasing",
                },
            }

        except Exception as e:
            return {'error': str(e)}

    def _get_expected_variability(self, variable: str, climate_zone: str) -> Dict[str, float]:
        """
        Get expected variability ranges for different climate zones
        """
        variability_ranges = {
            'temperature': {
                'equatorial': {'low': 1.0, 'high': 3.0},
                'tropical': {'low': 2.0, 'high': 5.0},
                'subtropical': {'low': 3.0, 'high': 8.0},
                'temperate': {'low': 5.0, 'high': 12.0},
                'polar': {'low': 8.0, 'high': 20.0}
            },
            'precipitation': {
                'equatorial': {'low': 20.0, 'high': 80.0},
                'tropical': {'low': 30.0, 'high': 120.0},
                'subtropical': {'low': 15.0, 'high': 60.0},
                'temperate': {'low': 10.0, 'high': 40.0},
                'polar': {'low': 5.0, 'high': 25.0}
            }
        }

        return variability_ranges.get(variable, {}).get(climate_zone, {'low': 0.0, 'high': 10.0})

    def _get_anomaly_threshold(self, data: np.ndarray, climate_zone: str) -> float:
        """
        Calculate anomaly threshold based on climate zone
        """
        std_dev = np.std(data)

        # Climate zone specific multipliers
        zone_multipliers = {
            'equatorial': 1.5,
            'tropical': 1.8,
            'subtropical': 2.0,
            'temperate': 2.2,
            'polar': 2.5
        }

        multiplier = zone_multipliers.get(climate_zone, 2.0)
        return float(std_dev * multiplier)

    def _ml_future_projections(self, data: np.ndarray, variable: str, 
                              location: Tuple[float, float]) -> Dict[str, Any]:
        """
        ML-based future projections with uncertainty quantification
        """
        try:
            # Create features for projection
            features = self._create_trend_features(data, location)

            if len(features) < 24:
                return self._project_future_trends(data, variable)  # Fallback to simple method

            # Prepare targets (future values)
            targets = data[12:]  # Shift by 12 months
            features = features[:-12]  # Remove last 12 features

            # Train ensemble models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=200, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=200, random_state=42),
                'linear': LinearRegression()
            }

            projections = {}

            for name, model in models.items():
                try:
                    # Train model
                    model.fit(features, targets)

                    # Create future features (next 5 years)
                    future_features = self._create_future_features(data, location, 60)  # 5 years

                    # Make predictions
                    future_predictions = model.predict(future_features)

                    # Calculate prediction intervals using bootstrap
                    prediction_intervals = self._calculate_prediction_intervals(
                        model, features, targets, future_features
                    )

                    projections[name] = {
                        'predictions': future_predictions.tolist(),
                        'prediction_intervals': prediction_intervals,
                        'model_score': float(model.score(features, targets))
                    }

                except Exception as e:
                    logger.warning(f"ML projection model {name} failed: {e}")
                    continue

            if projections:
                # Ensemble projection
                ensemble_pred = np.mean([p['predictions'] for p in projections.values()], axis=0)

                # Calculate ensemble uncertainty
                pred_std = np.std([p['predictions'] for p in projections.values()], axis=0)

                return {
                    'ml_projections': projections,
                    'ensemble_projection': ensemble_pred.tolist(),
                    'projection_uncertainty': pred_std.tolist(),
                    'projection_horizon_months': 60,
                    'confidence_level': float(np.mean([p['model_score'] for p in projections.values()])),
                    'methodology': 'ML Ensemble (RF + GB + Linear)'
                }
            else:
                return self._project_future_trends(data, variable)  # Fallback

        except Exception as e:
            logger.error(f"ML future projections error: {e}")
            return self._project_future_trends(data, variable)  # Fallback

    def _create_future_features(self, data: np.ndarray, location: Tuple[float, float], 
                               months_ahead: int) -> np.ndarray:
        """
        Create features for future time points
        """
        current_length = len(data)
        future_features = []

        for i in range(months_ahead):
            future_time = current_length + i
            feature_vector = []

            # Time-based features
            feature_vector.extend([
                future_time / (current_length + months_ahead),  # Normalized time
                ((future_time % 12)) / 12,  # Seasonal position
                np.sin(2 * np.pi * (future_time % 12) / 12),  # Seasonal sine
                np.cos(2 * np.pi * (future_time % 12) / 12),  # Seasonal cosine
            ])

            # Use recent historical context
            recent_data = data[-12:] if len(data) >= 12 else data
            feature_vector.extend([
                np.mean(recent_data),  # Recent average
                np.std(recent_data),   # Recent std
                data[-1],              # Last value
                np.mean(data[-6:]) if len(data) >= 6 else np.mean(data),  # 6-month average
            ])

            # Trend features (based on recent data)
            if len(data) >= 24:
                recent_slope = np.polyfit(range(12), data[-12:], 1)[0]
                long_slope = np.polyfit(range(24), data[-24:], 1)[0]
                feature_vector.extend([recent_slope, long_slope])
            else:
                feature_vector.extend([0.0, 0.0])

            # Location features
            feature_vector.extend([
                location[0] / 90.0,  # Normalized latitude
                location[1] / 180.0,  # Normalized longitude
                abs(location[0]) / 90.0,  # Distance from equator
            ])

            # Variability features
            variability = np.std(recent_data)
            range_val = np.max(recent_data) - np.min(recent_data)
            feature_vector.extend([variability, range_val])

            future_features.append(feature_vector)

        return np.array(future_features)

    def _calculate_prediction_intervals(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                                      X_future: np.ndarray, n_bootstrap: int = 100) -> Dict[str, List[float]]:
        """
        Calculate prediction intervals using bootstrap
        """
        try:
            predictions = []

            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]

                # Train model on bootstrap sample
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_boot, y_boot)

                # Predict
                pred = model_copy.predict(X_future)
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate percentiles
            lower_95 = np.percentile(predictions, 2.5, axis=0)
            lower_80 = np.percentile(predictions, 10, axis=0)
            upper_80 = np.percentile(predictions, 90, axis=0)
            upper_95 = np.percentile(predictions, 97.5, axis=0)

            return {
                'lower_95': lower_95.tolist(),
                'lower_80': lower_80.tolist(),
                'upper_80': upper_80.tolist(),
                'upper_95': upper_95.tolist()
            }

        except Exception as e:
            logger.error(f"Prediction intervals error: {e}")
            return {
                'lower_95': [0.0] * len(X_future),
                'lower_80': [0.0] * len(X_future),
                'upper_80': [0.0] * len(X_future),
                'upper_95': [0.0] * len(X_future)
            }

    def _validate_trend_models(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Validate trend models using walk-forward validation
        """
        try:
            if len(data) < 48:  # Need at least 4 years
                return {'error': 'Insufficient data for validation'}

            # Use last 2 years for validation
            validation_months = 24
            train_data = data[:-validation_months]
            test_data = data[-validation_months:]

            # Simple trend validation
            x_train = np.arange(len(train_data))
            slope, intercept, _, _, _ = stats.linregress(x_train, train_data)

            # Project trend
            x_test = np.arange(len(train_data), len(data))
            trend_predictions = slope * x_test + intercept

            # Calculate metrics
            mae = np.mean(np.abs(trend_predictions - test_data))
            rmse = np.sqrt(np.mean((trend_predictions - test_data) ** 2))
            mape = np.mean(np.abs((trend_predictions - test_data) / test_data)) * 100

            # Trend direction accuracy
            actual_trend = np.polyfit(range(len(test_data)), test_data, 1)[0]
            predicted_trend = slope
            trend_direction_correct = (actual_trend > 0) == (predicted_trend > 0)

            return {
                'validation_metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'trend_direction_accuracy': bool(trend_direction_correct.item()) if hasattr(trend_direction_correct, 'item') else bool(trend_direction_correct)                },
                'validation_period_months': validation_months,
                'trend_prediction_skill': float(max(0, 1 - rmse / np.std(test_data)))
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_trend_confidence(self, trend_results: Dict[str, Any]) -> float:
        """
        Calculate overall confidence in trend analysis
        """
        try:
            confidence_factors = []

            # Mann-Kendall confidence
            mk_result = trend_results.get('mann_kendall', {})
            if mk_result.get('is_significant', False):
                p_val = mk_result.get('p_value', 1.0)
                mk_confidence = 1.0 - p_val
                confidence_factors.append(mk_confidence)

            # Linear regression confidence
            lr_result = trend_results.get('linear_regression', {})
            r_squared = lr_result.get('r_squared', 0.0)
            confidence_factors.append(r_squared)

            # ML trend confidence
            ml_result = trend_results.get('ml_trend_detection', {})
            if 'ensemble_trend' in ml_result:
                ml_confidence = ml_result['ensemble_trend'].get('confidence', 0.0)
                confidence_factors.append(ml_confidence)

            # Polynomial analysis confidence
            poly_result = trend_results.get('polynomial_trends', {})
            if 'best_model_info' in poly_result:
                poly_r2 = poly_result['best_model_info'].get('r_squared', 0.0)
                confidence_factors.append(poly_r2)

            # Calculate weighted average
            if confidence_factors:
                return float(np.mean(confidence_factors))
            else:
                return 0.5  # Default moderate confidence

        except Exception as e:
            return 0.3  # Low confidence on error

    def _generate_enhanced_trend_summary(self, trend_results: Dict[str, Any], variable: str) -> str:
        """
        Generate comprehensive trend summary
        """
        try:
            summaries = []

            # Mann-Kendall summary
            mk_result = trend_results.get('mann_kendall', {})
            if mk_result.get('is_significant', False):
                direction = mk_result.get('trend_direction', 'unknown')
                confidence = mk_result.get('confidence', 'low')
                summaries.append(f"Mann-Kendall test shows {confidence} confidence {direction} trend")

            # Linear regression summary
            lr_result = trend_results.get('linear_regression', {})
            slope = lr_result.get('slope_per_year', 0)
            strength = lr_result.get('trend_strength', 'unknown')
            if abs(slope) > 0.01:
                summaries.append(f"Linear trend: {abs(slope):.3f} units/year ({strength})")

            # ML trend summary
            ml_result = trend_results.get('ml_trend_detection', {})
            if 'ensemble_trend' in ml_result:
                ml_direction = ml_result['ensemble_trend'].get('direction', 'stable')
                ml_confidence = ml_result['ensemble_trend'].get('confidence', 0)
                if ml_confidence > 0.6:
                    summaries.append(f"ML ensemble confirms {ml_direction} trend (confidence: {ml_confidence:.2f})")

            # Change point summary
            change_result = trend_results.get('change_point_detection', {})
            if isinstance(change_result, dict) and 'cusum' in change_result:
                if change_result['cusum'].get('change_point_detected', False):
                    summaries.append("Significant change point detected in time series")

            # Combine summaries
            if summaries:
                return f"{variable.title()}: " + "; ".join(summaries)
            else:
                return f"No significant trends detected in {variable}"

        except Exception as e:
            return f"Error generating summary for {variable}: {str(e)}"

    # Keep all your existing methods for compatibility
    def _mann_kendall_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Mann-Kendall trend test for non-parametric trend detection"""
        try:
            trend, h, p, z, tau, s, var_s, slope, intercept = mk.original_test(data)

            return {
                "trend_direction": trend,
                "is_significant": safe_bool(h),
                "p_value": float(p),
                "z_statistic": float(z),
                "tau": float(tau),
                "slope": float(slope),
                "confidence": "high" if p < 0.01 else "medium" if p < 0.05 else "low",
            }
        except Exception as e:
            return {'error': str(e)}

    def _linear_regression_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Linear regression trend analysis"""
        try:
            x = np.arange(len(data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

            # Convert monthly slope to annual
            annual_slope = slope * 12

            return {
                'slope_per_year': float(annual_slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'standard_error': float(std_err),
                'trend_strength': self._classify_trend_strength(abs(annual_slope), np.std(data))
            }
        except Exception as e:
            return {'error': str(e)}

    def _seasonal_trend_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trends by season"""
        try:
            years = len(data) // 12
            seasonal_data = data[:years * 12].reshape(years, 12)

            seasonal_trends = {}
            seasons = {
                'winter': [11, 0, 1],  # Dec, Jan, Feb
                'spring': [2, 3, 4],   # Mar, Apr, May
                'summer': [5, 6, 7],   # Jun, Jul, Aug
                'autumn': [8, 9, 10]   # Sep, Oct, Nov
            }

            for season_name, months in seasons.items():
                season_data = seasonal_data[:, months].mean(axis=1)

                if len(season_data) >= self.min_years_for_trend:
                    x = np.arange(len(season_data))
                    slope, _, r_value, p_value, _ = stats.linregress(x, season_data)

                    seasonal_trends[season_name] = {
                        "slope_per_year": float(slope),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "is_significant": safe_bool(p_value < self.significance_level),
                    }

            return seasonal_trends
        except Exception as e:
            return {'error': str(e)}

    def _detect_change_points(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect significant change points in the time series"""
        try:
            # Simple change point detection using cumulative sum
            cumsum = np.cumsum(data - np.mean(data))

            # Find maximum deviation point
            max_idx = np.argmax(np.abs(cumsum))
            max_deviation = abs(cumsum[max_idx])

            # Statistical significance test (simplified)
            threshold = 2 * np.std(data) * np.sqrt(len(data))
            is_significant = max_deviation > threshold

            change_year = 1994 + (max_idx // 12)  # Assuming data starts from 1994

            return {
                "change_point_detected": safe_bool(is_significant),
                "change_point_year": int(change_year) if is_significant else None,
                "change_magnitude": float(max_deviation),
                "significance_threshold": float(threshold),
            }
        except Exception as e:
            return {'error': str(e)}

    def _decadal_comparison(self, data: np.ndarray) -> Dict[str, Any]:
        """Compare recent decade with historical average"""
        try:
            years = len(data) // 12
            if years < 20:
                return {'error': 'Insufficient data for decadal comparison'}

            # Last 10 years vs previous decades
            recent_decade = data[-120:]  # Last 10 years (120 months)
            historical = data[:-120]     # All previous data

            recent_mean = np.mean(recent_decade)
            historical_mean = np.mean(historical)

            # Statistical test for difference
            t_stat, p_value = stats.ttest_ind(recent_decade, historical)

            return {
                "recent_decade_mean": float(recent_mean),
                "historical_mean": float(historical_mean),
                "difference": float(recent_mean - historical_mean),
                "percent_change": float(
                    (recent_mean - historical_mean) / historical_mean * 100
                ),
                "is_significantly_different": safe_bool(
                    p_value < self.significance_level
                ),
                "p_value": float(p_value),
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_climate_indicators(self, data: np.ndarray, variable: str) -> Dict[str, Any]:
        """Calculate climate change indicators specific to each variable"""
        try:
            if variable == 'temperature':
                return self._temperature_climate_indicators(data)
            elif variable == 'precipitation':
                return self._precipitation_climate_indicators(data)
            else:
                return self._general_climate_indicators(data)
        except Exception as e:
            return {'error': str(e)}

    def _temperature_climate_indicators(self, data: np.ndarray) -> Dict[str, Any]:
        """Temperature-specific climate indicators"""
        years = len(data) // 12
        annual_data = data[:years * 12].reshape(years, 12).mean(axis=1)

        # Warming indicators
        warming_rate = np.polyfit(range(len(annual_data)), annual_data, 1)[0]

        # Extreme temperature frequency
        hot_threshold = np.percentile(data, 90)
        cold_threshold = np.percentile(data, 10)

        recent_hot_freq = np.mean(data[-120:] > hot_threshold) * 100
        recent_cold_freq = np.mean(data[-120:] < cold_threshold) * 100

        return {
            'warming_rate_per_decade': float(warming_rate * 10),
            'recent_hot_days_percent': float(recent_hot_freq),
            'recent_cold_days_percent': float(recent_cold_freq),
            'temperature_variability_change': float(np.std(data[-120:]) - np.std(data[:-120]))
        }

    def _precipitation_climate_indicators(self, data: np.ndarray) -> Dict[str, Any]:
        """Precipitation-specific climate indicators"""
        # Precipitation intensity and frequency changes
        wet_threshold = np.percentile(data[data > 0], 75) if np.any(data > 0) else 1.0

        recent_intense_freq = np.mean(data[-120:] > wet_threshold) * 100
        historical_intense_freq = np.mean(data[:-120] > wet_threshold) * 100

        return {
            'intense_precipitation_change': float(recent_intense_freq - historical_intense_freq),
            'precipitation_variability_change': float(np.std(data[-120:]) - np.std(data[:-120])),
            'dry_spell_frequency': self._calculate_dry_spell_frequency(data)
        }

    def _general_climate_indicators(self, data: np.ndarray) -> Dict[str, Any]:
        """General climate indicators for other variables"""
        return {
            'variability_change': float(np.std(data[-120:]) - np.std(data[:-120])),
            'extreme_frequency_change': self._calculate_extreme_frequency_change(data)
        }

    def _calculate_dry_spell_frequency(self, data: np.ndarray) -> float:
        """Calculate frequency of dry spells"""
        dry_threshold = np.percentile(data, 25)
        dry_months = data < dry_threshold

        # Count consecutive dry periods
        dry_spells = 0
        in_spell = False

        for is_dry in dry_months[-120:]:  # Recent 10 years
            if is_dry and not in_spell:
                dry_spells += 1
                in_spell = True
            elif not is_dry:
                in_spell = False

        return float(dry_spells / 10)  # Spells per year

    def _calculate_extreme_frequency_change(self, data: np.ndarray) -> float:
        """Calculate change in extreme event frequency"""
        extreme_threshold = np.percentile(data, 95)

        recent_extreme_freq = np.mean(data[-120:] > extreme_threshold)
        historical_extreme_freq = np.mean(data[:-120] > extreme_threshold)

        return float(recent_extreme_freq - historical_extreme_freq)

    def _project_future_trends(self, data: np.ndarray, variable: str) -> Dict[str, Any]:
        """Project future trends based on historical analysis"""
        try:
            # Linear projection for next 10 years
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)

            # Project 10 years ahead (120 months)
            future_x = np.arange(len(data), len(data) + 120)
            future_values = slope * future_x + intercept

            return {
                'projected_change_10_years': float(slope * 120),
                'projected_mean_future': float(np.mean(future_values)),
                'uncertainty_range': float(2 * np.std(data)),
                'projection_method': 'linear_extrapolation'
            }
        except Exception as e:
            return {'error': str(e)}

    def _classify_trend_strength(self, slope: float, data_std: float) -> str:
        """Classify trend strength relative to data variability"""
        if data_std == 0:
            return "undefined"

        relative_slope = abs(slope) / data_std

        if relative_slope > 0.2:
            return 'very_strong'
        elif relative_slope > 0.1:
            return 'strong'
        elif relative_slope > 0.05:
            return 'moderate'
        elif relative_slope > 0.01:
            return 'weak'
        else:
            return 'negligible'

    def validate_trend_accuracy(self, historical_data: np.ndarray, validation_years: int = 5):
        """Validate trend accuracy using cross-validation"""
        # Split data for validation
        train_data = historical_data[: -validation_years * 12]
        test_data = historical_data[-validation_years * 12 :]

        # Calculate trend on training data
        trend_result = self._linear_regression_trend(train_data)
        slope = trend_result["slope_per_year"] / 12  # Monthly slope

        # Project forward and compare with actual
        projected = []
        for i in range(len(test_data)):
            projected_value = train_data[-1] + (slope * (i + 1))
            projected.append(projected_value)

        # Calculate accuracy metrics
        mae = np.mean(np.abs(np.array(projected) - test_data))
        rmse = np.sqrt(np.mean((np.array(projected) - test_data) ** 2))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "accuracy_score": max(0, 1 - (rmse / np.std(test_data))),
        }

    def _insufficient_data_result(self) -> Dict[str, Any]:
        """Return result for insufficient data"""
        return {
            'trend_analysis': {'error': 'Insufficient data for trend analysis'},
            'climate_indicators': {'error': 'Insufficient data'},
            'future_projections': {'error': 'Insufficient data'},
            'summary': 'Insufficient data for trend analysis'
        }

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            'trend_analysis': {'error': error_msg},
            'climate_indicators': {'error': error_msg},
            'future_projections': {'error': error_msg},
            'summary': f'Error in trend analysis: {error_msg}'
        }
