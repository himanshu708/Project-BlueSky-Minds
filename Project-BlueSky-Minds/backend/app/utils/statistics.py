import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def calculate_percentiles(
    data: np.ndarray, percentiles: List[float] = None
) -> Dict[str, float]:
    """
    Calculate percentiles for given data
    """
    if percentiles is None:
        percentiles = [5, 10, 25, 50, 75, 90, 95]

    result = {}
    for p in percentiles:
        result[f"p{int(p)}"] = float(np.percentile(data, p))

    return result


def calculate_return_periods(data: np.ndarray, values: List[float]) -> Dict[str, float]:
    """
    Calculate return periods for extreme values
    """
    sorted_data = np.sort(data)[::-1]  # Sort descending
    n = len(data)

    return_periods = {}
    for value in values:
        # Find rank of value
        rank = np.sum(sorted_data >= value) + 1
        return_period = n / rank if rank > 0 else np.inf
        return_periods[f"value_{value}"] = float(return_period)

    return return_periods


def detect_outliers(data: np.ndarray, method: str = "iqr") -> Dict[str, Any]:
    """
    Detect outliers using various methods
    """
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]

        return {
            "method": "IQR",
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(data) * 100,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers": outliers.tolist(),
        }

    elif method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        threshold = 3
        outliers = data[z_scores > threshold]

        return {
            "method": "Z-Score",
            "threshold": threshold,
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(data) * 100,
            "outliers": outliers.tolist(),
        }


def calculate_moving_averages(
    data: np.ndarray, windows: List[int] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate moving averages for different window sizes
    """
    if windows is None:
        windows = [3, 6, 12]  # 3, 6, 12 month moving averages

    moving_averages = {}
    for window in windows:
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window) / window, mode="valid")
            moving_averages[f"ma_{window}"] = ma

    return moving_averages


def seasonal_decomposition(data: np.ndarray, period: int = 12) -> Dict[str, np.ndarray]:
    """
    Simple seasonal decomposition (trend + seasonal + residual)
    """
    if len(data) < 2 * period:
        return {"error": "Insufficient data for seasonal decomposition"}

    # Calculate trend using moving average
    trend = np.convolve(data, np.ones(period) / period, mode="same")

    # Calculate seasonal component
    detrended = data - trend
    seasonal = np.zeros_like(data)

    for i in range(period):
        seasonal_values = detrended[i::period]
        seasonal[i::period] = np.mean(seasonal_values[~np.isnan(seasonal_values)])

    # Calculate residual
    residual = data - trend - seasonal

    return {
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "original": data,
    }


def calculate_autocorrelation(data: np.ndarray, max_lag: int = 24) -> Dict[str, Any]:
    """
    Calculate autocorrelation function
    """
    autocorr = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            if len(data) > lag:
                corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0.0)
            else:
                autocorr.append(0.0)

    return {
        "lags": list(range(max_lag + 1)),
        "autocorrelation": autocorr,
        "significant_lags": [i for i, ac in enumerate(autocorr) if abs(ac) > 0.2],
    }
