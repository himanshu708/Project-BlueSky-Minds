import numpy as np
from scipy import interpolate
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def interpolate_missing_data(
    data: np.ndarray, method: str = "linear", max_gap: int = 3
) -> Dict[str, Any]:
    """
    Interpolate missing data points
    """
    # Find missing data (NaN values)
    missing_mask = np.isnan(data)
    valid_mask = ~missing_mask

    if not np.any(missing_mask):
        return {
            "interpolated_data": data,
            "missing_count": 0,
            "interpolation_method": method,
            "gaps_filled": 0,
        }

    # Check gap sizes
    gaps = []
    in_gap = False
    gap_start = 0

    for i, is_missing in enumerate(missing_mask):
        if is_missing and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_missing and in_gap:
            gap_size = i - gap_start
            gaps.append((gap_start, i - 1, gap_size))
            in_gap = False

    # Handle gap at end
    if in_gap:
        gap_size = len(data) - gap_start
        gaps.append((gap_start, len(data) - 1, gap_size))

    # Filter gaps by maximum allowed size
    fillable_gaps = [gap for gap in gaps if gap[2] <= max_gap]

    interpolated_data = data.copy()
    gaps_filled = 0

    if method == "linear":
        # Linear interpolation
        valid_indices = np.where(valid_mask)[0]
        valid_values = data[valid_mask]

        if len(valid_indices) >= 2:
            f = interpolate.interp1d(
                valid_indices,
                valid_values,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            for gap_start, gap_end, gap_size in fillable_gaps:
                if gap_size <= max_gap:
                    gap_indices = np.arange(gap_start, gap_end + 1)
                    interpolated_data[gap_indices] = f(gap_indices)
                    gaps_filled += 1

    elif method == "spline":
        # Cubic spline interpolation
        valid_indices = np.where(valid_mask)[0]
        valid_values = data[valid_mask]

        if len(valid_indices) >= 4:  # Need at least 4 points for cubic spline
            try:
                f = interpolate.CubicSpline(valid_indices, valid_values)

                for gap_start, gap_end, gap_size in fillable_gaps:
                    if gap_size <= max_gap:
                        gap_indices = np.arange(gap_start, gap_end + 1)
                        interpolated_data[gap_indices] = f(gap_indices)
                        gaps_filled += 1
            except Exception as e:
                logger.warning(
                    f"Spline interpolation failed: {e}, falling back to linear"
                )
                return interpolate_missing_data(data, method="linear", max_gap=max_gap)

    elif method == "seasonal":
        # Seasonal interpolation (use same month from other years)
        period = 12  # Monthly data

        for gap_start, gap_end, gap_size in fillable_gaps:
            if gap_size <= max_gap:
                for i in range(gap_start, gap_end + 1):
                    month_index = i % period

                    # Find all valid values for this month
                    month_values = []
                    for j in range(month_index, len(data), period):
                        if not np.isnan(data[j]) and j != i:
                            month_values.append(data[j])

                    if month_values:
                        interpolated_data[i] = np.mean(month_values)

                gaps_filled += 1

    return {
        "interpolated_data": interpolated_data,
        "missing_count": np.sum(missing_mask),
        "interpolation_method": method,
        "gaps_filled": gaps_filled,
        "total_gaps": len(gaps),
        "fillable_gaps": len(fillable_gaps),
        "gap_details": gaps,
    }


def spatial_interpolation(
    coordinates: List[Tuple[float, float]],
    values: List[float],
    target_coords: Tuple[float, float],
    method: str = "idw",
) -> Dict[str, Any]:
    """
    Spatial interpolation for estimating values at new locations
    """
    coords_array = np.array(coordinates)
    values_array = np.array(values)
    target_lat, target_lon = target_coords

    if method == "idw":  # Inverse Distance Weighting
        # Calculate distances
        distances = np.sqrt(
            (coords_array[:, 0] - target_lat) ** 2
            + (coords_array[:, 1] - target_lon) ** 2
        )

        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)

        # Calculate weights (inverse distance squared)
        weights = 1 / (distances**2)
        weights = weights / np.sum(weights)

        # Interpolated value
        interpolated_value = np.sum(weights * values_array)

        return {
            "interpolated_value": float(interpolated_value),
            "method": "inverse_distance_weighting",
            "weights": weights.tolist(),
            "distances": distances.tolist(),
            "source_points": len(coordinates),
        }

    elif method == "nearest":
        # Nearest neighbor
        distances = np.sqrt(
            (coords_array[:, 0] - target_lat) ** 2
            + (coords_array[:, 1] - target_lon) ** 2
        )

        nearest_idx = np.argmin(distances)

        return {
            "interpolated_value": float(values_array[nearest_idx]),
            "method": "nearest_neighbor",
            "nearest_distance": float(distances[nearest_idx]),
            "nearest_coords": coordinates[nearest_idx],
            "source_points": len(coordinates),
        }


def temporal_interpolation(
    timestamps: List[int],
    values: List[float],
    target_timestamp: int,
    method: str = "linear",
) -> Dict[str, Any]:
    """
    Temporal interpolation for estimating values at new time points
    """
    timestamps_array = np.array(timestamps)
    values_array = np.array(values)

    if method == "linear":
        # Find surrounding points
        before_mask = timestamps_array <= target_timestamp
        after_mask = timestamps_array >= target_timestamp

        if not np.any(before_mask) or not np.any(after_mask):
            return {"error": "Target timestamp outside data range"}

        before_idx = np.where(before_mask)[0][-1]  # Last point before
        after_idx = np.where(after_mask)[0][0]  # First point after

        if before_idx == after_idx:
            # Exact match
            return {
                "interpolated_value": float(values_array[before_idx]),
                "method": "exact_match",
                "confidence": 1.0,
            }

        # Linear interpolation
        t1, t2 = timestamps_array[before_idx], timestamps_array[after_idx]
        v1, v2 = values_array[before_idx], values_array[after_idx]

        # Interpolation formula
        weight = (target_timestamp - t1) / (t2 - t1)
        interpolated_value = v1 + weight * (v2 - v1)

        return {
            "interpolated_value": float(interpolated_value),
            "method": "linear_temporal",
            "weight": float(weight),
            "before_point": {"timestamp": int(t1), "value": float(v1)},
            "after_point": {"timestamp": int(t2), "value": float(v2)},
        }


def quality_control_interpolation(
    data: np.ndarray, quality_flags: np.ndarray = None, max_consecutive_missing: int = 5
) -> Dict[str, Any]:
    """
    Quality-controlled interpolation with data validation
    """
    if quality_flags is None:
        quality_flags = np.ones(len(data))  # All good quality

    # Identify bad quality data
    bad_quality_mask = quality_flags == 0
    missing_mask = np.isnan(data) | bad_quality_mask

    # Check for consecutive missing data
    consecutive_missing = []
    count = 0

    for i, is_missing in enumerate(missing_mask):
        if is_missing:
            count += 1
        else:
            if count > 0:
                consecutive_missing.append(count)
            count = 0

    if count > 0:  # Handle missing data at end
        consecutive_missing.append(count)

    max_consecutive = max(consecutive_missing) if consecutive_missing else 0

    # Perform interpolation only if gaps are reasonable
    if max_consecutive <= max_consecutive_missing:
        interpolation_result = interpolate_missing_data(
            data, method="linear", max_gap=max_consecutive_missing
        )

        return {
            "success": True,
            "interpolated_data": interpolation_result["interpolated_data"],
            "quality_summary": {
                "total_points": len(data),
                "missing_points": np.sum(missing_mask),
                "bad_quality_points": np.sum(bad_quality_mask),
                "max_consecutive_missing": max_consecutive,
                "gaps_filled": interpolation_result["gaps_filled"],
            },
        }
    else:
        return {
            "success": False,
            "error": f"Too many consecutive missing points: {max_consecutive}",
            "quality_summary": {
                "total_points": len(data),
                "missing_points": np.sum(missing_mask),
                "max_consecutive_missing": max_consecutive,
            },
        }
