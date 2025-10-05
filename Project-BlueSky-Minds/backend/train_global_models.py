# Enhanced training script: train_global_models.py
import numpy as np
import pandas as pd
from itertools import product


class GlobalClimateTrainer:
    def __init__(self):
        self.global_grid_data = []

    async def create_global_training_grid(self):
        """Create a dense global grid for comprehensive coverage"""
        print("🌍 Creating global training grid...")

        # Create a 5-degree grid covering the entire Earth
        latitudes = np.arange(-85, 90, 5)  # Every 5 degrees latitude
        longitudes = np.arange(-180, 185, 5)  # Every 5 degrees longitude

        # This creates ~3,600 global points
        grid_points = []

        for lat, lon in product(latitudes, longitudes):
            # Skip extreme polar regions (no reliable data)
            if abs(lat) <= 85:
                grid_points.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "climate_zone": self._classify_climate_zone(lat, lon),
                    }
                )

        print(f"📍 Created {len(grid_points)} global grid points")

        # Sample representative points from each climate zone
        training_points = self._select_representative_points(grid_points)

        print(f"🎯 Selected {len(training_points)} representative training points")
        return training_points

    def _select_representative_points(self, grid_points):
        """Select representative points from each climate zone"""
        zones = {}

        # Group by climate zone
        for point in grid_points:
            zone = point["climate_zone"]
            if zone not in zones:
                zones[zone] = []
            zones[zone].append(point)

        # Select points from each zone
        selected_points = []

        for zone, points in zones.items():
            # Select every 3rd point to get good coverage without overloading
            step = max(1, len(points) // 20)  # Max 20 points per zone
            selected = points[::step]
            selected_points.extend(selected)
            print(f"  📍 {zone}: {len(selected)} points selected")

        return selected_points

    def _classify_climate_zone(self, lat, lon):
        """Enhanced climate classification"""
        abs_lat = abs(lat)

        # Primary classification by latitude
        if abs_lat < 5:
            base_zone = "equatorial"
        elif abs_lat < 23.5:
            base_zone = "tropical"
        elif abs_lat < 35:
            base_zone = "subtropical"
        elif abs_lat < 60:
            base_zone = "temperate"
        else:
            base_zone = "polar"

        # Add continental/maritime modifier
        if abs(lon) < 20 or abs(lon - 180) < 20:  # Near major water bodies
            modifier = "maritime"
        elif 30 < abs(lon) < 150:  # Continental interiors
            modifier = "continental"
        else:
            modifier = "coastal"

        return f"{base_zone}_{modifier}"
