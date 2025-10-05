from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import pandas as pd
import json
import io
import logging
from datetime import datetime

from app.services.nasa_data import NASADataService
from app.services.probability_engine import (
    AdvancedProbabilityEngine as ProbabilityEngine,
)

logger = logging.getLogger(__name__)
router = APIRouter()

nasa_service = NASADataService()
probability_engine = ProbabilityEngine()


@router.get("/csv")
async def export_csv(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    variables: str = Query("temperature", description="Comma-separated variables"),
    years_back: int = Query(30, ge=5, le=50),
):
    """
    Export weather probability data as CSV
    """
    try:
        # Parse variables
        variable_list = [v.strip() for v in variables.split(",")]

        # Fetch data
        end_year = datetime.now().year - 1
        start_year = end_year - years_back

        historical_data = await nasa_service.fetch_historical_data(
            latitude, longitude, start_year, end_year, variable_list
        )

        if not historical_data:
            raise HTTPException(status_code=404, detail="No data available")

        # Create DataFrame
        data_rows = []

        for variable, data in historical_data.items():
            years = len(data) // 12
            for year_idx in range(years):
                year = start_year + year_idx
                for month in range(12):
                    data_idx = year_idx * 12 + month
                    if data_idx < len(data):
                        data_rows.append(
                            {
                                "year": year,
                                "month": month + 1,
                                "variable": variable,
                                "value": data[data_idx],
                                "latitude": latitude,
                                "longitude": longitude,
                            }
                        )

        df = pd.DataFrame(data_rows)

        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_data_{latitude}_{longitude}_{timestamp}.csv"

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        raise HTTPException(status_code=500, detail="Export failed")


@router.get("/json")
async def export_json(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    variables: str = Query("temperature", description="Comma-separated variables"),
    years_back: int = Query(30, ge=5, le=50),
    include_analysis: bool = Query(True, description="Include probability analysis"),
):
    """
    Export weather data and analysis as JSON
    """
    try:
        variable_list = [v.strip() for v in variables.split(",")]

        # Fetch historical data
        end_year = datetime.now().year - 1
        start_year = end_year - years_back

        historical_data = await nasa_service.fetch_historical_data(
            latitude, longitude, start_year, end_year, variable_list
        )

        if not historical_data:
            raise HTTPException(status_code=404, detail="No data available")

        # Prepare export data
        export_data = {
            "metadata": {
                "location": {"latitude": latitude, "longitude": longitude},
                "time_period": {"start_year": start_year, "end_year": end_year},
                "variables": variable_list,
                "export_timestamp": datetime.now().isoformat(),
                "data_source": "NASA MERRA-2",
            },
            "raw_data": {},
        }

        # Add raw data
        for variable, data in historical_data.items():
            years = len(data) // 12
            monthly_data = []

            for year_idx in range(years):
                year = start_year + year_idx
                year_data = []
                for month in range(12):
                    data_idx = year_idx * 12 + month
                    if data_idx < len(data):
                        year_data.append(
                            {"month": month + 1, "value": float(data[data_idx])}
                        )
                monthly_data.append({"year": year, "monthly_values": year_data})

            export_data["raw_data"][variable] = monthly_data

        # Add analysis if requested
        if include_analysis:
            from datetime import date

            probability_results = probability_engine.calculate_weather_probabilities(
                historical_data, date.today(), (latitude, longitude)
            )

            export_data["analysis"] = probability_results

        # Convert to JSON
        json_content = json.dumps(export_data, indent=2)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_analysis_{latitude}_{longitude}_{timestamp}.json"

        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JSON export error: {e}")
        raise HTTPException(status_code=500, detail="Export failed")


@router.get("/summary")
async def export_summary_report(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    variables: str = Query("temperature,precipitation"),
    years_back: int = Query(30, ge=5, le=50),
):
    """
    Export a summary report with key statistics
    """
    try:
        variable_list = [v.strip() for v in variables.split(",")]

        # Fetch and analyze data
        end_year = datetime.now().year - 1
        start_year = end_year - years_back

        historical_data = await nasa_service.fetch_historical_data(
            latitude, longitude, start_year, end_year, variable_list
        )

        if not historical_data:
            raise HTTPException(status_code=404, detail="No data available")

        # Generate summary
        summary = {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "coordinates": f"{latitude:.4f}, {longitude:.4f}",
            },
            "analysis_period": f"{start_year} - {end_year} ({years_back} years)",
            "variables_analyzed": variable_list,
            "summary_statistics": {},
            "climate_trends": {},
            "generated_at": datetime.now().isoformat(),
        }

        # Calculate summary statistics for each variable
        for variable, data in historical_data.items():
            if len(data) > 0:
                import numpy as np

                summary["summary_statistics"][variable] = {
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "std_deviation": float(np.std(data)),
                    "minimum": float(np.min(data)),
                    "maximum": float(np.max(data)),
                    "data_points": len(data),
                    "percentiles": {
                        "10th": float(np.percentile(data, 10)),
                        "25th": float(np.percentile(data, 25)),
                        "75th": float(np.percentile(data, 75)),
                        "90th": float(np.percentile(data, 90)),
                    },
                }

                # Simple trend calculation
                if len(data) > 24:  # At least 2 years
                    x = np.arange(len(data))
                    slope = np.polyfit(x, data, 1)[0]
                    annual_trend = slope * 12  # Convert monthly to annual

                    summary["climate_trends"][variable] = {
                        "annual_trend": float(annual_trend),
                        "trend_direction": (
                            "increasing" if annual_trend > 0 else "decreasing"
                        ),
                        "trend_magnitude": (
                            "significant"
                            if abs(annual_trend) > np.std(data) * 0.1
                            else "minimal"
                        ),
                    }

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary report error: {e}")
        raise HTTPException(status_code=500, detail="Summary generation failed")
