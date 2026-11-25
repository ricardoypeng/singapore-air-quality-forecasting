"""Canonical column orders for aligned comparison inputs."""

from .features import EAST_CHANNELS, OTHER_REGIONS


QUALITY_CHANNELS = ("wind", "rain", "temperature", "humidity")
QUALITY_COLUMNS = tuple(f"{name}_fraction" for name in QUALITY_CHANNELS) + tuple(
    f"{name}_last_observation_age_minutes" for name in QUALITY_CHANNELS
)
RECENT_COLUMNS = ("wind_u_ms", "wind_v_ms", "reported_rainfall_mm",
                  "temperature_c", "relative_humidity_pct") + QUALITY_COLUMNS

BASE_COLUMNS = (
    tuple(f"{channel}_lag{lag}" for lag in range(23, -1, -1) for channel in EAST_CHANNELS)
    + tuple(
        f"{unit}_{function}_offset{offset}"
        for offset in range(7)
        for unit in ("hour", "weekday", "month")
        for function in ("sin", "cos")
    )
    + tuple(f"{region}_lag{lag}" for lag in range(23, -1, -1) for region in OTHER_REGIONS)
)
