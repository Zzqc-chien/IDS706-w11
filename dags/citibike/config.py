"""Settings shared by every pipeline step. All of them can be overridden with env vars."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# SQLAlchemy URI of the database that stores the cleaned data (not Airflow's own metadata DB).
PIPELINE_DB_URI = os.environ.get(
    "PIPELINE_DB_URI", "postgresql+psycopg://pipeline:pipeline@pipeline-db:5432/citibike"
)

# Citi Bike Jersey City / Hoboken system: small monthly files (a few MB), same schema as NYC.
TRIP_URL_PATTERNS = [
    "https://s3.amazonaws.com/tripdata/JC-{ym}-citibike-tripdata.csv.zip",
    "https://s3.amazonaws.com/tripdata/JC-{ym}-citibike-tripdata.zip",
]

# Open-Meteo historical weather API (free, no key). Coordinates: Jersey City.
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = float(os.environ.get("WEATHER_LAT", "40.7178"))
LONGITUDE = float(os.environ.get("WEATHER_LON", "-74.0431"))
TIMEZONE = "America/New_York"
WEATHER_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m"]

# Citi Bike publishes each month's file a few weeks after it ends, so scheduled runs
# look back: with LAG=2 and N=6, a run on 2026-11-07 processes 2026-04 .. 2026-09.
DEFAULT_N_MONTHS = int(os.environ.get("N_MONTHS", "6"))
PUBLISH_LAG_MONTHS = int(os.environ.get("PUBLISH_LAG_MONTHS", "2"))

HTTP_TIMEOUT = 120
