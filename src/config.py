"""Configuration settings for Gemini Audio STT & Speaker Diarization."""

import os
from pathlib import Path
from typing import Optional

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_SUBSET_DIR = DATA_DIR / "benchmark_subset"
BENCHMARK_AUDIO_DIR = BENCHMARK_SUBSET_DIR / "audio"
BENCHMARK_METADATA_PATH = BENCHMARK_SUBSET_DIR / "metadata.json"

RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
OPTIMIZED_RESULTS_DIR = RESULTS_DIR / "optimized"

# Ensure output directories exist
BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model identifiers
MODEL_2_5_FLASH = "gemini-2.5-flash"
MODEL_3_5_FLASH_LITE = "gemini-3.5-flash-lite"
MODEL_3_5_FLASH = "gemini-3.5-flash"

# Default benchmark models
BASELINE_MODELS = [MODEL_2_5_FLASH, MODEL_3_5_FLASH_LITE]

# GCP / Vertex AI configuration
DEFAULT_PROJECT_ID = "my-argolis-prj"
DEFAULT_LOCATION = "global"

def get_project_id() -> str:
    """Returns the GCP Project ID for Vertex AI."""
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("VERTEXAI_PROJECT")
        or os.environ.get("PROJECT_ID")
        or DEFAULT_PROJECT_ID
    )

def get_location() -> str:
    """Returns the GCP Location for Vertex AI."""
    return (
        os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("VERTEXAI_LOCATION")
        or os.environ.get("LOCATION")
        or DEFAULT_LOCATION
    )

def get_api_key() -> Optional[str]:
    """Returns the Gemini Developer API key if set."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Inference parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_THINKING_BUDGET = 0
DEFAULT_MAX_OUTPUT_TOKENS = 65536

# Request retry parameters
MAX_RETRIES = 5
RETRY_INITIAL_DELAY = 2.0
RETRY_MAX_DELAY = 60.0
RETRY_EXPONENTIAL_BASE = 2.0
REQUEST_TIMEOUT_SECONDS = 180.0
