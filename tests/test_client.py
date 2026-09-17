"""Unit tests for GeminiClient and network retry logic."""

import os
import pytest
from google.genai import errors

from src.client import GeminiClient, is_retryable_exception
from src.config import MODEL_2_5_FLASH, MODEL_3_5_FLASH_LITE


def test_client_initialization():
    """Verifies that GeminiClient initializes with valid default parameters."""
    client = GeminiClient()
    assert client.project_id is not None
    assert client.location is not None
    assert client.project_id in ["my-argolis-prj", "cloud-llm-preview1", os.environ.get("GOOGLE_CLOUD_PROJECT")]


def test_is_retryable_exception():
    """Verifies exception classification for retry policies."""
    # Transient server errors must retry
    srv_err = errors.ServerError(503, {"error": {"message": "Service Unavailable"}})
    assert is_retryable_exception(srv_err) is True

    # Rate limits must retry
    rate_err = errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED: Rate limit exceeded"}})
    assert is_retryable_exception(rate_err) is True

    # Generic client errors (e.g. 400 bad request) should not retry
    bad_req = errors.ClientError(400, {"error": {"message": "INVALID_ARGUMENT: Invalid parameter"}})
    assert is_retryable_exception(bad_req) is False

    # Connection, timeout, and OS/requests errors must retry
    assert is_retryable_exception(ConnectionError("Connection reset")) is True
    assert is_retryable_exception(TimeoutError("Read timed out")) is True
    assert is_retryable_exception(OSError("Socket closed")) is True


def test_client_live_connectivity():
    """Verifies live connectivity to Gemini 2.5 Flash and 3.5 Flash Lite."""
    client = GeminiClient()
    assert client.test_connection(MODEL_2_5_FLASH) is True
    assert client.test_connection(MODEL_3_5_FLASH_LITE) is True
