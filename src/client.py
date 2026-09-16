"""Gemini API client wrapper for live multimodal audio inference.

Supports:
- Vertex AI (project='cloud-llm-preview1', location='global')
- Developer API (via GEMINI_API_KEY / GOOGLE_API_KEY if present)
- Inline WAV audio byte transmission via types.Part.from_bytes
- Configurable thinking_budget (default: 0) and temperature (default: 0.0)
- Built-in tenacity exponential backoff retry for transient and rate-limit errors
- Execution timing and structured latency tracking
"""

import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import errors, types
import google.oauth2.credentials
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MAX_RETRIES,
    MODEL_2_5_FLASH,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_DELAY,
    get_api_key,
    get_location,
    get_project_id,
)

logger = logging.getLogger(__name__)


def is_retryable_exception(exc: BaseException) -> bool:
    """Determines whether an exception is transient and should be retried."""
    if isinstance(exc, (errors.ServerError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, errors.ClientError):
        code = getattr(exc, "code", None)
        message = str(exc).lower()
        if code in (429, 500, 502, 503, 504):
            return True
        if any(
            term in message
            for term in [
                "resource_exhausted",
                "rate_limit",
                "quota",
                "temporarily unavailable",
                "overloaded",
                "try again",
            ]
        ):
            return True
    return False


class GeminiClient:
    """Production client wrapper for Google Gemini multimodal audio models."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initializes the Gemini client.

        Args:
            project_id: GCP project ID for Vertex AI (defaults to config).
            location: GCP location for Vertex AI (defaults to config).
            api_key: Optional Gemini API key (defaults to config env vars).
        """
        self.project_id = project_id or get_project_id()
        self.location = location or get_location()
        self.api_key = api_key or get_api_key()
        self._client: Optional[genai.Client] = None
        self._token_timestamp: float = 0.0

    def _get_live_credentials(self):
        """Retrieves credentials via Application Default Credentials (ADC) with gcloud fallback."""
        try:
            import google.auth
            from google.auth.transport.requests import Request
            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            creds.refresh(Request())
            self._token_timestamp = time.time()
            return creds
        except Exception as e:
            logger.info(f"ADC refresh failed ({e}), falling back to gcloud auth print-access-token.")
            token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], text=True
            ).strip()
            self._token_timestamp = time.time()
            return google.oauth2.credentials.Credentials(token=token)

    def get_client(self, force_refresh: bool = False) -> genai.Client:
        """Returns or lazily creates a cached genai.Client instance.

        Refreshes the client if force_refresh is True or if the OAuth token
        is older than 45 minutes (2700 seconds).
        """
        needs_refresh = (
            force_refresh
            or self._client is None
            or (not self.api_key and (time.time() - self._token_timestamp > 2700))
        )

        if needs_refresh:
            if self.api_key:
                logger.info("Initializing GeminiClient via Developer API Key.")
                self._client = genai.Client(api_key=self.api_key)
            else:
                logger.info(
                    f"Initializing GeminiClient via Vertex AI (project={self.project_id}, location={self.location})."
                )
                creds = self._get_live_credentials()
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location,
                    credentials=creds,
                )
        return self._client

    def _generate_content_with_fallback(
        self, model: str, contents: Any, config: types.GenerateContentConfig
    ) -> Any:
        """Executes generate_content with 401 token refresh and 403 project fallback."""
        client = self.get_client()
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except errors.ClientError as ce:
            code = getattr(ce, "code", None)
            if code == 401:
                logger.warning("Received 401 unauthorized. Refreshing credentials and retrying...")
                client = self.get_client(force_refresh=True)
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            if code == 403 and self.project_id != "cloud-llm-preview1":
                logger.warning(
                    f"Received 403 PERMISSION_DENIED on project '{self.project_id}'. "
                    f"Falling back to 'cloud-llm-preview1'..."
                )
                self.project_id = "cloud-llm-preview1"
                client = self.get_client(force_refresh=True)
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            raise ce

    def test_connection(self, model: str = MODEL_2_5_FLASH) -> bool:
        """Runs a lightweight test query to verify live connectivity."""
        try:
            res = self._generate_content_with_fallback(
                model=model,
                contents="Ping",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.0,
                    max_output_tokens=10,
                ),
            )
            return bool(res and res.text)
        except Exception as e:
            logger.warning(f"Connection test failed for model {model}: {e}")
            return False

    def generate_with_audio(
        self,
        model: str,
        audio_source: Union[str, Path, bytes],
        prompt: str,
        system_instruction: Optional[str] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """Transcribes / processes audio via inline WAV bytes with retry.

        Args:
            model: Gemini model identifier (e.g. 'gemini-2.5-flash').
            audio_source: File path to WAV file or raw audio bytes.
            prompt: Text prompt accompanying the audio.
            system_instruction: Optional system instruction.
            thinking_budget: Thinking budget token count (0 disables thinking).
            temperature: Sampling temperature (0.0 for deterministic output).
            max_output_tokens: Maximum response tokens.
            response_mime_type: Optional response MIME type (e.g. 'application/json').
            response_schema: Optional OpenAPI/JSON response schema.

        Returns:
            Tuple of (raw response text string, execution latency in seconds).

        Raises:
            Exception: If all retries fail.
        """
        # Load audio bytes
        if isinstance(audio_source, (str, Path)):
            wav_path = Path(audio_source)
            if not wav_path.exists():
                raise FileNotFoundError(f"Audio file not found: {wav_path}")
            audio_bytes = wav_path.read_bytes()
        elif isinstance(audio_source, bytes):
            audio_bytes = audio_source
        else:
            raise TypeError(f"Unsupported audio source type: {type(audio_source)}")

        # Build inline audio part
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

        # Configure request
        config_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
            "system_instruction": system_instruction,
        }
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_kwargs)

        contents = [audio_part, prompt]

        @retry(
            retry=retry_if_exception(is_retryable_exception),
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential_jitter(
                initial=RETRY_INITIAL_DELAY, max=RETRY_MAX_DELAY
            ),
            reraise=True,
        )
        def _execute_call() -> Tuple[str, float]:
            start_time = time.perf_counter()
            response = self._generate_content_with_fallback(
                model=model,
                contents=contents,
                config=config,
            )
            latency = time.perf_counter() - start_time

            # Extract text safely
            text_out = response.text if response and response.text else ""
            return text_out.strip(), latency

        return _execute_call()

    def generate_text(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> Tuple[str, float]:
        """Generates text from text prompt with retry and latency tracking."""
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            system_instruction=system_instruction,
        )

        @retry(
            retry=retry_if_exception(is_retryable_exception),
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential_jitter(
                initial=RETRY_INITIAL_DELAY, max=RETRY_MAX_DELAY
            ),
            reraise=True,
        )
        def _execute_call() -> Tuple[str, float]:
            start_time = time.perf_counter()
            response = self._generate_content_with_fallback(
                model=model,
                contents=prompt,
                config=config,
            )
            latency = time.perf_counter() - start_time
            text_out = response.text if response and response.text else ""
            return text_out.strip(), latency

        return _execute_call()
