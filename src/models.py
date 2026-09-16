"""Pydantic data models for STT and Speaker Diarization evaluation."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    """Represents a single conversational turn attributed to a speaker."""
    model_config = ConfigDict(extra="allow")

    speaker: str = Field(description="Identifier for the speaker (e.g., 'Speaker 0', 'Agent')")
    text: str = Field(description="Spoken text or transcript for this turn")
    start_time: Optional[float] = Field(default=None, description="Start timestamp in seconds")
    end_time: Optional[float] = Field(default=None, description="End timestamp in seconds")


class SampleData(BaseModel):
    """Represents a curated benchmark audio sample with ground-truth annotations."""
    model_config = ConfigDict(extra="allow")

    sample_id: str = Field(description="Unique benchmark sample identifier (e.g., 'hindi_064')")
    audio_path: str = Field(description="Path to the audio WAV file")
    duration_seconds: float = Field(description="Audio duration in seconds")
    num_speakers: int = Field(description="Number of unique speakers in the sample")
    overlap_ratio: float = Field(default=0.0, description="Percentage of audio with overlapping speech (0-100 or fraction)")
    ground_truth_turns: List[Turn] = Field(default_factory=list, description="Ground truth conversational turns")

    # Optional metadata fields
    recording_id: Optional[str] = None
    language: Optional[str] = None
    dataset_type: Optional[str] = None
    num_segments: Optional[int] = None
    overlap_duration: Optional[float] = None
    turn_switches: Optional[int] = None
    audio_file: Optional[str] = None
    speaker_labels: Optional[List[str]] = None
    annotated_transcript: Optional[List[Dict[str, Any]]] = None
    reference_dialogue_raw: Optional[str] = None
    reference_dialogue_indic_clean: Optional[str] = None
    reference_dialogue_normalized: Optional[str] = None


class PipelinePrediction(BaseModel):
    """Represents the output of an STT + Diarization pipeline run on a sample."""
    model_config = ConfigDict(extra="allow")

    sample_id: str = Field(description="Unique sample identifier")
    model_id: str = Field(description="Model identifier (e.g., 'gemini-2.5-flash', 'gemini-3.5-flash-lite')")
    approach: str = Field(description="Pipeline approach (e.g., 'single_step', 'two_step', 'gepa')")
    predicted_turns: List[Turn] = Field(default_factory=list, description="Model-predicted conversational turns")
    raw_response: str = Field(default="", description="Raw response text from the API")
    latency_seconds: float = Field(default=0.0, description="Execution round-trip latency in seconds")


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for a sample or aggregate evaluation run."""
    model_config = ConfigDict(extra="allow")

    wer: float = Field(description="Word Error Rate (0.0 to 1.0+)")
    cer: float = Field(description="Character Error Rate (0.0 to 1.0+)")
    speaker_attribution_accuracy: float = Field(
        description="Word-level Speaker Attribution Accuracy via Hungarian matching (0.0 to 1.0)"
    )
    cpwer: float = Field(description="Concatenated Permutation Word Error Rate (0.0 to 1.0+)")
    diarization_gap: float = Field(
        description="Diarization Degradation Gap (cpWER - WER). Near 0 indicates minimal diarization error."
    )
    speaker_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Optimal Hungarian permutation mapping hypothesis speakers to reference speakers"
    )
    format_valid: bool = Field(default=True, description="Whether the model output conformed to required format")
