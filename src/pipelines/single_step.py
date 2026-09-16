"""Approach 1: Single-step end-to-end audio-to-diarized transcript pipeline."""

import logging
import re
from typing import List, Optional

from src.client import GeminiClient
from src.config import DEFAULT_TEMPERATURE, DEFAULT_THINKING_BUDGET, MODEL_2_5_FLASH
from src.models import PipelinePrediction, SampleData, Turn

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = """You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and accurately attribute each conversational turn to its speaker.

Output Format:
Output each speaker turn on a separate line strictly adhering to:
Speaker <ID>: <spoken transcript>

Example:
Speaker 0: हाँ, मैं बोल रहा हूँ।
Speaker 1: नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?

Rules:
1. Transcribe strictly in Devanagari script for Hindi speech.
2. Maintain consistent speaker labels throughout the audio clip (e.g., Speaker 0, Speaker 1).
3. Start a new turn whenever the speaker changes, including brief interruptions or overlapping turns.
4. Do not include timestamps, acoustic notes, explanations, summaries, or translations. Output only the speaker turns.
"""

DEFAULT_USER_PROMPT = (
    "Please transcribe the entire audio clip verbatim and attribute all speaker turns."
)


class SingleStepPipeline:
    """Baseline pipeline executing single-step multimodal audio diarization."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        user_prompt: str = DEFAULT_USER_PROMPT,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """Initializes the single-step pipeline.

        Args:
            client: Optional GeminiClient instance (lazily created if None).
            system_instruction: System prompt guiding transcription and diarization.
            user_prompt: User prompt accompanying the audio bytes.
            thinking_budget: Thinking token budget (0 for verbatim acoustic STT).
            temperature: Sampling temperature (0.0 for deterministic decoding).
        """
        self.client = client or GeminiClient()
        self.system_instruction = system_instruction
        self.user_prompt = user_prompt
        self.thinking_budget = thinking_budget
        self.temperature = temperature

    @classmethod
    def parse_turns(cls, raw_text: str) -> List[Turn]:
        """Parses model output text into structured Turn objects.

        Handles diverse formatting variations including markdown bolding,
        brackets, Indic speaker headers, and multi-line utterances.

        Args:
            raw_text: The raw output text returned by the model.

        Returns:
            List of Turn objects with speaker identifier and spoken text.
        """
        if not raw_text or not raw_text.strip():
            return []

        lines = raw_text.strip().split("\n")
        turns: List[Turn] = []
        current_speaker: Optional[str] = None
        current_text_parts: List[str] = []

        prefix_pattern = re.compile(
            r"^(?:[\*\#\-\s\[\(]*)"
            r"(Speaker\s*[\dA-Za-z]+|वक्ता\s*[\d\u0966-\u096fA-Za-z]+|Person\s*[\dA-Za-z]+|Participant\s*[\dA-Za-z]+|Agent|Customer|spk_[\dA-Za-z]+)"
            r"(?:[\*\#\s\]\)]*)[:\-\u2013]\s*(.*)$",
            re.IGNORECASE,
        )

        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue

            m = prefix_pattern.match(line_clean)
            if m:
                # Flush previous turn
                if current_speaker is not None and current_text_parts:
                    text_turn = " ".join(current_text_parts).strip()
                    if text_turn:
                        turns.append(Turn(speaker=current_speaker, text=text_turn))

                speaker_raw = m.group(1).strip()
                # Clean any stray formatting from speaker
                speaker_id = re.sub(r"[\[\]\*\#]", "", speaker_raw).strip()

                raw_line_text = m.group(2).strip()
                # Clean leading markdown asterisks or quotes
                clean_line_text = re.sub(r"^[\*\#\s\]\)]+", "", raw_line_text).strip()

                current_speaker = speaker_id
                current_text_parts = [clean_line_text] if clean_line_text else []
            else:
                if current_speaker is not None:
                    current_text_parts.append(line_clean)
                else:
                    # Check if line contains a speaker colon in another form
                    if ":" in line_clean:
                        parts = line_clean.split(":", 1)
                        header = parts[0].strip()
                        # If header looks like a speaker name (<= 3 words)
                        if 1 <= len(header.split()) <= 3 and not header.startswith("http"):
                            current_speaker = re.sub(r"[\[\]\*\#]", "", header).strip()
                            content = parts[1].strip()
                            clean_content = re.sub(r"^[\*\#\s\]\)]+", "", content).strip()
                            current_text_parts = [clean_content] if clean_content else []
                            continue

                    current_speaker = "Speaker 0"
                    current_text_parts.append(line_clean)

        if current_speaker is not None and current_text_parts:
            text_turn = " ".join(current_text_parts).strip()
            if text_turn:
                turns.append(Turn(speaker=current_speaker, text=text_turn))

        # Fallback if no turns could be formed
        if not turns and raw_text.strip():
            turns.append(Turn(speaker="Speaker 0", text=raw_text.strip()))

        return turns

    def run_sample(
        self,
        sample: SampleData,
        model_id: str = MODEL_2_5_FLASH,
    ) -> PipelinePrediction:
        """Executes Approach 1 on a single benchmark sample.

        Args:
            sample: SampleData instance containing audio_path and ground truth.
            model_id: Target Gemini model identifier.

        Returns:
            PipelinePrediction with turns, raw response, and latency.
        """
        logger.info(
            f"Running SingleStepPipeline on sample {sample.sample_id} with model {model_id}"
        )
        raw_response, latency = self.client.generate_with_audio(
            model=model_id,
            audio_source=sample.audio_path,
            prompt=self.user_prompt,
            system_instruction=self.system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
        )

        turns = self.parse_turns(raw_response)

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="single_step",
            predicted_turns=turns,
            raw_response=raw_response,
            latency_seconds=latency,
        )
