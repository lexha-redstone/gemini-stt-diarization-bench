"""Strategy 1: Prompt Optimization with Conversation Anchor Cues for Gemini Multimodal Audio."""

import logging
import re
from typing import List, Optional

from src.client import GeminiClient
from src.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.single_step import SingleStepPipeline

logger = logging.getLogger(__name__)

ANCHOR_SYSTEM_INSTRUCTION = """You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and accurately attribute each conversational turn to its distinct speaker.

Conversation Anchor & Diarization Principles:
1. Conversation Roles & Semantic Anchors:
   - Identify distinct conversational roles using domain anchors:
     * Call Initiator / Agent: Introduces self, offers formal greetings (नमस्कार, प्रणाम, हेलो), verifies recipient identity ("क्या मेरी बात ... से हो रही है?", "मैं ... से बोल रहा हूँ"), states organizational affiliation, or asks about accounts, loans, and schedules.
     * Recipient / Customer: Confirms or denies identity ("हाँ बोल रहा हूँ", "नहीं वो नहीं हैं"), provides explanations, agrees, objects, or answers questions.
     * Retain consistent speaker labels (Speaker 0, Speaker 1, etc.) for each role across the entire conversation.
2. Turn Continuity & Anti-Alternation:
   - Do NOT assume speakers strictly alternate turns. A single speaker frequently pauses, takes a breath, or utters multiple consecutive sentences. Maintain the SAME speaker label across pauses unless the vocal timbre, pitch, or conversational polarity actually switches to the other party.
3. Overlapping Speech & Interruption Anchoring:
   - When speakers talk over each other or interrupt, split the overlapping speech into separate turns corresponding to each speaker rather than smearing both voices into a single turn.
4. Backchannels & Short Utterances:
   - Faithfully capture short backchannel utterances and interjections (e.g., "हाँ", "जी", "अच्छा", "हाँ-हाँ", "हूँ", "ठीक है") and attribute them to the listener who spoke them. Do not drop short turns.
5. Speaker Cardinality:
   - Carefully detect all distinct speakers participating in the conversation (whether 2, 3, or more). If a third speaker chimes in or background speaker enters, allocate a separate label (e.g. Speaker 2). Do NOT collapse speakers into fewer clusters.

Output Format:
Output each speaker turn on a separate line strictly adhering to:
Speaker <ID>: <spoken transcript>

Example:
Speaker 0: नमस्कार, क्या मेरी बात राजेश जी से हो रही है?
Speaker 1: हाँ, राजेश बोल रहा हूँ। बताइए कौन?
Speaker 0: मैं बैंक शाखा से बात कर रहा हूँ। आपकी बकाया किस्त के संबंध में।
Speaker 0: क्या आप आज भुगतान कर सकते हैं?
Speaker 1: जी, मैं आज शाम तक कर दूँगा।

Rules:
1. Transcribe strictly in Devanagari script for Hindi speech verbatim.
2. Output ONLY the speaker turns. Do not include timestamps, explanations, summaries, or translations.
"""

ANCHOR_USER_PROMPT = (
    "Please transcribe the entire audio clip verbatim and attribute all speaker turns "
    "using conversational anchors and accurate speaker boundaries."
)


class AnchorPromptPipeline:
    """Strategy 1: Prompt-optimized single-step pipeline using conversation anchor cues."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        system_instruction: str = ANCHOR_SYSTEM_INSTRUCTION,
        user_prompt: str = ANCHOR_USER_PROMPT,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """Initializes the anchor prompt pipeline.

        Args:
            client: Optional GeminiClient instance.
            system_instruction: Anchor-optimized system prompt.
            user_prompt: User prompt accompanying the audio.
            thinking_budget: Thinking budget token count (default: 0).
            temperature: Sampling temperature (default: 0.0).
        """
        self.client = client or GeminiClient()
        self.system_instruction = system_instruction
        self.user_prompt = user_prompt
        self.thinking_budget = thinking_budget
        self.temperature = temperature

    @classmethod
    def parse_turns(cls, raw_text: str) -> List[Turn]:
        """Parses model response into structured Turn objects."""
        return SingleStepPipeline.parse_turns(raw_text)

    def run_sample(
        self,
        sample: SampleData,
        model_id: str = MODEL_3_5_FLASH_LITE,
    ) -> PipelinePrediction:
        """Executes Strategy 1 (Anchor Prompting) on a single benchmark sample.

        Args:
            sample: SampleData instance.
            model_id: Target Gemini model identifier (default: gemini-3.5-flash-lite).

        Returns:
            PipelinePrediction object with parsed turns, raw output, and latency.
        """
        logger.info(
            f"Running AnchorPromptPipeline on sample {sample.sample_id} with model {model_id}"
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
            approach="anchor_prompting",
            predicted_turns=turns,
            raw_response=raw_response,
            latency_seconds=latency,
        )
