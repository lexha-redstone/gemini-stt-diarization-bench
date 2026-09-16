"""Strategy C4: Acoustic Multi-Stage Verifier Pipeline for Pure Gemini 3.5 Flash Lite.

Architecture:
- Stage 1: Multimodal Audio Acoustic STT via `AcousticAnchorToken0Pipeline`
  Generates verbatim Devanagari transcription and preliminary acoustic speaker assignments.
- Stage 2: Pure 3.5 Lite Text-Only Targeted Consistency Verifier
  Operates on the sequential transcript text with preliminary speakers.
  Emits ONLY a minimal JSON diff of corrected turn indices (e.g. `{"corrections": [{"turn_id": 4, "speaker": "Speaker 0"}]}`).
  Zero rewriting of Devanagari words, preventing 100% of lexical drift and maintaining low latency.
- Python Merger: Applies corrections programmatically to immutable Stage 1 Turn objects.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.client import GeminiClient
from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.acoustic_token0 import AcousticAnchorToken0Pipeline

logger = logging.getLogger(__name__)

STAGE2_SYSTEM_INSTRUCTION = """You are an expert dialogue consistency verifier.
Below is a sequential Hindi dialogue with preliminary speaker attributions generated from acoustic audio cues.
The preliminary attributions are ~80% accurate, but may contain specific conversational anomalies.

YOUR TASK:
Review the turn sequence and output strictly a JSON object with turn numbers that must be corrected.

Correction Rules:
1. Conversational Stance & Debate Polarity:
   If Speaker A and Speaker B hold contrasting opinions or debate stances (e.g., in favor of vs. against a topic),
   ensure each turn is attributed to the speaker advocating that position. Never attribute an argument and its immediate counter-argument to the same speaker.
2. Interruption Resumption (A-B-A Sandwich):
   If Speaker A was speaking, Speaker B made a brief 1-2 word reaction (e.g., हाँ, जी, अच्छा, अरे, ठीक है, ओके),
   and Turn N immediately continues Speaker A's thought/sentence/question, ensure Turn N is attributed to Speaker A.
3. Sentence / Monologue Continuation:
   If Turn N begins with an incomplete clause, conjunction, or continuation continuing Turn N-1 without a genuine speaker change,
   ensure both turns have the SAME speaker.
4. Self-Acknowledgment:
   If a speaker uses an affirmative starter (e.g. हाँ, ओके, ठीक है, अच्छा) followed by their own continuation,
   ensure the speaker does not flip to another person.
5. Default Rule:
   If in doubt, DO NOT change the speaker. Trust the acoustic attribution. Most turns are already correct. Only correct obvious conversational inconsistencies.

Output Format:
Output strictly a JSON object with a list of corrections:
{
  "corrections": [
    {"turn_id": 4, "speaker": "Speaker 0"}
  ]
}
If no corrections are needed, output:
{"corrections": []}
"""


class AcousticMultiStagePipeline:
    """Candidate C4: Two-Stage Acoustic STT + Pure 3.5 Lite Text Verifier."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        self.client = client or GeminiClient()
        self.stage1_pipeline = AcousticAnchorToken0Pipeline(
            client=self.client,
            thinking_budget=thinking_budget,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _build_stage2_prompt(self, turns: List[Turn], profiles_text: str = "") -> str:
        """Formats Stage 1 turns into a numbered list for text verification."""
        prompt_parts = []
        if profiles_text and profiles_text.strip():
            prompt_parts.append(f"Speaker Context:\n{profiles_text.strip()}\n")

        lines = []
        for idx, turn in enumerate(turns, 1):
            lines.append(f"[Turn {idx}] {turn.speaker}: {turn.text}")
        prompt_parts.append("Input Dialogue:\n" + "\n".join(lines))
        prompt_parts.append("\nOutput JSON corrections:")
        return "\n".join(prompt_parts)

    def _parse_stage2_corrections(self, raw_text: str) -> List[Dict[str, Any]]:
        """Extracts JSON corrections array from model response."""
        if not raw_text or not raw_text.strip():
            return []

        # Find JSON block
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group(0))
            corrections = data.get("corrections", [])
            if isinstance(corrections, list):
                return corrections
        except Exception as e:
            logger.warning(f"Failed to parse Stage 2 JSON corrections: {e} (raw: {raw_text[:200]})")
        return []

    def run_sample(
        self,
        sample: SampleData,
        model_id: str = MODEL_3_5_FLASH_LITE,
        num_speakers: Optional[int] = None,
    ) -> PipelinePrediction:
        """Executes Candidate C4 two-stage pipeline on a benchmark sample.

        Args:
            sample: Benchmark SampleData instance.
            model_id: Target Gemini model identifier (default gemini-3.5-flash-lite).
            num_speakers: Optional speaker cardinality override.

        Returns:
            PipelinePrediction with turns, raw response, and combined latency.
        """
        # --- Stage 1: Multimodal Audio Acoustic STT ---
        stage1_pred = self.stage1_pipeline.run_sample(
            sample=sample,
            model_id=model_id,
            num_speakers=num_speakers,
        )

        stage1_turns = stage1_pred.predicted_turns
        if not stage1_turns or len(stage1_turns) <= 1:
            # Nothing to verify with fewer than 2 turns
            return PipelinePrediction(
                sample_id=sample.sample_id,
                model_id=model_id,
                approach="strategy_c4_multistage",
                predicted_turns=stage1_turns,
                raw_response=stage1_pred.raw_response,
                latency_seconds=stage1_pred.latency_seconds,
            )

        # Clone turns to preserve immutability
        final_turns = [Turn(speaker=t.speaker, text=t.text) for t in stage1_turns]

        # Extract profiles text from stage 1 raw response if present
        profiles_match = re.search(r"\[Speaker\s*Profiles\][\s\S]*?(?=\[Utterance\]|$)", stage1_pred.raw_response, re.IGNORECASE)
        profiles_text = profiles_match.group(0).strip() if profiles_match else ""

        # --- Stage 2: Pure 3.5 Lite Text Consistency Verifier ---
        stage2_prompt = self._build_stage2_prompt(final_turns, profiles_text=profiles_text)
        stage2_raw, stage2_latency = self.client.generate_text(
            model=model_id,
            prompt=stage2_prompt,
            system_instruction=STAGE2_SYSTEM_INSTRUCTION,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=512,  # Minimal token budget for fast diff emission
        )

        corrections = self._parse_stage2_corrections(stage2_raw)
        total_latency = stage1_pred.latency_seconds + stage2_latency

        # Programmatically apply corrections
        for corr in corrections:
            t_id = corr.get("turn_id")
            new_spk = corr.get("speaker")
            if isinstance(t_id, int) and isinstance(new_spk, str):
                spk_match = re.search(r"(\d+)", new_spk)
                if spk_match:
                    normalized_spk = f"Speaker {spk_match.group(1)}"
                else:
                    normalized_spk = new_spk.strip()

                idx = t_id - 1 if (1 <= t_id <= len(final_turns)) else t_id
                if 0 <= idx < len(final_turns):
                    logger.info(
                        f"Sample {sample.sample_id} Stage 2 correction: Turn {t_id} "
                        f"'{final_turns[idx].speaker}' -> '{normalized_spk}'"
                    )
                    final_turns[idx].speaker = normalized_spk

        combined_raw = (
            f"=== STAGE 1 (Acoustic STT) ===\n{stage1_pred.raw_response}\n\n"
            f"=== STAGE 2 (Text Verifier) ===\n{stage2_raw}"
        )

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="strategy_c4_multistage",
            predicted_turns=final_turns,
            raw_response=combined_raw,
            latency_seconds=total_latency,
        )


# Alias runner for benchmark runner naming convention
AcousticMultiStageRunner = AcousticMultiStagePipeline
