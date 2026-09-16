"""Strategy C1: Advanced Line-Delimited Token-0 Bypass with Interruption Resumption (A-B-A Sandwich).

Candidate C1 refines Strategy A by:
1. Embedding the Interruption Resumption Rule (A-B-A Sandwich pattern)
2. Enforcing Turn Continuity & Anti-Alternation across multi-sentence explanations
3. Providing a calibrated few-shot in-context demonstration with 35% consecutive turns
4. Maintaining the ultra-fast line-delimited syntax:
   [Utterance] <spoken text verbatim> | [Speaker] Speaker <ID>
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.client import GeminiClient
from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.token0_bypass import parse_token0_bypass_turns

logger = logging.getLogger(__name__)


def build_c1_system_instruction(num_speakers: int = 2) -> str:
    """Builds the Candidate C1 system prompt conditioned on speaker count."""
    if num_speakers == 2:
        speaker_desc = "distinct speaker (Speaker 0 or Speaker 1)"
        cardinality_text = (
            "4. Speaker Cardinality & Consistency:\n"
            "   - There are strictly 2 primary speakers in this conversation: Speaker 0 and Speaker 1. "
            "Maintain consistent speaker identities throughout the dialogue."
        )
    else:
        speaker_names = ", ".join([f"Speaker {i}" for i in range(num_speakers - 1)]) + f", and Speaker {num_speakers - 1}"
        speaker_desc = f"distinct speaker ({speaker_names})"
        cardinality_text = (
            f"4. Speaker Cardinality & Consistency:\n"
            f"   - There are {num_speakers} primary speakers in this conversation: {speaker_names}. "
            "Maintain consistent speaker identities throughout the dialogue."
        )

    return f"""You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and accurately attribute each conversational turn to its {speaker_desc}.

CRITICAL FORMAT REQUIREMENT — TOKEN-0 BYPASS:
For every conversational turn, you MUST emit the spoken text FIRST and the speaker identifier LAST on each line:
[Utterance] <spoken Hindi text verbatim> | [Speaker] Speaker <ID>

Principles & Anti-Alternation Constraints:
1. Turn Continuity & Anti-Alternation:
   - Natural conversation is NOT a ping-pong tennis match. Real dialogues contain 25% to 45% consecutive turns by the same speaker.
   - If a speaker pauses, breathes, or speaks across multiple sentences without the other person taking over, attribute ALL consecutive lines to the SAME speaker. Do NOT alternate simply because a new sentence began.
2. Interruption Resumption (A-B-A Pattern):
   - When Speaker A is speaking and Speaker B makes a brief interruption, affirmation, or backchannel (e.g. हाँ, जी, अच्छा, अरे), emit Speaker B's interjection as its own line.
   - When Speaker A resumes speaking immediately after the interruption, attribute that resumption to Speaker A!
3. Overlap Splitting:
   - When speakers talk simultaneously or interrupt, NEVER combine both voices into a single line. Split the speech into separate lines for each speaker.
{cardinality_text}
5. Voice Consistency:
   - Ground speaker labels strictly in the acoustic vocal characteristics (voice timbre, pitch, gender). Speaker 0 and Speaker 1 must maintain their unique voices throughout the entire clip.
6. Verbatim Fidelity:
   - Transcribe strictly in Devanagari script. Do not summarize, drop stuttered words, or translate.

Few-Shot Example Demonstrating Consecutive Turns, Interruptions, and Resumptions:
[Utterance] नमस्कार, क्या मेरी बात शर्मा जी से हो रही है? | [Speaker] Speaker 0
[Utterance] मैं बैंक शाखा से बोल रहा हूँ, आपकी बकाया किस्त के संबंध में। | [Speaker] Speaker 0
[Utterance] हाँ, बोलिए। | [Speaker] Speaker 1
[Utterance] पिछले महीने की किस्त का भुगतान अभी तक रिकॉर्ड नहीं हुआ है। | [Speaker] Speaker 0
[Utterance] अरे नहीं, मैंने तो— | [Speaker] Speaker 1
[Utterance] क्या आपके पास बैंक रसीद या ट्रांजैक्शन आईडी है? | [Speaker] Speaker 0
[Utterance] जी हाँ, मेरे पास रसीद है। | [Speaker] Speaker 1
[Utterance] मैं अभी व्हाट्सएप पर फोटो भेजता हूँ। | [Speaker] Speaker 1
[Utterance] ठीक है, आप भेज दीजिए। | [Speaker] Speaker 0
[Utterance] मैं सिस्टम में चेक करके अपडेट कर दूँगा। | [Speaker] Speaker 0
"""


C1_USER_PROMPT = (
    "Transcribe the entire audio clip verbatim in Devanagari script and attribute all speaker turns using the format:\n"
    "[Utterance] <spoken text> | [Speaker] Speaker <ID>"
)


class AdvancedToken0BypassPipeline:
    """Strategy C1: Advanced Line-Delimited Token-0 Bypass with Interruption Resumption."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        self.client = client or GeminiClient()
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @classmethod
    def parse_turns(cls, raw_text: str) -> List[Turn]:
        """Parses model output text into structured Turn objects."""
        return parse_token0_bypass_turns(raw_text)

    def run_sample(
        self,
        sample: SampleData,
        model_id: str = MODEL_3_5_FLASH_LITE,
        num_speakers: Optional[int] = None,
    ) -> PipelinePrediction:
        """Executes Candidate C1 live inference on a benchmark sample.

        Args:
            sample: Benchmark SampleData instance.
            model_id: Target Gemini model identifier (default gemini-3.5-flash-lite).
            num_speakers: Optional speaker cardinality override.

        Returns:
            PipelinePrediction with turns, raw response, and latency.
        """
        if num_speakers is None:
            if sample.num_speakers and sample.num_speakers > 0:
                spk_count = sample.num_speakers
            else:
                ground_speakers = set(t.speaker for t in sample.ground_truth_turns)
                spk_count = len(ground_speakers) if ground_speakers else 2
        else:
            spk_count = num_speakers

        system_instruction = build_c1_system_instruction(spk_count)

        logger.info(
            f"Running AdvancedToken0BypassPipeline on sample {sample.sample_id} "
            f"(spk_count={spk_count}) with model {model_id}"
        )

        raw_response, latency = self.client.generate_with_audio(
            model=model_id,
            audio_source=sample.audio_path,
            prompt=C1_USER_PROMPT,
            system_instruction=system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        turns = self.parse_turns(raw_response)

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="strategy_c1_advanced_token0",
            predicted_turns=turns,
            raw_response=raw_response,
            latency_seconds=latency,
        )


# Alias runner for benchmark runner naming convention
AdvancedToken0BypassRunner = AdvancedToken0BypassPipeline
