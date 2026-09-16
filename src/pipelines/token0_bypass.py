"""Strategy A: Advanced Single-Prompt Optimization with Token-0 Bypass Framing.

Inverts the autoregressive decoding sequence from:
  P(S_i | Audio, History) * P(W_i | S_i, Audio, History)  [Baseline: Token-0 Commitment Trap]
to:
  P(W_i | Audio, History) * P(S_i | W_i, Audio, History)  [Token-0 Bypass Schema]

Syntax:
  [Utterance] <spoken Hindi text verbatim> | [Speaker] Speaker <ID>

Enforces anti-alternation constraints, overlap splitting directives, backchannel preservation,
and incorporates few-shot Hindi overlap demonstrations for Gemini 3.5 Flash Lite.
"""

import logging
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

from src.client import GeminiClient
from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn

logger = logging.getLogger(__name__)

CANDIDATE_A_SYSTEM_INSTRUCTION = """You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and accurately attribute each conversational turn to its distinct speaker (Speaker 0 or Speaker 1).

CRITICAL FORMAT REQUIREMENT — TOKEN-0 BYPASS:
For every conversational turn, you MUST emit the spoken text FIRST and the speaker identifier LAST on each line.
Strictly adhere to this line-delimited syntax:
[Utterance] <spoken Hindi text verbatim> | [Speaker] Speaker <ID>

Principles & Anti-Alternation Constraints:
1. Turn Continuity & Anti-Alternation:
   - Conversation is NOT a strict alternating tennis match. Do NOT alternate between Speaker 0 and Speaker 1 if the same person is continuing their thought.
   - If a speaker pauses, breathes, or speaks across multiple sentences, attribute ALL consecutive turns to the SAME speaker.
2. Overlap & Interruption Splitting:
   - When speakers speak simultaneously or interrupt each other, NEVER combine both voices into a single line.
   - Split the speech into separate turns: emit the interrupted speaker's words as one turn, and the interrupting speaker's words on the next turn.
3. Backchannel Preservation:
   - Faithfully capture short acknowledgments, affirmations, and interjections (e.g., हाँ, जी, अच्छा, ठीक है, हूँ) as separate turns and attribute them to the listening speaker.
4. Speaker Cardinality & Consistency:
   - There are strictly 2 primary speakers in this conversation: Speaker 0 and Speaker 1. Maintain consistent speaker labels throughout the dialogue.
5. Verbatim Fidelity:
   - Transcribe strictly in Devanagari script. Do not summarize, translate, or drop stuttered words.

Few-Shot Example Demonstrating Consecutive Turns, Interruptions, and Backchannels:
[Utterance] नमस्कार, क्या मेरी बात शर्मा जी से हो रही है? | [Speaker] Speaker 0
[Utterance] मैं बैंक शाखा से बोल रहा हूँ। | [Speaker] Speaker 0
[Utterance] हाँ, बोलिए कौन? | [Speaker] Speaker 1
[Utterance] आपकी बकाया किस्त के संबंध में जानकारी चाहिए थी। | [Speaker] Speaker 0
[Utterance] अरे नहीं, मैंने तो पिछले हफ्ते ही— | [Speaker] Speaker 1
[Utterance] लेकिन हमारे सिस्टम में रिकॉर्ड अभी तक अपडेट नहीं हुआ है। | [Speaker] Speaker 0
[Utterance] जी, अच्छा। मैं रसीद दिखाता हूँ। | [Speaker] Speaker 1
[Utterance] ठीक है, आप व्हाट्सएप पर भेज दीजिए। | [Speaker] Speaker 0
"""

CANDIDATE_A_USER_PROMPT = (
    "Transcribe the entire audio clip verbatim in Devanagari script and attribute all "
    "speaker turns using the format:\n"
    "[Utterance] <spoken text> | [Speaker] Speaker <ID>"
)


def parse_token0_bypass_turns(raw_text: str) -> List[Turn]:
    """Parses Token-0 Bypass line-delimited format into structured Turn objects.

    Supported patterns:
    - `[Utterance] <text> | [Speaker] Speaker <ID>`
    - `[Utterance]: <text> | [Speaker]: Speaker <ID>`
    - `<text> | [Speaker] Speaker <ID>`
    - `<text> | Speaker <ID>`
    - Fallback: standard `Speaker <ID>: <text>` or un-attributed continuation lines.

    Args:
        raw_text: Raw output string from model inference.

    Returns:
        List of Turn objects with speaker identifier and spoken text.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.strip().split("\n")
    turns: List[Turn] = []
    buffered_text_parts: List[str] = []

    # Regex to detect speaker suffix on right side of pipe
    speaker_suffix_re = re.compile(
        r"(?:\[?Speaker\]?[:\s]*)?"
        r"(Speaker\s*[\dA-Za-z]+|\d+|[\u0966-\u096f]+|वक्ता\s*[\d\u0966-\u096fA-Za-z]+|Person\s*[\dA-Za-z]+|Participant\s*[\dA-Za-z]+|Agent|Customer|spk_[\dA-Za-z]+)",
        re.IGNORECASE,
    )

    # Prefix fallback pattern if model emitted traditional Speaker 0: prefix
    prefix_fallback_re = re.compile(
        r"^(?:[\*\#\-\s\[\(]*)"
        r"(Speaker\s*[\dA-Za-z]+|वक्ता\s*[\d\u0966-\u096fA-Za-z]+|Person\s*[\dA-Za-z]+|spk_[\dA-Za-z]+)"
        r"(?:[\*\#\s\]\)]*)[:\-\u2013]\s*(.*)$",
        re.IGNORECASE,
    )

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Strip leading bullet dashes or numbers (e.g. "- ", "1. ")
        line_clean = re.sub(r"^[-*•\d+.]+\s*", "", line_clean).strip()

        # Check if line contains the pipe delimiter separating utterance and speaker
        if "|" in line_clean:
            left, right = line_clean.rsplit("|", 1)
            right_clean = right.strip()
            sm = speaker_suffix_re.search(right_clean)

            if sm:
                speaker_raw = sm.group(1).strip()
                speaker_id = re.sub(r"[\[\]\*\#]", "", speaker_raw).strip()
                if speaker_id.isdigit():
                    speaker_id = f"Speaker {speaker_id}"

                # Clean utterance text from left part
                left_clean = left.strip()
                # Remove leading [Utterance], **[Utterance]**, Utterance:, etc.
                left_clean = re.sub(
                    r"^(?:[\*\#\-\s\[\(]*)(?:Utterance|utterance|वाक्य|चरण)[\*\#\s\]\)]*[:\-\u2013]?\s*",
                    "",
                    left_clean,
                    flags=re.IGNORECASE,
                ).strip()
                # Remove any remaining stray leading punctuation or markdown artifacts
                left_clean = re.sub(r"^[:\-\u2013\*\#\s\]\)]+", "", left_clean).strip()

                # Prepend any buffered text from previous lines
                if buffered_text_parts:
                    full_text = " ".join(buffered_text_parts + [left_clean]).strip()
                    buffered_text_parts = []
                else:
                    full_text = left_clean

                if full_text:
                    turns.append(Turn(speaker=speaker_id, text=full_text))
                continue

        # Check for fallback traditional prefix format
        fm = prefix_fallback_re.match(line_clean)
        if fm:
            # If we had buffered text, flush or attach
            speaker_id = re.sub(r"[\[\]\*\#]", "", fm.group(1)).strip()
            text_turn = fm.group(2).strip()
            if buffered_text_parts:
                buffered_text_parts = []
            if text_turn:
                turns.append(Turn(speaker=speaker_id, text=text_turn))
            continue

        # Check if line starts with [Utterance] without a pipe on the same line
        if re.match(r"^(?:\**\[?Utterance\]?[:\s]*\**)", line_clean, re.IGNORECASE):
            clean_part = re.sub(r"^(?:\**\[?Utterance\]?[:\s]*\**)", "", line_clean).strip()
            if clean_part:
                buffered_text_parts.append(clean_part)
            continue

        # Otherwise: line is continuation text
        if buffered_text_parts:
            buffered_text_parts.append(line_clean)
        elif turns:
            turns[-1].text += " " + line_clean
        else:
            buffered_text_parts.append(line_clean)

    # If any buffered text remains at the end without a turn
    if buffered_text_parts and not turns:
        turns.append(Turn(speaker="Speaker 0", text=" ".join(buffered_text_parts).strip()))
    elif buffered_text_parts and turns:
        turns[-1].text += " " + " ".join(buffered_text_parts).strip()

    # Fallback if no turns could be formed from non-empty string
    if not turns and raw_text.strip():
        turns.append(Turn(speaker="Speaker 0", text=raw_text.strip()))

    return turns


class Token0BypassPipeline:
    """Strategy A: Single-step pipeline utilizing Token-0 Bypass framing for Gemini 3.5 Flash Lite."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        system_instruction: str = CANDIDATE_A_SYSTEM_INSTRUCTION,
        user_prompt: str = CANDIDATE_A_USER_PROMPT,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        """Initializes the Token-0 Bypass Pipeline.

        Args:
            client: Optional GeminiClient instance.
            system_instruction: System prompt enforcing Token-0 bypass syntax.
            user_prompt: User prompt accompanying audio bytes.
            thinking_budget: Thinking budget token count (0 for pure acoustic STT).
            temperature: Sampling temperature (0.0 for deterministic decoding).
            max_output_tokens: Maximum tokens in generated response (default 8192).
        """
        self.client = client or GeminiClient()
        self.system_instruction = system_instruction
        self.user_prompt = user_prompt
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
    ) -> PipelinePrediction:
        """Executes Token-0 Bypass single-step pipeline on a benchmark sample.

        Args:
            sample: Benchmark SampleData instance.
            model_id: Target Gemini model identifier (default gemini-3.5-flash-lite).

        Returns:
            PipelinePrediction with turns, raw response, and latency.
        """
        logger.info(
            f"Running Token0BypassPipeline on sample {sample.sample_id} with model {model_id}"
        )
        raw_response, latency = self.client.generate_with_audio(
            model=model_id,
            audio_source=sample.audio_path,
            prompt=self.user_prompt,
            system_instruction=self.system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        turns = self.parse_turns(raw_response)

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="token0_bypass",
            predicted_turns=turns,
            raw_response=raw_response,
            latency_seconds=latency,
        )
