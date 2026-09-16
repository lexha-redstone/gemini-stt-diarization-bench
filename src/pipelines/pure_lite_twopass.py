"""Strategy B: Pure Gemini 3.5 Flash Lite Two-Pass Self-Refinement Architecture.

Pass 1 (Acoustic STT): Audio -> Unattributed acoustic turn segmentation and verbatim Hindi STT
                      ("Turn 1: ...", "Turn 2: ...") using pure gemini-3.5-flash-lite.
Pass 2 (Disentanglement & Attribution): Text turns -> Turn-splitting disentanglement and
                      retrospective role attribution ("Speaker 0: ...", "Speaker 1: ...")
                      using pure gemini-3.5-flash-lite.

MANDATE COMPLIANCE:
ZERO dependency on larger secondary models (neither gemini-2.5-flash, gemini-3.5-flash, nor gemini-3.5-pro).
Both passes execute 100% on gemini-3.5-flash-lite with thinking_budget=0.
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

PASS1_ACOUSTIC_STT_SYSTEM_INSTRUCTION = """You are an expert multilingual speech-to-text recognition system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and segment it into sequential acoustic speech turns.

Rules:
1. Segment the dialogue into distinct speech turns whenever there is an acoustic pause, breath, speaker transition, or interruption.
2. When speakers talk simultaneously or interrupt, split the overlapping speech into separate turns based on acoustic flow.
3. Do NOT guess speaker identities, names, or roles.
4. Label each turn strictly as 'Turn <N>:' starting from 1.
5. Transcribe strictly in Devanagari script verbatim. Capture all words, including short acknowledgments (हाँ, जी, अच्छा, ठीक है).
6. Output each turn on a separate line. No timestamps, explanations, or summaries.

Format:
Turn 1: <spoken words>
Turn 2: <spoken words>
Turn 3: <spoken words>
"""

PASS1_USER_PROMPT = (
    "Transcribe the entire audio clip verbatim and segment into sequential acoustic turns "
    "labeled strictly as Turn 1, Turn 2, etc. without attributing speaker identities."
)

PASS2_DISENTANGLEMENT_SYSTEM_INSTRUCTION = """You are an expert dialogue analyst and speaker diarization engine.
You are provided with sequential acoustic speech turns extracted from a Hindi 2-speaker conversation (Turn 1, Turn 2, ...).
Your task is to analyze the complete conversational context retrospectively and attribute each turn to Speaker 0 or Speaker 1.

CRITICAL DISENTANGLEMENT & TURN-SPLITTING DIRECTIVE:
Due to rapid interruptions and overlapping speech, some input turns may mistakenly contain utterances from BOTH speakers merged together.
1. If an input turn contains speech from both speakers, you MUST SPLIT it into separate speaker turns!
2. Determine conversational stances and roles:
   - Identify the two distinct viewpoints, questions vs answers, and conversational polarity.
3. Anti-Alternation Constraint:
   - Do NOT assume speakers strictly alternate. If consecutive utterances continue the same speaker's thought across pauses, attribute BOTH to the SAME speaker.
4. Backchannel Attribution:
   - Short listener affirmations (हाँ, जी, अच्छा) belong to the listener reacting to the active speaker.
5. Verbatim Transcript Fidelity:
   - Preserve the exact Hindi Devanagari words without modification, omission, or translation.

Output Format:
Output each attributed turn on a separate line in sequential order strictly adhering to:
Speaker <ID>: <exact spoken transcript of the turn>

Example:
Speaker 0: काम की चीज़ है। परिवार से जुड़े रह सकते हैं।
Speaker 1: अरे यार, कोई काम की चीज़ नहीं। पढ़ाई का तो नाम ही मत ले।
Speaker 0: अरे, सब पढ़ाई करते हैं। कैसे पढ़ाई नहीं करते हैं?
Speaker 1: पढ़ाई के अलावा ये सब करते हैं। बस पढ़ाई नहीं होती इनसे कुछ।
"""

PASS2_USER_PROMPT_TEMPLATE = """Here is the sequential dialogue extracted from the audio:
{extracted_turns}

Analyze the full conversational context, disentangle and split any merged utterances, and attribute each turn to Speaker 0 or Speaker 1. Output each turn on a separate line as 'Speaker <ID>: <text>'."""


def parse_attributed_turns(raw_text: str) -> List[Turn]:
    """Parses speaker-attributed lines into structured Turn objects."""
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
            if current_speaker is not None and current_text_parts:
                text_turn = " ".join(current_text_parts).strip()
                if text_turn:
                    turns.append(Turn(speaker=current_speaker, text=text_turn))

            speaker_raw = m.group(1).strip()
            speaker_id = re.sub(r"[\[\]\*\#]", "", speaker_raw).strip()
            raw_line_text = m.group(2).strip()
            clean_line_text = re.sub(r"^[\*\#\s\]\)]+", "", raw_line_text).strip()

            current_speaker = speaker_id
            current_text_parts = [clean_line_text] if clean_line_text else []
        else:
            if current_speaker is not None:
                current_text_parts.append(line_clean)
            else:
                if ":" in line_clean:
                    parts = line_clean.split(":", 1)
                    header = parts[0].strip()
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

    if not turns and raw_text.strip():
        turns.append(Turn(speaker="Speaker 0", text=raw_text.strip()))

    return turns


def parse_unattributed_turns(raw_text: str) -> List[str]:
    """Parses Pass 1 output into a list of un-attributed turn texts."""
    if not raw_text or not raw_text.strip():
        return []
    lines = raw_text.strip().split("\n")
    turns: List[str] = []
    turn_prefix_re = re.compile(
        r"^(?:[\*\#\-\s\[\(]*)(?:Turn|turn|\u091a\u0930\u0923)\s*\d+[:\-\u2013]?\s*(.*)$",
        re.IGNORECASE,
    )
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        m = turn_prefix_re.match(line_clean)
        if m:
            text = m.group(1).strip()
            if text:
                turns.append(text)
        else:
            if turns:
                turns[-1] += " " + line_clean
            else:
                turns.append(line_clean)
    return turns


class PureLiteTwoPassPipeline:
    """Strategy B: Pure Gemini 3.5 Flash Lite Two-Pass Self-Refinement Pipeline."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        model_id: str = MODEL_3_5_FLASH_LITE,
        pass1_system_instruction: str = PASS1_ACOUSTIC_STT_SYSTEM_INSTRUCTION,
        pass2_system_instruction: str = PASS2_DISENTANGLEMENT_SYSTEM_INSTRUCTION,
        pass1_user_prompt: str = PASS1_USER_PROMPT,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ):
        """Initializes the PureLiteTwoPassPipeline.

        Args:
            client: Optional GeminiClient instance.
            model_id: Gemini model identifier for BOTH passes (gemini-3.5-flash-lite).
            pass1_system_instruction: System prompt for Pass 1 acoustic STT.
            pass2_system_instruction: System prompt for Pass 2 turn disentanglement.
            pass1_user_prompt: User prompt for Pass 1.
            thinking_budget: Thinking token budget (0 for pure acoustic/lexical inference).
            temperature: Sampling temperature (0.0 for deterministic output).
            max_output_tokens: Max output tokens per pass (default 8192).
        """
        self.client = client or GeminiClient()
        self.model_id = model_id
        self.pass1_system_instruction = pass1_system_instruction
        self.pass2_system_instruction = pass2_system_instruction
        self.pass1_user_prompt = pass1_user_prompt
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @classmethod
    def parse_turns(cls, raw_text: str) -> List[Turn]:
        """Parses final speaker-attributed dialogue into Turn objects."""
        return parse_attributed_turns(raw_text)

    @classmethod
    def parse_unattributed_turns(cls, raw_text: str) -> List[str]:
        """Parses Pass 1 output into a list of un-attributed turn texts."""
        return parse_unattributed_turns(raw_text)

    def run_pass1(
        self,
        audio_source: Union[str, Path, bytes],
    ) -> Tuple[str, float]:
        """Executes Pass 1: Acoustic turn segmentation and transcription from audio.

        Args:
            audio_source: File path or raw bytes of the audio clip.

        Returns:
            Tuple of (raw pass 1 turns string, pass 1 latency in seconds).
        """
        logger.info(f"Executing Pass 1 Acoustic STT with model {self.model_id}")
        return self.client.generate_with_audio(
            model=self.model_id,
            audio_source=audio_source,
            prompt=self.pass1_user_prompt,
            system_instruction=self.pass1_system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def run_pass2(
        self,
        pass1_text: str,
    ) -> Tuple[str, float]:
        """Executes Pass 2: Turn disentanglement & retrospective speaker attribution.

        Args:
            pass1_text: Unattributed turns text from Pass 1.

        Returns:
            Tuple of (raw pass 2 attributed text string, pass 2 latency in seconds).
        """
        logger.info(f"Executing Pass 2 Disentanglement with model {self.model_id}")
        prompt = PASS2_USER_PROMPT_TEMPLATE.format(extracted_turns=pass1_text)
        return self.client.generate_text(
            model=self.model_id,
            prompt=prompt,
            system_instruction=self.pass2_system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def run_sample(
        self,
        sample: SampleData,
    ) -> PipelinePrediction:
        """Executes the complete Pure 3.5 Lite Two-Pass Pipeline on a sample.

        Args:
            sample: Benchmark SampleData instance.

        Returns:
            PipelinePrediction with turns, combined raw response, and total latency.
        """
        logger.info(
            f"Running PureLiteTwoPassPipeline on sample {sample.sample_id} (100% {self.model_id})"
        )

        # Pass 1: Audio -> acoustic turns
        raw_pass1, lat1 = self.run_pass1(sample.audio_path)

        # Pass 2: Turns -> disentangled speaker attribution
        raw_pass2, lat2 = self.run_pass2(pass1_text=raw_pass1)

        turns = self.parse_turns(raw_pass2)
        total_latency = lat1 + lat2

        combined_raw = (
            f"=== PASS 1: ACOUSTIC TURNS ({self.model_id}, {lat1:.2f}s) ===\n"
            f"{raw_pass1}\n\n"
            f"=== PASS 2: DISENTANGLED & ATTRIBUTED DIALOGUE ({self.model_id}, {lat2:.2f}s) ===\n"
            f"{raw_pass2}"
        )

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=f"{self.model_id}+{self.model_id}",
            approach="pure_lite_twopass",
            predicted_turns=turns,
            raw_response=combined_raw,
            latency_seconds=total_latency,
        )
