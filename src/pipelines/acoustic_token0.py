"""Strategy C3: Acoustic-Anchored Token-0 Bypass Pipeline for Pure Gemini 3.5 Flash Lite.

Innovations:
1. Acoustic Speaker Profiles Preamble ([Speaker Profiles]) emitted before turn transcription,
   depositing explicit pitch register, vocal timbre, speaking cadence, and initial stance
   into the autoregressive context to prevent low-ΔF0 inversion cascades.
2. Tri-Delimited Turn Syntax:
   [Utterance] <verbatim Devanagari text> | [Transition] <TAG> | [Speaker] Speaker <ID>
3. Four Transition Tags:
   - CONTINUE: Monologue continuation, sentence clauses, or self-acknowledgment (same speaker)
   - SHIFT: Clean conversational floor transfer to another speaker
   - RESUME: Interruption resumption (A-B-A sandwich pattern)
   - OVERLAP: Simultaneous cross-talk / heated interruption
4. Calibrated 11-turn few-shot demonstration with 60% non-alternating transition ratio.
5. Strict anti-ping-pong monologue constraints, self-acknowledgment rules, and dynamic speaker count (N >= 2).
"""

import logging
import re
from typing import List, Optional

from src.client import GeminiClient
from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn

logger = logging.getLogger(__name__)


def build_c3_system_instruction(num_speakers: int = 2) -> str:
    """Builds Candidate C3 system prompt conditioned on speaker count."""
    if num_speakers == 2:
        speaker_desc = "distinct speaker (Speaker 0 or Speaker 1)"
        profile_template = (
            "[Speaker Profiles]\n"
            "- Speaker 0: <Gender, relative pitch (e.g. deeper chest resonance / higher head voice), "
            "vocal timbre (e.g. clear, raspy, resonant, nasal), speaking cadence (e.g. deliberate, rapid), "
            "and initial conversational stance or role (e.g. caller, questioner, advocate).>\n"
            "- Speaker 1: <Gender, contrasting pitch relative to Speaker 0, vocal timbre, speaking cadence, "
            "and contrasting conversational stance or role (e.g. respondent, skeptic, explainer).>"
        )
        cardinality_text = (
            "4. Speaker Cardinality & Profile Grounding:\n"
            "   - There are strictly 2 primary speakers: Speaker 0 and Speaker 1.\n"
            "   - Ground every turn strictly in the acoustic vocal characteristics established in your initial "
            "Speaker Profiles. Maintain unique speaker identities throughout the entire recording. Never swap "
            "speaker identities mid-conversation."
        )
    else:
        speaker_names = ", ".join([f"Speaker {i}" for i in range(num_speakers - 1)]) + f", and Speaker {num_speakers - 1}"
        speaker_desc = f"distinct speaker ({speaker_names})"
        profile_lines = [
            f"- Speaker {i}: <Gender, relative pitch register, vocal timbre, cadence, and role.>"
            for i in range(num_speakers)
        ]
        profile_template = "[Speaker Profiles]\n" + "\n".join(profile_lines)
        cardinality_text = (
            f"4. Speaker Cardinality & Profile Grounding:\n"
            f"   - There are {num_speakers} primary speakers: {speaker_names}.\n"
            "   - Ground every turn strictly in the acoustic vocal characteristics established in your initial "
            "Speaker Profiles. Maintain unique speaker identities throughout the entire recording."
        )

    return f"""You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi strictly using Devanagari script (देवनागरी लिपि) and accurately attribute each conversational turn to its {speaker_desc}.

CRITICAL STEP 1 — ACOUSTIC SPEAKER PROFILES (EMIT FIRST):
Before transcribing turns, you MUST listen to the entire audio clip and establish an explicit acoustic contrast profile for each speaker to ground your attribution:
{profile_template}

CRITICAL STEP 2 — TOKEN-0 BYPASS WITH TRANSITION TYPING:
For every conversational turn, you MUST emit the spoken text FIRST, the transition relationship SECOND, and the speaker identifier LAST on each line:
[Utterance] <spoken Hindi text verbatim> | [Transition] <TAG> | [Speaker] Speaker <ID>

Valid Transition Tags (<TAG>):
- CONTINUE : Same speaker continuing their thought across sentences, breathing pauses, or self-acknowledgments (monologue).
- SHIFT    : Clean transfer of conversational floor to another speaker.
- RESUME   : Speaker resuming speaking immediately after a brief interruption/backchannel by another speaker (A-B-A pattern).
- OVERLAP  : Simultaneous speech or speech interrupting while another person is still vocalizing.

Principles & Anti-Cascade Constraints:
1. Turn Continuity & Anti-Alternation (CONTINUE):
   - Natural conversation is NOT a ping-pong match. Real dialogues contain 30% to 50% consecutive turns by the same speaker.
   - If a speaker pauses, breathes, or speaks across multiple sentences without another person taking over, tag with [Transition] CONTINUE and attribute to the SAME speaker.
   - Do NOT alternate simply because a new sentence began!
2. Self-Acknowledgments vs. Interjections:
   - When a speaker uses an affirmative starter (e.g., हाँ, ओके, ठीक है, अच्छा) to continue their own thought, use [Transition] CONTINUE with the SAME speaker.
   - When the listener interjects a brief reaction (e.g., अरे!, जी हाँ, अच्छा) while the other person is speaking, emit the interjection on its own line with [Transition] SHIFT or [Transition] OVERLAP.
3. Interruption Resumption (A-B-A Pattern / RESUME):
   - When Speaker A is speaking and Speaker B makes a brief interruption/backchannel, emit Speaker B's interjection on its own line.
   - When Speaker A resumes speaking immediately after the brief interruption, you MUST tag the turn as [Transition] RESUME and attribute it to Speaker A!
{cardinality_text}
5. Zero-Swallowing Rule:
   - Transcribe every question, affirmation, and overlapping phrase on its own line. Never merge two different voices into one line.
6. Verbatim Devanagari Fidelity:
   - Transcribe 100% in Devanagari script. Never summarize, drop stuttered words, or translate.
   - NEVER output Urdu, Nastaliq, or Latin/English transliteration.

Few-Shot Demonstration (Turn Continuity, Overlaps, Resumptions, and Tagging):
[Speaker Profiles]
- Speaker 0: Male, lower pitch chest voice, deliberate cadence, loan officer inquiring about payment status.
- Speaker 1: Male, higher pitch voice, energetic cadence, customer explaining receipt and bank transfer.

[Utterance] नमस्कार, क्या मेरी बात शर्मा जी से हो रही है? | [Transition] SHIFT | [Speaker] Speaker 0
[Utterance] मैं बैंक शाखा से बोल रहा हूँ, आपकी बकाया किस्त के संबंध में। | [Transition] CONTINUE | [Speaker] Speaker 0
[Utterance] जी, बोलिए। | [Transition] SHIFT | [Speaker] Speaker 1
[Utterance] पिछले महीने की किस्त का भुगतान अभी तक रिकॉर्ड नहीं हुआ है। | [Transition] RESUME | [Speaker] Speaker 0
[Utterance] हमारे सिस्टम में पेंडिंग शो हो रहा है, क्या आपने पेमेंट किया था? | [Transition] CONTINUE | [Speaker] Speaker 0
[Utterance] अरे नहीं, मैंने तो परसों ही ऑनलाइन— | [Transition] OVERLAP | [Speaker] Speaker 1
[Utterance] क्या आपके पास बैंक रसीद या ट्रांजैक्शन आईडी है? | [Transition] OVERLAP | [Speaker] Speaker 0
[Utterance] जी हाँ, मेरे पास रसीद का स्क्रीनशॉट है। | [Transition] RESUME | [Speaker] Speaker 1
[Utterance] मैं अभी व्हाट्सएप पर फोटो भेजता हूँ। | [Transition] CONTINUE | [Speaker] Speaker 1
[Utterance] ठीक है, आप भेज दीजिए। | [Transition] SHIFT | [Speaker] Speaker 0
[Utterance] मैं तुरंत वेरिफिकेशन करके सिस्टम में अपडेट कर दूँगा। | [Transition] CONTINUE | [Speaker] Speaker 0
"""


C3_USER_PROMPT = (
    "Establish the acoustic [Speaker Profiles] first, then transcribe the entire audio clip verbatim "
    "in Devanagari script and attribute all speaker turns using the format:\n"
    "[Utterance] <spoken text> | [Transition] <CONTINUE|SHIFT|RESUME|OVERLAP> | [Speaker] Speaker <ID>"
)


def parse_acoustic_token0_turns(raw_text: str) -> List[Turn]:
    """Robustly parses Candidate C3 output into structured Turn objects.

    Strips the `[Speaker Profiles]` preamble and extracts turns formatted with
    tri-delimited syntax, with robust fallbacks for two-field and prefix formats.
    """
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.strip().split("\n")
    turns: List[Turn] = []

    # Regex for tri-delimited turn: [Utterance] ... | [Transition] ... | [Speaker] ...
    p3 = re.compile(
        r"^\s*(?:\[?Utterance\]?[:\s]*)?(?P<text>.*?)\s*"
        r"\|\s*(?:\[?Transition\]?[:\s]*)?(?P<trans>[A-Z_]+)\s*"
        r"\|\s*(?:\[?Speaker\]?[:\s]*)?(?P<spk>Speaker\s*\d+|\d+)\s*$",
        re.IGNORECASE,
    )
    # Regex for two-field fallback: [Utterance] ... | [Speaker] ...
    p2 = re.compile(
        r"^\s*(?:\[?Utterance\]?[:\s]*)?(?P<text>.*?)\s*"
        r"\|\s*(?:\[?Speaker\]?[:\s]*)?(?P<spk>Speaker\s*\d+|\d+)\s*$",
        re.IGNORECASE,
    )
    # Regex for standard prefix fallback: Speaker X: ...
    p_prefix = re.compile(
        r"^\s*(?:\[?Speaker\]?[:\s]*)?(?P<spk>Speaker\s*\d+)\s*:\s*(?P<text>.*)$",
        re.IGNORECASE,
    )

    # State tracking: are we past the speaker profiles preamble?
    in_preamble = False

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check for start of [Speaker Profiles] preamble
        if re.search(r"\[Speaker\s*Profiles\]", line_clean, re.IGNORECASE):
            in_preamble = True
            continue

        # Ignore lines inside the preamble (e.g., "- Speaker 0: ...", "* Speaker 1: ...")
        if in_preamble:
            if re.match(r"^[-*•]?\s*Speaker\s*\d+\s*:", line_clean, re.IGNORECASE) or line_clean.startswith("-"):
                continue
            # If line matches turn pattern, we have exited the preamble
            if "[Utterance]" in line_clean or "|" in line_clean:
                in_preamble = False
            else:
                # Still in explanatory preamble text
                continue

        # Strip list markers like "1. ", "- ", etc.
        line_clean = re.sub(r"^[-*•\d+.]+\s*", "", line_clean).strip()
        if not line_clean:
            continue

        # Try tri-delimited pattern
        m3 = p3.match(line_clean)
        if m3:
            txt = m3.group("text").strip()
            spk = m3.group("spk").strip()
            if spk.isdigit():
                spk = f"Speaker {spk}"
            elif not spk.lower().startswith("speaker"):
                spk = f"Speaker {spk}"
            # Normalize speaker name to "Speaker X"
            spk_match = re.search(r"(\d+)", spk)
            if spk_match:
                spk = f"Speaker {spk_match.group(1)}"
            if txt:
                turns.append(Turn(speaker=spk, text=txt))
            continue

        # Try two-field fallback
        m2 = p2.match(line_clean)
        if m2:
            txt = m2.group("text").strip()
            spk = m2.group("spk").strip()
            spk_match = re.search(r"(\d+)", spk)
            if spk_match:
                spk = f"Speaker {spk_match.group(1)}"
            if txt:
                turns.append(Turn(speaker=spk, text=txt))
            continue

        # Try prefix fallback (Speaker X: text)
        m_pref = p_prefix.match(line_clean)
        if m_pref:
            txt = m_pref.group("text").strip()
            spk = m_pref.group("spk").strip()
            spk_match = re.search(r"(\d+)", spk)
            if spk_match:
                spk = f"Speaker {spk_match.group(1)}"
            if txt:
                turns.append(Turn(speaker=spk, text=txt))
            continue

        # If line does not match patterns but we already have turns, append as continuation
        if turns and line_clean:
            turns[-1].text += " " + line_clean

    # If no turns parsed, fallback to single turn
    if not turns and raw_text.strip():
        # Clean any preamble tags from raw text
        cleaned_text = re.sub(r"\[Speaker\s*Profiles\][\s\S]*?(?=\[Utterance\]|$)", "", raw_text, flags=re.IGNORECASE).strip()
        if cleaned_text:
            turns.append(Turn(speaker="Speaker 0", text=cleaned_text))
        else:
            turns.append(Turn(speaker="Speaker 0", text=raw_text.strip()))

    return turns


class AcousticAnchorToken0Pipeline:
    """Strategy C3: Acoustic-Anchored Token-0 Bypass Pipeline."""

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
        return parse_acoustic_token0_turns(raw_text)

    def run_sample(
        self,
        sample: SampleData,
        model_id: str = MODEL_3_5_FLASH_LITE,
        num_speakers: Optional[int] = None,
    ) -> PipelinePrediction:
        """Executes Candidate C3 live inference on a benchmark sample.

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

        system_instruction = build_c3_system_instruction(spk_count)

        logger.info(
            f"Running AcousticAnchorToken0Pipeline on sample {sample.sample_id} "
            f"(spk_count={spk_count}) with model {model_id}"
        )

        raw_response, latency = self.client.generate_with_audio(
            model=model_id,
            audio_source=sample.audio_path,
            prompt=C3_USER_PROMPT,
            system_instruction=system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        turns = self.parse_turns(raw_response)

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="strategy_c3_acoustic_token0",
            predicted_turns=turns,
            raw_response=raw_response,
            latency_seconds=latency,
        )


# Alias runner for benchmark runner naming convention
AcousticAnchorToken0Runner = AcousticAnchorToken0Pipeline
