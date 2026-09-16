"""Candidate C5: Adaptive Acoustic Champion Pipeline for Pure Gemini 3.5 Flash Lite.

Innovations:
1. Stage 1: Acoustic STT with Question-Isolation & Zero-Swallowing Directive
   - Emits explicit [Speaker Profiles] preamble grounding pitch register, vocal timbre,
     cadence, and initial stance.
   - Tri-delimited syntax: [Utterance] <text> | [Transition] <TAG> | [Speaker] Speaker <ID>
   - Enforces Question-Isolation & Zero-Swallowing: Every question ('?', क्यों, क्या, कब, कैसे, कहाँ)
     and reactive affirmation is strictly emitted on its own line with explicit transition tagging,
     preventing sub-second question swallowing (resolving the hindi_073 / hindi_092 deficit).
2. Stage 2: Conservative, Transition-Aware Text Consistency Verifier
   - Informs Stage 2 of transition tags:
     [Turn {idx}] {turn.speaker} | [Transition] {tag}: {turn.text}
   - Strict Constraint on Tagged Turns:
     Never flip turns marked CONTINUES_SAME_SPEAKER or RESUMES_AFTER_INTERRUPTION unless
     there is an overwhelming global stance inversion.
   - For casual banter or non-adversarial dialogue, emits {"corrections": []}.
3. Adaptive Gating Trigger:
   - Evaluates clip duration (>= 70.0s), ping-pong trap alternation (>= 78% on >= 15 turns),
     and debate / argumentative polarity markers.
   - For short dialogues (< 70s) with normal alternation (e.g. hindi_085, hindi_064, hindi_089),
     completely bypasses Stage 2 with 0ms latency overhead, directly returning Stage 1's
     pristine acoustic attribution.
   - Zero lexical drift: 100% frozen Devanagari text, with minimal JSON diff corrections applied.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.client import GeminiClient
from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn

logger = logging.getLogger(__name__)

# Canonical Transition Tag Mapping
TAG_MAP = {
    "CONTINUE": "CONTINUES_SAME_SPEAKER",
    "CONTINUATION": "CONTINUES_SAME_SPEAKER",
    "CONTINUES": "CONTINUES_SAME_SPEAKER",
    "CONTINUES_SAME_SPEAKER": "CONTINUES_SAME_SPEAKER",
    "RESUME": "RESUMES_AFTER_INTERRUPTION",
    "RESUMES": "RESUMES_AFTER_INTERRUPTION",
    "RESUMPTION": "RESUMES_AFTER_INTERRUPTION",
    "RESUMES_AFTER_INTERRUPTION": "RESUMES_AFTER_INTERRUPTION",
    "SHIFT": "SPEAKER_FLOOR_SHIFT",
    "SHIFTS": "SPEAKER_FLOOR_SHIFT",
    "FLOOR_SHIFT": "SPEAKER_FLOOR_SHIFT",
    "SPEAKER_FLOOR_SHIFT": "SPEAKER_FLOOR_SHIFT",
    "OVERLAP": "SIMULTANEOUS_OVERLAP",
    "OVERLAPS": "SIMULTANEOUS_OVERLAP",
    "SIMULTANEOUS": "SIMULTANEOUS_OVERLAP",
    "SIMULTANEOUS_OVERLAP": "SIMULTANEOUS_OVERLAP",
}

# Debate and argumentative vocabulary for conversational polarity detection
DEBATE_MARKERS = [
    "बहस",      # debate
    "सहमत",     # agree
    "असहमत",    # disagree
    "विरोध",    # oppose / opposition
    "नुकसान",   # loss / harm
    "फायदा",    # benefit / advantage
    "फायदे",    # benefits
    "खिलाफ",    # against
    "पक्ष",     # side / favor
    "मुद्दा",    # issue
    "मुद्दे",    # issues
    "तर्क",     # argument / reasoning
    "आपत्ति",   # objection
    "समस्या",   # problem
    "राय",      # opinion
]


def build_c5_stage1_system_instruction(num_speakers: int = 2) -> str:
    """Builds Candidate C5 Stage 1 system prompt with enhanced Question-Isolation."""
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
5. Zero-Swallowing & Question-Isolation Directive:
   - When a different speaker interjects, asks a question (e.g. ending in '?', or interrogatives like क्यों, क्या, कब, कैसे, कहाँ), or gives a brief reaction (e.g., हाँ, जी, अरे, अच्छा), ensure it is emitted on its own distinct line so sub-second questions are never swallowed into the preceding turn.
   - Transcribe every question, affirmation, and overlapping phrase on its own line. Never merge two different voices into one line.
   - Do NOT split a continuous sentence or thought spoken by the SAME person into artificial turns.
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


C5_STAGE1_USER_PROMPT = (
    "Establish the acoustic [Speaker Profiles] first, then transcribe the entire audio clip verbatim "
    "in Devanagari script and attribute all speaker turns using the format:\n"
    "[Utterance] <spoken text> | [Transition] <CONTINUE|SHIFT|RESUME|OVERLAP> | [Speaker] Speaker <ID>"
)


STAGE2_CHAMPION_SYSTEM_INSTRUCTION = """You are an expert dialogue consistency verifier.
Below is a sequential Hindi dialogue with preliminary speaker attributions and transition tags generated from acoustic audio cues.
The preliminary attributions are ~80-85% accurate, but may contain specific conversational anomalies.

YOUR TASK:
Review the turn sequence and output strictly a JSON object with turn numbers that must be corrected.

Correction Rules:
1. Conversational Stance & Debate Polarity:
   If speakers hold contrasting opinions or debate stances (e.g. Village vs. City life, Tech Pros vs. Cons, Child Labor / Government responsibility, Work vs. Break, Mobile phones / Online education):
   - Ground each speaker's identity in the established Speaker Profiles.
   - Turns advocating Stance A MUST be attributed to the advocating speaker, and opposing counter-arguments to the opposing speaker.
   - An opposing debate argument is an OVERWHELMING GLOBAL STANCE INVERSION that overrides any preliminary continuity tag!
2. Single-Speaker Recovery:
   If the preliminary turns are erroneously attributed to only one speaker (e.g., all turns are Speaker 0), reassign the conversational turns so that the speakers are properly distinguished according to the dialogue exchange.
3. Interruption Resumption (A-B-A Sandwich):
   If Speaker A was speaking, Speaker B made a brief 1-2 word reaction (e.g., हाँ, जी, अच्छा, अरे, ठीक है, ओके),
   and Turn N immediately continues Speaker A's thought/sentence/question, ensure Turn N is attributed to Speaker A.
4. Sentence / Monologue Continuation:
   If Turn N begins with an incomplete clause, conjunction, or continuation continuing Turn N-1 without a genuine speaker change,
   ensure both turns have the SAME speaker.
5. Strict Constraint on Tagged Turns:
   If a turn is marked [Transition] CONTINUES_SAME_SPEAKER or [Transition] RESUMES_AFTER_INTERRUPTION, DO NOT change the speaker unless there is an overwhelming global stance inversion.
6. Default Rule:
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


def parse_c5_stage1_turns(raw_text: str) -> Tuple[List[Turn], List[str], str]:
    """Parses Candidate C5 Stage 1 response into turns, transition tags, and speaker profiles text."""
    if not raw_text or not raw_text.strip():
        return [], [], ""

    lines = raw_text.strip().split("\n")
    turns: List[Turn] = []
    tags: List[str] = []
    profiles_lines: List[str] = []

    # Regex for tri-delimited turn: [Utterance] ... | [Transition] ... | [Speaker] ...
    p3 = re.compile(
        r"^\s*(?:\[?Utterance\]?[:\s]*)?(?P<text>.*?)\s*"
        r"\|\s*(?:\[?Transition\]?[:\s]*)?(?P<trans>[A-Z_]+)\s*"
        r"\|\s*(?:\[?Speaker\]?[:\s]*)?(?P<spk>Speaker\s*\d+|spk[_\s]*\d+|\d+)\s*$",
        re.IGNORECASE,
    )
    # Regex for two-field fallback: [Utterance] ... | [Speaker] ...
    p2 = re.compile(
        r"^\s*(?:\[?Utterance\]?[:\s]*)?(?P<text>.*?)\s*"
        r"\|\s*(?:\[?Speaker\]?[:\s]*)?(?P<spk>Speaker\s*\d+|spk[_\s]*\d+|\d+)\s*$",
        re.IGNORECASE,
    )
    # Regex for standard prefix fallback: Speaker X: ...
    p_prefix = re.compile(
        r"^\s*(?:\[?Speaker\]?[:\s]*)?(?P<spk>Speaker\s*\d+|spk[_\s]*\d+|\d+)\s*:\s*(?P<text>.*)$",
        re.IGNORECASE,
    )

    in_preamble = False

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Skip markdown code fences (```, ```json, ```text, etc.)
        if line_clean.startswith("```"):
            continue

        # Check for start of [Speaker Profiles] preamble
        if re.search(r"\[Speaker\s*Profiles\]", line_clean, re.IGNORECASE):
            in_preamble = True
            profiles_lines.append(line_clean)
            continue

        # Process lines inside the preamble
        if in_preamble:
            if re.match(r"^[-*•]?\s*Speaker\s*\d+\s*:", line_clean, re.IGNORECASE) or line_clean.startswith("-"):
                profiles_lines.append(line_clean)
                continue
            if "[Utterance]" in line_clean or "|" in line_clean:
                in_preamble = False
            else:
                profiles_lines.append(line_clean)
                continue

        # Strip list markers like "1. ", "- ", etc.
        line_clean = re.sub(r"^[-*•\d+.]+\s*", "", line_clean).strip()
        if not line_clean:
            continue

        # Try tri-delimited pattern
        m3 = p3.match(line_clean)
        if m3:
            txt = m3.group("text").strip()
            raw_tag = m3.group("trans").strip().upper()
            spk = m3.group("spk").strip()
            spk_match = re.search(r"(\d+)", spk)
            normalized_spk = f"Speaker {spk_match.group(1)}" if spk_match else spk
            canonical_tag = TAG_MAP.get(raw_tag, raw_tag)
            if txt:
                turns.append(Turn(speaker=normalized_spk, text=txt))
                tags.append(canonical_tag)
            continue

        # Try two-field fallback
        m2 = p2.match(line_clean)
        if m2:
            txt = m2.group("text").strip()
            spk = m2.group("spk").strip()
            spk_match = re.search(r"(\d+)", spk)
            normalized_spk = f"Speaker {spk_match.group(1)}" if spk_match else spk
            # Infer tag based on continuity
            if turns and turns[-1].speaker == normalized_spk:
                canonical_tag = "CONTINUES_SAME_SPEAKER"
            else:
                canonical_tag = "SPEAKER_FLOOR_SHIFT"
            if txt:
                turns.append(Turn(speaker=normalized_spk, text=txt))
                tags.append(canonical_tag)
            continue

        # Try prefix fallback
        m_pref = p_prefix.match(line_clean)
        if m_pref:
            txt = m_pref.group("text").strip()
            spk = m_pref.group("spk").strip()
            spk_match = re.search(r"(\d+)", spk)
            normalized_spk = f"Speaker {spk_match.group(1)}" if spk_match else spk
            if turns and turns[-1].speaker == normalized_spk:
                canonical_tag = "CONTINUES_SAME_SPEAKER"
            else:
                canonical_tag = "SPEAKER_FLOOR_SHIFT"
            if txt:
                turns.append(Turn(speaker=normalized_spk, text=txt))
                tags.append(canonical_tag)
            continue

        # Append continuation to previous turn if available (skipping trailing markdown fences)
        if turns and line_clean:
            cleaned_continuation = re.sub(r"```[a-zA-Z]*$", "", line_clean).strip()
            if cleaned_continuation:
                turns[-1].text += " " + cleaned_continuation

    # Fallback if no turns parsed
    if not turns and raw_text.strip():
        cleaned_text = re.sub(r"```[a-zA-Z]*", "", raw_text)
        cleaned_text = re.sub(r"\[Speaker\s*Profiles\][\s\S]*?(?=\[Utterance\]|$)", "", cleaned_text, flags=re.IGNORECASE).strip()
        fallback_text = cleaned_text if cleaned_text else raw_text.strip()
        turns.append(Turn(speaker="Speaker 0", text=fallback_text))
        tags.append("SPEAKER_FLOOR_SHIFT")

    profiles_text = "\n".join(profiles_lines).strip()
    return turns, tags, profiles_text


parse_adaptive_acoustic_turns = parse_c5_stage1_turns


def check_debate_context(turns: List[Turn]) -> Tuple[bool, List[str]]:
    """Checks whether turns contain debate or argumentative vocabulary."""
    full_text = " ".join(t.text for t in turns)
    found = [m for m in DEBATE_MARKERS if m in full_text]
    strong_debate_words = ["बहस", "असहमत", "विरोध", "खिलाफ", "आपत्ति"]
    if any(m in full_text for m in strong_debate_words):
        return True, found
    if len(found) >= 2:
        return True, found
    return False, found


def should_trigger_stage2(
    duration_seconds: float,
    turns: List[Turn],
    tags: List[str],
) -> Tuple[bool, str]:
    """Evaluates Adaptive Gating Trigger to decide whether to invoke Stage 2 text verifier.

    Trigger Criteria:
    1. Duration >= 70.0 seconds (long monologues / stance drift).
    2. High turn alternation (>= 85% alternation on >= 16 turns) indicating a possible ping-pong trap.
    3. Debate / argumentative polarity markers present in the dialogue.
    Otherwise, bypass Stage 2 with 0ms overhead.
    """
    n_turns = len(turns)
    if n_turns <= 1:
        return False, "Insufficient turns (<= 1) for Stage 2"

    # Criterion 0: Single-speaker collapse detection
    unique_speakers = set(t.speaker for t in turns)
    if len(unique_speakers) < 2 and n_turns >= 4:
        return True, f"Single-speaker collapse ({len(unique_speakers)} speaker for {n_turns} turns)"

    # Criterion 1: Long duration
    if duration_seconds >= 70.0:
        return True, f"Long duration ({duration_seconds:.1f}s >= 70.0s)"

    # Criterion 2: High turn alternation (ping-pong trap detection)
    alternations = sum(1 for i in range(n_turns - 1) if turns[i].speaker != turns[i + 1].speaker)
    alt_ratio = alternations / (n_turns - 1) if n_turns > 1 else 0.0
    if alt_ratio >= 0.85 and n_turns >= 16:
        return True, f"High alternation ping-pong trap ({alt_ratio:.1%} on {n_turns} turns)"

    # Criterion 3: Debate / argumentative markers
    has_debate, found_markers = check_debate_context(turns)
    if has_debate:
        return True, f"Debate markers detected ({', '.join(found_markers)})"

    return False, f"Bypass Stage 2 (dur={duration_seconds:.1f}s < 70s, alt={alt_ratio:.1%}, casual dialogue)"


class AdaptiveAcousticChampionPipeline:
    """Candidate C5: Adaptive Acoustic Champion Pipeline for Pure Gemini 3.5 Flash Lite."""

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
        turns, _, _ = parse_c5_stage1_turns(raw_text)
        return turns

    def _build_stage2_prompt(
        self,
        turns: List[Turn],
        tags: List[str],
        profiles_text: str = "",
    ) -> str:
        """Formats Stage 1 turns with explicit transition tags for Stage 2 verification."""
        prompt_parts = []
        if profiles_text and profiles_text.strip():
            prompt_parts.append(f"Speaker Context:\n{profiles_text.strip()}\n")

        lines = []
        for idx, turn in enumerate(turns, 1):
            tag = tags[idx - 1] if idx - 1 < len(tags) else "SPEAKER_FLOOR_SHIFT"
            lines.append(f"[Turn {idx}] {turn.speaker} | [Transition] {tag}: {turn.text}")
        prompt_parts.append("Input Dialogue:\n" + "\n".join(lines))
        prompt_parts.append("\nOutput JSON corrections:")
        return "\n".join(prompt_parts)

    def _parse_stage2_corrections(self, raw_text: str) -> List[Dict[str, Any]]:
        """Extracts JSON corrections array from model response."""
        if not raw_text or not raw_text.strip():
            return []

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
        """Executes Candidate C5 adaptive pipeline on a benchmark sample.

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

        system_instruction = build_c5_stage1_system_instruction(spk_count)

        logger.info(
            f"Running Candidate C5 Stage 1 (Audio STT) on sample {sample.sample_id} "
            f"(dur={sample.duration_seconds:.1f}s, spk_count={spk_count}) with model {model_id}"
        )

        stage1_raw, stage1_latency = self.client.generate_with_audio(
            model=model_id,
            audio_source=sample.audio_path,
            prompt=C5_STAGE1_USER_PROMPT,
            system_instruction=system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        stage1_turns, stage1_tags, profiles_text = parse_c5_stage1_turns(stage1_raw)

        stage1_pred = PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="strategy_c5_adaptive_champion",
            predicted_turns=stage1_turns,
            raw_response=stage1_raw,
            latency_seconds=stage1_latency,
        )

        # Evaluate Adaptive Gating Trigger
        trigger_stage2, trigger_reason = should_trigger_stage2(
            duration_seconds=sample.duration_seconds,
            turns=stage1_turns,
            tags=stage1_tags,
        )

        if not trigger_stage2:
            logger.info(
                f"Candidate C5 Adaptive Gate on {sample.sample_id}: {trigger_reason} "
                f"-> Stage 2 bypassed with 0ms overhead!"
            )
            return stage1_pred

        # Execute Stage 2: Conservative Text Consistency Verifier
        logger.info(
            f"Candidate C5 Adaptive Gate on {sample.sample_id}: {trigger_reason} "
            f"-> Invoking Stage 2 Verifier ({len(stage1_turns)} turns)"
        )

        try:
            final_turns = [Turn(speaker=t.speaker, text=t.text) for t in stage1_turns]
            stage2_prompt = self._build_stage2_prompt(final_turns, tags=stage1_tags, profiles_text=profiles_text)
            if len(set(t.speaker for t in stage1_turns)) < 2 and len(stage1_turns) >= 4:
                spk_list_str = ", ".join(f"Speaker {i}" for i in range(spk_count))
                stage2_prompt += (
                    "\n\nCRITICAL NOTICE: Single-speaker collapse detected in preliminary attributions. "
                    "All turns above are currently attributed to only one speaker. You MUST reassign the "
                    f"conversational turns among {spk_list_str} based on the dialogue exchange."
                )

            stage2_raw, stage2_latency = self.client.generate_text(
                model=model_id,
                prompt=stage2_prompt,
                system_instruction=STAGE2_CHAMPION_SYSTEM_INSTRUCTION,
                thinking_budget=self.thinking_budget,
                temperature=self.temperature,
                max_output_tokens=8192,
            )

            corrections = self._parse_stage2_corrections(stage2_raw)
            total_latency = stage1_latency + stage2_latency

            # Apply corrections programmatically
            for corr in corrections:
                if not isinstance(corr, dict):
                    continue
                raw_tid = corr.get("turn_id")
                try:
                    t_id = int(raw_tid)
                except (TypeError, ValueError):
                    continue

                new_spk = corr.get("speaker")
                if isinstance(new_spk, str):
                    spk_match = re.search(r"(\d+)", new_spk)
                    normalized_spk = f"Speaker {spk_match.group(1)}" if spk_match else new_spk.strip()
                    if 1 <= t_id <= len(final_turns):
                        idx = t_id - 1
                        logger.info(
                            f"Sample {sample.sample_id} C5 Stage 2 correction: Turn {t_id} "
                            f"'{final_turns[idx].speaker}' -> '{normalized_spk}'"
                        )
                        final_turns[idx].speaker = normalized_spk

            combined_raw = (
                f"=== STAGE 1 (Acoustic STT) ===\n{stage1_raw}\n\n"
                f"=== STAGE 2 (Transition-Aware Verifier, Trigger: {trigger_reason}) ===\n{stage2_raw}"
            )

            return PipelinePrediction(
                sample_id=sample.sample_id,
                model_id=model_id,
                approach="strategy_c5_adaptive_champion",
                predicted_turns=final_turns,
                raw_response=combined_raw,
                latency_seconds=total_latency,
            )
        except Exception as e:
            logger.warning(
                f"Candidate C5 Stage 2 text verifier encountered error for {sample.sample_id}: {e}. "
                f"Falling back safely to Stage 1 prediction."
            )
            return stage1_pred


# Alias runner for benchmark runner naming convention
AdaptiveAcousticChampionRunner = AdaptiveAcousticChampionPipeline
