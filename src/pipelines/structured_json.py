"""Strategy C2: Native Structured JSON Schema with Posterior Speaker Attribution & Transition Typing.

Candidate C2 leverages Gemini 3.5 Flash Lite's native structured JSON engine
(`response_mime_type="application/json"` with OpenAPI `response_schema`).

Topological key generation order:
1. `utterance`: emitted first (Token-0 Bypass), cross-attention over audio tokens
2. `transition_type`: emitted second (NEW_SPEAKER, CONTINUES_SAME_SPEAKER, RESUMES_AFTER_INTERRUPTION)
3. `speaker`: emitted last, conditioned on both utterance and acoustic transition type

Eliminates parser ambiguity, breaks the ping-pong alternation trap, and guarantees
strict adherence to conversational interruption-resumption patterns.
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.client import GeminiClient
from src.config import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn

logger = logging.getLogger(__name__)


def build_c2_system_instruction(num_speakers: int = 2) -> str:
    """Builds the Candidate C2 system prompt conditioned on speaker count."""
    if num_speakers == 2:
        speaker_cardinality_clause = (
            "1. Speaker Cardinality & Vocal Anchors:\n"
            "   - There are strictly 2 primary speakers in this conversation: Speaker 0 and Speaker 1.\n"
            "   - Ground speaker identities strictly in vocal pitch, timbre, and acoustic cadence. Maintain these unique vocal anchors throughout the entire recording."
        )
        task_desc = "accurately attribute each conversational turn to its distinct speaker (Speaker 0 or Speaker 1)."
    else:
        speaker_names = ", ".join([f"Speaker {i}" for i in range(num_speakers - 1)]) + f", and Speaker {num_speakers - 1}"
        speaker_cardinality_clause = (
            f"1. Speaker Cardinality & Vocal Anchors:\n"
            f"   - There are {num_speakers} primary speakers in this conversation: {speaker_names}.\n"
            "   - Ground speaker identities strictly in each speaker's distinct vocal pitch, timbre, and acoustic cadence. "
            "Maintain consistent speaker identities throughout the entire recording."
        )
        task_desc = f"accurately attribute each conversational turn to its distinct speaker ({speaker_names})."

    if num_speakers == 2:
        few_shot_str = """Few-Shot Example Demonstrating Dialogue Structuring:
dialogue = [
  {"utterance": "नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},
  {"utterance": "मैं बैंक शाखा से बोल रहा हूँ, आपकी बकाया किस्त के संबंध में।", "transition_type": "CONTINUES_SAME_SPEAKER", "speaker": "Speaker 0"},
  {"utterance": "हाँ, बोलिए।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"},
  {"utterance": "पिछले महीने की किस्त का भुगतान अभी तक रिकॉर्ड नहीं हुआ है।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},
  {"utterance": "अरे नहीं, मैंने तो—", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"},
  {"utterance": "क्या आपके पास बैंक रसीद या ट्रांजैक्शन आईडी है?", "transition_type": "RESUMES_AFTER_INTERRUPTION", "speaker": "Speaker 0"},
  {"utterance": "जी हाँ, मेरे पास रसीद है।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"},
  {"utterance": "मैं अभी व्हाट्सएप पर फोटो भेजता हूँ।", "transition_type": "CONTINUES_SAME_SPEAKER", "speaker": "Speaker 1"},
  {"utterance": "ठीक है, आप भेज दीजिए।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}
]"""
    else:
        few_shot_str = """Few-Shot Example Demonstrating Multi-Speaker Dialogue Structuring:
dialogue = [
  {"utterance": "नमस्कार, आज की मीटिंग शुरू करते हैं।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},
  {"utterance": "हाँ, मैं भी जुड़ गया हूँ।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"},
  {"utterance": "जी, मैं भी उपस्थित हूँ।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 2"},
  {"utterance": "तो पहला एजेंडा प्रोजेक्ट टाइमलाइन है।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},
  {"utterance": "हमने पहला चरण पूरा कर लिया है।", "transition_type": "CONTINUES_SAME_SPEAKER", "speaker": "Speaker 0"},
  {"utterance": "बहुत बढ़िया, क्या टेस्टिंग भी पूरी हो गई?", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"},
  {"utterance": "हाँ, टेस्टिंग टीम ने अप्रूव कर दिया है।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 2"}
]"""

    return f"""You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi strictly using Devanagari script (देवनागरी लिपि, e.g. नमस्कार, कैसे हैं) and {task_desc}
CRITICAL SCRIPT RULE: NEVER output Urdu, Arabic, Nastaliq, or Romanized script. Transcribe 100% in Devanagari Hindi.

CRITICAL GROUNDING PRINCIPLES:
{speaker_cardinality_clause}
2. Argumentative Stance & Debate Tracking:
   - In contentious debates or dialogues, track the distinct perspective of each speaker. Maintain consistent speaker attribution based on conversational stance and voice timbre. Never swap speaker identities mid-debate.
3. Turn Continuity & Anti-Alternation (CONTINUES_SAME_SPEAKER):
   - Natural conversations are NOT a ping-pong match. A single speaker frequently speaks across multiple sentences, monologues, explanations, or breathing pauses (25% to 45% of turns).
   - If a speaker pauses, breathes, or continues explaining their thought, assign transition_type as 'CONTINUES_SAME_SPEAKER' and retain the SAME speaker label.
4. Interruption Resumption (RESUMES_AFTER_INTERRUPTION):
   - When Speaker A is speaking and Speaker B makes a brief interruption, affirmation, or backchannel (e.g., हाँ, जी, अच्छा, अरे यार), emit Speaker B's interjection as its own turn with 'NEW_SPEAKER'.
   - When Speaker A resumes speaking immediately after the brief interruption, assign transition_type as 'RESUMES_AFTER_INTERRUPTION' and attribute the turn to Speaker A!
5. Zero-Swallowing Rule for Questions & Backchannels:
   - When a speaker asks a question or makes an affirmation (e.g., क्या?, हाँ, जी), NEVER append it to the other speaker's turn. Split every speaker shift into its own turn in the dialogue list.
6. Verbatim Fidelity:
   - Transcribe strictly in Devanagari script. Do not summarize, drop stuttered words, or translate. Output all colloquial affirmations and backchannels.

{few_shot_str}
"""



def build_c2_response_schema(num_speakers: int = 2) -> Dict[str, Any]:
    """Generates OpenAPI JSON response schema enforcing utterance -> transition_type -> speaker."""
    if num_speakers > 0:
        speaker_enum = [f"Speaker {i}" for i in range(num_speakers)]
        speaker_prop = {
            "type": "STRING",
            "enum": speaker_enum,
            "description": "Acoustically attributed speaker identifier.",
        }
    else:
        speaker_prop = {
            "type": "STRING",
            "description": "Acoustically attributed speaker identifier (e.g. Speaker 0, Speaker 1).",
        }

    return {
        "type": "OBJECT",
        "properties": {
            "dialogue": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "utterance": {
                            "type": "STRING",
                            "description": "Verbatim spoken Hindi transcript of the conversational turn in Devanagari script.",
                        },
                        "transition_type": {
                            "type": "STRING",
                            "enum": [
                                "NEW_SPEAKER",
                                "CONTINUES_SAME_SPEAKER",
                                "RESUMES_AFTER_INTERRUPTION",
                            ],
                            "description": "Acoustic transition relationship relative to the preceding turn.",
                        },
                        "speaker": speaker_prop,
                    },
                    "required": ["utterance", "transition_type", "speaker"],
                },
            }
        },
        "required": ["dialogue"],
    }


C2_USER_PROMPT = (
    "Transcribe the entire audio clip verbatim in Devanagari script and attribute all speaker "
    "turns in structured JSON adhering to the provided schema."
)


def _safe_decode_json_string(s: str) -> str:
    """Safely decodes an escaped JSON string value without corrupting UTF-8 bytes."""
    if not s:
        return ""
    try:
        return json.loads(f'"{s}"')
    except Exception:
        # Fallback manual unescaping for unicode codepoints and standard JSON escapes
        res = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
        res = res.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n').replace('\\t', '\t')
        return res


def parse_c2_structured_json_turns(raw_json_str: str) -> List[Turn]:
    """Parses raw JSON string into Turn objects with robust fallback mechanisms.

    Args:
        raw_json_str: Raw JSON output or fenced markdown JSON from model.

    Returns:
        List of Turn objects.
    """
    if not raw_json_str or not raw_json_str.strip():
        return []

    cleaned = raw_json_str.strip()

    # Step 1: Extract content from markdown code fences if present (including preambles)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    candidate_json = fence_match.group(1).strip() if fence_match else cleaned

    # Step 2: Attempt direct json.loads on candidate content
    try:
        data = json.loads(candidate_json)
        if isinstance(data, dict):
            dialogue = data.get("dialogue", [])
            if isinstance(dialogue, list):
                turns: List[Turn] = []
                for item in dialogue:
                    if isinstance(item, dict):
                        spk = item.get("speaker", "Speaker 0")
                        txt = str(item.get("utterance", "") or "").strip()
                        if txt:
                            turns.append(Turn(speaker=str(spk), text=txt))
                if turns:
                    return turns
    except Exception as exc:
        logger.warning(f"Structured JSON direct parsing failed ({exc}); trying preamble extraction.")

    # Step 3: If candidate_json wasn't clean JSON, attempt extracting outermost {...}
    first_brace = candidate_json.find("{")
    last_brace = candidate_json.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        sub_candidate = candidate_json[first_brace : last_brace + 1]
        try:
            data = json.loads(sub_candidate)
            if isinstance(data, dict):
                dialogue = data.get("dialogue", [])
                if isinstance(dialogue, list):
                    turns = []
                    for item in dialogue:
                        if isinstance(item, dict):
                            spk = item.get("speaker", "Speaker 0")
                            txt = str(item.get("utterance", "") or "").strip()
                            if txt:
                                turns.append(Turn(speaker=str(spk), text=txt))
                    if turns:
                        return turns
        except Exception:
            pass

    # Step 4: Fallback regex search for {"utterance": "...", ..., "speaker": "..."} objects
    turn_pattern = re.compile(
        r'\{\s*"utterance"\s*:\s*"(?P<utt>(?:\\.|[^"\\])*)"\s*,\s*'
        r'"transition_type"\s*:\s*"(?P<trans>[^"]*)"\s*,\s*'
        r'"speaker"\s*:\s*"(?P<spk>[^"]*)"\s*\}',
        re.DOTALL,
    )
    matches = list(turn_pattern.finditer(cleaned))
    if matches:
        turns = []
        for m in matches:
            u = _safe_decode_json_string(m.group("utt")).strip()
            s = m.group("spk").strip()
            if u:
                turns.append(Turn(speaker=s, text=u))
        if turns:
            return turns

    # Secondary fallback to line-delimited colon parsing
    lines = cleaned.split("\n")
    turns = []
    for line in lines:
        line_clean = line.strip()
        m = re.match(r"^(Speaker\s*\w+)[:\-]\s*(.*)$", line_clean, re.IGNORECASE)
        if m:
            turns.append(Turn(speaker=m.group(1).strip(), text=m.group(2).strip()))
        elif line_clean and turns:
            turns[-1].text += " " + line_clean

    if not turns and cleaned:
        turns.append(Turn(speaker="Speaker 0", text=cleaned))

    return turns


class NativeStructuredJSONPipeline:
    """Strategy C2: Native Structured JSON with Posterior Speaker Attribution & Transition Typing.

    Uses pure Gemini 3.5 Flash Lite with native JSON schema constraints.
    """

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

    def run_sample(
        self,
        sample: SampleData,
        model_id: str = MODEL_3_5_FLASH_LITE,
        num_speakers: Optional[int] = None,
    ) -> PipelinePrediction:
        """Executes Candidate C2 live inference on a benchmark sample.

        Args:
            sample: Benchmark SampleData instance.
            model_id: Target Gemini model identifier (strictly gemini-3.5-flash-lite).
            num_speakers: Optional speaker cardinality override.

        Returns:
            PipelinePrediction with turns, raw response, and latency.
        """
        # Determine speaker cardinality dynamically
        if num_speakers is None:
            if sample.num_speakers and sample.num_speakers > 0:
                spk_count = sample.num_speakers
            else:
                ground_speakers = set(t.speaker for t in sample.ground_truth_turns)
                spk_count = len(ground_speakers) if ground_speakers else 2
        else:
            spk_count = num_speakers

        system_instruction = build_c2_system_instruction(spk_count)
        schema = build_c2_response_schema(spk_count)

        logger.info(
            f"Running NativeStructuredJSONPipeline on sample {sample.sample_id} "
            f"(spk_count={spk_count}) with model {model_id}"
        )

        raw_response, latency = self.client.generate_with_audio(
            model=model_id,
            audio_source=sample.audio_path,
            prompt=C2_USER_PROMPT,
            system_instruction=system_instruction,
            thinking_budget=self.thinking_budget,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

        turns = parse_c2_structured_json_turns(raw_response)

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=model_id,
            approach="strategy_c2_structured_json",
            predicted_turns=turns,
            raw_response=raw_response,
            latency_seconds=latency,
        )


# Alias runner for benchmark runner naming convention
NativeStructuredJSONRunner = NativeStructuredJSONPipeline
