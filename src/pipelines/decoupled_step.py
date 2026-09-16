"""Strategy 2: Two-Step Decoupled Architecture for Multilingual STT & Speaker Diarization.

Step 1: Acoustic STT Turn Extraction (Audio -> Unattributed turns 'Turn 1: ...', 'Turn 2: ...')
        Eliminates token-0 commitment dilemma and acoustic capacity bottlenecks.
Step 2: Global Context Role & Speaker Attribution (Text turns -> Diarized 'Speaker 0: ...')
        Global retrospective reasoning over the entire dialogue resolves parity cascades
        and assigns consistent speaker identities using conversation anchors.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.client import GeminiClient
from src.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH,
    MODEL_3_5_FLASH_LITE,
)
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.single_step import SingleStepPipeline

logger = logging.getLogger(__name__)

DEFAULT_STEP1_SYSTEM_INSTRUCTION = """You are an expert multilingual speech-to-text recognition system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and segment it into sequential acoustic speech turns.

Crucial Instructions:
1. Segment the dialogue into distinct speech turns whenever there is an acoustic pause, breath, speaker transition, or interruption.
2. When speakers talk simultaneously or interrupt, split the overlapping speech into separate turns based on acoustic flow.
3. Do NOT guess speaker identities, names, or roles.
4. Label each turn strictly as 'Turn <N>:' starting from 1.
5. Transcribe strictly in Devanagari script verbatim. Capture all words, including short acknowledgments, affirmations, and backchannels (e.g., हाँ, जी, अच्छा, ठीक है).
6. Output each turn on a separate line.

Format:
Turn 1: <spoken words>
Turn 2: <spoken words>
Turn 3: <spoken words>

Rules:
- Do not output speaker names (no 'Speaker 0', no 'Agent', etc.).
- Output ONLY the numbered turns. No timestamps or explanations.
"""

DEFAULT_STEP1_USER_PROMPT = (
    "Transcribe the entire audio clip verbatim and segment into sequential acoustic turns "
    "labeled strictly as Turn 1, Turn 2, etc. without attributing speaker identities."
)

DEFAULT_STEP2_SYSTEM_INSTRUCTION = """You are an expert dialogue analyst and speaker diarization engine.
You are provided with a sequence of verbatim acoustic speech turns extracted from a Hindi multi-speaker dialogue (Turn 1, Turn 2, ...).
Your task is to analyze the complete conversational flow retrospectively and attribute each turn to its true speaker (Speaker 0, Speaker 1, and Speaker 2 if present).

Conversational Anchor & Attribution Principles:
1. Conversational Roles & Semantic Anchors:
   - Call Initiator / Agent / Caller: Offers formal greetings (e.g. नमस्कार, प्रणाम, हेलो), introduces themselves or their institution, asks to confirm the identity of the interlocutor ("क्या मेरी बात ... जी से हो रही है?"), inquires about accounts, dues, schedules, or procedures.
   - Call Recipient / Customer: Confirms or denies identity ("हाँ बोल रहा हूँ", "कौन बोल रहा है?"), reacts, provides reasons/excuses, agrees, objects, or asks questions.
   - Third Speaker (if present): Interjects with background remarks, second opinion, or distinct viewpoint.
2. Global Conversational Coherence & Q&A Polarity:
   - A question, demand, or proposal posed by Speaker A is typically answered or acknowledged by Speaker B.
   - Retain speaker consistency across the entire conversation: do NOT flip speaker identities mid-dialogue.
3. Consecutive Turn Continuity & Anti-Alternation:
   - If consecutive turns (e.g. Turn N and Turn N+1) represent the continuation of the same speaker's thought across a pause or breath, attribute BOTH turns to the SAME speaker.
4. Backchannels & Reactions:
   - Short listener affirmations ("हाँ", "जी", "अच्छा") belong to the listening interlocutor reacting to the active speaker.
5. Verbatim Text Fidelity:
   - Preserve the EXACT Hindi Devanagari words and phrasing of each turn without modification, summarization, translation, or omission.

Output Format:
Output each attributed turn on a separate line in sequential order strictly adhering to:
Speaker <ID>: <exact spoken transcript of the turn>

Example:
Speaker 0: नमस्कार, क्या मेरी बात मोहन जी से हो रही है?
Speaker 1: हाँ, बोलिए कौन?
Speaker 0: मैं बैंक शाखा से बात कर रहा हूँ।
Speaker 0: आपकी किस्त की तारीख निकल चुकी है।
Speaker 1: जी, मुझे पता है। कल जमा करवा दूँगा।
"""

DEFAULT_STEP2_USER_PROMPT_TEMPLATE = """Here is the sequential un-attributed dialogue extracted from the audio:
{extracted_turns}

Analyze the full conversational context and attribute each turn to Speaker 0, Speaker 1, or Speaker 2. Output each turn on a separate line as 'Speaker <ID>: <text>'."""


class TwoStepDecoupledPipeline:
    """Strategy 2: Two-Step Decoupled Pipeline (Acoustic Turn STT + LLM Role Attribution)."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        step1_model_id: str = MODEL_3_5_FLASH_LITE,
        step2_model_id: str = MODEL_3_5_FLASH,
        step1_system_instruction: str = DEFAULT_STEP1_SYSTEM_INSTRUCTION,
        step2_system_instruction: str = DEFAULT_STEP2_SYSTEM_INSTRUCTION,
        step1_user_prompt: str = DEFAULT_STEP1_USER_PROMPT,
        step1_thinking_budget: int = DEFAULT_THINKING_BUDGET,
        step2_thinking_budget: int = DEFAULT_THINKING_BUDGET,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """Initializes the TwoStepDecoupledPipeline.

        Args:
            client: Optional GeminiClient instance.
            step1_model_id: Model used for Step 1 acoustic transcription (default: gemini-3.5-flash-lite).
            step2_model_id: Model used for Step 2 text role attribution (default: gemini-3.5-flash).
            step1_system_instruction: Step 1 system instruction for acoustic turn extraction.
            step2_system_instruction: Step 2 system instruction for global role attribution.
            step1_user_prompt: User prompt for Step 1.
            step1_thinking_budget: Thinking budget for Step 1 (default: 0).
            step2_thinking_budget: Thinking budget for Step 2 (default: 0).
            temperature: Sampling temperature (default: 0.0).
        """
        self.client = client or GeminiClient()
        self.step1_model_id = step1_model_id
        self.step2_model_id = step2_model_id
        self.step1_system_instruction = step1_system_instruction
        self.step2_system_instruction = step2_system_instruction
        self.step1_user_prompt = step1_user_prompt
        self.step1_thinking_budget = step1_thinking_budget
        self.step2_thinking_budget = step2_thinking_budget
        self.temperature = temperature

    @classmethod
    def parse_turns(cls, raw_text: str) -> List[Turn]:
        """Parses final speaker-attributed dialogue into Turn objects."""
        return SingleStepPipeline.parse_turns(raw_text)

    @classmethod
    def parse_unattributed_turns(cls, raw_text: str) -> List[str]:
        """Parses Step 1 output into a list of un-attributed turn texts."""
        if not raw_text or not raw_text.strip():
            return []
        lines = raw_text.strip().split("\n")
        turns = []
        turn_prefix_re = re.compile(r"^(?:[\*\#\-\s\[\(]*)(?:Turn|turn|\u091a\u0930\u0923)\s*\d+[:\-\u2013]?\s*(.*)$", re.IGNORECASE)
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

    def run_step1(
        self,
        audio_source: Union[str, Path, bytes],
        model_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Executes Step 1: Acoustic STT turn extraction from audio.

        Args:
            audio_source: File path or raw bytes of the audio clip.
            model_id: Model to use (defaults to self.step1_model_id).

        Returns:
            Tuple of (raw step 1 turns string, step 1 latency in seconds).
        """
        target_model = model_id or self.step1_model_id
        logger.info(f"Executing Step 1 Acoustic STT with model {target_model}")
        return self.client.generate_with_audio(
            model=target_model,
            audio_source=audio_source,
            prompt=self.step1_user_prompt,
            system_instruction=self.step1_system_instruction,
            thinking_budget=self.step1_thinking_budget,
            temperature=self.temperature,
        )

    def run_step2(
        self,
        step1_text: str,
        model_id: Optional[str] = None,
        num_speakers_hint: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Executes Step 2: Global semantic role & speaker attribution on extracted turns.

        Args:
            step1_text: Unattributed turns text from Step 1.
            model_id: Model to use (defaults to self.step2_model_id).
            num_speakers_hint: Optional hint about total number of speakers in dialogue.

        Returns:
            Tuple of (raw step 2 attributed text string, step 2 latency in seconds).
        """
        target_model = model_id or self.step2_model_id
        logger.info(f"Executing Step 2 Global Role Attribution with model {target_model}")

        prompt = DEFAULT_STEP2_USER_PROMPT_TEMPLATE.format(extracted_turns=step1_text)
        if num_speakers_hint:
            prompt += f"\nNote: This conversation contains approximately {num_speakers_hint} distinct speakers."

        return self.client.generate_text(
            model=target_model,
            prompt=prompt,
            system_instruction=self.step2_system_instruction,
            thinking_budget=self.step2_thinking_budget,
            temperature=self.temperature,
        )

    def run_sample(
        self,
        sample: SampleData,
        step1_model_id: Optional[str] = None,
        step2_model_id: Optional[str] = None,
    ) -> PipelinePrediction:
        """Executes the complete Two-Step Decoupled Pipeline on a benchmark sample.

        Args:
            sample: SampleData benchmark instance.
            step1_model_id: Optional override for Step 1 model.
            step2_model_id: Optional override for Step 2 model.

        Returns:
            PipelinePrediction with attributed turns, dual-step raw response, and total latency.
        """
        m1 = step1_model_id or self.step1_model_id
        m2 = step2_model_id or self.step2_model_id
        logger.info(
            f"Running TwoStepDecoupledPipeline on sample {sample.sample_id} "
            f"(Step 1: {m1}, Step 2: {m2})"
        )

        # Step 1: Acoustic turn extraction
        raw_step1, lat1 = self.run_step1(sample.audio_path, model_id=m1)

        # Step 2: Global semantic role attribution
        raw_step2, lat2 = self.run_step2(
            step1_text=raw_step1,
            model_id=m2,
            num_speakers_hint=sample.num_speakers,
        )

        turns = self.parse_turns(raw_step2)
        total_latency = lat1 + lat2

        combined_raw = (
            f"=== STEP 1: ACOUSTIC TURNS ({m1}, {lat1:.2f}s) ===\n"
            f"{raw_step1}\n\n"
            f"=== STEP 2: ATTRIBUTED DIALOGUE ({m2}, {lat2:.2f}s) ===\n"
            f"{raw_step2}"
        )

        return PipelinePrediction(
            sample_id=sample.sample_id,
            model_id=f"{m1}+{m2}",
            approach="two_step_decoupled",
            predicted_turns=turns,
            raw_response=combined_raw,
            latency_seconds=total_latency,
        )
