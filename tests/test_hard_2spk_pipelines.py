"""Unit tests for Hard 2-Speaker High-Overlap Pipelines.

Covers:
1. Token0BypassPipeline (Strategy A):
   - Token-0 bypass syntax parsing: `[Utterance] <text> | [Speaker] Speaker <ID>`
   - Formatting variations: brackets, colons, markdown bolding, bullets, multi-line utterances
   - Fallback parsing for traditional prefixes
   - Pipeline contract, thinking_budget=0, model_id=gemini-3.5-flash-lite
2. PureLiteTwoPassPipeline (Strategy B):
   - Pass 1 unattributed turn parsing: `Turn 1: ...`, `Turn 2: ...`
   - Pass 2 attributed dialogue parsing: `Speaker 0: ...`, `Speaker 1: ...`
   - Turn splitting and anti-alternation parsing fidelity
   - Pure 3.5 Flash Lite guarantee: ZERO larger model dependency
   - Dual-pass latency accumulation and combined raw response formatting
"""

from unittest.mock import MagicMock
import pytest

from src.config import MODEL_3_5_FLASH_LITE
from src.models import SampleData, Turn
from src.pipelines.pure_lite_twopass import (
    PASS1_ACOUSTIC_STT_SYSTEM_INSTRUCTION,
    PASS2_DISENTANGLEMENT_SYSTEM_INSTRUCTION,
    PureLiteTwoPassPipeline,
    parse_attributed_turns,
    parse_unattributed_turns,
)
from src.pipelines.token0_bypass import (
    CANDIDATE_A_SYSTEM_INSTRUCTION,
    Token0BypassPipeline,
    parse_token0_bypass_turns,
)


# =========================================================================
# Tests for Strategy A: Token0BypassPipeline
# =========================================================================

def test_token0_bypass_standard_parsing():
    """Tests parsing canonical Token-0 Bypass line-delimited format."""
    raw_text = (
        "[Utterance] नमस्कार, क्या मेरी बात शर्मा जी से हो रही है? | [Speaker] Speaker 0\n"
        "[Utterance] मैं बैंक शाखा से बोल रहा हूँ। | [Speaker] Speaker 0\n"
        "[Utterance] हाँ, बोलिए कौन? | [Speaker] Speaker 1\n"
        "[Utterance] आपकी बकाया किस्त के संबंध में जानकारी चाहिए थी। | [Speaker] Speaker 0"
    )
    turns = parse_token0_bypass_turns(raw_text)
    assert len(turns) == 4

    assert turns[0].speaker == "Speaker 0"
    assert turns[0].text == "नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?"

    # Verifies consecutive turns by same speaker (Anti-Alternation preservation)
    assert turns[1].speaker == "Speaker 0"
    assert turns[1].text == "मैं बैंक शाखा से बोल रहा हूँ।"

    assert turns[2].speaker == "Speaker 1"
    assert turns[2].text == "हाँ, बोलिए कौन?"

    assert turns[3].speaker == "Speaker 0"
    assert turns[3].text == "आपकी बकाया किस्त के संबंध में जानकारी चाहिए थी।"


def test_token0_bypass_formatting_variations():
    """Tests parsing with markdown bolding, colons, bullets, and bracket variations."""
    raw_text = (
        "- **[Utterance]**: हाँ जी, बोलिए। | **[Speaker]**: Speaker 1\n"
        "1. Utterance: मैं सुन रहा हूँ। | Speaker: Speaker 0\n"
        "[Utterance] ठीक है, बाद में मिलते हैं। | Speaker 1\n"
        "ज़रूर, अपना ख्याल रखिएगा। | [Speaker] Speaker 0"
    )
    turns = parse_token0_bypass_turns(raw_text)
    assert len(turns) == 4

    assert turns[0].speaker == "Speaker 1"
    assert turns[0].text == "हाँ जी, बोलिए।"

    assert turns[1].speaker == "Speaker 0"
    assert turns[1].text == "मैं सुन रहा हूँ।"

    assert turns[2].speaker == "Speaker 1"
    assert turns[2].text == "ठीक है, बाद में मिलते हैं।"

    assert turns[3].speaker == "Speaker 0"
    assert turns[3].text == "ज़रूर, अपना ख्याल रखिएगा।"


def test_token0_bypass_multiline_buffering():
    """Tests handling of utterances split across multiple lines before the pipe delimiter."""
    raw_text = (
        "[Utterance] यह वाक्य बहुत लम्बा है\n"
        "और यह दूसरी पंक्ति में जारी रहता है। | [Speaker] Speaker 0\n"
        "[Utterance] समझ गया सर। | [Speaker] Speaker 1"
    )
    turns = parse_token0_bypass_turns(raw_text)
    assert len(turns) == 2
    assert turns[0].speaker == "Speaker 0"
    assert "यह वाक्य बहुत लम्बा है" in turns[0].text
    assert "और यह दूसरी पंक्ति में जारी रहता है।" in turns[0].text
    assert turns[1].speaker == "Speaker 1"
    assert turns[1].text == "समझ गया सर।"


def test_token0_bypass_fallback_prefix():
    """Tests fallback when the model emits traditional Speaker <ID>: prefixes."""
    raw_text = (
        "Speaker 0: नमस्ते, कैसे हैं आप?\n"
        "Speaker 1: मैं ठीक हूँ, आप बताइए।"
    )
    turns = parse_token0_bypass_turns(raw_text)
    assert len(turns) == 2
    assert turns[0].speaker == "Speaker 0"
    assert turns[0].text == "नमस्ते, कैसे हैं आप?"
    assert turns[1].speaker == "Speaker 1"
    assert turns[1].text == "मैं ठीक हूँ, आप बताइए।"


def test_token0_bypass_empty_and_whitespace():
    """Tests that empty or whitespace strings return an empty list."""
    assert parse_token0_bypass_turns("") == []
    assert parse_token0_bypass_turns("   \n\n  \t  ") == []


def test_token0_bypass_pipeline_execution():
    """Tests Token0BypassPipeline run_sample with mocked client."""
    mock_client = MagicMock()
    mock_response = (
        "[Utterance] नमस्ते | [Speaker] Speaker 0\n"
        "[Utterance] नमस्कार जी | [Speaker] Speaker 1"
    )
    mock_client.generate_with_audio.return_value = (mock_response, 3.42)

    pipeline = Token0BypassPipeline(client=mock_client)
    assert pipeline.thinking_budget == 0
    assert pipeline.temperature == 0.0
    assert "[Utterance]" in pipeline.system_instruction

    sample = SampleData(
        sample_id="hindi_064",
        audio_path="data/hard_2spk_subset/audio/hindi_064.wav",
        duration_seconds=60.2,
        num_speakers=2,
        overlap_ratio=14.33,
        ground_truth_turns=[Turn(speaker="Speaker 0", text="नमस्ते")],
    )

    pred = pipeline.run_sample(sample, model_id=MODEL_3_5_FLASH_LITE)
    assert pred.sample_id == "hindi_064"
    assert pred.model_id == MODEL_3_5_FLASH_LITE
    assert pred.approach == "token0_bypass"
    assert pred.latency_seconds == 3.42
    assert len(pred.predicted_turns) == 2
    assert pred.predicted_turns[0].speaker == "Speaker 0"
    assert pred.predicted_turns[1].speaker == "Speaker 1"

    # Verify client call parameters
    mock_client.generate_with_audio.assert_called_once_with(
        model=MODEL_3_5_FLASH_LITE,
        audio_source=sample.audio_path,
        prompt=pipeline.user_prompt,
        system_instruction=pipeline.system_instruction,
        thinking_budget=0,
        temperature=0.0,
        max_output_tokens=pipeline.max_output_tokens,
    )


# =========================================================================
# Tests for Strategy B: PureLiteTwoPassPipeline
# =========================================================================

def test_pure_lite_pass1_unattributed_parsing():
    """Tests parsing Pass 1 numbered acoustic turns."""
    raw_pass1 = (
        "Turn 1: नमस्ते सर, क्या हाल है?\n"
        "Turn 2: सब ठीक है, आप बताइए।\n"
        "Turn 3: मैं भी ठीक हूँ।"
    )
    unattributed = parse_unattributed_turns(raw_pass1)
    assert len(unattributed) == 3
    assert unattributed[0] == "नमस्ते सर, क्या हाल है?"
    assert unattributed[1] == "सब ठीक है, आप बताइए।"
    assert unattributed[2] == "मैं भी ठीक हूँ।"


def test_pure_lite_pass2_attributed_parsing():
    """Tests parsing Pass 2 speaker-attributed dialogue."""
    raw_pass2 = (
        "Speaker 0: नमस्ते सर, क्या हाल है?\n"
        "Speaker 1: सब ठीक है, आप बताइए।\n"
        "Speaker 0: मैं भी ठीक हूँ।"
    )
    turns = parse_attributed_turns(raw_pass2)
    assert len(turns) == 3
    assert turns[0].speaker == "Speaker 0"
    assert turns[0].text == "नमस्ते सर, क्या हाल है?"
    assert turns[1].speaker == "Speaker 1"
    assert turns[1].text == "सब ठीक है, आप बताइए।"
    assert turns[2].speaker == "Speaker 0"
    assert turns[2].text == "मैं भी ठीक हूँ।"


def test_pure_lite_turn_splitting_parsing():
    """Tests that split turns from Pass 2 disentanglement are parsed correctly."""
    # Pass 1 had a merged turn; Pass 2 split it into two turns
    raw_pass2 = (
        "Speaker 0: काम की चीज़ है। परिवार से जुड़े रह सकते हैं।\n"
        "Speaker 1: अरे यार, कोई काम की चीज़ नहीं।\n"
        "Speaker 0: अरे, सब पढ़ाई करते हैं। कैसे नहीं करते?\n"
        "Speaker 1: पढ़ाई के अलावा ये सब करते हैं।"
    )
    turns = PureLiteTwoPassPipeline.parse_turns(raw_pass2)
    assert len(turns) == 4
    assert turns[0].speaker == "Speaker 0"
    assert turns[1].speaker == "Speaker 1"
    assert turns[2].speaker == "Speaker 0"
    assert turns[3].speaker == "Speaker 1"


def test_pure_lite_zero_larger_model_constraint():
    """Verifies that PureLiteTwoPassPipeline uses ONLY gemini-3.5-flash-lite."""
    pipeline = PureLiteTwoPassPipeline()
    assert pipeline.model_id == MODEL_3_5_FLASH_LITE
    assert "2.5" not in pipeline.model_id
    assert "pro" not in pipeline.model_id.lower()
    # Ensure neither Pass 1 nor Pass 2 references larger models
    assert pipeline.model_id == "gemini-3.5-flash-lite"
    assert pipeline.thinking_budget == 0


def test_pure_lite_twopass_pipeline_execution():
    """Tests PureLiteTwoPassPipeline run_sample with mocked client."""
    mock_client = MagicMock()
    pass1_output = (
        "Turn 1: नमस्ते\n"
        "Turn 2: नमस्कार जी"
    )
    pass2_output = (
        "Speaker 0: नमस्ते\n"
        "Speaker 1: नमस्कार जी"
    )
    mock_client.generate_with_audio.return_value = (pass1_output, 3.10)
    mock_client.generate_text.return_value = (pass2_output, 1.50)

    pipeline = PureLiteTwoPassPipeline(client=mock_client)

    sample = SampleData(
        sample_id="hindi_064",
        audio_path="data/hard_2spk_subset/audio/hindi_064.wav",
        duration_seconds=60.2,
        num_speakers=2,
        overlap_ratio=14.33,
        ground_truth_turns=[Turn(speaker="Speaker 0", text="नमस्ते")],
    )

    pred = pipeline.run_sample(sample)
    assert pred.sample_id == "hindi_064"
    assert pred.model_id == f"{MODEL_3_5_FLASH_LITE}+{MODEL_3_5_FLASH_LITE}"
    assert pred.approach == "pure_lite_twopass"
    assert round(pred.latency_seconds, 2) == 4.60
    assert len(pred.predicted_turns) == 2
    assert pred.predicted_turns[0].speaker == "Speaker 0"
    assert pred.predicted_turns[1].speaker == "Speaker 1"

    # Verify both calls used MODEL_3_5_FLASH_LITE
    mock_client.generate_with_audio.assert_called_once_with(
        model=MODEL_3_5_FLASH_LITE,
        audio_source=sample.audio_path,
        prompt=pipeline.pass1_user_prompt,
        system_instruction=pipeline.pass1_system_instruction,
        thinking_budget=0,
        temperature=0.0,
        max_output_tokens=pipeline.max_output_tokens,
    )
    assert mock_client.generate_text.call_count == 1
    call_args = mock_client.generate_text.call_args[1]
    assert call_args["model"] == MODEL_3_5_FLASH_LITE
    assert call_args["thinking_budget"] == 0
