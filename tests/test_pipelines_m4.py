"""Unit and integration tests for Milestone 4 Creative Iteration Pipelines.

Tests:
- Candidate C3 (`src/pipelines/acoustic_token0.py`):
  - System prompt generation (2 speakers and dynamic N speakers)
  - Parser robustness across tri-delimited, fallback, and noisy preamble formats
  - Preamble stripping integrity
- Candidate C4 (`src/pipelines/acoustic_multistage.py`):
  - Numbered turn prompt generation
  - JSON diff parser robustness
  - Text-preserving correction application
"""

import pytest

from src.models import Turn
from src.pipelines.acoustic_multistage import (
    STAGE2_SYSTEM_INSTRUCTION,
    AcousticMultiStagePipeline,
)
from src.pipelines.acoustic_token0 import (
    AcousticAnchorToken0Pipeline,
    build_c3_system_instruction,
    parse_acoustic_token0_turns,
)


class TestAcousticToken0PromptConstruction:
    """Tests system prompt construction for Candidate C3."""

    def test_2_speaker_prompt_construction(self):
        prompt = build_c3_system_instruction(num_speakers=2)
        assert "[Speaker Profiles]" in prompt
        assert "Speaker 0" in prompt
        assert "Speaker 1" in prompt
        assert "CONTINUE" in prompt
        assert "SHIFT" in prompt
        assert "RESUME" in prompt
        assert "OVERLAP" in prompt
        assert "[Utterance]" in prompt
        assert "[Transition]" in prompt
        assert "[Speaker]" in prompt
        # Verify 11-turn few-shot is present
        assert "नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?" in prompt
        assert "तुरंत वेरिफिकेशन करके सिस्टम में अपडेट कर दूँगा।" in prompt

    def test_dynamic_n_speaker_prompt_construction(self):
        prompt_4spk = build_c3_system_instruction(num_speakers=4)
        assert "[Speaker Profiles]" in prompt_4spk
        assert "Speaker 0" in prompt_4spk
        assert "Speaker 1" in prompt_4spk
        assert "Speaker 2" in prompt_4spk
        assert "Speaker 3" in prompt_4spk
        assert "4 primary speakers" in prompt_4spk


class TestAcousticToken0Parser:
    """Tests parsing engine for Candidate C3."""

    def test_parse_standard_tri_delimited_with_preamble(self):
        raw = """[Speaker Profiles]
- Speaker 0: Male, deep voice, calm cadence, bank manager.
- Speaker 1: Female, bright timbre, rapid cadence, customer.

[Utterance] नमस्कार, मैं बैंक से बात कर रहा हूँ। | [Transition] SHIFT | [Speaker] Speaker 0
[Utterance] क्या आपने किस्त जमा की है? | [Transition] CONTINUE | [Speaker] Speaker 0
[Utterance] जी, मैंने कल ही कर दी थी। | [Transition] SHIFT | [Speaker] Speaker 1
[Utterance] अरे हाँ, अब दिख रहा है। | [Transition] RESUME | [Speaker] Speaker 0
"""
        turns = parse_acoustic_token0_turns(raw)
        assert len(turns) == 4
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्कार, मैं बैंक से बात कर रहा हूँ।"
        assert turns[1].speaker == "Speaker 0"
        assert turns[1].text == "क्या आपने किस्त जमा की है?"
        assert turns[2].speaker == "Speaker 1"
        assert turns[2].text == "जी, मैंने कल ही कर दी थी।"
        assert turns[3].speaker == "Speaker 0"
        assert turns[3].text == "अरे हाँ, अब दिख रहा है।"

    def test_parse_strips_various_preamble_formats(self):
        raw = """Here is the transcript:
[Speaker Profiles]
* Speaker 0: Lower pitch voice, male.
* Speaker 1: Higher pitch voice, female.

1. [Utterance] हेलो, कौन बोल रहा है? | [Transition] SHIFT | [Speaker] Speaker 0
2. [Utterance] मैं बोल रही हूँ। | [Transition] SHIFT | [Speaker] Speaker 1
"""
        turns = parse_acoustic_token0_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "हेलो, कौन बोल रहा है?"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "मैं बोल रही हूँ।"

    def test_parse_two_field_fallback(self):
        raw = """[Utterance] पहला वाक्य | [Speaker] Speaker 0
[Utterance] दूसरा वाक्य | [Speaker] Speaker 1
"""
        turns = parse_acoustic_token0_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "पहला वाक्य"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "दूसरा वाक्य"

    def test_parse_prefix_fallback(self):
        raw = """Speaker 0: पहला वाक्य
Speaker 1: दूसरा वाक्य
"""
        turns = parse_acoustic_token0_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "पहला वाक्य"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "दूसरा वाक्य"

    def test_parse_empty_or_whitespace(self):
        assert parse_acoustic_token0_turns("") == []
        assert parse_acoustic_token0_turns("   \n\n  ") == []

    def test_multiline_continuation(self):
        raw = """[Utterance] यह एक बहुत लंबा वाक्य है | [Transition] CONTINUE | [Speaker] Speaker 0
जो अगली लाइन पर भी जारी रहता है।
[Utterance] अगला वाक्य | [Transition] SHIFT | [Speaker] Speaker 1
"""
        turns = parse_acoustic_token0_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "जो अगली लाइन पर भी जारी रहता है।" in turns[0].text


class TestAcousticMultiStagePipeline:
    """Tests Stage 2 consistency verifier logic in Candidate C4."""

    def test_build_stage2_prompt(self):
        pipeline = AcousticMultiStagePipeline.__new__(AcousticMultiStagePipeline)
        turns = [
            Turn(speaker="Speaker 0", text="नमस्ते"),
            Turn(speaker="Speaker 1", text="हाँ, कहिए"),
        ]
        prompt = pipeline._build_stage2_prompt(turns)
        assert "[Turn 1] Speaker 0: नमस्ते" in prompt
        assert "[Turn 2] Speaker 1: हाँ, कहिए" in prompt

    def test_parse_stage2_corrections_valid(self):
        pipeline = AcousticMultiStagePipeline.__new__(AcousticMultiStagePipeline)
        raw_json = '{"corrections": [{"turn_id": 2, "speaker": "Speaker 0"}]}'
        corrections = pipeline._parse_stage2_corrections(raw_json)
        assert len(corrections) == 1
        assert corrections[0]["turn_id"] == 2
        assert corrections[0]["speaker"] == "Speaker 0"

    def test_parse_stage2_corrections_markdown_wrapped(self):
        pipeline = AcousticMultiStagePipeline.__new__(AcousticMultiStagePipeline)
        raw = """```json
{
  "corrections": [
    {"turn_id": 3, "speaker": "Speaker 1"}
  ]
}
```"""
        corrections = pipeline._parse_stage2_corrections(raw)
        assert len(corrections) == 1
        assert corrections[0]["turn_id"] == 3
        assert corrections[0]["speaker"] == "Speaker 1"

    def test_parse_stage2_corrections_empty(self):
        pipeline = AcousticMultiStagePipeline.__new__(AcousticMultiStagePipeline)
        raw = '{"corrections": []}'
        corrections = pipeline._parse_stage2_corrections(raw)
        assert corrections == []

    def test_parse_stage2_corrections_invalid_fallback(self):
        pipeline = AcousticMultiStagePipeline.__new__(AcousticMultiStagePipeline)
        assert pipeline._parse_stage2_corrections("Invalid JSON string") == []
        assert pipeline._parse_stage2_corrections("") == []


from src.pipelines.adaptive_acoustic_champion import (
    STAGE2_CHAMPION_SYSTEM_INSTRUCTION,
    AdaptiveAcousticChampionPipeline,
    build_c5_stage1_system_instruction,
    check_debate_context,
    parse_c5_stage1_turns,
    should_trigger_stage2,
)


class TestAdaptiveAcousticChampionPromptConstruction:
    """Tests system prompt construction for Candidate C5."""

    def test_c5_2_speaker_prompt_construction(self):
        prompt = build_c5_stage1_system_instruction(num_speakers=2)
        assert "[Speaker Profiles]" in prompt
        assert "Speaker 0" in prompt
        assert "Speaker 1" in prompt
        assert "Zero-Swallowing & Question-Isolation Directive" in prompt
        assert "sub-second questions are never swallowed" in prompt
        assert "[Utterance]" in prompt
        assert "[Transition]" in prompt
        assert "[Speaker]" in prompt
        assert "CONTINUE" in prompt
        assert "SHIFT" in prompt
        assert "RESUME" in prompt
        assert "OVERLAP" in prompt

    def test_c5_dynamic_n_speaker_prompt_construction(self):
        prompt_3spk = build_c5_stage1_system_instruction(num_speakers=3)
        assert "3 primary speakers" in prompt_3spk
        assert "Speaker 0" in prompt_3spk
        assert "Speaker 1" in prompt_3spk
        assert "Speaker 2" in prompt_3spk


class TestAdaptiveAcousticChampionParser:
    """Tests Stage 1 parser with transition tag extraction for Candidate C5."""

    def test_parse_tri_delimited_with_tags_and_preamble(self):
        raw = """[Speaker Profiles]
- Speaker 0: Female, clear timbre, questioner.
- Speaker 1: Female, bright tone, respondent.

[Utterance] हाँ यार, कचरे वाली गाड़ी आज नहीं आई। | [Transition] SHIFT | [Speaker] Speaker 0
[Utterance] क्यों नहीं आई आज? | [Transition] SHIFT | [Speaker] Speaker 1
[Utterance] पता नहीं, पूछना पड़ेगा। | [Transition] CONTINUE | [Speaker] Speaker 0
[Utterance] अरे हाँ, सच में। | [Transition] RESUME | [Speaker] Speaker 1
"""
        turns, tags, profiles = parse_c5_stage1_turns(raw)
        assert len(turns) == 4
        assert len(tags) == 4
        assert "[Speaker Profiles]" in profiles
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "हाँ यार, कचरे वाली गाड़ी आज नहीं आई।"
        assert tags[0] == "SPEAKER_FLOOR_SHIFT"

        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "क्यों नहीं आई आज?"
        assert tags[1] == "SPEAKER_FLOOR_SHIFT"

        assert turns[2].speaker == "Speaker 0"
        assert tags[2] == "CONTINUES_SAME_SPEAKER"

        assert turns[3].speaker == "Speaker 1"
        assert tags[3] == "RESUMES_AFTER_INTERRUPTION"

    def test_parse_fallback_handling(self):
        raw = """Speaker 0: पहला वाक्य
Speaker 0: दूसरा वाक्य
Speaker 1: तीसरा वाक्य
"""
        turns, tags, profiles = parse_c5_stage1_turns(raw)
        assert len(turns) == 3
        assert turns[0].speaker == "Speaker 0"
        assert tags[0] == "SPEAKER_FLOOR_SHIFT"
        assert turns[1].speaker == "Speaker 0"
        assert tags[1] == "CONTINUES_SAME_SPEAKER"
        assert turns[2].speaker == "Speaker 1"
        assert tags[2] == "SPEAKER_FLOOR_SHIFT"


class TestAdaptiveGatingTrigger:
    """Tests Adaptive Gating Trigger decision logic."""

    def test_long_duration_triggers_stage2(self):
        turns = [Turn(speaker=f"Speaker {i % 2}", text=f"Turn {i}") for i in range(10)]
        tags = ["SPEAKER_FLOOR_SHIFT"] * 10
        # 300.0s clip (e.g. hindi_062, hindi_084)
        trigger, reason = should_trigger_stage2(duration_seconds=300.0, turns=turns, tags=tags)
        assert trigger is True
        assert "70.0s" in reason

    def test_short_casual_dialogue_bypasses_stage2(self):
        # 60s clip with normal alternation and no debate (e.g. hindi_085, hindi_064, hindi_089)
        turns = [
            Turn(speaker="Speaker 0", text="यार आज कितनी शॉपिंग कर ली"),
            Turn(speaker="Speaker 0", text="मेरे पैसे ही नहीं बचे यार बिल्कुल भी"),
            Turn(speaker="Speaker 1", text="अरे सच में? कितना खर्च कर दिया?"),
            Turn(speaker="Speaker 0", text="बहुत सारा"),
            Turn(speaker="Speaker 1", text="अच्छा ठीक है"),
        ]
        tags = [
            "SPEAKER_FLOOR_SHIFT",
            "CONTINUES_SAME_SPEAKER",
            "SPEAKER_FLOOR_SHIFT",
            "SPEAKER_FLOOR_SHIFT",
            "SPEAKER_FLOOR_SHIFT",
        ]
        trigger, reason = should_trigger_stage2(duration_seconds=60.1, turns=turns, tags=tags)
        assert trigger is False
        assert "Bypass Stage 2" in reason

    def test_ping_pong_trap_triggers_stage2(self):
        # 16 alternating turns (alt_ratio = 100%)
        turns = [Turn(speaker=f"Speaker {i % 2}", text=f"Word {i}") for i in range(16)]
        tags = ["SPEAKER_FLOOR_SHIFT"] * 16
        trigger, reason = should_trigger_stage2(duration_seconds=60.2, turns=turns, tags=tags)
        assert trigger is True
        assert "ping-pong trap" in reason

    def test_debate_markers_trigger_stage2(self):
        # 60s clip containing debate vocabulary (e.g. hindi_066)
        turns = [
            Turn(speaker="Speaker 0", text="मोबाइल के बहुत नुकसान हो रहे हैं"),
            Turn(speaker="Speaker 1", text="हाँ सहमत हूँ पर इसके फायदे भी हैं"),
        ]
        tags = ["SPEAKER_FLOOR_SHIFT", "SPEAKER_FLOOR_SHIFT"]
        trigger, reason = should_trigger_stage2(duration_seconds=60.5, turns=turns, tags=tags)
        assert trigger is True
        assert "Debate markers detected" in reason

    def test_check_debate_context_detection(self):
        turns_debate = [
            Turn(speaker="Speaker 0", text="इस मुद्दे पर हमारी बहस है और हम विरोध करते हैं"),
        ]
        has_debate, markers = check_debate_context(turns_debate)
        assert has_debate is True
        assert "बहस" in markers
        assert "विरोध" in markers

        turns_casual = [
            Turn(speaker="Speaker 0", text="चाय पीते हैं और बात करते हैं"),
        ]
        has_debate, markers = check_debate_context(turns_casual)
        assert has_debate is False
        assert markers == []


class TestAdaptiveAcousticChampionStage2:
    """Tests Stage 2 prompt construction and transition tagging for Candidate C5."""

    def test_stage2_prompt_includes_transition_tags(self):
        pipeline = AdaptiveAcousticChampionPipeline.__new__(AdaptiveAcousticChampionPipeline)
        turns = [
            Turn(speaker="Speaker 0", text="नमस्ते"),
            Turn(speaker="Speaker 0", text="मैं बैंक से बोल रहा हूँ"),
            Turn(speaker="Speaker 1", text="हाँ कहिए"),
        ]
        tags = ["SPEAKER_FLOOR_SHIFT", "CONTINUES_SAME_SPEAKER", "SPEAKER_FLOOR_SHIFT"]
        prompt = pipeline._build_stage2_prompt(turns, tags=tags, profiles_text="Speaker Profiles...")
        assert "[Turn 1] Speaker 0 | [Transition] SPEAKER_FLOOR_SHIFT: नमस्ते" in prompt
        assert "[Turn 2] Speaker 0 | [Transition] CONTINUES_SAME_SPEAKER: मैं बैंक से बोल रहा हूँ" in prompt
        assert "[Turn 3] Speaker 1 | [Transition] SPEAKER_FLOOR_SHIFT: हाँ कहिए" in prompt
        assert "Speaker Context:" in prompt
        assert "Speaker Profiles..." in prompt

    def test_stage2_system_instruction_contains_strict_constraints(self):
        assert "CONTINUES_SAME_SPEAKER" in STAGE2_CHAMPION_SYSTEM_INSTRUCTION
        assert "RESUMES_AFTER_INTERRUPTION" in STAGE2_CHAMPION_SYSTEM_INSTRUCTION
        assert "DO NOT change the speaker unless there is an overwhelming global stance inversion" in STAGE2_CHAMPION_SYSTEM_INSTRUCTION
        assert '{"corrections": []}' in STAGE2_CHAMPION_SYSTEM_INSTRUCTION


import json
from pathlib import Path

class TestGeneralizationCertificationC5:
    """Tests Candidate C5 generalization artifacts and certification logic."""

    def test_compute_metrics_slice_logic(self):
        from scripts.run_generalization_c5 import compute_metrics_slice

        empty_res = compute_metrics_slice([])
        assert empty_res["count"] == 0
        assert empty_res["mean_speaker_attribution_accuracy"] == 0.0

        mock_records = [
            {"metrics": {"wer": 0.20, "cer": 0.10, "speaker_attribution_accuracy": 0.80, "cpwer": 0.30, "diarization_gap": 0.10}, "latency_seconds": 4.0},
            {"metrics": {"wer": 0.25, "cer": 0.12, "speaker_attribution_accuracy": 0.90, "cpwer": 0.35, "diarization_gap": 0.10}, "latency_seconds": 5.0},
        ]
        res = compute_metrics_slice(mock_records)
        assert res["count"] == 2
        assert res["mean_speaker_attribution_accuracy"] == 0.85
        assert res["mean_wer"] == 0.225
        assert res["mean_latency_seconds"] == 4.5
        assert res["outlier_count"] == 0

    def test_c5_generalization_predictions_artifact_integrity(self):
        preds_path = Path(__file__).resolve().parent.parent / "results" / "optimized" / "predictions_strategy_c5_generalization.json"
        assert preds_path.exists(), f"Missing prediction file: {preds_path}"

        with open(preds_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) == 20, f"Expected 20 samples, got {len(records)}"

        sample_ids = [r["sample_id"] for r in records]
        assert len(set(sample_ids)) == 20, "Duplicate sample IDs detected!"

        for r in records:
            assert r["model_id"] == "gemini-3.5-flash-lite", f"Invalid model_id: {r['model_id']}"
            assert r["approach"] == "strategy_c5_adaptive_champion"
            assert isinstance(r["latency_seconds"], (int, float)) and r["latency_seconds"] > 0
            assert "raw_response" in r and len(r["raw_response"]) > 0
            assert "predicted_turns" in r and len(r["predicted_turns"]) > 0
            assert "metrics" in r
            m = r["metrics"]
            assert 0.0 <= m["speaker_attribution_accuracy"] <= 1.0
            assert 0.0 <= m["wer"]
            assert 0.0 <= m["cer"]
            assert 0.0 <= m["cpwer"]

    def test_c5_generalization_summary_accuracy_and_slices(self):
        summary_path = Path(__file__).resolve().parent.parent / "results" / "optimized" / "generalization_summary.json"
        assert summary_path.exists(), f"Missing summary file: {summary_path}"

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        macro = summary["macro_metrics"]
        assert summary["total_samples"] == 20
        assert summary["evaluated_model"] == "gemini-3.5-flash-lite"
        assert macro["outlier_count"] == 0

        # Assert performance targets and zero regression
        assert macro["mean_speaker_attribution_accuracy"] >= 0.75  # 77.75% achieved vs 69.04% baseline
        assert macro["mean_wer"] <= 0.245                          # 23.49% achieved
        assert macro["mean_latency_seconds"] < 6.0                 # 4.34s achieved

        # Slices
        slices = summary["slices"]
        assert slices["2_speakers"]["count"] == 15
        assert slices["2_speakers"]["mean_speaker_attribution_accuracy"] >= 0.75
        assert slices["3_speakers"]["count"] == 5
        assert slices["3_speakers"]["mean_speaker_attribution_accuracy"] >= 0.70
        assert slices["4_speakers"]["count"] == 0

        # Delta vs Default 3.5 Lite
        deltas = summary["baseline_comparison"]["candidate_c5_delta_vs_3_5_default"]
        assert deltas["saa_gain"] > 0, "Regression detected vs Default 3.5 Lite baseline!"
        assert deltas["diarization_gap_reduction"] > 0, "Diarization Gap did not improve!"
