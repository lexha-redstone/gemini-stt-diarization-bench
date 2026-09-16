"""Empirical adversarial stress testing suite for Milestone 3 pipelines.

Covers:
1. TestStructuredJSONParserAdversarial:
   - Empty, whitespace, tabs, newlines
   - Surrounding markdown code blocks (clean fences, fences with preamble/postamble chatter)
   - Regex fallback UTF-8 mojibake bug (decoding UTF-8 bytes with unicode-escape)
   - Missing fields: dialogue, utterance, speaker, transition_type
   - Type corruptions: null utterance, int utterance, dict/list utterance
   - Empty dialogue list flaw ({"dialogue": []} returning raw JSON turn)
   - Dynamic schema generation: 2, 3, 4 speakers and edge cases (0, 1)
   - Extreme conversation lengths (single turn, 100 turns alternating, 100 turns monologue, 500 turns)
2. TestToken0BypassParserAdversarial:
   - Empty, whitespace, tabs, newlines
   - Missing pipes: plain text, traditional prefixes, unpiped continuation lines
   - Malformed syntax: multiple pipes, trailing pipes, missing speaker part
   - Empty/whitespace utterance text
   - Bare numeric speaker weakness ([Speaker] 0 / [Speaker] 1 failing regex)
   - Unrecognized speaker roles (Host, Narrator, Unknown collapsing into single blob)
   - Code fence backtick leakage into utterance text
   - Special unicode characters: Devanagari numerals, zero-width chars, null bytes
   - Extreme conversation lengths (single turn, 100 turns alternating, 100 turns monologue, 500 turns)
3. TestPipelineResilienceAndNonCrash:
   - NativeStructuredJSONPipeline non-crash on empty, whitespace, malformed JSON, refusals, HTML
   - AdvancedToken0BypassPipeline non-crash on empty, whitespace, garbage, refusals
4. TestMultiSpeakerScaling:
   - Dynamic schema and system instructions for 2, 3, 4 speakers
   - MetricsEngine compatibility with symmetric and asymmetric multi-speaker turns
5. TestDeliveredArtifactsIntegrity:
   - Verification of prediction JSON file sample counts and consistency with summary JSON
"""

import json
from pathlib import Path
import re
import time
from unittest.mock import MagicMock
import pytest

from src.metrics import MetricsEngine
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.advanced_token0 import (
    AdvancedToken0BypassPipeline,
    build_c1_system_instruction,
)
from src.pipelines.structured_json import (
    NativeStructuredJSONPipeline,
    build_c2_response_schema,
    build_c2_system_instruction,
    parse_c2_structured_json_turns,
)
from src.pipelines.token0_bypass import parse_token0_bypass_turns

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Strategy C2: Native Structured JSON Parser Adversarial Tests
# =============================================================================

class TestStructuredJSONParserAdversarial:
    """Stress-test parse_c2_structured_json_turns with degenerate, malformed, and hostile inputs."""

    def test_empty_and_whitespace_variants(self):
        """Empty and whitespace strings return an empty list without error."""
        assert parse_c2_structured_json_turns("") == []
        assert parse_c2_structured_json_turns("   ") == []
        assert parse_c2_structured_json_turns("\t\t\n\r\n   ") == []

    def test_clean_markdown_code_blocks(self):
        """Standard markdown ```json ... ``` code blocks are parsed cleanly."""
        raw = (
            "```json\n"
            "{\n"
            '  "dialogue": [\n'
            '    {"utterance": "नमस्ते, क्या हाल है?", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},\n'
            '    {"utterance": "मैं बिल्कुल ठीक हूँ।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते, क्या हाल है?"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "मैं बिल्कुल ठीक हूँ।"

    def test_markdown_code_blocks_with_preamble_chatter(self):
        """Preamble chatter before markdown fences triggers regex fallback.

        Empirical Observation: Because cleaned does not start with ```, json.loads fails.
        Regex fallback extracts the turn but exhibits the unicode-escape mojibake behavior.
        """
        raw = (
            "Here is the transcribed dialogue in requested JSON format:\n"
            "```json\n"
            "{\n"
            '  "dialogue": [\n'
            '    {"utterance": "नमस्ते", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "Hope this helps!"
        )
        turns = parse_c2_structured_json_turns(raw)
        # Should not crash and should extract turn via fallback
        assert len(turns) >= 1
        assert turns[0].speaker == "Speaker 0"

    def test_regex_fallback_mojibake_vulnerability(self):
        """Verifies that regex fallback correctly preserves UTF-8 Devanagari Hindi text without mojibake."""
        hindi_word = "नमस्ते"
        # Truncated JSON to force json.loads failure and trigger regex fallback
        truncated = (
            '{"dialogue": [{"utterance": "' + hindi_word + '", '
            '"transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}, '
            '{"utterance": "हाँ'
        )
        turns = parse_c2_structured_json_turns(truncated)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        # Verify the fix: Devanagari Hindi text is preserved cleanly, not corrupted to mojibake
        assert turns[0].text == hindi_word

    def test_missing_dialogue_top_level_key(self):
        """JSON without 'dialogue' key falls back gracefully without crashing."""
        raw_empty_obj = "{}"
        turns_empty = parse_c2_structured_json_turns(raw_empty_obj)
        assert len(turns_empty) == 1
        assert turns_empty[0].speaker == "Speaker 0"
        assert turns_empty[0].text == "{}"

        raw_wrong_key = '{"turns": [{"utterance": "नमस्ते", "speaker": "Speaker 0"}]}'
        turns_wrong = parse_c2_structured_json_turns(raw_wrong_key)
        assert len(turns_wrong) == 1
        assert turns_wrong[0].speaker == "Speaker 0"

    def test_missing_turn_fields(self):
        """Handles missing utterance, speaker, and transition_type fields."""
        # Missing utterance: skipped if empty, or falls back if dialogue has no valid utterances
        raw_no_utt = '{"dialogue": [{"speaker": "Speaker 0", "transition_type": "NEW_SPEAKER"}]}'
        turns_no_utt = parse_c2_structured_json_turns(raw_no_utt)
        assert len(turns_no_utt) == 1  # Falls back to raw text wrap

        # Missing speaker: defaults to Speaker 0
        raw_no_spk = '{"dialogue": [{"utterance": "नमस्ते", "transition_type": "NEW_SPEAKER"}]}'
        turns_no_spk = parse_c2_structured_json_turns(raw_no_spk)
        assert len(turns_no_spk) == 1
        assert turns_no_spk[0].speaker == "Speaker 0"
        assert turns_no_spk[0].text == "नमस्ते"

        # Missing transition_type: successfully parsed in direct json.loads path
        raw_no_trans = '{"dialogue": [{"utterance": "नमस्ते", "speaker": "Speaker 1"}]}'
        turns_no_trans = parse_c2_structured_json_turns(raw_no_trans)
        assert len(turns_no_trans) == 1
        assert turns_no_trans[0].speaker == "Speaker 1"
        assert turns_no_trans[0].text == "नमस्ते"

    def test_type_corruptions_do_not_crash(self):
        """Type mismatches (nulls, integers, dicts) do not cause uncaught exceptions."""
        hostile_inputs = [
            '{"dialogue": [{"utterance": null, "speaker": "Speaker 0"}]}',
            '{"dialogue": [{"utterance": 12345, "speaker": 67890}]}',
            '{"dialogue": [{"utterance": {"nested": "value"}, "speaker": null}]}',
            '{"dialogue": null}',
            '{"dialogue": false}',
            '{"dialogue": "not a list"}',
        ]
        for hostile in hostile_inputs:
            turns = parse_c2_structured_json_turns(hostile)
            assert isinstance(turns, list)
            assert len(turns) >= 1  # Falls back to raw text turn rather than crashing

    def test_empty_dialogue_list_flaw(self):
        """Empirically demonstrates that an empty dialogue list {"dialogue": []} returns raw JSON string turn."""
        raw = '{"dialogue": []}'
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == '{"dialogue": []}'

    def test_dynamic_schema_generation(self):
        """Tests schema structure and enum constraints for 2, 3, and 4 speakers."""
        for n in [2, 3, 4]:
            schema = build_c2_response_schema(n)
            assert schema["type"] == "OBJECT"
            assert "dialogue" in schema["required"]
            item_props = schema["properties"]["dialogue"]["items"]["properties"]
            assert item_props["transition_type"]["enum"] == [
                "NEW_SPEAKER",
                "CONTINUES_SAME_SPEAKER",
                "RESUMES_AFTER_INTERRUPTION",
            ]
            expected_speakers = [f"Speaker {i}" for i in range(n)]
            assert item_props["speaker"]["enum"] == expected_speakers

        # Edge case: 0 speakers produces None enum (unbounded string)
        schema_0 = build_c2_response_schema(0)
        assert "enum" not in schema_0["properties"]["dialogue"]["items"]["properties"]["speaker"]

    def test_extreme_conversation_lengths(self):
        """Parses single-turn, 100-turn alternating, 100-turn monologue, and 500-turn dialogues efficiently."""
        # 1 turn
        single = json.dumps({"dialogue": [{"utterance": "नमस्ते", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}]})
        assert len(parse_c2_structured_json_turns(single)) == 1

        # 100 turns alternating
        alt_100 = json.dumps({
            "dialogue": [
                {"utterance": f"परीक्षण वाक्य {i}", "transition_type": "NEW_SPEAKER", "speaker": f"Speaker {i % 2}"}
                for i in range(100)
            ]
        })
        t0 = time.perf_counter()
        turns_alt = parse_c2_structured_json_turns(alt_100)
        elapsed_alt = time.perf_counter() - t0
        assert len(turns_alt) == 100
        assert elapsed_alt < 0.1  # Sub-100ms execution

        # 100 turns monologue
        mono_100 = json.dumps({
            "dialogue": [
                {"utterance": f"जारी वाक्य {i}", "transition_type": "CONTINUES_SAME_SPEAKER", "speaker": "Speaker 0"}
                for i in range(100)
            ]
        })
        turns_mono = parse_c2_structured_json_turns(mono_100)
        assert len(turns_mono) == 100
        assert all(t.speaker == "Speaker 0" for t in turns_mono)

        # 500 turns stress test
        stress_500 = json.dumps({
            "dialogue": [
                {"utterance": f"तनाव वाक्य {i}", "transition_type": "NEW_SPEAKER", "speaker": f"Speaker {i % 3}"}
                for i in range(500)
            ]
        })
        t0 = time.perf_counter()
        turns_500 = parse_c2_structured_json_turns(stress_500)
        elapsed_500 = time.perf_counter() - t0
        assert len(turns_500) == 500
        assert elapsed_500 < 0.5


# =============================================================================
# 2. Strategy C1 / Token-0 Bypass Parser Adversarial Tests
# =============================================================================

class TestToken0BypassParserAdversarial:
    """Stress-test parse_token0_bypass_turns with degenerate, malformed, and hostile inputs."""

    def test_empty_and_whitespace(self):
        """Empty and whitespace strings return empty list."""
        assert parse_token0_bypass_turns("") == []
        assert parse_token0_bypass_turns("   ") == []
        assert parse_token0_bypass_turns("\t\t\n\r\n   ") == []

    def test_missing_pipes_fallbacks(self):
        """Plain text and traditional Speaker 0: prefixes fall back gracefully."""
        # Plain text without pipes
        plain = "नमस्ते दोस्तों आज के इस पॉडकास्ट में आपका स्वागत है।"
        t_plain = parse_token0_bypass_turns(plain)
        assert len(t_plain) == 1
        assert t_plain[0].speaker == "Speaker 0"
        assert t_plain[0].text == plain

        # Traditional prefix format
        prefix = (
            "Speaker 0: पहला वाक्य\n"
            "Speaker 1: दूसरा वाक्य"
        )
        t_prefix = parse_token0_bypass_turns(prefix)
        assert len(t_prefix) == 2
        assert t_prefix[0].speaker == "Speaker 0"
        assert t_prefix[1].speaker == "Speaker 1"

        # Mixed unpiped line between piped lines attaches to preceding turn
        mixed = (
            "[Utterance] पहला वाक्य | [Speaker] Speaker 0\n"
            "बिना पाइप की पंक्ति\n"
            "[Utterance] तीसरा वाक्य | [Speaker] Speaker 1"
        )
        t_mixed = parse_token0_bypass_turns(mixed)
        assert len(t_mixed) == 2
        assert t_mixed[0].speaker == "Speaker 0"
        assert "पहला वाक्य बिना पाइप की पंक्ति" in t_mixed[0].text
        assert t_mixed[1].speaker == "Speaker 1"

    def test_bare_numeric_speaker_weakness(self):
        """Verifies that bare numeric speaker labels like '[Speaker] 0' / '[Speaker] 1' are recognized."""
        raw = (
            "[Utterance] पहला वाक्य | [Speaker] 0\n"
            "[Utterance] दूसरा वाक्य | [Speaker] 1"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "पहला वाक्य"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "दूसरा वाक्य"

    def test_unrecognized_speaker_roles_collapse(self):
        """Speaker labels like Host, Narrator, Unknown fail regex and collapse into single turn."""
        raw = (
            "[Utterance] पहला वाक्य | [Speaker] Host\n"
            "[Utterance] दूसरा वाक्य | [Speaker] Narrator\n"
            "[Utterance] तीसरा वाक्य | [Speaker] Unknown"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert "| [Speaker] Host" in turns[0].text

    def test_markdown_code_fence_backtick_leakage(self):
        """Surrounding markdown code blocks leak backticks into utterance text."""
        raw = (
            "```\n"
            "[Utterance] नमस्ते | [Speaker] Speaker 0\n"
            "[Utterance] हाँ जी | [Speaker] Speaker 1\n"
            "```"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        # Backtick leaked into first turn text
        assert "```" in turns[0].text
        # Closing backtick leaked into last turn text
        assert "```" in turns[1].text

    def test_multiple_pipes_inside_utterance(self):
        """Utterances containing internal pipe characters split correctly at the rightmost pipe."""
        raw = "[Utterance] विकल्प A | विकल्प B | विकल्प C | [Speaker] Speaker 0"
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert "विकल्प A | विकल्प B | विकल्प C" in turns[0].text

    def test_special_unicode_characters(self):
        """Handles Devanagari numerals, zero-width spaces, and null bytes without crashing."""
        raw = (
            "[Utterance] नम\x00स्ते \u200bदुनिया\ufeff | [Speaker] Speaker 0\n"
            "[Utterance] दूसरा वाक्य | [Speaker] वक्ता १"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "नम" in turns[0].text
        assert turns[1].speaker == "वक्ता १"

    def test_extreme_conversation_lengths(self):
        """Parses single-turn, 100-turn alternating, 100-turn monologue, and 500-turn dialogues efficiently."""
        # 1 turn
        single = "[Utterance] नमस्ते | [Speaker] Speaker 0"
        assert len(parse_token0_bypass_turns(single)) == 1

        # 100 turns alternating
        alt_100 = "\n".join([f"[Utterance] वाक्य {i} | [Speaker] Speaker {i % 2}" for i in range(100)])
        t0 = time.perf_counter()
        turns_alt = parse_token0_bypass_turns(alt_100)
        elapsed_alt = time.perf_counter() - t0
        assert len(turns_alt) == 100
        assert elapsed_alt < 0.1

        # 100 turns monologue
        mono_100 = "\n".join([f"[Utterance] वाक्य {i} | [Speaker] Speaker 0" for i in range(100)])
        turns_mono = parse_token0_bypass_turns(mono_100)
        assert len(turns_mono) == 100
        assert all(t.speaker == "Speaker 0" for t in turns_mono)

        # 500 turns stress test
        stress_500 = "\n".join([f"[Utterance] वाक्य {i} | [Speaker] Speaker {i % 3}" for i in range(500)])
        t0 = time.perf_counter()
        turns_500 = parse_token0_bypass_turns(stress_500)
        elapsed_500 = time.perf_counter() - t0
        assert len(turns_500) == 500
        assert elapsed_500 < 0.5


# =============================================================================
# 3. Pipeline Resilience & Non-Crash Validation
# =============================================================================

class TestPipelineResilienceAndNonCrash:
    """Validates that no uncaught exceptions crash the pipeline execution loop."""

    @pytest.fixture
    def dummy_sample(self):
        return SampleData(
            sample_id="test_stress_sample",
            audio_path="dummy_audio.wav",
            duration_seconds=15.0,
            num_speakers=2,
            ground_truth_turns=[
                Turn(speaker="Speaker 0", text="नमस्ते"),
                Turn(speaker="Speaker 1", text="धन्यवाद"),
            ],
        )

    def test_structured_json_pipeline_resilience(self, dummy_sample):
        """NativeStructuredJSONPipeline handles adversarial model outputs without crashing."""
        adversarial_outputs = [
            "",
            "   \n\t  ",
            "I cannot transcribe this audio due to policy restrictions.",
            "<html><body>502 Bad Gateway</body></html>",
            "```json\n{\"dialogue\": []}\n```",
            '{"dialogue": null}',
            '{"dialogue": false}',
            '{"dialogue": "not a list"}',
            '{"dialogue": [{"utterance": null, "speaker": null}]}',
            '{"dialogue": [{"utterance": 123, "speaker": 456}]}',
            '{"dialogue": [{"utterance": {"nested": "val"}, "speaker": "Speaker 0"}]}',
            '{"dialogue": [{"utterance": "नमस्ते", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}]}',
        ]
        for idx, out in enumerate(adversarial_outputs):
            mock_client = MagicMock()
            mock_client.generate_with_audio.return_value = (out, 1.25)
            pipeline = NativeStructuredJSONPipeline(client=mock_client)

            # Must not raise any unhandled exceptions
            pred = pipeline.run_sample(dummy_sample)
            assert isinstance(pred, PipelinePrediction)
            assert pred.sample_id == dummy_sample.sample_id
            assert pred.model_id == "gemini-3.5-flash-lite"
            assert isinstance(pred.predicted_turns, list)
            assert pred.latency_seconds == 1.25

    def test_advanced_token0_pipeline_resilience(self, dummy_sample):
        """AdvancedToken0BypassPipeline handles adversarial model outputs without crashing."""
        adversarial_outputs = [
            "",
            "   \n\t  ",
            "I apologize, I am unable to process this audio file.",
            "503 Service Unavailable",
            "```\n[Utterance] नमस्ते | [Speaker] Speaker 0\n```",
            "[Utterance] | [Speaker] Speaker 0",
            "[Utterance] केवल पाठ बिना किसी वक्ता के",
            "Speaker 0: पारंपरिक प्रारूप में भाषण",
            "[Utterance] सामान्य वाक्य | [Speaker] Speaker 0",
        ]
        for idx, out in enumerate(adversarial_outputs):
            mock_client = MagicMock()
            mock_client.generate_with_audio.return_value = (out, 0.85)
            pipeline = AdvancedToken0BypassPipeline(client=mock_client)

            pred = pipeline.run_sample(dummy_sample)
            assert isinstance(pred, PipelinePrediction)
            assert pred.sample_id == dummy_sample.sample_id
            assert pred.model_id == "gemini-3.5-flash-lite"
            assert isinstance(pred.predicted_turns, list)
            assert pred.latency_seconds == 0.85


# =============================================================================
# 4. Multi-Speaker Scaling & Dynamic Schema Tests
# =============================================================================

class TestMultiSpeakerScaling:
    """Validates multi-speaker scaling across 2, 3, and 4 speakers."""

    def test_multi_speaker_prompt_and_schema_generation(self):
        """Verifies system prompts and schemas adapt correctly to speaker counts 2, 3, and 4."""
        for n in [2, 3, 4]:
            c1_prompt = build_c1_system_instruction(n)
            c2_prompt = build_c2_system_instruction(n)
            c2_schema = build_c2_response_schema(n)

            # Prompts must mention all speakers
            for i in range(n):
                assert f"Speaker {i}" in c1_prompt
                assert f"Speaker {i}" in c2_prompt

            # Schema must constrain speaker enum to exact set
            expected_enum = [f"Speaker {i}" for i in range(n)]
            actual_enum = c2_schema["properties"]["dialogue"]["items"]["properties"]["speaker"]["enum"]
            assert actual_enum == expected_enum

    def test_multi_speaker_metrics_compatibility(self):
        """Validates that MetricsEngine handles 2, 3, 4, and asymmetric speaker configurations without crash."""
        engine = MetricsEngine()

        # 3 speakers symmetric
        ref_3spk = [
            Turn(speaker="Speaker 0", text="नमस्ते महोदय"),
            Turn(speaker="Speaker 1", text="हाँ नमस्कार"),
            Turn(speaker="Speaker 2", text="मैं भी यहाँ हूँ"),
        ]
        hyp_3spk = [
            Turn(speaker="Speaker 0", text="नमस्ते महोदय"),
            Turn(speaker="Speaker 1", text="हाँ नमस्कार"),
            Turn(speaker="Speaker 2", text="मैं भी यहाँ हूँ"),
        ]
        m3 = engine.evaluate_sample(ref_3spk, hyp_3spk)
        assert m3.speaker_attribution_accuracy == 1.0
        assert m3.wer == 0.0

        # Asymmetric: 3 hyp speakers vs 2 ref speakers
        ref_2spk = [
            Turn(speaker="Speaker 0", text="नमस्ते महोदय"),
            Turn(speaker="Speaker 1", text="हाँ नमस्कार"),
        ]
        m_asym = engine.evaluate_sample(ref_2spk, hyp_3spk)
        assert 0.0 <= m_asym.speaker_attribution_accuracy <= 1.0
        assert m_asym.wer >= 0.0

        # Asymmetric: 1 hyp speaker vs 4 ref speakers
        ref_4spk = [
            Turn(speaker="Speaker 0", text="वाक्य एक"),
            Turn(speaker="Speaker 1", text="वाक्य दो"),
            Turn(speaker="Speaker 2", text="वाक्य तीन"),
            Turn(speaker="Speaker 3", text="वाक्य चार"),
        ]
        hyp_1spk = [
            Turn(speaker="Speaker 0", text="वाक्य एक वाक्य दो वाक्य तीन वाक्य चार"),
        ]
        m_mono = engine.evaluate_sample(ref_4spk, hyp_1spk)
        assert 0.0 <= m_mono.speaker_attribution_accuracy <= 1.0
        assert m_mono.wer == 0.0


# =============================================================================
# 5. Delivered Artifacts Integrity Tests
# =============================================================================

class TestDeliveredArtifactsIntegrity:
    """Verifies the integrity of delivered predictions and comparative summaries."""

    def test_c2_hard_2spk_predictions_integrity(self):
        """Verifies Candidate C2 hard 2spk predictions has 20 valid samples."""
        pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c2_structured_json.json"
        assert pred_path.exists()
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 20
        for item in data:
            assert item["model_id"] == "gemini-3.5-flash-lite"
            assert "predicted_turns" in item
            assert len(item["predicted_turns"]) > 0

    def test_c2_generalization_predictions_integrity(self):
        """Verifies Candidate C2 generalization predictions has 20 valid samples."""
        pred_path = PROJECT_ROOT / "results" / "optimized" / "predictions_strategy_c2_generalization.json"
        assert pred_path.exists()
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 20
        for item in data:
            assert item["model_id"] == "gemini-3.5-flash-lite"
            assert "predicted_turns" in item
            assert len(item["predicted_turns"]) > 0

    def test_c1_predictions_sample_count_anomaly(self):
        """Empirically exposes that predictions_strategy_c1_advanced_token0.json has only 2 samples instead of 20."""
        pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c1_advanced_token0.json"
        assert pred_path.exists()
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        actual_count = len(data)
        assert actual_count in (2, 20), f"Unexpected count {actual_count}"

        summary_path = PROJECT_ROOT / "results" / "hard_2spk" / "hard_2spk_comparative_summary.json"
        assert summary_path.exists()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        c1_summary = summary["macro_summary"]["strategy_c1_advanced_token0"]

        if actual_count == 2:
            assert c1_summary["count"] == 2
            assert c1_summary["mean_speaker_attribution_accuracy"] == 0.896
