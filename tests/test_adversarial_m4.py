"""Empirical adversarial stress testing suite for Milestone 4 Candidate C5 pipeline.

Covers:
1. TestC5MalformedAndNoisyOutputsAdversarial:
   - Empty, whitespace, null, tabs, newlines
   - Clean markdown code fences and backtick leakage vulnerability
   - Preamble chatter and corrupted [Speaker Profiles]
   - Irregular speaker IDs (Speaker 00, speaker 1, bare digits, SPK_2 vulnerability)
   - Missing transition tags, two-field and prefix fallbacks
   - Malformed pipe delimiters and empty utterance handling
   - Unicode anomalies (Devanagari numerals, zero-width chars, emojis)
2. TestC5ExtremeDialoguesAdversarial:
   - 100+ turns single-speaker monologues
   - 100+ turns rapid alternating ping-pong
   - Extreme scale 500-turn throughput and linear time complexity
   - Preamble-only outputs with zero utterance content
3. TestC5AdaptiveGatingTriggerAdversarial:
   - Insufficient turns (<= 1) bypass
   - Duration boundary conditions (69.9s vs 70.0s, negative, zero)
   - Single-speaker collapse detection trigger
   - High alternation threshold boundaries (15 turns vs 16 turns, 80% vs 85%)
   - Colloquial ping-pong triggers
   - Debate context detection (strong single-word vs weak multi-word markers)
4. TestC5Stage2CorrectionInjectionAdversarial:
   - Valid JSON corrections
   - Markdown-wrapped JSON corrections
   - Multiple JSON blocks / greedy regex behavior
   - Malformed, truncated, and empty JSON fallbacks
   - Out-of-bounds (9999) and negative (-1) turn IDs
   - Index collision: turn_id 0 and turn_id 1 both modifying index 0
   - Boolean turn IDs (turn_id True modifying index 0)
   - String turn IDs safely ignored
   - Duplicate and conflicting corrections for the same turn
   - Non-dict items in corrections list triggering AttributeError crash in run_sample
5. TestC5PipelineResilienceAndExecutionAdversarial:
   - Normal mock execution emits valid PipelinePrediction
   - Empty Stage 1 response emits valid PipelinePrediction
   - Model refusal Stage 1 response emits valid PipelinePrediction
   - HTML error page Stage 1 response emits valid PipelinePrediction
   - Stage 2 malformed or empty responses preserve Stage 1 turns without crash
   - Single-speaker collapse injects critical notice into Stage 2 prompt
6. TestC5DynamicCardinalityScalingAdversarial:
   - Prompt generation for N = 2, 3, 4, 5 speakers
   - Prompt generation edge cases N = 0, 1 (malformed phrasing)
   - MetricsEngine Hungarian SAA compatibility across multi-speaker predictions
"""

import json
from pathlib import Path
import re
import time
from unittest.mock import MagicMock
import pytest

from src.metrics import MetricsEngine
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.adaptive_acoustic_champion import (
    STAGE2_CHAMPION_SYSTEM_INSTRUCTION,
    AdaptiveAcousticChampionPipeline,
    build_c5_stage1_system_instruction,
    check_debate_context,
    parse_c5_stage1_turns,
    should_trigger_stage2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Malformed and Noisy Raw Outputs
# =============================================================================

class TestC5MalformedAndNoisyOutputsAdversarial:
    """Stress-tests parse_c5_stage1_turns against noisy, malformed, and irregular formats."""

    def test_empty_and_whitespace_inputs(self):
        """Empty, whitespace, tabs, and newlines return empty structures without error."""
        assert parse_c5_stage1_turns("") == ([], [], "")
        assert parse_c5_stage1_turns("   ") == ([], [], "")
        assert parse_c5_stage1_turns("\n\n\t\r\n   ") == ([], [], "")

    def test_clean_markdown_code_fences(self):
        """Markdown code blocks ``` ... ``` parse internal utterances cleanly."""
        raw = (
            "```\n"
            "[Speaker Profiles]\n"
            "- Speaker 0: Male, chest voice\n"
            "- Speaker 1: Female, higher pitch\n\n"
            "[Utterance] नमस्कार, क्या हाल है? | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "[Utterance] मैं ठीक हूँ। | [Transition] SHIFT | [Speaker] Speaker 1\n"
            "```"
        )
        turns, tags, profiles = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्कार, क्या हाल है?"
        assert tags[0] == "SPEAKER_FLOOR_SHIFT"
        assert turns[1].speaker == "Speaker 1"
        assert tags[1] == "SPEAKER_FLOOR_SHIFT"

    def test_markdown_code_fence_backtick_leakage_vulnerability(self):
        """Verifies that closing code fence backticks do NOT leak into the final turn text.

        Remediated: Standalone markdown code fences (```) are skipped by the parser,
        ensuring pristine transcription text without trailing backticks.
        """
        raw = (
            "```\n"
            "[Utterance] पहला वाक्य | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "[Utterance] दूसरा वाक्य | [Transition] SHIFT | [Speaker] Speaker 1\n"
            "```"
        )
        turns, tags, profiles = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[1].text == "दूसरा वाक्य"
        assert not turns[1].text.endswith("```")

    def test_preamble_chatter_and_corrupted_profiles(self):
        """Handles verbose conversational intro text and various profile bullet formats."""
        raw = (
            "Here is the verbatim transcription with acoustic profile grounding:\n\n"
            "[Speaker Profiles]\n"
            "* Speaker 0: Low pitch, male, calm loan officer.\n"
            "• Speaker 1: High pitch, female, rapid borrower.\n\n"
            "Dialogue turns follow below:\n"
            "1. [Utterance] नमस्ते जी | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "2. [Utterance] हाँ जी नमस्कार | [Transition] SHIFT | [Speaker] Speaker 1\n"
        )
        turns, tags, profiles = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते जी"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "हाँ जी नमस्कार"
        assert "Speaker 0" in profiles
        assert "Speaker 1" in profiles

    def test_irregular_speaker_ids_speaker_00(self):
        """Speaker 00 is captured and preserved."""
        raw = (
            "[Utterance] नमस्ते | [Transition] SHIFT | [Speaker] Speaker 00\n"
            "[Utterance] हाँ | [Transition] SHIFT | [Speaker] Speaker 01\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 00"
        assert turns[1].speaker == "Speaker 01"

    def test_irregular_speaker_ids_lowercase(self):
        """Lowercase 'speaker 1' is normalized to 'Speaker 1'."""
        raw = (
            "[Utterance] नमस्ते | [Transition] SHIFT | [Speaker] speaker 0\n"
            "[Utterance] हाँ | [Transition] SHIFT | [Speaker] speaker 1\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[1].speaker == "Speaker 1"

    def test_irregular_speaker_ids_bare_digits(self):
        """Bare digits in speaker position ('0', '1') are normalized to 'Speaker 0'."""
        raw = (
            "[Utterance] नमस्ते | [Transition] SHIFT | [Speaker] 0\n"
            "[Utterance] हाँ | [Transition] SHIFT | [Speaker] 1\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[1].speaker == "Speaker 1"

    def test_irregular_speaker_ids_spk_2_vulnerability(self):
        """Verifies that 'SPK_2' irregular speaker format is cleanly parsed and normalized.

        Remediated: Regex p3, p2, and p_prefix support 'SPK_\\d+' labels, normalizing
        'SPK_2' to 'Speaker 2' instead of swallowing it into the preceding turn.
        """
        raw = (
            "[Utterance] नमस्ते | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "[Utterance] क्या हाल है | [Transition] SHIFT | [Speaker] SPK_2\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते"
        assert turns[1].speaker == "Speaker 2"
        assert turns[1].text == "क्या हाल है"

    def test_missing_transition_tags_two_field_fallback(self):
        """Two-field syntax [Utterance] ... | [Speaker] ... is parsed via p2 fallback."""
        raw = (
            "[Utterance] पहला वाक्य | [Speaker] Speaker 0\n"
            "[Utterance] दूसरा वाक्य | [Speaker] Speaker 0\n"
            "[Utterance] तीसरा वाक्य | [Speaker] Speaker 1\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 3
        assert turns[0].speaker == "Speaker 0"
        assert tags[0] == "SPEAKER_FLOOR_SHIFT"
        assert turns[1].speaker == "Speaker 0"
        assert tags[1] == "CONTINUES_SAME_SPEAKER"
        assert turns[2].speaker == "Speaker 1"
        assert tags[2] == "SPEAKER_FLOOR_SHIFT"

    def test_prefix_fallback(self):
        """Traditional Speaker 0: ... prefix syntax is parsed via p_prefix fallback."""
        raw = (
            "Speaker 0: पहला वाक्य\n"
            "Speaker 1: दूसरा वाक्य\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[1].speaker == "Speaker 1"

    def test_unpiped_plain_text_fallback(self):
        """Unstructured plain text falls back to a single Speaker 0 turn."""
        raw = "यह एक असंरचित वाक्य है जो किसी भी प्रारूप का पालन नहीं करता।"
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == raw
        assert tags[0] == "SPEAKER_FLOOR_SHIFT"

    def test_malformed_pipes_and_empty_utterance(self):
        """Lines with empty utterance text are skipped, preserving tag alignment."""
        raw = (
            "[Utterance] | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "[Utterance] वैध वाक्य | [Transition] SHIFT | [Speaker] Speaker 1\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 1
        assert len(tags) == 1
        assert turns[0].speaker == "Speaker 1"
        assert turns[0].text == "वैध वाक्य"

    def test_unicode_anomalies(self):
        """Preserves Devanagari numerals, zero-width characters, and emoji."""
        raw = (
            "[Utterance] संख्या १२३४५ और \u200cज़ीरो विड्थ | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "[Utterance] इमोजी 😀 सहित वाक्य | [Transition] SHIFT | [Speaker] Speaker 1\n"
        )
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 2
        assert "१२३४५" in turns[0].text
        assert "\u200c" in turns[0].text
        assert "😀" in turns[1].text


# =============================================================================
# 2. Extreme Dialogues
# =============================================================================

class TestC5ExtremeDialoguesAdversarial:
    """Stress-tests dialogue scaling, monologue cascades, and extreme lengths."""

    def test_single_speaker_monologue_100_turns(self):
        """Parses a 100-turn single-speaker monologue cleanly."""
        lines = [
            f"[Utterance] वाक्य संख्या {i} जारी है | [Transition] CONTINUE | [Speaker] Speaker 0"
            for i in range(100)
        ]
        raw = "\n".join(lines)
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 100
        assert all(t.speaker == "Speaker 0" for t in turns)
        assert all(tag == "CONTINUES_SAME_SPEAKER" for tag in tags)

    def test_rapid_ping_pong_100_turns(self):
        """Parses 100 rapidly alternating turns."""
        lines = [
            f"[Utterance] टर्न {i} प्रतिक्रिया | [Transition] SHIFT | [Speaker] Speaker {i % 2}"
            for i in range(100)
        ]
        raw = "\n".join(lines)
        turns, tags, _ = parse_c5_stage1_turns(raw)
        assert len(turns) == 100
        assert len(tags) == 100
        for i in range(100):
            assert turns[i].speaker == f"Speaker {i % 2}"

    def test_extreme_scale_500_turns(self):
        """Scales to 500 turns in sub-second time (O(N) verification)."""
        lines = [
            f"[Utterance] दीर्घावधि संवाद टर्न {i} | [Transition] SHIFT | [Speaker] Speaker {i % 3}"
            for i in range(500)
        ]
        raw = "\n".join(lines)
        t0 = time.perf_counter()
        turns, tags, _ = parse_c5_stage1_turns(raw)
        elapsed = time.perf_counter() - t0
        assert len(turns) == 500
        assert len(tags) == 500
        assert elapsed < 0.5, f"Parsing 500 turns took too long: {elapsed:.3f}s"

    def test_preamble_only_with_no_utterances(self):
        """Speaker Profiles preamble without subsequent utterances falls back gracefully."""
        raw = (
            "[Speaker Profiles]\n"
            "- Speaker 0: Male, lower chest voice\n"
            "- Speaker 1: Female, bright tone\n"
        )
        turns, tags, profiles = parse_c5_stage1_turns(raw)
        # Should not crash; falls back to raw text turn
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"


# =============================================================================
# 3. Adaptive Gating Trigger
# =============================================================================

class TestC5AdaptiveGatingTriggerAdversarial:
    """Stress-tests should_trigger_stage2 and check_debate_context decision boundaries."""

    def test_insufficient_turns_bypass(self):
        """0 turns or 1 turn unconditionally bypass Stage 2."""
        assert should_trigger_stage2(120.0, [], []) == (False, "Insufficient turns (<= 1) for Stage 2")
        single_turn = [Turn(speaker="Speaker 0", text="अकेला वाक्य")]
        assert should_trigger_stage2(120.0, single_turn, ["SPEAKER_FLOOR_SHIFT"]) == (
            False,
            "Insufficient turns (<= 1) for Stage 2",
        )

    def test_duration_boundary_conditions(self):
        """Tests exact 70.0s boundary condition."""
        turns = [Turn(speaker=f"Speaker {i % 2}", text=f"बातचीत {i}") for i in range(5)]
        tags = ["SPEAKER_FLOOR_SHIFT"] * 5

        # 69.9s bypasses
        trig_69, _ = should_trigger_stage2(69.9, turns, tags)
        assert trig_69 is False

        # 70.0s triggers
        trig_70, reason_70 = should_trigger_stage2(70.0, turns, tags)
        assert trig_70 is True
        assert ">= 70.0s" in reason_70

        # Negative and zero duration do not crash
        trig_neg, _ = should_trigger_stage2(-10.0, turns, tags)
        assert trig_neg is False

    def test_single_speaker_collapse_trigger(self):
        """Single-speaker collapse (>= 4 turns with 1 speaker) triggers Stage 2."""
        turns_4 = [Turn(speaker="Speaker 0", text=f"वाक्य {i}") for i in range(4)]
        trig, reason = should_trigger_stage2(30.0, turns_4, ["CONTINUES_SAME_SPEAKER"] * 4)
        assert trig is True
        assert "Single-speaker collapse" in reason

        # 3 turns does not trigger collapse
        turns_3 = [Turn(speaker="Speaker 0", text=f"वाक्य {i}") for i in range(3)]
        trig_3, _ = should_trigger_stage2(30.0, turns_3, ["CONTINUES_SAME_SPEAKER"] * 3)
        assert trig_3 is False

    def test_high_alternation_ping_pong_boundary(self):
        """Alternation ratio boundary: requires alt_ratio >= 0.85 AND n_turns >= 16."""
        # 15 turns alternating (100% alt) -> does NOT trigger (n_turns < 16)
        turns_15 = [Turn(speaker=f"Speaker {i % 2}", text=f"बात {i}") for i in range(15)]
        trig_15, _ = should_trigger_stage2(45.0, turns_15, ["SPEAKER_FLOOR_SHIFT"] * 15)
        assert trig_15 is False

        # 16 turns alternating (100% alt) -> TRIGGERS
        turns_16 = [Turn(speaker=f"Speaker {i % 2}", text=f"बात {i}") for i in range(16)]
        trig_16, reason_16 = should_trigger_stage2(45.0, turns_16, ["SPEAKER_FLOOR_SHIFT"] * 16)
        assert trig_16 is True
        assert "ping-pong trap" in reason_16

        # 16 turns with 50% alternation -> does NOT trigger
        turns_mixed = [Turn(speaker=f"Speaker {i // 2 % 2}", text=f"बात {i}") for i in range(16)]
        trig_mixed, _ = should_trigger_stage2(45.0, turns_mixed, ["SPEAKER_FLOOR_SHIFT"] * 16)
        assert trig_mixed is False

    def test_colloquial_ping_pong_keywords(self):
        """Benchmark-specific keywords (बनारस, etc.) do NOT trigger Stage 2 for short casual dialogues."""
        turns = [
            Turn(speaker="Speaker 0", text="बनारस के घाट पर चलते हैं"),
            Turn(speaker="Speaker 1", text="हाँ चलो"),
        ]
        trig, reason = should_trigger_stage2(30.0, turns, ["SPEAKER_FLOOR_SHIFT"] * 2)
        assert trig is False
        assert "Bypass Stage 2" in reason

    def test_debate_markers_strong_vs_weak(self):
        """Single strong debate word triggers; weak debate words require 2+ occurrences."""
        # Strong marker: 'विरोध' alone triggers
        turns_strong = [Turn(speaker="Speaker 0", text="हम इस नीति का कड़ा विरोध करते हैं")]
        has_deb, markers = check_debate_context(turns_strong)
        assert has_deb is True
        assert "विरोध" in markers

        # Weak marker: 'मुद्दा' alone does NOT trigger
        turns_weak_single = [Turn(speaker="Speaker 0", text="यह एक महत्वपूर्ण मुद्दा है")]
        has_deb_single, markers_single = check_debate_context(turns_weak_single)
        assert has_deb_single is False

        # Weak markers: 'मुद्दा' + 'फायदा' triggers
        turns_weak_multi = [
            Turn(speaker="Speaker 0", text="यह एक महत्वपूर्ण मुद्दा है"),
            Turn(speaker="Speaker 1", text="इससे क्या फायदा होगा?"),
        ]
        has_deb_multi, markers_multi = check_debate_context(turns_weak_multi)
        assert has_deb_multi is True
        assert "मुद्दा" in markers_multi
        assert "फायदा" in markers_multi


# =============================================================================
# 4. Stage 2 Correction Injection and Robustness
# =============================================================================

class TestC5Stage2CorrectionInjectionAdversarial:
    """Stress-tests Stage 2 JSON parsing, hostile inputs, and correction logic."""

    @pytest.fixture
    def pipeline(self):
        return AdaptiveAcousticChampionPipeline(client=MagicMock())

    def test_valid_json_corrections(self, pipeline):
        """Parses standard valid JSON corrections."""
        raw = '{"corrections": [{"turn_id": 2, "speaker": "Speaker 0"}]}'
        corrections = pipeline._parse_stage2_corrections(raw)
        assert len(corrections) == 1
        assert corrections[0]["turn_id"] == 2
        assert corrections[0]["speaker"] == "Speaker 0"

    def test_markdown_wrapped_json_corrections(self, pipeline):
        """Parses JSON enclosed in markdown code fences."""
        raw = (
            "```json\n"
            "{\n"
            '  "corrections": [\n'
            '    {"turn_id": 5, "speaker": "Speaker 1"}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        corrections = pipeline._parse_stage2_corrections(raw)
        assert len(corrections) == 1
        assert corrections[0]["turn_id"] == 5

    def test_multiple_json_blocks_greedy_regex_behavior(self, pipeline):
        """Demonstrates behavior when multiple JSON objects appear in Stage 2 response.

        Empirical Observation: Greedy regex r'\\{[\\s\\S]*\\}' matches from the first '{'
        to the last '}', joining separate JSON blocks into invalid JSON syntax and falling back to [].
        """
        raw = (
            '{"thought": "Turn 2 should be Speaker 0"}\n'
            '{"corrections": [{"turn_id": 2, "speaker": "Speaker 0"}]}'
        )
        corrections = pipeline._parse_stage2_corrections(raw)
        assert corrections == []

    def test_malformed_truncated_and_empty_json(self, pipeline):
        """Truncated JSON and invalid strings fall back to empty list without raising."""
        assert pipeline._parse_stage2_corrections("") == []
        assert pipeline._parse_stage2_corrections("   ") == []
        assert pipeline._parse_stage2_corrections("No JSON here") == []
        assert pipeline._parse_stage2_corrections('{"corrections": [') == []
        assert pipeline._parse_stage2_corrections("<html>502 Bad Gateway</html>") == []

    def test_out_of_bounds_and_negative_turn_ids(self):
        """Out-of-bounds (9999) and negative turn IDs are safely ignored during correction."""
        turns = [Turn(speaker="Speaker 0", text="बात 1"), Turn(speaker="Speaker 1", text="बात 2")]
        corrections = [
            {"turn_id": 9999, "speaker": "Speaker 1"},
            {"turn_id": -1, "speaker": "Speaker 0"},
            {"turn_id": -50, "speaker": "Speaker 1"},
        ]
        # Simulate correction loop
        for corr in corrections:
            t_id = corr.get("turn_id")
            new_spk = corr.get("speaker")
            idx = t_id - 1 if (1 <= t_id <= len(turns)) else t_id
            if 0 <= idx < len(turns):
                turns[idx].speaker = new_spk

        assert turns[0].speaker == "Speaker 0"
        assert turns[1].speaker == "Speaker 1"

    def test_turn_id_zero_and_one_index_collision(self):
        """Empirically demonstrates that turn_id = 0 and turn_id = 1 both modify index 0.

        In adaptive_acoustic_champion.py:548:
        idx = t_id - 1 if (1 <= t_id <= len(final_turns)) else t_id
        When t_id == 0, 1 <= 0 <= len is False -> idx = 0.
        When t_id == 1, 1 <= 1 <= len is True  -> idx = 1 - 1 = 0.
        Both target turns[0], causing an index collision.
        """
        turns = [Turn(speaker="Speaker 0", text="बात 1"), Turn(speaker="Speaker 1", text="बात 2")]
        # Apply turn_id = 0
        corr_0 = {"turn_id": 0, "speaker": "Speaker 1"}
        idx_0 = corr_0["turn_id"] - 1 if (1 <= corr_0["turn_id"] <= len(turns)) else corr_0["turn_id"]
        assert idx_0 == 0

        # Apply turn_id = 1
        corr_1 = {"turn_id": 1, "speaker": "Speaker 1"}
        idx_1 = corr_1["turn_id"] - 1 if (1 <= corr_1["turn_id"] <= len(turns)) else corr_1["turn_id"]
        assert idx_1 == 0

    def test_boolean_turn_id_behavior(self):
        """In Python, isinstance(True, int) is True, so turn_id=True behaves like turn_id=1."""
        turns = [Turn(speaker="Speaker 0", text="बात 1"), Turn(speaker="Speaker 1", text="बात 2")]
        corr = {"turn_id": True, "speaker": "Speaker 1"}
        t_id = corr.get("turn_id")
        new_spk = corr.get("speaker")
        if isinstance(t_id, int) and isinstance(new_spk, str):
            idx = t_id - 1 if (1 <= t_id <= len(turns)) else t_id
            if 0 <= idx < len(turns):
                turns[idx].speaker = new_spk
        assert turns[0].speaker == "Speaker 1"

    def test_string_turn_id_safety(self):
        """String turn IDs ('1') are rejected by isinstance(t_id, int)."""
        turns = [Turn(speaker="Speaker 0", text="बात 1")]
        corr = {"turn_id": "1", "speaker": "Speaker 1"}
        t_id = corr.get("turn_id")
        assert not isinstance(t_id, int)
        # Turn remains unmodified
        assert turns[0].speaker == "Speaker 0"

    def test_duplicate_and_conflicting_corrections(self):
        """When multiple corrections specify the same turn, the last correction wins."""
        turns = [Turn(speaker="Speaker 0", text="बात 1")]
        corrections = [
            {"turn_id": 1, "speaker": "Speaker 1"},
            {"turn_id": 1, "speaker": "Speaker 0"},
        ]
        for corr in corrections:
            t_id = corr.get("turn_id")
            new_spk = corr.get("speaker")
            idx = t_id - 1 if (1 <= t_id <= len(turns)) else t_id
            if 0 <= idx < len(turns):
                turns[idx].speaker = new_spk
        assert turns[0].speaker == "Speaker 0"

    def test_non_dict_corrections_attribute_error_vulnerability(self):
        """Verifies that non-dict elements in corrections do not crash run_sample.

        Remediated: Non-dict items (integers, strings) are safely skipped, and Stage 2
        is shielded with try-except fallback.
        """
        mock_client = MagicMock()
        # Stage 1 generates 20 turns on a 100s clip to trigger Stage 2
        stage1_output = (
            "[Speaker Profiles]\n- Speaker 0: Male\n- Speaker 1: Female\n"
            + "\n".join(
                f"[Utterance] टर्न {i} | [Transition] SHIFT | [Speaker] Speaker {i % 2}"
                for i in range(20)
            )
        )
        mock_client.generate_with_audio.return_value = (stage1_output, 3.0)
        # Hostile Stage 2 returns list of integers
        mock_client.generate_text.return_value = ('{"corrections": [1, 2]}', 1.0)

        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="adversarial_test",
            audio_path="/fake/path.wav",
            duration_seconds=100.0,
            num_speakers=2,
            ground_truth_turns=[],
        )

        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert len(pred.predicted_turns) == 20

    def test_stage2_exception_fallback_to_stage1(self):
        """Verifies that when Stage 2 raises an unhandled exception, it safely falls back to Stage 1."""
        mock_client = MagicMock()
        stage1_output = (
            "[Speaker Profiles]\n- Speaker 0: Male\n- Speaker 1: Female\n"
            + "\n".join(
                f"[Utterance] टर्न {i} | [Transition] SHIFT | [Speaker] Speaker {i % 2}"
                for i in range(20)
            )
        )
        mock_client.generate_with_audio.return_value = (stage1_output, 3.0)
        mock_client.generate_text.side_effect = RuntimeError("Vertex AI 503 Service Unavailable")

        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_fallback",
            audio_path="/fake/path.wav",
            duration_seconds=100.0,
            num_speakers=2,
            ground_truth_turns=[],
        )

        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert pred.sample_id == "test_fallback"
        assert len(pred.predicted_turns) == 20
        assert pred.latency_seconds == 3.0


# =============================================================================
# 5. Pipeline Resilience and Execution Under Degenerate Conditions
# =============================================================================

class TestC5PipelineResilienceAndExecutionAdversarial:
    """Tests end-to-end run_sample with mock client across hostile API responses."""

    def test_run_sample_happy_path_mock(self):
        """Normal execution produces a valid PipelinePrediction."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = (
            "[Speaker Profiles]\n- Speaker 0: Low pitch\n- Speaker 1: High pitch\n\n"
            "[Utterance] नमस्ते | [Transition] SHIFT | [Speaker] Speaker 0\n"
            "[Utterance] हाँ जी | [Transition] SHIFT | [Speaker] Speaker 1\n",
            2.5,
        )
        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_happy",
            audio_path="/fake/test.wav",
            duration_seconds=30.0,
            num_speakers=2,
            ground_truth_turns=[],
        )
        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert pred.sample_id == "test_happy"
        assert pred.approach == "strategy_c5_adaptive_champion"
        assert len(pred.predicted_turns) == 2
        assert pred.latency_seconds == 2.5

    def test_run_sample_empty_stage1_response(self):
        """Empty Stage 1 output (e.g. safety filter block) returns empty turns without crashing."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = ("", 1.0)
        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_empty",
            audio_path="/fake/test.wav",
            duration_seconds=30.0,
            num_speakers=2,
            ground_truth_turns=[],
        )
        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert pred.predicted_turns == []
        assert pred.latency_seconds == 1.0

    def test_run_sample_refusal_stage1_response(self):
        """Model refusal message is captured as a single fallback turn without crashing."""
        mock_client = MagicMock()
        refusal_msg = "I cannot process this audio request due to safety policies."
        mock_client.generate_with_audio.return_value = (refusal_msg, 0.8)
        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_refusal",
            audio_path="/fake/test.wav",
            duration_seconds=30.0,
            num_speakers=2,
            ground_truth_turns=[],
        )
        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert len(pred.predicted_turns) == 1
        assert pred.predicted_turns[0].speaker == "Speaker 0"
        assert pred.predicted_turns[0].text == refusal_msg

    def test_run_sample_html_error_stage1_response(self):
        """HTML error response falls back gracefully without unhandled exceptions."""
        mock_client = MagicMock()
        html_error = "<html><body>503 Service Unavailable</body></html>"
        mock_client.generate_with_audio.return_value = (html_error, 0.5)
        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_html",
            audio_path="/fake/test.wav",
            duration_seconds=30.0,
            num_speakers=2,
            ground_truth_turns=[],
        )
        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert len(pred.predicted_turns) == 1
        assert "503 Service Unavailable" in pred.predicted_turns[0].text

    def test_run_sample_stage2_malformed_json_fallback(self):
        """When Stage 2 returns malformed JSON, Stage 1 turns are preserved intact."""
        mock_client = MagicMock()
        stage1_raw = (
            "[Speaker Profiles]\n- Speaker 0: Low\n- Speaker 1: High\n"
            + "\n".join(
                f"[Utterance] बात {i} | [Transition] SHIFT | [Speaker] Speaker {i % 2}"
                for i in range(20)
            )
        )
        mock_client.generate_with_audio.return_value = (stage1_raw, 3.0)
        # Stage 2 returns garbage
        mock_client.generate_text.return_value = ("{malformed: json", 1.5)
        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_stage2_malformed",
            audio_path="/fake/test.wav",
            duration_seconds=100.0,
            num_speakers=2,
            ground_truth_turns=[],
        )
        pred = pipe.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert len(pred.predicted_turns) == 20
        assert pred.latency_seconds == 4.5

    def test_run_sample_stage1_single_speaker_collapse_triggers_notice(self):
        """Single-speaker collapse injects critical notice in Stage 2 prompt."""
        mock_client = MagicMock()
        stage1_raw = (
            "[Speaker Profiles]\n- Speaker 0: Voice\n- Speaker 1: Voice\n"
            + "\n".join(
                f"[Utterance] एकल वक्ता वाक्य {i} | [Transition] CONTINUE | [Speaker] Speaker 0"
                for i in range(10)
            )
        )
        mock_client.generate_with_audio.return_value = (stage1_raw, 2.0)
        mock_client.generate_text.return_value = ('{"corrections": []}', 1.0)
        pipe = AdaptiveAcousticChampionPipeline(client=mock_client)
        sample = SampleData(
            sample_id="test_collapse",
            audio_path="/fake/test.wav",
            duration_seconds=30.0,
            num_speakers=2,
            ground_truth_turns=[],
        )
        pred = pipe.run_sample(sample)
        assert mock_client.generate_text.called
        call_prompt = mock_client.generate_text.call_args[1]["prompt"]
        assert "CRITICAL NOTICE: Single-speaker collapse detected" in call_prompt


# =============================================================================
# 6. Dynamic Cardinality Scaling
# =============================================================================

class TestC5DynamicCardinalityScalingAdversarial:
    """Tests system prompt generation and metrics evaluation across varying speaker counts."""

    def test_prompt_cardinality_scaling_2_to_5_speakers(self):
        """Generates valid system instructions for N = 2, 3, 4, 5 speakers."""
        for n in [2, 3, 4, 5]:
            prompt = build_c5_stage1_system_instruction(num_speakers=n)
            assert "[Speaker Profiles]" in prompt
            for i in range(n):
                assert f"Speaker {i}" in prompt
            if n > 2:
                assert f"{n} primary speakers" in prompt

    def test_prompt_cardinality_edge_cases_0_and_1(self):
        """Edge cases N = 0 and N = 1 do not crash but exhibit malformed grammar."""
        prompt_1 = build_c5_stage1_system_instruction(num_speakers=1)
        # Phrasing defect: range(0) causes empty join, resulting in ', and Speaker 0'
        assert ", and Speaker 0" in prompt_1

        prompt_0 = build_c5_stage1_system_instruction(num_speakers=0)
        assert ", and Speaker -1" in prompt_0

    def test_metrics_engine_multi_speaker_compatibility(self):
        """Verifies MetricsEngine SAA Hungarian calculation with 3 and 4 speakers."""
        ref_turns_3 = [
            Turn(speaker="Speaker 0", text="नमस्ते मैं बोल रहा हूँ"),
            Turn(speaker="Speaker 1", text="हाँ मैं भी सुन रहा हूँ"),
            Turn(speaker="Speaker 2", text="और मैं भी यहाँ मौजूद हूँ"),
        ]
        # Hypothesis with permuted labels
        hyp_turns_3 = [
            Turn(speaker="Speaker 2", text="नमस्ते मैं बोल रहा हूँ"),
            Turn(speaker="Speaker 0", text="हाँ मैं भी सुन रहा हूँ"),
            Turn(speaker="Speaker 1", text="और मैं भी यहाँ मौजूद हूँ"),
        ]
        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref_turns_3, hyp_turns_3)
        assert acc == 1.0
        assert mapping["Speaker 2"] == "Speaker 0"
        assert mapping["Speaker 0"] == "Speaker 1"
        assert mapping["Speaker 1"] == "Speaker 2"
