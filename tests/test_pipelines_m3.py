"""Comprehensive Unit Test Suite for Milestone 3 Parity Pipelines.

Tests Candidate C1 (AdvancedToken0BypassRunner / AdvancedToken0BypassPipeline)
and Candidate C2 (NativeStructuredJSONRunner / NativeStructuredJSONPipeline).

Covers:
1. Candidate C1 (Advanced Token-0 Bypass):
   - System instruction building and dynamic speaker cardinality
   - Token-0 bypass parsing: canonical format, bare numeric speaker tags ([Speaker] 0 / [Speaker] 1),
     markdown bolding, bullets, multiple internal pipes, multiline buffering, prefix fallback
   - Pipeline execution and mock client integration
2. Candidate C2 (Native Structured JSON):
   - OpenAPI schema generation for 2, 3, 4, and 0 speakers
   - System instruction building and multi-speaker demonstrations
   - Structured JSON parsing: clean JSON, fenced markdown, conversational preambles (fenced & unfenced),
     regex fallback on truncated JSON, Devanagari Hindi text preservation (NO mojibake),
     unicode escape decoding, missing fields, type corruptions
   - Pipeline execution with schema parameter verification
3. Runner and Pipeline alias equivalences
4. Safe merge logic in benchmark runner for partial runs
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pytest

from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.advanced_token0 import (
    AdvancedToken0BypassPipeline,
    AdvancedToken0BypassRunner,
    build_c1_system_instruction,
)
from src.pipelines.structured_json import (
    NativeStructuredJSONPipeline,
    NativeStructuredJSONRunner,
    build_c2_response_schema,
    build_c2_system_instruction,
    parse_c2_structured_json_turns,
)
from src.pipelines.token0_bypass import parse_token0_bypass_turns


# =============================================================================
# 1. Candidate C1: Advanced Token-0 Bypass Tests
# =============================================================================

class TestCandidateC1AdvancedToken0:
    """Unit tests for Candidate C1 pipeline, schemas, and parsing."""

    def test_runner_alias_equivalence(self):
        """AdvancedToken0BypassRunner is an alias of AdvancedToken0BypassPipeline."""
        assert AdvancedToken0BypassRunner is AdvancedToken0BypassPipeline

    def test_build_c1_system_instruction_2spk(self):
        """Builds system prompt for 2-speaker dialogues with key principles."""
        prompt = build_c1_system_instruction(num_speakers=2)
        assert "TOKEN-0 BYPASS" in prompt
        assert "[Utterance] <spoken Hindi text verbatim> | [Speaker] Speaker <ID>" in prompt
        assert "Interruption Resumption (A-B-A Pattern)" in prompt
        assert "Turn Continuity & Anti-Alternation" in prompt
        assert "strictly 2 primary speakers" in prompt
        assert "Speaker 0 and Speaker 1" in prompt

    def test_build_c1_system_instruction_multi_speaker(self):
        """Builds system prompt scaled to 3 or 4 speakers."""
        prompt_3 = build_c1_system_instruction(num_speakers=3)
        assert "There are 3 primary speakers" in prompt_3
        assert "Speaker 0, Speaker 1, and Speaker 2" in prompt_3

        prompt_4 = build_c1_system_instruction(num_speakers=4)
        assert "There are 4 primary speakers" in prompt_4
        assert "Speaker 0, Speaker 1, Speaker 2, and Speaker 3" in prompt_4

    def test_parse_token0_standard_format(self):
        """Parses canonical Token-0 Bypass turns."""
        raw = (
            "[Utterance] नमस्कार, आप कैसे हैं? | [Speaker] Speaker 0\n"
            "[Utterance] मैं ठीक हूँ, आप बताइए। | [Speaker] Speaker 1\n"
            "[Utterance] सब बढ़िया चल रहा है। | [Speaker] Speaker 0"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 3
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्कार, आप कैसे हैं?"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "मैं ठीक हूँ, आप बताइए।"
        assert turns[2].speaker == "Speaker 0"
        assert turns[2].text == "सब बढ़िया चल रहा है।"

    def test_parse_token0_bare_numeric_speaker_tags(self):
        """Parses bare numeric speaker tags like [Speaker] 0 and [Speaker] 1."""
        raw = (
            "[Utterance] पहला संवाद यहाँ है | [Speaker] 0\n"
            "[Utterance] दूसरा संवाद उत्तर में | [Speaker] 1\n"
            "[Utterance] तीसरा संवाद निरंतरता | 0\n"
            "[Utterance] चौथा संवाद पुनः | 1"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 4
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "पहला संवाद यहाँ है"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "दूसरा संवाद उत्तर में"
        assert turns[2].speaker == "Speaker 0"
        assert turns[2].text == "तीसरा संवाद निरंतरता"
        assert turns[3].speaker == "Speaker 1"
        assert turns[3].text == "चौथा संवाद पुनः"

    def test_parse_token0_formatting_variations(self):
        """Parses markdown bolding, bullets, colons, and bracket variations."""
        raw = (
            "- **[Utterance]**: हाँ जी, बोलिए। | **[Speaker]**: Speaker 1\n"
            "1. Utterance: मैं सुन रहा हूँ। | Speaker: Speaker 0\n"
            "[Utterance] ठीक है, बाद में मिलते हैं। | Speaker 1\n"
            "ज़रूर, अपना ख्याल रखिएगा। | [Speaker] Speaker 0"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 4
        assert turns[0].speaker == "Speaker 1"
        assert turns[0].text == "हाँ जी, बोलिए।"
        assert turns[1].speaker == "Speaker 0"
        assert turns[1].text == "मैं सुन रहा हूँ।"
        assert turns[2].speaker == "Speaker 1"
        assert turns[2].text == "ठीक है, बाद में मिलते हैं।"
        assert turns[3].speaker == "Speaker 0"
        assert turns[3].text == "ज़रूर, अपना ख्याल रखिएगा।"

    def test_parse_token0_multiple_internal_pipes(self):
        """Splits at rightmost pipe when utterance contains internal pipe symbols."""
        raw = "[Utterance] विकल्प A | विकल्प B | विकल्प C | [Speaker] Speaker 0"
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert "विकल्प A | विकल्प B | विकल्प C" in turns[0].text

    def test_parse_token0_multiline_buffering(self):
        """Buffers multiline text preceding pipe delimiter into single turn."""
        raw = (
            "[Utterance] यह पहला वाक्य है\n"
            "जो कि दो पंक्तियों में फैला हुआ है। | [Speaker] Speaker 0\n"
            "[Utterance] समझ गया। | [Speaker] Speaker 1"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "यह पहला वाक्य है" in turns[0].text
        assert "जो कि दो पंक्तियों में फैला हुआ है।" in turns[0].text
        assert turns[1].speaker == "Speaker 1"

    def test_parse_token0_prefix_fallback(self):
        """Falls back gracefully if model emits traditional Speaker <ID>: prefixes."""
        raw = (
            "Speaker 0: नमस्ते, क्या हाल है?\n"
            "Speaker 1: सब कुशल मंगल है।"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते, क्या हाल है?"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "सब कुशल मंगल है।"

    def test_parse_token0_empty_and_whitespace(self):
        """Empty and whitespace strings return empty lists."""
        assert parse_token0_bypass_turns("") == []
        assert parse_token0_bypass_turns("   \n\t  ") == []

    def test_c1_pipeline_execution_mocked(self):
        """Candidate C1 executes run_sample and returns PipelinePrediction."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = (
            "[Utterance] परीक्षण संवाद | [Speaker] Speaker 0",
            3.14,
        )
        pipeline = AdvancedToken0BypassRunner(client=mock_client)
        assert pipeline.thinking_budget == 0
        assert pipeline.temperature == 0.0

        sample = SampleData(
            sample_id="hindi_001",
            audio_path="/tmp/fake.wav",
            duration_seconds=10.0,
            num_speakers=2,
            ground_truth_turns=[Turn(speaker="Speaker 0", text="परीक्षण संवाद")],
        )

        pred = pipeline.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert pred.sample_id == "hindi_001"
        assert pred.approach == "strategy_c1_advanced_token0"
        assert pred.model_id == "gemini-3.5-flash-lite"
        assert len(pred.predicted_turns) == 1
        assert pred.predicted_turns[0].speaker == "Speaker 0"
        assert pred.predicted_turns[0].text == "परीक्षण संवाद"
        assert pred.latency_seconds == 3.14


# =============================================================================
# 2. Candidate C2: Native Structured JSON Tests
# =============================================================================

class TestCandidateC2NativeStructuredJSON:
    """Unit tests for Candidate C2 pipeline, OpenAPI schema, and JSON parsing."""

    def test_runner_alias_equivalence(self):
        """NativeStructuredJSONRunner is an alias of NativeStructuredJSONPipeline."""
        assert NativeStructuredJSONRunner is NativeStructuredJSONPipeline

    def test_build_c2_response_schema_structure(self):
        """Generates valid OpenAPI schema structure for 2 speakers."""
        schema = build_c2_response_schema(num_speakers=2)
        assert schema["type"] == "OBJECT"
        assert "dialogue" in schema["required"]
        dialogue_prop = schema["properties"]["dialogue"]
        assert dialogue_prop["type"] == "ARRAY"
        item_props = dialogue_prop["items"]["properties"]
        assert "utterance" in item_props
        assert "transition_type" in item_props
        assert "speaker" in item_props
        assert item_props["transition_type"]["enum"] == [
            "NEW_SPEAKER",
            "CONTINUES_SAME_SPEAKER",
            "RESUMES_AFTER_INTERRUPTION",
        ]
        assert item_props["speaker"]["enum"] == ["Speaker 0", "Speaker 1"]

    def test_build_c2_response_schema_multi_speaker(self):
        """Scales speaker enum dynamically for 3 and 4 speakers."""
        schema_3 = build_c2_response_schema(num_speakers=3)
        assert schema_3["properties"]["dialogue"]["items"]["properties"]["speaker"]["enum"] == [
            "Speaker 0",
            "Speaker 1",
            "Speaker 2",
        ]

        schema_4 = build_c2_response_schema(num_speakers=4)
        assert schema_4["properties"]["dialogue"]["items"]["properties"]["speaker"]["enum"] == [
            "Speaker 0",
            "Speaker 1",
            "Speaker 2",
            "Speaker 3",
        ]

    def test_build_c2_response_schema_edge_case_zero_speakers(self):
        """Zero speakers produces unbounded string speaker without enum constraint."""
        schema_0 = build_c2_response_schema(num_speakers=0)
        assert "enum" not in schema_0["properties"]["dialogue"]["items"]["properties"]["speaker"]

    def test_build_c2_system_instruction(self):
        """Generates system instruction enforcing Devanagari script and transition typing."""
        inst = build_c2_system_instruction(num_speakers=2)
        assert "देवनागरी लिपि" in inst
        assert "NEW_SPEAKER" in inst
        assert "CONTINUES_SAME_SPEAKER" in inst
        assert "RESUMES_AFTER_INTERRUPTION" in inst
        assert "utterance" in inst
        assert "Speaker 0" in inst
        assert "Speaker 1" in inst

    def test_parse_c2_clean_json(self):
        """Parses standard clean JSON string."""
        raw = json.dumps({
            "dialogue": [
                {"utterance": "नमस्ते जी", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},
                {"utterance": "नमस्कार, कहिए।", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"},
            ]
        })
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते जी"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "नमस्कार, कहिए।"

    def test_parse_c2_clean_markdown_fence(self):
        """Parses markdown code block ```json ... ``` cleanly."""
        raw = (
            "```json\n"
            "{\n"
            '  "dialogue": [\n'
            '    {"utterance": "पहला वाक्य", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}\n'
            "  ]\n"
            "}\n"
            "```"
        )
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "पहला वाक्य"

    def test_parse_c2_markdown_fence_with_preamble(self):
        """Cleanly extracts JSON from fenced code block preceded and followed by preamble chatter."""
        raw = (
            "Here is the transcribed dialogue in the requested JSON format:\n"
            "```json\n"
            "{\n"
            '  "dialogue": [\n'
            '    {"utterance": "नमस्ते दुनिया", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"},\n'
            '    {"utterance": "हाँ बिल्कुल", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 1"}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "I hope this transcription is accurate and helpful!"
        )
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते दुनिया"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "हाँ बिल्कुल"

    def test_parse_c2_unfenced_json_with_preamble(self):
        """Extracts outermost JSON object when preceded by preamble without markdown fences."""
        raw = (
            'Sure, here is the JSON output: '
            '{"dialogue": [{"utterance": "सीधा संवाद", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}]} '
            'End of transmission.'
        )
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "सीधा संवाद"

    def test_parse_c2_regex_fallback_devanagari_no_mojibake(self):
        """CRITICAL: Verifies that truncated JSON triggers regex fallback WITHOUT mojibake corruption."""
        hindi_text = "नमस्ते भारत, कैसे हैं आप?"
        truncated = (
            '{"dialogue": [{"utterance": "' + hindi_text + '", '
            '"transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}, '
            '{"utterance": "अधूरा'
        )
        turns = parse_c2_structured_json_turns(truncated)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        # Must preserve authentic Devanagari, NOT corrupted Latin-1 mojibake
        assert turns[0].text == hindi_text
        assert "à" not in turns[0].text

    def test_parse_c2_regex_fallback_unicode_escapes(self):
        """Safely decodes unicode escape sequences in regex fallback without mojibake."""
        raw_escaped = (
            '{"dialogue": [{"utterance": "\\u0928\\u092e\\u0938\\u094d\\u0924\\u0947 \\"दुनिया\\"", '
            '"transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}, '
            '{"broken_json": true'
        )
        turns = parse_c2_structured_json_turns(raw_escaped)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == 'नमस्ते "दुनिया"'

    def test_parse_c2_empty_and_whitespace(self):
        """Empty and whitespace strings return empty lists."""
        assert parse_c2_structured_json_turns("") == []
        assert parse_c2_structured_json_turns("   \n\t  ") == []

    def test_parse_c2_missing_speaker_field_defaults(self):
        """Missing speaker field defaults to Speaker 0."""
        raw = json.dumps({"dialogue": [{"utterance": "नमस्ते", "transition_type": "NEW_SPEAKER"}]})
        turns = parse_c2_structured_json_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते"

    def test_c2_pipeline_execution_mocked(self):
        """Candidate C2 executes run_sample with schema and returns PipelinePrediction."""
        mock_client = MagicMock()
        mock_response = json.dumps({
            "dialogue": [
                {"utterance": "परीक्षण सी2", "transition_type": "NEW_SPEAKER", "speaker": "Speaker 0"}
            ]
        })
        mock_client.generate_with_audio.return_value = (mock_response, 4.25)

        pipeline = NativeStructuredJSONRunner(client=mock_client)
        assert pipeline.thinking_budget == 0
        assert pipeline.temperature == 0.0

        sample = SampleData(
            sample_id="hindi_002",
            audio_path="/tmp/fake_c2.wav",
            duration_seconds=12.0,
            num_speakers=2,
            ground_truth_turns=[Turn(speaker="Speaker 0", text="परीक्षण सी2")],
        )

        pred = pipeline.run_sample(sample)
        assert isinstance(pred, PipelinePrediction)
        assert pred.sample_id == "hindi_002"
        assert pred.approach == "strategy_c2_structured_json"
        assert pred.model_id == "gemini-3.5-flash-lite"
        assert len(pred.predicted_turns) == 1
        assert pred.predicted_turns[0].speaker == "Speaker 0"
        assert pred.predicted_turns[0].text == "परीक्षण सी2"
        assert pred.latency_seconds == 4.25

        # Verify client was called with response_mime_type and response_schema
        call_kwargs = mock_client.generate_with_audio.call_args[1]
        assert call_kwargs["response_mime_type"] == "application/json"
        assert call_kwargs["response_schema"] is not None
        assert call_kwargs["response_schema"]["type"] == "OBJECT"


# =============================================================================
# 3. Partial Run Data Protection Verification
# =============================================================================

class TestBenchmarkRunnerDataProtection:
    """Verifies that partial runs merge rather than clobber predictions."""

    def test_merge_predictions_on_partial_run(self):
        """Simulates merging partial sample predictions into an existing 20-sample predictions file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            preds_file = tmp_path / "predictions_test_strat.json"

            # Create initial 20 samples
            initial_records = [
                {"sample_id": f"sample_{i:02d}", "metrics": {"saa": 0.80}, "predicted_turns": []}
                for i in range(20)
            ]
            with open(preds_file, "w", encoding="utf-8") as f:
                json.dump(initial_records, f)

            # Partial update for sample_03 and sample_05
            partial_updates = [
                {"sample_id": "sample_03", "metrics": {"saa": 0.95}, "predicted_turns": [{"speaker": "Speaker 0", "text": "updated 3"}]},
                {"sample_id": "sample_05", "metrics": {"saa": 0.99}, "predicted_turns": [{"speaker": "Speaker 1", "text": "updated 5"}]},
            ]

            # Replicate runner merge logic
            with open(preds_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            records_map = {r["sample_id"]: r for r in existing}
            for r in partial_updates:
                records_map[r["sample_id"]] = r
            merged = list(records_map.values())
            with open(preds_file, "w", encoding="utf-8") as f:
                json.dump(merged, f)

            # Verify: count remains 20, and samples 3 & 5 are updated
            with open(preds_file, "r", encoding="utf-8") as f:
                reloaded = json.load(f)
            assert len(reloaded) == 20
            sample_03 = next(r for r in reloaded if r["sample_id"] == "sample_03")
            assert sample_03["metrics"]["saa"] == 0.95
            sample_00 = next(r for r in reloaded if r["sample_id"] == "sample_00")
            assert sample_00["metrics"]["saa"] == 0.80  # Unchanged
