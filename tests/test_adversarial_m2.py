"""Empirical adversarial stress testing suite for Milestone 2 pipelines and metrics.

Covers:
1. TestToken0BypassParserEdgeCases:
   - Empty, whitespace, tabs, newlines, null bytes, zero-width chars
   - Missing pipes: plain text, traditional prefixes, missing pipe in some/all lines
   - Inverted brackets: [Speaker 0] | [Utterance] text, Speaker 0 | text
   - Non-standard speaker labels: Devanagari numerals, spk_X, Person X, Participant X, Agent/Customer
   - Multiline turns, trailing turns, internal pipes inside utterances
   - Consecutive same-speaker turns (anti-alternation)
   - LLM anomalies: code fences, JSON blocks, refusals, asterisks, hashtags
2. TestPureLiteTwoPassParserEdgeCases:
   - Pass 1: unattributed acoustic turns with Turn N:, Turn १:, चरण N:, plain text
   - Pass 2: attributed turns with Speaker N:, वक्ता N:, multiline, turn splitting
3. TestPipelineExecutionAndFallback:
   - Token0BypassPipeline non-crash on empty, whitespace, malformed, and refusal responses
   - PureLiteTwoPassPipeline non-crash on Pass 1/Pass 2 empty or garbage outputs
   - Pure 3.5 Lite constraint verification (model_id, thinking_budget=0)
4. TestMetricsEngineAdversarialEdgeCases:
   - 0 predicted vs N reference turns
   - N predicted vs 0 reference turns
   - 0 predicted and 0 reference turns
   - Punctuation-only turns that normalize to empty
   - Extreme speaker count disparity (10 hyp vs 2 ref, 1 hyp vs 5 ref)
   - Tie-breaking in confusion matrix and cost matrix
   - Mathematical invariants: 0<=SAA<=1, WER>=0, cpWER>=0, diarization_gap>=0
5. TestLivePredictionsDataIntegrity:
   - Reparsing audit and metric recalculation audit of live prediction JSON files
   - Cross-check against hard_2spk_comparative_summary.json
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import EvaluationMetrics, SampleData, Turn
from src.normalizer import IndicTextNormalizer
from src.pipelines.pure_lite_twopass import (
    PureLiteTwoPassPipeline,
    parse_attributed_turns,
    parse_unattributed_turns,
)
from src.pipelines.token0_bypass import (
    Token0BypassPipeline,
    parse_token0_bypass_turns,
)


# =============================================================================
# 1. Strategy A: Token-0 Bypass Parser Edge Cases
# =============================================================================

class TestToken0BypassParserEdgeCases:
    """Stress-test parse_token0_bypass_turns with degenerate, malformed, and hostile inputs."""

    def test_empty_and_whitespace_variants(self):
        """Empty, whitespace, tabs, and newlines return empty list."""
        assert parse_token0_bypass_turns("") == []
        assert parse_token0_bypass_turns("   ") == []
        assert parse_token0_bypass_turns("\t\t\n\r\n  \t  ") == []

    def test_null_bytes_and_invisible_characters(self):
        """Null bytes and zero-width characters do not crash parser."""
        raw = "[Utterance] नम\x00स्ते \u200bदुनिया\ufeff | [Speaker] Speaker 0"
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert "नम" in turns[0].text
        assert "दुनिया" in turns[0].text

    def test_missing_pipes_plain_text(self):
        """Plain Hindi text without any pipe falls back gracefully to Speaker 0."""
        raw = "नमस्ते दोस्तों आज के इस पॉडकास्ट में आपका स्वागत है।"
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == raw

    def test_missing_pipes_traditional_prefixes(self):
        """Traditional Speaker 0: prefixes without pipes are parsed via fallback."""
        raw = (
            "Speaker 0: नमस्ते, क्या मेरी बात शर्मा जी से हो रही है?\n"
            "Speaker 1: जी हाँ, बोलिए क्या बात है?\n"
            "Speaker 0: आपकी ईएमआई का भुगतान बाकी है।"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 3
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "नमस्ते, क्या मेरी बात शर्मा जी से हो रही है?"
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "जी हाँ, बोलिए क्या बात है?"
        assert turns[2].speaker == "Speaker 0"

    def test_missing_pipes_utterance_tag_only(self):
        """Lines starting with [Utterance] without pipes buffer and form valid turns."""
        raw = (
            "[Utterance] यह पहला वाक्य है\n"
            "[Utterance] यह दूसरा वाक्य है"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) >= 1
        assert turns[0].speaker == "Speaker 0"
        assert "पहला वाक्य" in turns[0].text

    def test_partial_missing_pipes_in_dialogue(self):
        """Some lines have pipes while other lines do not (continuation / missing pipe)."""
        raw = (
            "[Utterance] नमस्ते जी | [Speaker] Speaker 0\n"
            "यह बात ध्यान देने योग्य है कि ऋण समय पर चुकाया जाना चाहिए।\n"
            "[Utterance] बिल्कुल सही कहा आपने | [Speaker] Speaker 1"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "ऋण समय पर चुकाया जाना चाहिए" in turns[0].text
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "बिल्कुल सही कहा आपने"

    def test_internal_pipes_in_utterance_text(self):
        """Utterances containing pipes internally (e.g. options A | B) split on last pipe."""
        raw = (
            "[Utterance] विकल्प १ | विकल्प २ | विकल्प ३ | [Speaker] Speaker 0\n"
            "[Utterance] मैं विकल्प १ चुनता हूँ | [Speaker] Speaker 1"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "विकल्प १ | विकल्प २ | विकल्प ३" in turns[0].text
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "मैं विकल्प १ चुनता हूँ"

    def test_devanagari_numerals_and_labels(self):
        """Devanagari numerals and वक्ता labels are correctly extracted."""
        raw = (
            "[Utterance] नमस्ते | [Speaker] वक्ता ०\n"
            "[Utterance] नमस्कार जी | [Speaker] वक्ता १\n"
            "[Utterance] क्या हालचाल? | [Speaker] Speaker ०\n"
            "[Utterance] सब बढ़िया | [Speaker] Speaker १"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 4
        assert turns[0].speaker == "वक्ता ०"
        assert turns[0].text == "नमस्ते"
        assert turns[1].speaker == "वक्ता १"
        assert turns[1].text == "नमस्कार जी"
        assert turns[2].speaker == "Speaker ०"
        assert turns[2].text == "क्या हालचाल?"
        assert turns[3].speaker == "Speaker १"
        assert turns[3].text == "सब बढ़िया"

    def test_various_speaker_prefixes(self):
        """Supports Person X, Participant X, spk_X, Agent, and Customer."""
        raw = (
            "[Utterance] हेल्लो | [Speaker] Person 1\n"
            "[Utterance] हाँ जी | [Speaker] Participant 2\n"
            "[Utterance] क्या समस्या है? | [Speaker] Agent\n"
            "[Utterance] इंटरनेट बंद है | [Speaker] Customer\n"
            "[Utterance] जाँच रहा हूँ | [Speaker] spk_01"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 5
        assert turns[0].speaker == "Person 1"
        assert turns[1].speaker == "Participant 2"
        assert turns[2].speaker == "Agent"
        assert turns[3].speaker == "Customer"
        assert turns[4].speaker == "spk_01"

    def test_inverted_brackets_graceful_fallback(self):
        """Inverted order [Speaker 0] | [Utterance] text does not crash and yields turns."""
        raw = (
            "[Speaker] Speaker 0 | [Utterance] नमस्ते\n"
            "[Speaker] Speaker 1 | [Utterance] नमस्कार"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) >= 1
        assert all(isinstance(t, Turn) for t in turns)
        assert all(len(t.text) > 0 for t in turns)

    def test_multiline_buffering_before_pipe(self):
        """Long utterance split across 4 lines before pipe emits as single turn."""
        raw = (
            "[Utterance] यह एक बहुत लम्बा वाक्य है\n"
            "जो कई पंक्तियों में फैला हुआ है\n"
            "और अंत में जाकर समाप्त होता है\n"
            "यहाँ पर। | [Speaker] Speaker 0\n"
            "[Utterance] समझ आ गया। | [Speaker] Speaker 1"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "यह एक बहुत लम्बा वाक्य है" in turns[0].text
        assert "जो कई पंक्तियों में फैला हुआ है" in turns[0].text
        assert "यहाँ पर।" in turns[0].text
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "समझ आ गया।"

    def test_consecutive_same_speaker_turns_preserved(self):
        """Preserves consecutive same-speaker turns (anti-alternation constraint)."""
        raw = "\n".join([
            f"[Utterance] वाक्य संख्या {i} यहाँ है। | [Speaker] Speaker 0"
            for i in range(10)
        ])
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 10
        for i, t in enumerate(turns):
            assert t.speaker == "Speaker 0"
            assert f"वाक्य संख्या {i}" in t.text

    def test_markdown_code_fences_and_headers(self):
        """Markdown code fences, bolding, and headers are parsed without crash."""
        raw = (
            "```markdown\n"
            "### Dialogue Transcript\n"
            "- **[Utterance]**: नमस्ते | **[Speaker]**: Speaker 0\n"
            "- **[Utterance]**: नमस्कार जी | **[Speaker]**: Speaker 1\n"
            "```"
        )
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "नमस्ते" in turns[0].text
        assert turns[1].speaker == "Speaker 1"
        assert "नमस्कार जी" in turns[1].text

    def test_refusal_and_hallucinated_text(self):
        """Model refusal or generic hallucination falls back to a single turn gracefully."""
        raw = "I am an AI assistant and cannot transcribe this audio."
        turns = parse_token0_bypass_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == raw


# =============================================================================
# 2. Strategy B: Pure Lite Two-Pass Parser Edge Cases
# =============================================================================

class TestPureLiteTwoPassParserEdgeCases:
    """Stress-test parse_unattributed_turns and parse_attributed_turns."""

    def test_pass1_empty_and_whitespace(self):
        """Pass 1 parser returns empty list on empty/whitespace."""
        assert parse_unattributed_turns("") == []
        assert parse_unattributed_turns("   \n\t  ") == []

    def test_pass1_devanagari_numerals_and_charan(self):
        """Pass 1 handles Devanagari numerals and Hindi 'चरण' prefix."""
        raw = (
            "Turn १: पहला वाक्य\n"
            "Turn २: दूसरा वाक्य\n"
            "चरण ३: तीसरा वाक्य"
        )
        turns = parse_unattributed_turns(raw)
        assert len(turns) == 3
        assert turns[0] == "पहला वाक्य"
        assert turns[1] == "दूसरा वाक्य"
        assert turns[2] == "तीसरा वाक्य"

    def test_pass1_plain_paragraphs_no_prefixes(self):
        """Pass 1 without Turn N: prefixes groups continuation text into single segment."""
        raw = (
            "यह पहला वाक्य है।\n"
            "यह दूसरा वाक्य है।"
        )
        turns = parse_unattributed_turns(raw)
        assert len(turns) == 1
        assert "पहला वाक्य" in turns[0]
        assert "दूसरा वाक्य" in turns[0]

    def test_pass2_empty_and_whitespace(self):
        """Pass 2 parser returns empty list on empty/whitespace."""
        assert parse_attributed_turns("") == []
        assert parse_attributed_turns("   \n\t  ") == []

    def test_pass2_devanagari_speaker_labels(self):
        """Pass 2 handles वक्ता ० and वक्ता १."""
        raw = (
            "वक्ता ०: नमस्ते, कैसे हैं?\n"
            "वक्ता १: मैं ठीक हूँ।"
        )
        turns = parse_attributed_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "वक्ता ०"
        assert turns[0].text == "नमस्ते, कैसे हैं?"
        assert turns[1].speaker == "वक्ता १"
        assert turns[1].text == "मैं ठीक हूँ।"

    def test_pass2_multiline_continuation(self):
        """Pass 2 handles utterances continuing over multiple un-prefixed lines."""
        raw = (
            "Speaker 0: यह पहला वाक्य है\n"
            "जो कि आगे भी जारी रहता है\n"
            "Speaker 1: हाँ जी, मैंने सुन लिया।"
        )
        turns = parse_attributed_turns(raw)
        assert len(turns) == 2
        assert turns[0].speaker == "Speaker 0"
        assert "यह पहला वाक्य है" in turns[0].text
        assert "जो कि आगे भी जारी रहता है" in turns[0].text
        assert turns[1].speaker == "Speaker 1"
        assert turns[1].text == "हाँ जी, मैंने सुन लिया।"

    def test_pass2_consecutive_same_speaker_turns(self):
        """Pass 2 preserves consecutive turns by same speaker."""
        raw = (
            "Speaker 0: पहला विचार।\n"
            "Speaker 0: दूसरा विचार इसी वक्ता का।\n"
            "Speaker 1: ठीक है।"
        )
        turns = parse_attributed_turns(raw)
        assert len(turns) == 3
        assert turns[0].speaker == "Speaker 0"
        assert turns[0].text == "पहला विचार।"
        assert turns[1].speaker == "Speaker 0"
        assert turns[1].text == "दूसरा विचार इसी वक्ता का।"
        assert turns[2].speaker == "Speaker 1"
        assert turns[2].text == "ठीक है।"

    def test_pass2_colon_fallback_single_line(self):
        """Pass 2 colon fallback correctly captures non-standard header on line 1."""
        raw = "Doctor: मरीज की हालत स्थिर है।"
        turns = parse_attributed_turns(raw)
        assert len(turns) == 1
        assert turns[0].speaker == "Doctor"
        assert turns[0].text == "मरीज की हालत स्थिर है।"


# =============================================================================
# 3. Pipeline Fallback & Non-Crash Invariants
# =============================================================================

class TestPipelineExecutionAndFallback:
    """Stress-test pipeline execution under simulated API failure/edge responses."""

    def test_token0_bypass_pipeline_empty_api_response(self):
        """Pipeline returns empty predicted_turns when API returns empty string."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = ("", 0.5)

        pipeline = Token0BypassPipeline(client=mock_client)
        sample = SampleData(
            sample_id="dummy_01",
            audio_path="dummy.wav",
            duration_seconds=10.0,
            num_speakers=2,
            overlap_ratio=5.0,
            ground_truth_turns=[Turn(speaker="Speaker 0", text="नमस्ते")],
        )

        pred = pipeline.run_sample(sample)
        assert pred.sample_id == "dummy_01"
        assert pred.predicted_turns == []
        assert pred.latency_seconds == 0.5

    def test_token0_bypass_pipeline_malformed_api_response(self):
        """Pipeline falls back gracefully on unparseable refusal response."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = (
            "ERROR: Audio rate limit exceeded or unprocessable.", 1.2
        )

        pipeline = Token0BypassPipeline(client=mock_client)
        sample = SampleData(
            sample_id="dummy_02",
            audio_path="dummy.wav",
            duration_seconds=10.0,
            num_speakers=2,
            overlap_ratio=5.0,
            ground_truth_turns=[Turn(speaker="Speaker 0", text="नमस्ते")],
        )

        pred = pipeline.run_sample(sample)
        assert len(pred.predicted_turns) == 1
        assert pred.predicted_turns[0].speaker == "Speaker 0"
        assert "Audio rate limit" in pred.predicted_turns[0].text

    def test_pure_lite_twopass_empty_pass1_fallback(self):
        """PureLiteTwoPassPipeline does not crash when Pass 1 emits empty string."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = ("", 1.0)
        mock_client.generate_text.return_value = ("", 0.5)

        pipeline = PureLiteTwoPassPipeline(client=mock_client)
        sample = SampleData(
            sample_id="dummy_03",
            audio_path="dummy.wav",
            duration_seconds=10.0,
            num_speakers=2,
            overlap_ratio=5.0,
            ground_truth_turns=[Turn(speaker="Speaker 0", text="नमस्ते")],
        )

        pred = pipeline.run_sample(sample)
        assert pred.sample_id == "dummy_03"
        assert pred.predicted_turns == []
        assert pred.latency_seconds == 1.5

    def test_pure_lite_twopass_empty_pass2_fallback(self):
        """PureLiteTwoPassPipeline does not crash when Pass 2 emits empty string."""
        mock_client = MagicMock()
        mock_client.generate_with_audio.return_value = ("Turn 1: नमस्ते", 1.0)
        mock_client.generate_text.return_value = ("", 0.5)

        pipeline = PureLiteTwoPassPipeline(client=mock_client)
        sample = SampleData(
            sample_id="dummy_04",
            audio_path="dummy.wav",
            duration_seconds=10.0,
            num_speakers=2,
            overlap_ratio=5.0,
            ground_truth_turns=[Turn(speaker="Speaker 0", text="नमस्ते")],
        )

        pred = pipeline.run_sample(sample)
        assert pred.sample_id == "dummy_04"
        assert pred.predicted_turns == []
        assert pred.latency_seconds == 1.5

    def test_pure_lite_pure_model_invariants(self):
        """PureLiteTwoPassPipeline strictly uses gemini-3.5-flash-lite on BOTH passes."""
        pipeline = PureLiteTwoPassPipeline()
        assert pipeline.model_id == "gemini-3.5-flash-lite"
        assert pipeline.thinking_budget == 0
        assert pipeline.temperature == 0.0


# =============================================================================
# 4. Metric Engine Calculations on Edge-Case Predictions
# =============================================================================

class TestMetricsEngineAdversarialEdgeCases:
    """Stress-test MetricsEngine with extreme boundary condition predictions."""

    def test_zero_predicted_turns_vs_reference(self):
        """0 predicted turns against non-empty ground truth turns."""
        ref = [
            Turn(speaker="Speaker 0", text="नमस्ते क्या हाल है"),
            Turn(speaker="Speaker 1", text="सब ठीक है धन्यवाद"),
        ]
        hyp = []

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        assert metrics.wer == 1.0
        assert metrics.cer == 1.0
        assert metrics.speaker_attribution_accuracy == 0.0
        assert metrics.cpwer == 1.0
        assert metrics.diarization_gap == 0.0
        assert metrics.format_valid is False

    def test_reference_empty_vs_predicted_turns(self):
        """Non-empty predicted turns against empty ground truth turns."""
        ref = []
        hyp = [Turn(speaker="Speaker 0", text="नमस्ते भारत")]

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        assert metrics.wer == 1.0
        assert metrics.cer == 1.0
        assert metrics.speaker_attribution_accuracy == 0.0
        assert metrics.cpwer == 1.0
        assert metrics.diarization_gap == 0.0
        assert metrics.format_valid is True

    def test_both_reference_and_predicted_turns_empty(self):
        """Both reference and hypothesis are empty."""
        ref = []
        hyp = []

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        assert metrics.wer == 0.0
        assert metrics.cer == 0.0
        assert metrics.speaker_attribution_accuracy == 1.0
        assert metrics.cpwer == 0.0
        assert metrics.diarization_gap == 0.0
        assert metrics.format_valid is False

    def test_turns_normalizing_to_empty_strings(self):
        """Turns containing only punctuation or filler tags that normalize to empty."""
        ref = [Turn(speaker="Speaker 0", text="। ॥ , . ! ?")]
        hyp = [Turn(speaker="Speaker 0", text="[Laughter] <applause>")]

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        # Normalizer strips punctuation and tags, leaving both texts empty
        assert metrics.wer == 0.0
        assert metrics.cer == 0.0
        assert metrics.speaker_attribution_accuracy == 1.0
        assert metrics.cpwer == 0.0
        assert metrics.diarization_gap == 0.0

    def test_extreme_speaker_count_disparity_10_hyp_vs_2_ref(self):
        """Hypothesis hallucinates 10 speakers for 2 reference speakers."""
        ref = [
            Turn(speaker="Speaker 0", text="एक दो तीन चार पाँच छह सात आठ नौ दस"),
            Turn(speaker="Speaker 1", text="ग्यारह बारह तेरह चौदह पंद्रह"),
        ]
        # Split across 10 hypothesis speakers
        hyp = [
            Turn(speaker=f"Speaker {i}", text=f"शब्द {i}")
            for i in range(10)
        ]

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        assert 0.0 <= metrics.speaker_attribution_accuracy <= 1.0
        assert metrics.wer > 0.0
        assert metrics.cpwer > 0.0
        assert metrics.diarization_gap >= 0.0
        assert metrics.format_valid is True

    def test_extreme_speaker_count_disparity_1_hyp_vs_5_ref(self):
        """Hypothesis collapses 5 reference speakers into a single speaker."""
        ref = [
            Turn(speaker=f"Speaker {i}", text=f"बातचीत संख्या {i}")
            for i in range(5)
        ]
        hyp = [
            Turn(speaker="Speaker 0", text=" ".join(f"बातचीत संख्या {i}" for i in range(5)))
        ]

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        # WER is 0.0 because all words match perfectly
        assert metrics.wer == pytest.approx(0.0)
        # SAA matches only the largest reference speaker (1 out of 5, each 3 words = 3/15 = 0.20)
        assert metrics.speaker_attribution_accuracy == pytest.approx(0.20)
        # cpWER reflects 4 unmatched speakers
        assert metrics.cpwer > 0.0
        assert metrics.diarization_gap >= 0.0

    def test_tie_breaking_equal_costs_stability(self):
        """Confusion matrix with identical counts breaks ties deterministically."""
        ref = [
            Turn(speaker="Speaker 0", text="समान शब्द संख्या"),
            Turn(speaker="Speaker 1", text="समान शब्द संख्या"),
        ]
        hyp = [
            Turn(speaker="Speaker A", text="समान शब्द संख्या"),
            Turn(speaker="Speaker B", text="समान शब्द संख्या"),
        ]

        metrics = MetricsEngine.evaluate_sample(ref, hyp, normalize=True)
        assert metrics.wer == pytest.approx(0.0)
        assert metrics.speaker_attribution_accuracy == pytest.approx(1.0)
        assert metrics.cpwer == pytest.approx(0.0)
        assert metrics.diarization_gap == pytest.approx(0.0)
        assert len(metrics.speaker_mapping) == 2

    def test_mathematical_invariants_property_test(self):
        """Verifies mathematical invariants across randomized edge cases."""
        vocab = ["नमस्ते", "आप", "कैसे", "हैं", "सब", "ठीक", "है"]
        import random
        rng = random.Random(42)

        for _ in range(30):
            n_ref = rng.randint(1, 5)
            n_hyp = rng.randint(1, 5)

            ref_turns = [
                Turn(speaker=f"spk_{rng.randint(0, 2)}", text=" ".join(rng.choices(vocab, k=rng.randint(1, 6))))
                for _ in range(n_ref)
            ]
            hyp_turns = [
                Turn(speaker=f"hyp_{rng.randint(0, 3)}", text=" ".join(rng.choices(vocab, k=rng.randint(1, 6))))
                for _ in range(n_hyp)
            ]

            metrics = MetricsEngine.evaluate_sample(ref_turns, hyp_turns, normalize=True)
            assert 0.0 <= metrics.speaker_attribution_accuracy <= 1.0
            assert metrics.wer >= 0.0
            assert metrics.cpwer >= 0.0
            assert metrics.diarization_gap >= 0.0
            assert metrics.diarization_gap == pytest.approx(max(0.0, metrics.cpwer - metrics.wer))


# =============================================================================
# 5. Live Predictions Data Integrity & Audit
# =============================================================================

class TestLivePredictionsDataIntegrity:
    """Audits the raw prediction JSON files in results/hard_2spk/."""

    @pytest.fixture(scope="class")
    @classmethod
    def dataset_samples(cls):
        """Loads the ground-truth samples from data/hard_2spk_subset/."""
        return DatasetLoader.load_benchmark_subset(subset_dir="data/hard_2spk_subset")

    def test_strategy_a_token0_bypass_raw_predictions_integrity(self, dataset_samples):
        """Audits Strategy A raw prediction file: reparsing fidelity and metric recalculation."""
        path = Path("results/hard_2spk/predictions_strategy_a_token0_bypass.json")
        assert path.exists(), f"Prediction file missing at {path}"

        with open(path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        assert len(predictions) == 20
        samples_by_id = {s.sample_id: s for s in dataset_samples}

        for p in predictions:
            sid = p["sample_id"]
            assert sid in samples_by_id
            assert p["approach"] == "token0_bypass"
            assert p["model_id"] == "gemini-3.5-flash-lite"
            assert p["latency_seconds"] > 0.0
            assert len(p["predicted_turns"]) > 0

            # 1. Verify parser determinism: raw_response parses to predicted_turns
            reparsed = parse_token0_bypass_turns(p["raw_response"])
            assert len(reparsed) == len(p["predicted_turns"])
            for t_rep, t_sto in zip(reparsed, p["predicted_turns"]):
                assert t_rep.speaker == t_sto["speaker"]
                assert t_rep.text == t_sto["text"]

            # 2. Verify metric engine recalculation matches stored metrics
            gt_turns = samples_by_id[sid].ground_truth_turns
            hyp_turns = [Turn(**t) for t in p["predicted_turns"]]
            recalc = MetricsEngine.evaluate_sample(gt_turns, hyp_turns, normalize=True)
            stored = p["metrics"]

            assert abs(recalc.wer - stored["wer"]) < 0.001
            assert abs(recalc.speaker_attribution_accuracy - stored["speaker_attribution_accuracy"]) < 0.001
            assert abs(recalc.cpwer - stored["cpwer"]) < 0.001
            assert abs(recalc.diarization_gap - stored["diarization_gap"]) < 0.001

    def test_strategy_b_pure_lite_twopass_raw_predictions_integrity(self, dataset_samples):
        """Audits Strategy B raw prediction file: pass 2 reparsing and metric recalculation."""
        path = Path("results/hard_2spk/predictions_strategy_b_pure_lite_twopass.json")
        assert path.exists(), f"Prediction file missing at {path}"

        with open(path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

        assert len(predictions) == 20
        samples_by_id = {s.sample_id: s for s in dataset_samples}

        for p in predictions:
            sid = p["sample_id"]
            assert sid in samples_by_id
            assert p["approach"] == "pure_lite_twopass"
            assert p["model_id"] == "gemini-3.5-flash-lite+gemini-3.5-flash-lite"
            assert p["latency_seconds"] > 0.0
            assert len(p["predicted_turns"]) > 0

            # 1. Verify pass 2 parsing determinism
            raw = p["raw_response"]
            assert "=== PASS 1: ACOUSTIC TURNS" in raw
            assert "=== PASS 2: DISENTANGLED & ATTRIBUTED DIALOGUE" in raw
            pass2_text = raw.split("=== PASS 2: DISENTANGLED & ATTRIBUTED DIALOGUE")[1].split("===", 1)[-1].strip()
            reparsed = parse_attributed_turns(pass2_text)
            assert len(reparsed) == len(p["predicted_turns"])
            for t_rep, t_sto in zip(reparsed, p["predicted_turns"]):
                assert t_rep.speaker == t_sto["speaker"]
                assert t_rep.text == t_sto["text"]

            # 2. Verify metric engine recalculation matches stored metrics
            gt_turns = samples_by_id[sid].ground_truth_turns
            hyp_turns = [Turn(**t) for t in p["predicted_turns"]]
            recalc = MetricsEngine.evaluate_sample(gt_turns, hyp_turns, normalize=True)
            stored = p["metrics"]

            assert abs(recalc.wer - stored["wer"]) < 0.001
            assert abs(recalc.speaker_attribution_accuracy - stored["speaker_attribution_accuracy"]) < 0.001
            assert abs(recalc.cpwer - stored["cpwer"]) < 0.001
            assert abs(recalc.diarization_gap - stored["diarization_gap"]) < 0.001

    def test_comparative_summary_macro_aggregation(self):
        """Verifies macro averages in hard_2spk_comparative_summary.json against raw predictions."""
        summary_path = Path("results/hard_2spk/hard_2spk_comparative_summary.json")
        assert summary_path.exists()

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        macro = summary["macro_summary"]

        # Cross-verify Strategy A
        with open("results/hard_2spk/predictions_strategy_a_token0_bypass.json") as f:
            preds_a = json.load(f)
        mean_saa_a = sum(p["metrics"]["speaker_attribution_accuracy"] for p in preds_a) / 20
        mean_gap_a = sum(p["metrics"]["diarization_gap"] for p in preds_a) / 20
        assert abs(mean_saa_a - macro["strategy_a_token0_bypass"]["mean_speaker_attribution_accuracy"]) < 0.001
        assert abs(mean_gap_a - macro["strategy_a_token0_bypass"]["mean_diarization_gap"]) < 0.001

        # Cross-verify Strategy B
        with open("results/hard_2spk/predictions_strategy_b_pure_lite_twopass.json") as f:
            preds_b = json.load(f)
        mean_saa_b = sum(p["metrics"]["speaker_attribution_accuracy"] for p in preds_b) / 20
        mean_gap_b = sum(p["metrics"]["diarization_gap"] for p in preds_b) / 20
        assert abs(mean_saa_b - macro["strategy_b_pure_lite_twopass"]["mean_speaker_attribution_accuracy"]) < 0.001
        assert abs(mean_gap_b - macro["strategy_b_pure_lite_twopass"]["mean_diarization_gap"]) < 0.001
