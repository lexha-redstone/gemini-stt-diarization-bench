"""Tier 2: Boundary & Corner Cases E2E Tests (>=5 tests per category).

Covers:
- Category 1: Empty & Minimal Inputs (5 tests)
- Category 2: Timestamp & Duration Anomalies (5 tests)
- Category 3: Overlap & Alignment Extremes (5 tests)
- Category 4: Speaker Tag & Naming Inconsistencies (5 tests)
- Category 5: Unicode & Formatting Extremes (5 tests)
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import Turn, SampleData, EvaluationMetrics
from src.normalizer import IndicTextNormalizer
from src.metrics import MetricsEngine


# =====================================================================
# Category 1: Empty & Minimal Inputs
# =====================================================================

def test_b1_empty_reference_wer():
    """Verify WER handles empty reference string safely without crashing."""
    engine = MetricsEngine()
    ref = ""
    hyp = "कुछ शब्द यहाँ हैं"
    # When reference is empty and hypothesis has words, WER is either 1.0 or insertions count
    wer = engine.compute_wer(ref, hyp)
    assert wer >= 0.0


def test_b1_empty_hypothesis_wer():
    """Verify WER handles empty hypothesis (100% deletions)."""
    engine = MetricsEngine()
    ref = "यह वाक्य पूरी तरह छूट गया"
    hyp = ""
    wer = engine.compute_wer(ref, hyp)
    assert wer == 1.0, f"Expected 1.0 WER for empty hypothesis, got {wer}"


def test_b1_both_empty_wer():
    """Verify WER for both reference and hypothesis empty returns 0.0."""
    engine = MetricsEngine()
    wer = engine.compute_wer("", "")
    assert wer == 0.0, f"Expected 0.0 WER for empty strings, got {wer}"


def test_b1_normalizer_empty_and_whitespace():
    """Verify normalizer handles empty and whitespace-only strings."""
    norm = IndicTextNormalizer()
    assert norm.normalize("") == ""
    assert norm.normalize("   ") == ""
    assert norm.normalize("\t\n  \r\n") == ""


def test_b1_empty_turn_lists_evaluate_sample():
    """Verify evaluate_sample handles empty turn lists gracefully."""
    engine = MetricsEngine()
    res = engine.evaluate_sample([], [])
    assert isinstance(res, EvaluationMetrics)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


# =====================================================================
# Category 2: Timestamp & Duration Anomalies
# =====================================================================

def test_b2_zero_duration_turn():
    """Verify Turn with zero duration (start_time == end_time)."""
    turn = Turn(speaker="Spk1", text="तुरंत", start_time=10.0, end_time=10.0)
    assert turn.start_time == turn.end_time == 10.0
    engine = MetricsEngine()
    res = engine.evaluate_sample([turn], [turn])
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_b2_inverted_timestamps_in_turn():
    """Verify turn handling when end_time < start_time (anomaly seen in raw data)."""
    # Should instantiate or handle safely without crashing downstream evaluation
    turn = Turn(speaker="Spk1", text="उल्टा समय", start_time=15.0, end_time=10.0)
    assert turn.start_time == 15.0
    assert turn.end_time == 10.0


def test_b2_timestamp_exceeding_duration():
    """Verify turns whose timestamps exceed declared duration."""
    sample = SampleData(
        sample_id="overflow_sample",
        audio_path="dummy.wav",
        duration_seconds=60.0,
        num_speakers=1,
        ground_truth_turns=[
            Turn(speaker="Spk1", text="अंतिम वाक्य", start_time=59.5, end_time=60.4)
        ]
    )
    assert sample.ground_truth_turns[0].end_time > sample.duration_seconds


def test_b2_extreme_text_length_normalizer():
    """Verify normalizer efficiency and stability on a very long text (>10k chars)."""
    norm = IndicTextNormalizer()
    repeated_sentence = "यह एक बहुत लंबा परीक्षण वाक्य है जिसमें देवनागरी मात्राएँ शामिल हैं। " * 200
    assert len(repeated_sentence) > 10000
    cleaned = norm.normalize(repeated_sentence)
    assert len(cleaned) > 5000
    assert "।" not in cleaned


def test_b2_single_speaker_monologue():
    """Verify SAA and cpWER on a monologue (only 1 speaker)."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Speaker 1", text="पहला वाक्य"),
        Turn(speaker="Speaker 1", text="दूसरा वाक्य"),
        Turn(speaker="Speaker 1", text="तीसरा वाक्य")
    ]
    hyp_turns = [
        Turn(speaker="Speaker 1", text="पहला वाक्य"),
        Turn(speaker="Speaker 1", text="दूसरा वाक्य"),
        Turn(speaker="Speaker 1", text="तीसरा वाक्य")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.speaker_attribution_accuracy == 1.0
    assert res.cpwer == 0.0
    assert res.diarization_gap == 0.0


# =====================================================================
# Category 3: Overlap & Alignment Extremes
# =====================================================================

def test_b3_full_100_percent_overlap():
    """Verify handling when two speakers talk simultaneously over the same time interval."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Speaker 1", text="मैं बोल रहा हूँ", start_time=1.0, end_time=3.0),
        Turn(speaker="Speaker 2", text="मैं भी बोल रहा हूँ", start_time=1.0, end_time=3.0)
    ]
    hyp_turns = [
        Turn(speaker="Speaker 1", text="मैं बोल रहा हूँ", start_time=1.0, end_time=3.0),
        Turn(speaker="Speaker 2", text="मैं भी बोल रहा हूँ", start_time=1.0, end_time=3.0)
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_b3_zero_percent_overlap():
    """Verify handling when turns are strictly sequential with long pauses."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Speaker 1", text="नमस्ते सर", start_time=0.0, end_time=2.0),
        Turn(speaker="Speaker 2", text="हाँ जी बताइए", start_time=10.0, end_time=12.0)
    ]
    res = engine.evaluate_sample(ref_turns, ref_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_b3_oversegmented_hypothesis():
    """Verify word alignment when hypothesis splits 1 reference turn into 4 short turns."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Speaker 1", text="एक दो तीन चार पाँच छह सात आठ")
    ]
    hyp_turns = [
        Turn(speaker="Speaker 1", text="एक दो"),
        Turn(speaker="Speaker 1", text="तीन चार"),
        Turn(speaker="Speaker 1", text="पाँच छह"),
        Turn(speaker="Speaker 1", text="सात आठ")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_b3_undersegmented_hypothesis():
    """Verify word alignment when hypothesis merges multiple reference turns of the same speaker."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Speaker 1", text="पहला भाग"),
        Turn(speaker="Speaker 1", text="दूसरा भाग"),
        Turn(speaker="Speaker 1", text="तीसरा भाग")
    ]
    hyp_turns = [
        Turn(speaker="Speaker 1", text="पहला भाग दूसरा भाग तीसरा भाग")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_b3_unequal_speaker_counts():
    """Verify Hungarian matching when hypothesis introduces an unreferenced extra speaker."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="A", text="नमस्ते सर"),
        Turn(speaker="B", text="हाँ जी")
    ]
    hyp_turns = [
        Turn(speaker="A", text="नमस्ते सर"),
        Turn(speaker="B", text="हाँ"),
        Turn(speaker="C", text="जी")  # 'जी' attributed to spurious speaker C
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    # SAA should be less than 1.0 because word 'जी' was attributed to C instead of B
    assert res.speaker_attribution_accuracy < 1.0


# =====================================================================
# Category 4: Speaker Tag & Naming Inconsistencies
# =====================================================================

def test_b4_unmapped_speaker_labels():
    """Verify Hungarian matching with non-standard speaker strings."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Agent_HDFC_01", text="नमस्ते"),
        Turn(speaker="Customer_Borrower_42", text="हाँ")
    ]
    hyp_turns = [
        Turn(speaker="SPEAKER_A", text="नमस्ते"),
        Turn(speaker="SPEAKER_B", text="हाँ")
    ]
    saa, mapping = engine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
    assert saa == 1.0
    assert mapping.get("SPEAKER_A") == "Agent_HDFC_01"
    assert mapping.get("SPEAKER_B") == "Customer_Borrower_42"


def test_b4_hypothesis_single_speaker_collapse():
    """Verify behavior when model collapses all utterances onto a single speaker."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="A", text="एक दो तीन चार पाँच"),  # 5 words
        Turn(speaker="B", text="छह सात आठ नौ दस")       # 5 words
    ]
    hyp_turns = [
        Turn(speaker="A", text="एक दो तीन चार पाँच"),
        Turn(speaker="A", text="छह सात आठ नौ दस")      # collapsed onto A
    ]
    saa, mapping = engine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
    # 5 out of 10 words correctly mapped to A, 5 misattributed to B
    assert abs(saa - 0.5) < 1e-4


def test_b4_alternating_single_word_turns():
    """Verify rapid single-word turn switching."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="A", text="हाँ"),
        Turn(speaker="B", text="नहीं"),
        Turn(speaker="A", text="क्यों"),
        Turn(speaker="B", text="बस")
    ]
    res = engine.evaluate_sample(ref_turns, ref_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_b4_speaker_names_with_special_chars():
    """Verify speaker labels with spaces, colons, brackets, and hyphens."""
    engine = MetricsEngine()
    ref_turns = [Turn(speaker="Speaker: [01]", text="नमस्ते"), Turn(speaker="Speaker: [02]", text="हाँ")]
    hyp_turns = [Turn(speaker="Speaker-A", text="नमस्ते"), Turn(speaker="Speaker-B", text="हाँ")]
    saa, mapping = engine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
    assert saa == 1.0


def test_b4_case_variant_speaker_labels():
    """Verify case sensitivity and mapping between 'speaker 1' and 'SPEAKER 1'."""
    engine = MetricsEngine()
    ref_turns = [Turn(speaker="speaker 1", text="नमस्ते")]
    hyp_turns = [Turn(speaker="SPEAKER 1", text="नमस्ते")]
    saa, mapping = engine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
    assert saa == 1.0


# =====================================================================
# Category 5: Unicode & Formatting Extremes
# =====================================================================

def test_b5_pure_punctuation_string():
    """Verify that string of pure punctuation and symbols normalizes to empty string."""
    norm = IndicTextNormalizer()
    text = "। ॥ , . ? ! ; : - _ / \\ @ # $ % ^ & * ( ) [ ] { }"
    result = norm.normalize(text)
    assert result == ""


def test_b5_pure_latin_text_in_indic_normalizer():
    """Verify normalizer handles pure English text with lowercase normalization."""
    norm = IndicTextNormalizer()
    text = "Hello World! This is a test."
    result = norm.normalize(text)
    assert result == "hello world this is a test"


def test_b5_mixed_indic_and_arabic_numerals():
    """Verify Devanagari and Arabic numbers are preserved."""
    norm = IndicTextNormalizer()
    text = "खाता संख्या १२३४ और पिन 5678"
    result = norm.normalize(text)
    assert "१२३४" in result
    assert "5678" in result


def test_b5_only_filler_tokens():
    """Verify string consisting exclusively of non-speech tokens normalizes to empty."""
    norm = IndicTextNormalizer()
    text = "[Unintelligible] <laughter> [stutters] <coughing> <noise>"
    result = norm.normalize(text)
    assert result == ""


def test_b5_nested_brackets_and_tags():
    """Verify nested or consecutive bracketed tags are cleanly stripped."""
    norm = IndicTextNormalizer()
    text = "नमस्ते [[noise]] दुनिया <<laughter>> फिर मिलेंगे"
    result = norm.normalize(text)
    assert result == "नमस्ते दुनिया फिर मिलेंगे"
