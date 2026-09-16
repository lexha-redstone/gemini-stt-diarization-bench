"""Tier 1: Feature Coverage E2E Tests (>=5 tests per feature).

Covers:
- Feature 1: Benchmark Dataset Curation & Audio Extraction (5 tests)
- Feature 2: Pydantic Data Models & Metadata Schema Validation (5 tests)
- Feature 3: Indic Text Normalizer (5 tests)
- Feature 4: Programmatic Metrics Engine (WER & CER) (5 tests)
- Feature 5: Hungarian SAA, cpWER & Sample Evaluation (5 tests)
"""
import os
import sys
import wave
import pytest
from pydantic import ValidationError

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import Turn, SampleData, PipelinePrediction, EvaluationMetrics
from src.normalizer import IndicTextNormalizer
from src.metrics import MetricsEngine
from src.dataset import DatasetLoader

load_benchmark_subset = DatasetLoader.load_benchmark_subset


# =====================================================================
# Feature 1: Benchmark Dataset Curation & Audio Extraction
# =====================================================================

def test_f1_dataset_subset_count():
    """Verify that exactly 20 curated benchmark samples are loaded."""
    samples = load_benchmark_subset()
    assert len(samples) == 20, f"Expected exactly 20 benchmark samples, got {len(samples)}"


def test_f1_audio_files_exist_on_disk():
    """Verify that all 20 WAV audio files exist on disk and have non-zero size."""
    samples = load_benchmark_subset()
    assert len(samples) == 20
    for sample in samples:
        audio_path = os.path.join(PROJECT_ROOT, sample.audio_path)
        assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
        file_size = os.path.getsize(audio_path)
        assert file_size > 100_000, f"Audio file {audio_path} is suspiciously small: {file_size} bytes"


def test_f1_audio_format_pcm_16khz_mono():
    """Verify that all 20 audio files are 16kHz, mono, 16-bit PCM WAV."""
    samples = load_benchmark_subset()
    for sample in samples:
        audio_path = os.path.join(PROJECT_ROOT, sample.audio_path)
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            duration = num_frames / float(frame_rate)
            
            assert channels == 1, f"Expected 1 channel (mono) in {sample.sample_id}, got {channels}"
            assert sample_width == 2, f"Expected 16-bit (2-byte) PCM in {sample.sample_id}, got {sample_width}"
            assert frame_rate == 16000, f"Expected 16000 Hz in {sample.sample_id}, got {frame_rate}"
            assert duration > 50.0, f"Expected duration > 50s for {sample.sample_id}, got {duration:.2f}s"


def test_f1_dataset_unique_recording_ids():
    """Verify that all 20 samples originate from 20 distinct recording IDs."""
    import json
    metadata_path = os.path.join(PROJECT_ROOT, "data/benchmark_subset/metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)
    recording_ids = [m["recording_id"] for m in meta_list]
    assert len(recording_ids) == 20
    assert len(set(recording_ids)) == 20, f"Duplicate recording_ids detected: {recording_ids}"


def test_f1_dataset_duration_and_overlap_distribution():
    """Verify dataset duration boundaries and presence of overlapping speech."""
    samples = load_benchmark_subset()
    durations = [s.duration_seconds for s in samples]
    overlaps = [s.overlap_ratio for s in samples]
    
    # Assert durations are within operational range (50s to 250s)
    assert min(durations) >= 50.0, f"Minimum duration {min(durations)}s is under 50s"
    assert max(durations) <= 250.0, f"Maximum duration {max(durations)}s exceeds 250s"
    
    # Assert positive overlap ratios across the benchmark
    assert any(o > 5.0 for o in overlaps), "Expected samples with substantial (>5%) overlap ratio"
    assert sum(overlaps) / len(overlaps) >= 2.0, "Expected mean overlap ratio >= 2.0%"


# =====================================================================
# Feature 2: Pydantic Data Models & Metadata Schema Validation
# =====================================================================

def test_f2_turn_model_valid():
    """Verify valid instantiation of Turn model."""
    turn = Turn(speaker="Speaker 1", text="नमस्ते, क्या हाल है?", start_time=0.5, end_time=2.1)
    assert turn.speaker == "Speaker 1"
    assert turn.text == "नमस्ते, क्या हाल है?"
    assert turn.start_time == 0.5
    assert turn.end_time == 2.1


def test_f2_turn_model_validation_rejects_missing():
    """Verify that Turn model rejects missing required fields."""
    with pytest.raises(ValidationError):
        # speaker is required
        Turn(text="Missing speaker")
    with pytest.raises(ValidationError):
        # text is required
        Turn(speaker="Speaker 1")


def test_f2_sample_data_model_valid():
    """Verify valid instantiation of SampleData model."""
    sample = SampleData(
        sample_id="test_sample_001",
        audio_path="data/benchmark_subset/audio/test.wav",
        duration_seconds=60.0,
        num_speakers=2,
        overlap_ratio=10.5,
        ground_truth_turns=[
            Turn(speaker="Agent", text="नमस्ते"),
            Turn(speaker="Customer", text="हाँ जी")
        ]
    )
    assert sample.sample_id == "test_sample_001"
    assert sample.num_speakers == 2
    assert len(sample.ground_truth_turns) == 2


def test_f2_pipeline_prediction_model_valid():
    """Verify valid instantiation of PipelinePrediction model."""
    pred = PipelinePrediction(
        sample_id="test_sample_001",
        model_id="gemini-3.5-flash-lite",
        approach="single_step",
        predicted_turns=[
            Turn(speaker="Speaker 1", text="नमस्ते")
        ],
        raw_response="Speaker 1: नमस्ते",
        latency_seconds=1.23
    )
    assert pred.model_id == "gemini-3.5-flash-lite"
    assert pred.latency_seconds == 1.23
    assert len(pred.predicted_turns) == 1


def test_f2_evaluation_metrics_model_valid():
    """Verify valid instantiation of EvaluationMetrics model."""
    metrics = EvaluationMetrics(
        wer=0.05,
        cer=0.02,
        speaker_attribution_accuracy=0.95,
        cpwer=0.08,
        diarization_gap=0.03,
        speaker_mapping={"Speaker 1": "Agent"},
        format_valid=True
    )
    assert metrics.wer == 0.05
    assert metrics.speaker_attribution_accuracy == 0.95
    assert metrics.diarization_gap == 0.03
    assert metrics.format_valid is True


# =====================================================================
# Feature 3: Indic Text Normalizer
# =====================================================================

def test_f3_normalizer_preserves_devanagari_matras():
    """Verify that IndicTextNormalizer preserves Devanagari vowel signs (Mc) and marks (Mn)."""
    norm = IndicTextNormalizer()
    text = "हाँ, मैं बात कर रहा हूँ।"
    # 'ा' (Mc), 'ँ' (Mn), 'ै' (Mn), 'ू' (Mn) must NOT be stripped
    result = norm.normalize(text)
    assert result == "हाँ मैं बात कर रहा हूँ", f"Matras stripped: {result}"


def test_f3_normalizer_removes_punctuation_and_danda():
    """Verify that IndicTextNormalizer removes Latin punctuation and Devanagari danda."""
    norm = IndicTextNormalizer()
    text = "नमस्ते! क्या हाल है? सब ठीक है। बिल्कुल॥"
    result = norm.normalize(text)
    assert result == "नमस्ते क्या हाल है सब ठीक है बिल्कुल", f"Punctuation remaining: {result}"


def test_f3_normalizer_removes_invisible_unicode():
    """Verify that zero-width joiners, non-joiners, and BOM are eliminated."""
    norm = IndicTextNormalizer()
    text = "\ufeffनमस्ते \u200b\u200cदुनिया\u200d"
    result = norm.normalize(text)
    assert result == "नमस्ते दुनिया", f"Invisible characters retained: {result!r}"
    assert "\ufeff" not in result
    assert "\u200b" not in result
    assert "\u200c" not in result
    assert "\u200d" not in result


def test_f3_normalizer_strips_non_speech_tokens():
    """Verify that bracketed and angle-bracketed acoustic tags are stripped."""
    norm = IndicTextNormalizer()
    text = "[Unintelligible] मैं कल [stutters] आऊंगा <laughter> ठीक है <coughing>"
    result = norm.normalize(text)
    assert result == "मैं कल आऊंगा ठीक है", f"Non-speech tokens remaining: {result}"


def test_f3_normalizer_handles_english_parenthetical_glosses():
    """Verify parenthetical transliteration glosses (e.g. (shopping)) are handled."""
    norm = IndicTextNormalizer()
    text = "मैंने शॉपिंग (shopping) की और पेमेंट (payment) कर दिया"
    result = norm.normalize(text)
    # The normalizer removes parenthetical transliteration glosses for Indic evaluation
    assert "shopping" not in result
    assert "payment" not in result
    assert "शॉपिंग" in result
    assert "पेमेंट" in result


# =====================================================================
# Feature 4: Programmatic Metrics Engine (WER & CER)
# =====================================================================

def test_f4_wer_identical_transcripts():
    """Verify that WER for identical reference and hypothesis is 0.0."""
    engine = MetricsEngine()
    ref = "यह एक परीक्षण वाक्य है"
    hyp = "यह एक परीक्षण वाक्य है"
    wer = engine.compute_wer(ref, hyp)
    assert wer == 0.0, f"Expected 0.0 WER for identical strings, got {wer}"


def test_f4_wer_completely_disjoint():
    """Verify that WER for completely disjoint strings is 1.0 (or greater if extra words)."""
    engine = MetricsEngine()
    ref = "लाल सेब मीठा है"
    hyp = "हरा पत्ता कड़वा था"
    wer = engine.compute_wer(ref, hyp)
    # 4 substitutions out of 4 words = 1.0
    assert wer == 1.0, f"Expected 1.0 WER for 4/4 substitutions, got {wer}"


def test_f4_wer_known_edit_distance():
    """Verify exact mathematical WER for known substitution, insertion, and deletion."""
    engine = MetricsEngine()
    ref = "एक दो तीन चार पाँच"  # 5 reference words
    # Replace 'दो' with 'दस' (1 sub), delete 'चार' (1 del), insert 'छह' (1 ins)
    hyp = "एक दस तीन पाँच छह"
    # Edit distance = 1 sub + 1 del + 1 ins = 3 edits. WER = 3 / 5 = 0.6
    wer = engine.compute_wer(ref, hyp)
    assert abs(wer - 0.6) < 1e-4, f"Expected 0.6 WER, got {wer}"


def test_f4_cer_known_edit_distance():
    """Verify character error rate (CER) calculation."""
    engine = MetricsEngine()
    ref = "नमस्ते"
    hyp = "नमस्ते"
    cer_identical = engine.compute_cer(ref, hyp)
    assert cer_identical == 0.0, f"Expected 0.0 CER, got {cer_identical}"
    
    # 1 character difference
    hyp2 = "नमस्त"
    cer_diff = engine.compute_cer(ref, hyp2)
    assert cer_diff > 0.0, f"Expected positive CER, got {cer_diff}"


def test_f4_wer_with_devanagari_sentences():
    """Verify WER computation on realistic Hindi sentences after normalization."""
    engine = MetricsEngine()
    ref = "हाँ, मैं बात कर रहा हूँ।"
    hyp = "हाँ मैं बात कर रहा हूँ"
    # Even though ref has punctuation, normalizer should make them equal
    wer = engine.compute_wer(ref, hyp)
    assert wer == 0.0, f"Expected 0.0 WER with punctuation normalized, got {wer}"


# =====================================================================
# Feature 5: Hungarian SAA, cpWER & Sample Evaluation
# =====================================================================

def test_f5_saa_perfect_match():
    """Verify that identical speaker turns yield 100% SAA (1.0)."""
    engine = MetricsEngine()
    turns = [
        Turn(speaker="Speaker 1", text="नमस्ते सर"),
        Turn(speaker="Speaker 2", text="हाँ बताइए"),
        Turn(speaker="Speaker 1", text="आपका पेमेंट ड्यू है")
    ]
    saa, mapping = engine.compute_speaker_attribution_accuracy(turns, turns)
    assert saa == 1.0, f"Expected 1.0 SAA for identical turns, got {saa}"
    assert mapping.get("Speaker 1") == "Speaker 1"
    assert mapping.get("Speaker 2") == "Speaker 2"


def test_f5_saa_inverted_speaker_labels():
    """Verify Hungarian matching correctly recovers 100% SAA on inverted speaker tags."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="spk_0", text="नमस्ते सर"),
        Turn(speaker="spk_1", text="हाँ बताइए"),
        Turn(speaker="spk_0", text="आपका पेमेंट ड्यू है")
    ]
    # Hypothesis has inverted labels 'Speaker B' for spk_0 and 'Speaker A' for spk_1
    hyp_turns = [
        Turn(speaker="Speaker B", text="नमस्ते सर"),
        Turn(speaker="Speaker A", text="हाँ बताइए"),
        Turn(speaker="Speaker B", text="आपका पेमेंट ड्यू है")
    ]
    saa, mapping = engine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
    assert saa == 1.0, f"Hungarian matching failed to recover inverted labels, got SAA={saa}"
    assert mapping.get("Speaker B") == "spk_0"
    assert mapping.get("Speaker A") == "spk_1"


def test_f5_saa_three_speaker_permutation():
    """Verify Hungarian matching on 3 speakers with cyclical label permutation."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="A", text="पहला वक्ता बोल रहा है"),
        Turn(speaker="B", text="दूसरा वक्ता उत्तर दे रहा है"),
        Turn(speaker="C", text="तीसरा वक्ता हस्तक्षेप करता है")
    ]
    # Hypothesis permutes: A->X, B->Y, C->Z
    hyp_turns = [
        Turn(speaker="X", text="पहला वक्ता बोल रहा है"),
        Turn(speaker="Y", text="दूसरा वक्ता उत्तर दे रहा है"),
        Turn(speaker="Z", text="तीसरा वक्ता हस्तक्षेप करता है")
    ]
    saa, mapping = engine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
    assert saa == 1.0, f"Expected 1.0 SAA for 3-speaker permutation, got {saa}"
    assert mapping.get("X") == "A"
    assert mapping.get("Y") == "B"
    assert mapping.get("Z") == "C"


def test_f5_cpwer_and_diarization_gap():
    """Verify cpWER and Diarization Degradation Gap (cpWER - WER)."""
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="spk_0", text="नमस्ते"),
        Turn(speaker="spk_1", text="हाँ")
    ]
    # 1. Perfect diarization: gap should be 0.0
    hyp_perfect = [
        Turn(speaker="spk_0", text="नमस्ते"),
        Turn(speaker="spk_1", text="हाँ")
    ]
    res_perfect = engine.evaluate_sample(ref_turns, hyp_perfect)
    assert res_perfect.wer == 0.0
    assert res_perfect.cpwer == 0.0
    assert abs(res_perfect.diarization_gap) < 1e-4

    # 2. Confused speakers (wrong turn attribution): cpWER should be higher than WER
    hyp_confused = [
        Turn(speaker="spk_1", text="नमस्ते"),  # attributed to spk_1
        Turn(speaker="spk_1", text="हाँ")      # also attributed to spk_1, spk_0 dropped
    ]
    res_confused = engine.evaluate_sample(ref_turns, hyp_confused)
    assert res_confused.wer == 0.0  # All words match globally
    assert res_confused.cpwer > 0.0  # cpWER penalizes collapsed speakers
    assert res_confused.diarization_gap > 0.0


def test_f5_evaluate_sample_structure():
    """Verify that evaluate_sample returns a fully populated EvaluationMetrics instance."""
    engine = MetricsEngine()
    ref_turns = [Turn(speaker="Agent", text="नमस्कार"), Turn(speaker="Customer", text="हाँ जी")]
    hyp_turns = [Turn(speaker="Agent", text="नमस्कार"), Turn(speaker="Customer", text="हाँ जी")]
    
    result = engine.evaluate_sample(ref_turns, hyp_turns)
    assert isinstance(result, EvaluationMetrics)
    assert hasattr(result, "wer")
    assert hasattr(result, "cer")
    assert hasattr(result, "speaker_attribution_accuracy")
    assert hasattr(result, "cpwer")
    assert hasattr(result, "diarization_gap")
    assert hasattr(result, "speaker_mapping")
    assert hasattr(result, "format_valid")
    assert result.format_valid is True
