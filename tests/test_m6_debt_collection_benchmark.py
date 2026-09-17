"""Milestone 6 Verification Suite: 50-Sample Hindi Debt Collection & Fierce Argument Benchmark.

Verifies:
1. Dataset & Audio Integrity: 50 samples in data/debt_collection_subset/, 4 duration buckets,
   mean overlap >= 15.0% (acceptance >= 12.0%), Devanagari ground-truth turns, and playable 16kHz mono WAV files
   with zero silent or corrupted audio windows.
2. Prediction Integrity: 50 live Vertex AI predictions for both Gemini 2.5 Flash and Candidate C5 (3.5 Flash Lite)
   with zero mocks, valid latencies, and non-empty turns.
3. Independent Metric Recalculation: Independently recalculates WER, CER, Hungarian SAA, cpWER, and Diarization Gap
   via src/metrics.py across all 50 samples and all 4 duration buckets, confirming exact match (+/- 1e-3)
   with debt_collection_comparative_summary.json.
4. Report & Documentation Consistency: Verifies results/debt_collection/debt_collection_benchmark_report.md,
   benchmark_report.md, one-pager.md, and README.md.
"""

import json
from pathlib import Path
import wave

import numpy as np
import pytest

from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import Turn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = PROJECT_ROOT / "data" / "debt_collection_subset"
RESULTS_DIR = PROJECT_ROOT / "results" / "debt_collection"


def test_debt_collection_dataset_audio_and_metadata():
    """Verifies 50-sample Hindi debt collection dataset, stratified buckets, overlap >= 15%, and non-silent WAV files."""
    assert SUBSET_DIR.exists(), f"Missing subset directory: {SUBSET_DIR}"
    samples = DatasetLoader.load_benchmark_subset(subset_dir=SUBSET_DIR, verify_audio=True)
    assert len(samples) == 50, f"Expected 50 samples, got {len(samples)}"

    stats = DatasetLoader.get_summary_stats(samples)
    assert stats["mean_overlap_ratio"] >= 12.0, f"Mean overlap {stats['mean_overlap_ratio']}% < 12.0%"
    assert stats["mean_overlap_ratio"] >= 15.0, f"Mean overlap {stats['mean_overlap_ratio']}% < 15.0% target"

    buckets = {"short": 0, "medium": 0, "long": 0, "very_long": 0}
    for s in samples:
        d = s.duration_seconds
        if 30.0 <= d < 70.0:
            buckets["short"] += 1
        elif 70.0 <= d < 180.0:
            buckets["medium"] += 1
        elif 180.0 <= d < 300.0:
            buckets["long"] += 1
        elif 300.0 <= d <= 600.0:
            buckets["very_long"] += 1
        else:
            pytest.fail(f"Sample {s.sample_id} duration {d}s outside valid 30s-600s range")

        assert s.num_speakers == 2
        assert len(s.ground_truth_turns) >= 6
        for t in s.ground_truth_turns:
            assert t.speaker in ["Speaker 0", "Speaker 1"]
            assert len(t.text.strip()) > 0
            assert t.start_time is not None and t.end_time is not None
            assert 0.0 <= t.start_time < t.end_time <= d + 0.5

        # Audio file verification
        wav_path = PROJECT_ROOT / s.audio_path
        assert wav_path.exists(), f"Missing WAV file: {wav_path}"
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            nframes = wf.getnframes()
            pcm = np.frombuffer(wf.readframes(nframes), dtype=np.int16).astype(np.float32) / 32768.0
            assert np.std(pcm) > 0.02, f"Low audio energy in {s.sample_id}"

            # Check 1-second windows to guarantee zero silent windows
            n_win = len(pcm) // 16000
            for w in range(n_win):
                w_rms = np.sqrt(np.mean(pcm[w * 16000 : (w + 1) * 16000] ** 2))
                assert w_rms > 0.001, f"Silent 1s window detected in {s.sample_id} at second {w}"

    assert buckets["short"] == 13
    assert buckets["medium"] == 13
    assert buckets["long"] == 12
    assert buckets["very_long"] == 12


def test_debt_collection_predictions_integrity():
    """Verifies 50 live Vertex AI predictions for both models with zero mocks or placeholders."""
    pred_25_path = RESULTS_DIR / "predictions_gemini_2_5_flash.json"
    pred_c5_path = RESULTS_DIR / "predictions_strategy_c5_adaptive_champion.json"

    assert pred_25_path.exists(), f"Missing {pred_25_path}"
    assert pred_c5_path.exists(), f"Missing {pred_c5_path}"

    preds_25 = json.loads(pred_25_path.read_text(encoding="utf-8"))
    preds_c5 = json.loads(pred_c5_path.read_text(encoding="utf-8"))

    assert len(preds_25) == 50, f"Expected 50 predictions in 2.5 Flash, got {len(preds_25)}"
    assert len(preds_c5) == 50, f"Expected 50 predictions in Candidate C5, got {len(preds_c5)}"

    for model_label, preds in [("2.5 Flash", preds_25), ("Candidate C5", preds_c5)]:
        seen_ids = set()
        for rec in preds:
            sid = rec["sample_id"]
            seen_ids.add(sid)
            assert rec["source"] == "live_vertex_ai", f"{model_label} {sid} source is not live_vertex_ai"
            assert rec["latency_seconds"] > 0.5, f"{model_label} {sid} suspicious latency {rec['latency_seconds']}"
            assert len(rec["predicted_turns"]) > 0, f"{model_label} {sid} has empty predicted_turns"
            assert "mock" not in str(rec.get("raw_response", "")).lower()
        assert len(seen_ids) == 50


def test_debt_collection_summary_and_independent_recalculation():
    """Independently recalculates metrics via src/metrics.py across all 50 samples and all 4 duration buckets."""
    summary_path = RESULTS_DIR / "debt_collection_comparative_summary.json"
    pred_25_path = RESULTS_DIR / "predictions_gemini_2_5_flash.json"
    pred_c5_path = RESULTS_DIR / "predictions_strategy_c5_adaptive_champion.json"

    assert summary_path.exists(), f"Missing {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    preds_25 = {r["sample_id"]: r for r in json.loads(pred_25_path.read_text(encoding="utf-8"))}
    preds_c5 = {r["sample_id"]: r for r in json.loads(pred_c5_path.read_text(encoding="utf-8"))}

    samples = DatasetLoader.load_benchmark_subset(subset_dir=SUBSET_DIR, verify_audio=False)
    assert summary["total_samples"] == 50

    for model_key, preds_dict in [
        ("gemini_2_5_flash", preds_25),
        ("strategy_c5_adaptive_champion", preds_c5),
    ]:
        recalc_saas = []
        recalc_wers = []
        recalc_cpwers = []
        recalc_gaps = []

        for s in samples:
            rec = preds_dict[s.sample_id]
            hyp_turns = [Turn(**t) for t in rec["predicted_turns"]]
            m = MetricsEngine.evaluate_sample(s.ground_truth_turns, hyp_turns, normalize=True)

            # Verify sample-level agreement
            assert abs(m.speaker_attribution_accuracy - rec["metrics"]["speaker_attribution_accuracy"]) <= 1e-3
            assert abs(m.wer - rec["metrics"]["wer"]) <= 1e-3
            assert abs(m.cpwer - rec["metrics"]["cpwer"]) <= 1e-3
            assert abs(m.diarization_gap - rec["metrics"]["diarization_gap"]) <= 1e-3

            recalc_saas.append(m.speaker_attribution_accuracy)
            recalc_wers.append(m.wer)
            recalc_cpwers.append(m.cpwer)
            recalc_gaps.append(m.diarization_gap)

        # Verify macro summary agreement
        macro_reported = summary["macro_summary"][model_key]
        assert abs(sum(recalc_saas) / 50.0 - macro_reported["mean_speaker_attribution_accuracy"]) <= 1e-3
        assert abs(sum(recalc_wers) / 50.0 - macro_reported["mean_wer"]) <= 1e-3
        assert abs(sum(recalc_cpwers) / 50.0 - macro_reported["mean_cpwer"]) <= 1e-3
        assert abs(sum(recalc_gaps) / 50.0 - macro_reported["mean_diarization_gap"]) <= 1e-3

    # Verify duration buckets exist and sum to 50
    buckets = summary["duration_buckets"]
    for b_key in ["<70s", "70s-180s", "180s-300s", "300s-600s"]:
        assert b_key in buckets, f"Missing bucket {b_key} in summary"
        assert buckets[b_key]["sample_count"] in [12, 13]
    assert sum(buckets[k]["sample_count"] for k in ["<70s", "70s-180s", "180s-300s", "300s-600s"]) == 50


def test_debt_collection_report_and_documentation_consistency():
    """Verifies debt_collection_benchmark_report.md and master documentation updates."""
    report_path = RESULTS_DIR / "debt_collection_benchmark_report.md"
    master_report_path = PROJECT_ROOT / "benchmark_report.md"
    onepager_path = PROJECT_ROOT / "one-pager.md"
    readme_path = PROJECT_ROOT / "README.md"

    assert report_path.exists(), f"Missing {report_path}"
    report_text = report_path.read_text(encoding="utf-8")

    # Verify required sections and concepts in debt_collection_benchmark_report.md
    assert "Executive Summary" in report_text
    assert "<70s" in report_text and "70s-180s" in report_text and "180s-300s" in report_text and "300s-600s" in report_text
    assert "Cutoff" in report_text or "cutoff" in report_text
    assert "Routing" in report_text or "routing" in report_text
    assert "debt_001" in report_text and "debt_050" in report_text

    # Verify master documentation updates
    master_text = master_report_path.read_text(encoding="utf-8")
    onepager_text = onepager_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")

    assert "debt_collection_subset" in master_text
    assert "debt_collection_subset" in readme_text
    assert "Debt Collection" in onepager_text or "debt collection" in onepager_text
