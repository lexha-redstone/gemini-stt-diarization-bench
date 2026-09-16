"""Automated verification suite for Milestone 5: Complete 37-Sample 2-Speaker Benchmark."""

import json
import wave
from pathlib import Path
import pytest

from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import Turn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_2SPK_DIR = PROJECT_ROOT / "data" / "all_2spk_subset"
RESULTS_ALL_2SPK_DIR = PROJECT_ROOT / "results" / "all_2spk"


def test_all_2spk_dataset_metadata_and_audio():
    """Verifies that data/all_2spk_subset/ contains all 37 2-speaker samples with valid 16kHz mono WAV files."""
    metadata_path = ALL_2SPK_DIR / "metadata.json"
    assert metadata_path.exists(), f"Missing {metadata_path}"

    samples = DatasetLoader.load_benchmark_subset(ALL_2SPK_DIR)
    assert len(samples) == 37, f"Expected 37 samples in all_2spk_subset, got {len(samples)}"

    sample_ids = set()
    for s in samples:
        assert s.num_speakers == 2, f"Sample {s.sample_id} has num_speakers={s.num_speakers}, expected 2"
        assert s.duration_seconds > 0, f"Invalid duration for {s.sample_id}"
        assert len(s.ground_truth_turns) > 0, f"Empty ground_truth_turns for {s.sample_id}"
        sample_ids.add(s.sample_id)

        wav_path = ALL_2SPK_DIR / "audio" / f"{s.sample_id}.wav"
        assert wav_path.exists(), f"Audio file missing: {wav_path}"
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected mono audio for {s.sample_id}"
            assert wf.getframerate() == 16000, f"Expected 16kHz audio for {s.sample_id}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit PCM for {s.sample_id}"

    assert len(sample_ids) == 37, "Expected 37 unique sample IDs"


def test_all_2spk_predictions_integrity():
    """Verifies that predictions for both Gemini 2.5 Flash and Candidate C5 exist for all 37 samples with zero mocks."""
    pred_25_path = RESULTS_ALL_2SPK_DIR / "predictions_gemini_2_5_flash.json"
    pred_c5_path = RESULTS_ALL_2SPK_DIR / "predictions_strategy_c5_adaptive_champion.json"
    summary_path = RESULTS_ALL_2SPK_DIR / "all_2spk_comparative_summary.json"

    assert pred_25_path.exists(), f"Missing {pred_25_path}"
    assert pred_c5_path.exists(), f"Missing {pred_c5_path}"
    assert summary_path.exists(), f"Missing {summary_path}"

    with open(pred_25_path, "r", encoding="utf-8") as f:
        preds_25 = json.load(f)
    with open(pred_c5_path, "r", encoding="utf-8") as f:
        preds_c5 = json.load(f)

    assert len(preds_25) == 37, f"Expected 37 predictions in 2.5 Flash file, got {len(preds_25)}"
    assert len(preds_c5) == 37, f"Expected 37 predictions in Candidate C5 file, got {len(preds_c5)}"

    samples = DatasetLoader.load_benchmark_subset(ALL_2SPK_DIR)
    expected_ids = {s.sample_id for s in samples}

    ids_25 = {p["sample_id"] for p in preds_25}
    ids_c5 = {p["sample_id"] for p in preds_c5}

    assert ids_25 == expected_ids, f"2.5 Flash sample IDs mismatch: {ids_25 ^ expected_ids}"
    assert ids_c5 == expected_ids, f"Candidate C5 sample IDs mismatch: {ids_c5 ^ expected_ids}"

    for p in preds_25 + preds_c5:
        assert p.get("latency_seconds", 0) > 0, f"Invalid latency for {p['sample_id']}"
        assert len(p.get("predicted_turns", [])) > 0, f"Empty predicted_turns for {p['sample_id']}"
        assert p.get("raw_response", "").strip() != "", f"Empty raw_response for {p['sample_id']}"
        assert p.get("source") in ("reused_hard_2spk", "reused_indic_subset", "live_vertex_ai")


def test_all_2spk_independent_metric_recalculation():
    """Independently recalculates metrics via src/metrics.py across all 37 samples and verifies macro summary."""
    pred_25_path = RESULTS_ALL_2SPK_DIR / "predictions_gemini_2_5_flash.json"
    pred_c5_path = RESULTS_ALL_2SPK_DIR / "predictions_strategy_c5_adaptive_champion.json"
    summary_path = RESULTS_ALL_2SPK_DIR / "all_2spk_comparative_summary.json"

    with open(pred_25_path, "r", encoding="utf-8") as f:
        preds_25 = {p["sample_id"]: p for p in json.load(f)}
    with open(pred_c5_path, "r", encoding="utf-8") as f:
        preds_c5 = {p["sample_id"]: p for p in json.load(f)}
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    samples = {s.sample_id: s for s in DatasetLoader.load_benchmark_subset(ALL_2SPK_DIR)}

    for model_key, preds_map in [
        ("gemini_2_5_flash", preds_25),
        ("strategy_c5_adaptive_champion", preds_c5),
    ]:
        saa_list = []
        wer_list = []
        cer_list = []
        cpwer_list = []
        gap_list = []

        for sid, sample in samples.items():
            rec = preds_map[sid]
            hyp_turns = [Turn(**t) for t in rec["predicted_turns"]]
            eval_m = MetricsEngine.evaluate_sample(sample.ground_truth_turns, hyp_turns, normalize=True)

            assert abs(eval_m.speaker_attribution_accuracy - rec["metrics"]["speaker_attribution_accuracy"]) < 1e-3
            assert abs(eval_m.wer - rec["metrics"]["wer"]) < 1e-3
            assert abs(eval_m.cpwer - rec["metrics"]["cpwer"]) < 1e-3

            saa_list.append(rec["metrics"]["speaker_attribution_accuracy"])
            wer_list.append(rec["metrics"]["wer"])
            cer_list.append(rec["metrics"]["cer"])
            cpwer_list.append(rec["metrics"]["cpwer"])
            gap_list.append(rec["metrics"]["diarization_gap"])

        macro = summary["macro_summary"][model_key]
        assert macro["count"] == 37
        assert abs(macro["mean_speaker_attribution_accuracy"] - round(sum(saa_list) / 37, 4)) < 1e-3
        assert abs(macro["mean_diarization_gap"] - round(sum(gap_list) / 37, 4)) < 1e-3


def test_benchmark_report_and_documentation_consistency():
    """Verifies that benchmark_report.md, one-pager.md, and README.md match the authoritative JSON results."""
    import re

    pred_25_path = RESULTS_ALL_2SPK_DIR / "predictions_gemini_2_5_flash.json"
    pred_c5_path = RESULTS_ALL_2SPK_DIR / "predictions_strategy_c5_adaptive_champion.json"
    hard_c5_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c5_adaptive_champion.json"

    with open(pred_25_path, "r", encoding="utf-8") as f:
        preds_25 = {p["sample_id"]: p for p in json.load(f)}
    with open(pred_c5_path, "r", encoding="utf-8") as f:
        preds_c5 = {p["sample_id"]: p for p in json.load(f)}
    with open(hard_c5_path, "r", encoding="utf-8") as f:
        hard_c5 = {p["sample_id"]: p for p in json.load(f)}

    report_text = (PROJECT_ROOT / "benchmark_report.md").read_text(encoding="utf-8")
    onepager_text = (PROJECT_ROOT / "one-pager.md").read_text(encoding="utf-8")
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    creative_report_text = (
        PROJECT_ROOT / "results" / "hard_2spk" / "hard_2spk_creative_parity_report.md"
    ).read_text(encoding="utf-8")

    # Ensure outlier attribution points to hindi_089 (where WER = 6.2287), not hindi_084
    assert "1 (`hindi_089`)" in report_text
    assert "1 (`hindi_089`)" in readme_text
    assert "hindi_089" in onepager_text
    assert "1 (`hindi_084`)" not in report_text
    assert "1 (`hindi_084`)" not in readme_text

    # Ensure accurate test count (368 tests) and no stale 367 references
    assert "368/368 Tests Passing" in report_text
    assert "368 tests" in readme_text
    assert "367" not in report_text
    assert "367" not in readme_text

    # Ensure one-pager.md states 57 evaluated clips and not stale 37 clips for outlier immunity across all benchmarks
    assert "across all 57 evaluated clips" in onepager_text
    assert "across all 57 clips" in onepager_text
    assert "(**0 outliers** across all 37 clips)" not in onepager_text

    # Ensure exact corpus duration (8,064.95s) consistency across README.md and benchmark_report.md
    assert "8,064.95s" in readme_text
    assert "8,064.9s " not in readme_text

    # Ensure Stage 2 selective text verification trigger rate is 55% (11/20 clips) in creative report
    assert "Audio + Sel. Text (55%)" in creative_report_text
    assert "Audio + Sel. Text (45%)" not in creative_report_text

    # Verify Strategy C1 authoritative figures (78.76% SAA, 19.37% Gap, 24.77% WER, 43.70% cpWER, 4.667s)
    assert "| **Strategy C1: Advanced Token-0** | `gemini-3.5-flash-lite` | Yes | 78.76% | 19.37% | 24.77% | 43.70% | 4.667s |" in report_text
    assert "| **Strategy C1: Advanced Token-0** | `gemini-3.5-flash-lite` | Yes | 78.76% | 19.37% | 24.77% | 43.70% | 4.667s |" in readme_text

    # Verify all 8 rows of Section 5.1 macro/slice summary table in benchmark_report.md against summary JSON
    summary_path = RESULTS_ALL_2SPK_DIR / "all_2spk_comparative_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    expected_sec51_slices = [
        ("Full 37-Sample Corpus", "gemini_2_5_flash", summary_data["macro_summary"]["gemini_2_5_flash"]),
        ("Full 37-Sample Corpus", "strategy_c5_adaptive_champion", summary_data["macro_summary"]["strategy_c5_adaptive_champion"]),
        ("Hard High-Overlap Slice", "gemini_2_5_flash", summary_data["slices"]["hard_20_high_overlap"]["gemini_2_5_flash"]),
        ("Hard High-Overlap Slice", "strategy_c5_adaptive_champion", summary_data["slices"]["hard_20_high_overlap"]["strategy_c5_adaptive_champion"]),
        ("Moderate-Overlap Slice", "gemini_2_5_flash", summary_data["slices"]["remaining_17_moderate_overlap"]["gemini_2_5_flash"]),
        ("Moderate-Overlap Slice", "strategy_c5_adaptive_champion", summary_data["slices"]["remaining_17_moderate_overlap"]["strategy_c5_adaptive_champion"]),
        ("Newly Evaluated Live Slice", "gemini_2_5_flash", summary_data["slices"]["newly_evaluated_16_live"]["gemini_2_5_flash"]),
        ("Newly Evaluated Live Slice", "strategy_c5_adaptive_champion", summary_data["slices"]["newly_evaluated_16_live"]["strategy_c5_adaptive_champion"]),
    ]
    for slice_label, model_key, m in expected_sec51_slices:
        saa_str = f"{m['mean_speaker_attribution_accuracy'] * 100:.2f}%"
        gap_str = f"{m['mean_diarization_gap'] * 100:.2f}%"
        raw_wer_str = f"{m['mean_wer'] * 100:.2f}%"
        norm_wer_str = f"{m['mean_wer_normalized'] * 100:.2f}%"
        raw_cer_str = f"{m['mean_cer'] * 100:.2f}%"
        norm_cer_str = f"{m['mean_cer_normalized'] * 100:.2f}%"
        raw_cpwer_str = f"{m['mean_cpwer'] * 100:.2f}%"
        norm_cpwer_str = f"{m['mean_cpwer_normalized'] * 100:.2f}%"
        lat_str = f"{m['mean_latency_seconds']:.3f}s"
        speedup_str = f"{m['speedup_vs_2_5']:.2f}×"
        assert saa_str in report_text, f"Missing {saa_str} for {slice_label} {model_key}"
        assert gap_str in report_text, f"Missing {gap_str} for {slice_label} {model_key}"
        assert raw_wer_str in report_text, f"Missing {raw_wer_str} for {slice_label} {model_key}"
        assert norm_wer_str in report_text, f"Missing {norm_wer_str} for {slice_label} {model_key}"
        assert raw_cer_str in report_text, f"Missing {raw_cer_str} for {slice_label} {model_key}"
        assert norm_cer_str in report_text, f"Missing {norm_cer_str} for {slice_label} {model_key}"
        assert raw_cpwer_str in report_text, f"Missing {raw_cpwer_str} for {slice_label} {model_key}"
        assert norm_cpwer_str in report_text, f"Missing {norm_cpwer_str} for {slice_label} {model_key}"
        assert lat_str in report_text, f"Missing {lat_str} for {slice_label} {model_key}"
        assert speedup_str in report_text, f"Missing {speedup_str} for {slice_label} {model_key}"


    # Verify all 20 sample rows in Section 4.1 of benchmark_report.md and Section 5.1 of creative report
    sec41_pattern = re.compile(
        r"\|\s*`(hindi_\d+)`\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)%\s*\|\s*(?:\*\*)?([\d\.]+)%(?:\*\*)?\s*\|"
        r"\s*(?:\*\*)?([\d\.]+)%(?:\*\*)?\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)s\s*\|"
    )
    for doc_name, doc_content in [
        ("benchmark_report.md", report_text),
        ("hard_2spk_creative_parity_report.md", creative_report_text),
    ]:
        rows_20 = sec41_pattern.findall(doc_content)
        assert len(rows_20) == 20, f"Expected 20 rows in {doc_name}, found {len(rows_20)}"
        for sid, dur, ov, saa, gap, wer, cpwer, lat in rows_20:
            item = hard_c5[sid]
            assert abs(float(dur) - item["duration_seconds"]) < 0.1, f"Dur mismatch on {sid} in {doc_name}"
            assert abs(float(ov) - item["overlap_ratio"]) < 0.02, f"Ov mismatch on {sid} in {doc_name}"
            assert abs(float(saa) - item["metrics"]["speaker_attribution_accuracy"] * 100) < 0.02, f"SAA mismatch on {sid} in {doc_name}"
            assert abs(float(gap) - item["metrics"]["diarization_gap"] * 100) < 0.02, f"Gap mismatch on {sid} in {doc_name}"
            assert abs(float(wer) - item["metrics"]["wer"] * 100) < 0.02, f"WER mismatch on {sid} in {doc_name}"
            assert abs(float(cpwer) - item["metrics"]["cpwer"] * 100) < 0.02, f"cpWER mismatch on {sid} in {doc_name}"
            assert abs(float(lat) - item["latency_seconds"]) < 0.02, f"Lat mismatch on {sid} in {doc_name}"

    # Verify all 37 sample rows in Section 5.3 of benchmark_report.md match JSON values
    row_pattern = re.compile(
        r"\|\s*\d+\s*\|\s*`(hindi_\d+)`\s*\|\s*([\d\.]+)s\s*\|\s*([\d\.]+)%\s*\|"
        r"\s*[^\|]+\|\s*([\d\.]+)%\s*\|\s*(?:\*\*)?([\d\.]+)%(?:\*\*)?\s*\|"
        r"\s*(?:\*\*)?([\d\.]+)%\*?(?:\*\*)?\s*\|\s*(?:\*\*)?([\d\.]+)%(?:\*\*)?\s*\|"
        r"\s*([\d\.]+)s\s*\|\s*(?:\*\*)?([\d\.]+)s(?:\*\*)?"
    )
    rows = row_pattern.findall(report_text)
    assert len(rows) == 37, f"Expected 37 rows in benchmark_report.md section 5.3, found {len(rows)}"

    for sid, dur, ov, saa25, saac5, wer25, werc5, lat25, latc5 in rows:
        d25 = preds_25[sid]
        dc5 = preds_c5[sid]
        assert abs(float(dur) - d25["duration_seconds"]) < 0.02, f"Duration mismatch on {sid}"
        assert abs(float(ov) - d25["overlap_ratio"]) < 0.02, f"Overlap mismatch on {sid}"
        assert abs(float(saa25) - d25["metrics"]["speaker_attribution_accuracy"] * 100) < 0.02, f"SAA25 mismatch on {sid}"
        assert abs(float(saac5) - dc5["metrics"]["speaker_attribution_accuracy"] * 100) < 0.02, f"SAAC5 mismatch on {sid}"
        assert abs(float(wer25) - d25["metrics"]["wer"] * 100) < 0.02, f"WER25 mismatch on {sid}"
        assert abs(float(werc5) - dc5["metrics"]["wer"] * 100) < 0.02, f"WERC5 mismatch on {sid}"
        assert abs(float(lat25) - d25["latency_seconds"]) < 0.02, f"Lat25 mismatch on {sid}"
        assert abs(float(latc5) - dc5["latency_seconds"]) < 0.02, f"LatC5 mismatch on {sid}"


