"""Unit tests for scripts/run_full_benchmark.py and comparative summary generation."""

import json
from pathlib import Path
import pytest

from scripts.run_full_benchmark import compile_comparative_summary
from src.config import (
    BASELINE_RESULTS_DIR,
    OPTIMIZED_RESULTS_DIR,
    RESULTS_DIR,
)


def test_compile_comparative_summary_structure(tmp_path):
    """Tests that compile_comparative_summary produces a valid JSON structure."""
    out_file = tmp_path / "test_summary.json"
    summary = compile_comparative_summary(
        baseline_dir=BASELINE_RESULTS_DIR,
        optimized_dir=OPTIMIZED_RESULTS_DIR,
        output_path=out_file,
    )

    assert out_file.exists()
    assert "macro_summary" in summary
    assert "slices" in summary
    assert "recovery_summary" in summary
    assert "sample_level_comparisons" in summary

    ms = summary["macro_summary"]
    assert "gemini_2_5_flash_baseline" in ms
    assert "gemini_3_5_flash_lite_default" in ms
    assert "strategy1_anchor_prompting" in ms
    assert "strategy2_two_step_decoupled" in ms

    # Check that Strategy 2 SAA beats 2.5 Flash Baseline
    b25_saa = ms["gemini_2_5_flash_baseline"]["mean_speaker_attribution_accuracy"]
    s2_saa = ms["strategy2_two_step_decoupled"]["mean_speaker_attribution_accuracy"]
    assert s2_saa > b25_saa, f"Strategy 2 SAA ({s2_saa}) must exceed 2.5 Flash SAA ({b25_saa})"

    # Check that Strategy 1 WER beats 2.5 Flash normalized WER
    b25_wer = ms["gemini_2_5_flash_baseline"]["mean_wer_normalized"]
    s1_wer = ms["strategy1_anchor_prompting"]["mean_wer"]
    assert s1_wer < b25_wer, f"Strategy 1 WER ({s1_wer}) must be lower than 2.5 Flash norm WER ({b25_wer})"

    # Check speedup
    assert ms["strategy1_anchor_prompting"]["latency_speedup_vs_2_5"] >= 2.0
    assert ms["strategy2_two_step_decoupled"]["latency_speedup_vs_2_5"] >= 1.2


def test_comparative_summary_persisted_file():
    """Tests that the canonical results/comparative_summary.json exists and is valid."""
    summary_path = RESULTS_DIR / "comparative_summary.json"
    assert summary_path.exists(), "results/comparative_summary.json must exist"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_samples"] == 20
    assert len(data["sample_level_comparisons"]) == 20
    assert data["recovery_summary"]["recovery_rate_pct"] == 100.0
