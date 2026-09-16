#!/usr/bin/env python3
"""Independent Metric Recalculation & Statistical Verification for Milestone 4.

Author: m4_challenger_2
Mission:
1. Re-read raw prediction JSON files from disk:
   - results/hard_2spk/predictions_strategy_c5_adaptive_champion.json
   - results/optimized/predictions_strategy_c5_generalization.json
   - (and Cycle 1 candidates C3, C4)
2. Independently recalculate all metrics from scratch using `src/metrics.py`:
   - Hungarian Speaker Attribution Accuracy (SAA)
   - Word Error Rate (WER)
   - Character Error Rate (CER)
   - Concatenated Permutation WER (cpWER)
   - Diarization Degradation Gap (cpWER - WER)
   - Mean Latency
3. Verify that recalculated values match reported macro summaries:
   - Hard 2-speaker benchmark (20 clips): SAA = 87.10%, Diarization Gap = 12.69%, WER = 24.86%, Latency = 5.811s.
   - Generalization benchmark (20 clips): SAA = 77.75%, Diarization Gap = 24.05%, WER = 23.49%, Latency = 4.339s.
4. Comprehensive integrity audit:
   - Zero empty, missing, or fabricated predictions
   - Pure model constraint (model_id == 'gemini-3.5-flash-lite')
   - Raw response content audit vs parsed turn structure
   - Multi-speaker slice stratification (2-speaker vs 3-speaker)
"""

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import MetricsEngine
from src.models import Turn


def load_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def recompute_metrics_for_dataset(
    pred_path: Path,
    metadata_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Recomputes sample-level metrics and macro averages independently."""
    preds = load_json(pred_path)
    metadata = load_json(metadata_path)
    gt_map = {item["sample_id"]: item for item in metadata}

    sample_results = []
    for pred in preds:
        sid = pred["sample_id"]
        assert sid in gt_map, f"Sample {sid} not found in ground truth metadata!"
        gt_item = gt_map[sid]

        ref_turns = [Turn(**t) for t in gt_item["ground_truth_turns"]]
        hyp_turns = [Turn(**t) for t in pred["predicted_turns"]]

        # Recompute independently from scratch via MetricsEngine
        eval_metrics = MetricsEngine.evaluate_sample(ref_turns, hyp_turns, normalize=True)

        sample_results.append({
            "sample_id": sid,
            "latency_seconds": pred.get("latency_seconds", 0.0),
            "num_speakers": pred.get("num_speakers", gt_item.get("num_speakers", 2)),
            "overlap_ratio": pred.get("overlap_ratio", gt_item.get("overlap_ratio", 0.0)),
            "stored_metrics": pred.get("metrics", {}),
            "raw_response": pred.get("raw_response", ""),
            "model_id": pred.get("model_id", ""),
            "approach": pred.get("approach", ""),
            "predicted_turns": hyp_turns,
            "recomputed_metrics_raw": {
                "wer": eval_metrics.wer,
                "cer": eval_metrics.cer,
                "speaker_attribution_accuracy": eval_metrics.speaker_attribution_accuracy,
                "cpwer": eval_metrics.cpwer,
                "diarization_gap": eval_metrics.diarization_gap,
                "speaker_mapping": eval_metrics.speaker_mapping,
                "format_valid": eval_metrics.format_valid,
            },
            "recomputed_metrics_rounded": {
                "wer": round(eval_metrics.wer, 4),
                "cer": round(eval_metrics.cer, 4),
                "speaker_attribution_accuracy": round(eval_metrics.speaker_attribution_accuracy, 4),
                "cpwer": round(eval_metrics.cpwer, 4),
                "diarization_gap": round(eval_metrics.diarization_gap, 4),
                "speaker_mapping": eval_metrics.speaker_mapping,
                "format_valid": eval_metrics.format_valid,
            },
        })

    def calc_macro(from_rounded: bool = False) -> Dict[str, Any]:
        key = "recomputed_metrics_rounded" if from_rounded else "recomputed_metrics_raw"
        wers = [s[key]["wer"] for s in sample_results]
        cers = [s[key]["cer"] for s in sample_results]
        saas = [s[key]["speaker_attribution_accuracy"] for s in sample_results]
        cpwers = [s[key]["cpwer"] for s in sample_results]
        gaps = [s[key]["diarization_gap"] for s in sample_results]
        latencies = [s["latency_seconds"] for s in sample_results]

        n = len(sample_results)
        normal_wers = [w for w in wers if w <= 2.0]
        normal_cers = [c for c in cers if c <= 2.0]
        normal_cpwers = [cp for cp in cpwers if cp <= 2.0]

        return {
            "count": n,
            "mean_wer": round(sum(wers) / n, 4) if n else 0.0,
            "mean_wer_normalized": round(sum(normal_wers) / len(normal_wers), 4) if normal_wers else 0.0,
            "mean_cer": round(sum(cers) / n, 4) if n else 0.0,
            "mean_cer_normalized": round(sum(normal_cers) / len(normal_cers), 4) if normal_cers else 0.0,
            "mean_speaker_attribution_accuracy": round(sum(saas) / n, 4) if n else 0.0,
            "mean_cpwer": round(sum(cpwers) / n, 4) if n else 0.0,
            "mean_cpwer_normalized": round(sum(normal_cpwers) / len(normal_cpwers), 4) if normal_cpwers else 0.0,
            "mean_diarization_gap": round(sum(gaps) / n, 4) if n else 0.0,
            "mean_latency_seconds": round(sum(latencies) / n, 3) if n else 0.0,
            "outlier_count": len(wers) - len(normal_wers),
        }

    macro_from_raw = calc_macro(from_rounded=False)
    macro_from_rounded = calc_macro(from_rounded=True)

    return sample_results, macro_from_raw, macro_from_rounded


class TestMilestone4IndependentMetricRecalculation:
    """Comprehensive metric verification suite for M4 Champion and Generalization."""

    def test_recalculate_strategy_c5_hard_2spk(self):
        """Verify independent recalculation of Candidate C5 on the 20 hard 2-speaker clips."""
        pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c5_adaptive_champion.json"
        meta_path = PROJECT_ROOT / "data" / "hard_2spk_subset" / "metadata.json"
        summary_path = PROJECT_ROOT / "results" / "hard_2spk" / "hard_2spk_comparative_summary.json"

        samples, macro_raw, macro_rounded = recompute_metrics_for_dataset(pred_path, meta_path)
        assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"

        # 1. Check sample-level match between stored and recomputed metrics (4-decimal precision)
        for s in samples:
            stored = s["stored_metrics"]
            recomp = s["recomputed_metrics_rounded"]
            for k in ["wer", "cer", "speaker_attribution_accuracy", "cpwer", "diarization_gap"]:
                assert stored[k] == recomp[k], (
                    f"Sample {s['sample_id']} metric {k} mismatch: stored={stored[k]}, recomp={recomp[k]}"
                )

        # 2. Check macro summary comparison
        summary = load_json(summary_path)
        reported = summary["macro_summary"]["strategy_c5_adaptive_champion"]

        # Expected reported values:
        # SAA: 87.10% (0.8710)
        # Diarization Gap: 12.69% (0.1269)
        # WER: 24.86% (0.2486)
        # Mean Latency: 5.811s
        assert macro_rounded["mean_speaker_attribution_accuracy"] == 0.8710
        assert macro_rounded["mean_diarization_gap"] == 0.1269
        assert macro_rounded["mean_wer"] == 0.2486
        assert macro_rounded["mean_cer"] == 0.1361
        assert macro_rounded["mean_cpwer"] == 0.3710
        assert macro_rounded["mean_latency_seconds"] == 5.811
        assert macro_rounded["outlier_count"] == 0

        # Verify against comparative summary JSON fields directly
        assert reported["mean_speaker_attribution_accuracy"] == 0.871
        assert reported["mean_diarization_gap"] == 0.1269
        assert reported["mean_wer"] == 0.2486
        assert reported["mean_cer"] == 0.1361
        assert reported["mean_cpwer"] == 0.371
        assert reported["mean_latency_seconds"] == 5.811

        # Check acceptance criteria compliance
        assert macro_rounded["mean_speaker_attribution_accuracy"] >= 0.850, "SAA below 85.0% threshold"
        assert macro_rounded["mean_speaker_attribution_accuracy"] >= 0.865, "SAA below 86.5% stretch target"
        assert macro_rounded["mean_diarization_gap"] <= 0.145, "Diarization Gap above 14.5% threshold"
        assert macro_rounded["mean_diarization_gap"] < 0.1316, "Diarization Gap did not beat 2.5 Flash (13.16%)"
        assert macro_rounded["mean_latency_seconds"] < 7.033, "Latency exceeds 2.5 Flash (7.033s)"

    def test_recalculate_strategy_c5_generalization(self):
        """Verify independent recalculation of Candidate C5 on the 20-clip Generalization dataset."""
        pred_path = PROJECT_ROOT / "results" / "optimized" / "predictions_strategy_c5_generalization.json"
        meta_path = PROJECT_ROOT / "data" / "indic_diarbench_subset" / "metadata.json"
        summary_path = PROJECT_ROOT / "results" / "optimized" / "generalization_summary.json"

        samples, macro_raw, macro_rounded = recompute_metrics_for_dataset(pred_path, meta_path)
        assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"

        # 1. Check sample-level match
        for s in samples:
            stored = s["stored_metrics"]
            recomp = s["recomputed_metrics_rounded"]
            for k in ["wer", "cer", "speaker_attribution_accuracy", "cpwer", "diarization_gap"]:
                assert stored[k] == recomp[k], (
                    f"Sample {s['sample_id']} metric {k} mismatch: stored={stored[k]}, recomp={recomp[k]}"
                )

        # 2. Check reported macro metrics
        summary = load_json(summary_path)
        reported = summary["macro_metrics"]

        # Expected reported values:
        # SAA: 77.75% (0.7775)
        # Diarization Gap: 24.05% (0.2405)
        # WER: 23.49% (0.2349)
        # CER: 13.53% (0.1353)
        # cpWER: 47.54% (0.4754)
        # Mean Latency: 4.339s
        assert macro_rounded["mean_speaker_attribution_accuracy"] == 0.7775
        assert macro_rounded["mean_diarization_gap"] == 0.2405
        assert macro_rounded["mean_wer"] == 0.2349
        assert macro_rounded["mean_cer"] == 0.1353
        assert macro_rounded["mean_cpwer"] == 0.4754
        assert macro_rounded["mean_latency_seconds"] == 4.339
        assert macro_rounded["outlier_count"] == 0

        assert reported["mean_speaker_attribution_accuracy"] == 0.7775
        assert reported["mean_diarization_gap"] == 0.2405
        assert reported["mean_wer"] == 0.2349
        assert reported["mean_cer"] == 0.1353
        assert reported["mean_cpwer"] == 0.4754
        assert reported["mean_latency_seconds"] == 4.339

        # Confirm Zero Regression vs Default 3.5 Flash Lite baseline (69.04% SAA)
        default_baseline_saa = 0.6904
        assert macro_rounded["mean_speaker_attribution_accuracy"] > default_baseline_saa + 0.08, (
            f"Gain over baseline is less than expected: {macro_rounded['mean_speaker_attribution_accuracy']} vs {default_baseline_saa}"
        )

    def test_recalculate_generalization_speaker_slices(self):
        """Verify multi-speaker slice stratification (15 2-speaker clips and 5 3-speaker clips)."""
        pred_path = PROJECT_ROOT / "results" / "optimized" / "predictions_strategy_c5_generalization.json"
        meta_path = PROJECT_ROOT / "data" / "indic_diarbench_subset" / "metadata.json"
        summary_path = PROJECT_ROOT / "results" / "optimized" / "generalization_summary.json"

        samples, _, _ = recompute_metrics_for_dataset(pred_path, meta_path)
        summary = load_json(summary_path)

        spk2_samples = [s for s in samples if s["num_speakers"] == 2]
        spk3_samples = [s for s in samples if s["num_speakers"] == 3]

        assert len(spk2_samples) == 15, f"Expected 15 2-speaker clips, found {len(spk2_samples)}"
        assert len(spk3_samples) == 5, f"Expected 5 3-speaker clips, found {len(spk3_samples)}"

        # 2-speaker slice checks
        rep_2 = summary["slices"]["2_speakers"]
        mean_saa_2 = round(sum(s["recomputed_metrics_rounded"]["speaker_attribution_accuracy"] for s in spk2_samples) / 15, 4)
        mean_gap_2 = round(sum(s["recomputed_metrics_rounded"]["diarization_gap"] for s in spk2_samples) / 15, 4)
        mean_wer_2 = round(sum(s["recomputed_metrics_rounded"]["wer"] for s in spk2_samples) / 15, 4)
        mean_lat_2 = round(sum(s["latency_seconds"] for s in spk2_samples) / 15, 3)

        assert mean_saa_2 == 0.7853 == rep_2["mean_speaker_attribution_accuracy"]
        assert mean_gap_2 == 0.2193 == rep_2["mean_diarization_gap"]
        assert mean_wer_2 == 0.2454 == rep_2["mean_wer"]
        assert mean_lat_2 == 4.039 == rep_2["mean_latency_seconds"]

        # 3-speaker slice checks
        rep_3 = summary["slices"]["3_speakers"]
        mean_saa_3 = round(sum(s["recomputed_metrics_rounded"]["speaker_attribution_accuracy"] for s in spk3_samples) / 5, 4)
        mean_gap_3 = round(sum(s["recomputed_metrics_rounded"]["diarization_gap"] for s in spk3_samples) / 5, 4)
        mean_wer_3 = round(sum(s["recomputed_metrics_rounded"]["wer"] for s in spk3_samples) / 5, 4)
        mean_lat_3 = round(sum(s["latency_seconds"] for s in spk3_samples) / 5, 3)

        assert mean_saa_3 == 0.7542 == rep_3["mean_speaker_attribution_accuracy"]
        assert mean_gap_3 == 0.3040 == rep_3["mean_diarization_gap"]
        assert mean_wer_3 == 0.2033 == rep_3["mean_wer"]
        assert mean_lat_3 == 5.241 == rep_3["mean_latency_seconds"]

        # Note cpWER rounding nuance: sum of rounded is 0.5074, sum of raw is 0.5073
        mean_cpwer_3_rnd = round(sum(s["recomputed_metrics_rounded"]["cpwer"] for s in spk3_samples) / 5, 4)
        mean_cpwer_3_raw = round(sum(s["recomputed_metrics_raw"]["cpwer"] for s in spk3_samples) / 5, 4)
        assert rep_3["mean_cpwer"] in (mean_cpwer_3_rnd, mean_cpwer_3_raw)

    def test_prediction_data_integrity_and_anti_fabrication(self):
        """Forensic audit: verify no missing, empty, or fabricated prediction entries."""
        for pred_file in [
            PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c5_adaptive_champion.json",
            PROJECT_ROOT / "results" / "optimized" / "predictions_strategy_c5_generalization.json",
        ]:
            preds = load_json(pred_file)
            assert len(preds) == 20, f"{pred_file.name}: Expected 20 predictions, got {len(preds)}"

            sample_ids = [p["sample_id"] for p in preds]
            assert len(set(sample_ids)) == 20, f"{pred_file.name}: Duplicate sample_ids found!"

            for p in preds:
                sid = p["sample_id"]
                assert p["model_id"] == "gemini-3.5-flash-lite", f"{sid}: illegal model {p['model_id']}"
                assert p["approach"] == "strategy_c5_adaptive_champion"
                assert p["latency_seconds"] > 1.0, f"{sid}: suspiciously low latency {p['latency_seconds']}"
                assert p["latency_seconds"] < 30.0, f"{sid}: excessive latency {p['latency_seconds']}"

                raw_resp = p["raw_response"]
                assert len(raw_resp.strip()) > 50, f"{sid}: raw_response too short or empty"
                assert "=== STAGE 1" in raw_resp or "[Utterance]" in raw_resp, (
                    f"{sid}: raw_response missing C5 prompt syntax headers"
                )

                turns = p["predicted_turns"]
                assert len(turns) >= 2, f"{sid}: suspiciously low turn count ({len(turns)})"

                # Check turn syntax and absence of fabrication
                for t_idx, t in enumerate(turns):
                    spk = t["speaker"]
                    txt = t["text"]
                    assert re.match(r"^Speaker \d+$", spk), f"{sid} turn {t_idx} invalid speaker: {spk}"
                    assert len(txt.strip()) > 0, f"{sid} turn {t_idx} has empty text"

                # Grounding check: verify that words from predicted turns appear in raw response
                for t in turns[:3]:
                    words = [w for w in t["text"].split() if len(w) > 2]
                    if words:
                        assert any(w in raw_resp for w in words), (
                            f"{sid}: Turn words {words} not found anywhere in raw_response!"
                        )

    def test_cycle1_candidates_c3_and_c4_recalculation(self):
        """Verify Candidate C3 and C4 benchmark data in results/hard_2spk/."""
        meta_path = PROJECT_ROOT / "data" / "hard_2spk_subset" / "metadata.json"
        summary_path = PROJECT_ROOT / "results" / "hard_2spk" / "hard_2spk_comparative_summary.json"
        summary = load_json(summary_path)["macro_summary"]

        # Candidate C3
        c3_pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c3_acoustic_token0.json"
        c3_samples, _, c3_macro = recompute_metrics_for_dataset(c3_pred_path, meta_path)
        assert len(c3_samples) == 20
        assert c3_macro["mean_speaker_attribution_accuracy"] == 0.7975
        assert c3_macro["mean_diarization_gap"] == 0.2149
        assert c3_macro["mean_wer"] == 0.2410
        assert summary["strategy_c3_acoustic_token0"]["mean_speaker_attribution_accuracy"] == 0.7975

        # Candidate C4
        c4_pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c4_multistage.json"
        c4_samples, _, c4_macro = recompute_metrics_for_dataset(c4_pred_path, meta_path)
        assert len(c4_samples) == 20
        assert c4_macro["mean_speaker_attribution_accuracy"] == 0.7899
        assert c4_macro["mean_diarization_gap"] == 0.2041
        assert c4_macro["mean_wer"] == 0.2396
        assert summary["strategy_c4_multistage"]["mean_speaker_attribution_accuracy"] == 0.7899

        # Verify C5 exceeds both C3 and C4 significantly
        c5_pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c5_adaptive_champion.json"
        _, _, c5_macro = recompute_metrics_for_dataset(c5_pred_path, meta_path)
        assert c5_macro["mean_speaker_attribution_accuracy"] - c3_macro["mean_speaker_attribution_accuracy"] > 0.07
        assert c5_macro["mean_speaker_attribution_accuracy"] - c4_macro["mean_speaker_attribution_accuracy"] > 0.08
        assert c3_macro["mean_diarization_gap"] - c5_macro["mean_diarization_gap"] > 0.08
        assert c4_macro["mean_diarization_gap"] - c5_macro["mean_diarization_gap"] > 0.07

    def test_hungarian_bipartite_mapping_soundness(self):
        """Verify mathematical integrity of Hungarian bipartite speaker mapping."""
        meta_path = PROJECT_ROOT / "data" / "hard_2spk_subset" / "metadata.json"
        c5_pred_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_c5_adaptive_champion.json"

        metadata = load_json(meta_path)
        preds = load_json(c5_pred_path)
        gt_map = {m["sample_id"]: m for m in metadata}

        for p in preds:
            sid = p["sample_id"]
            gt = gt_map[sid]
            ref_turns = [Turn(**t) for t in gt["ground_truth_turns"]]
            hyp_turns = [Turn(**t) for t in p["predicted_turns"]]

            acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref_turns, hyp_turns)
            assert 0.0 <= acc <= 1.0, f"{sid}: invalid accuracy {acc}"

            # Mapping must be injective (no two hyp speakers mapped to same ref speaker)
            ref_mapped = list(mapping.values())
            assert len(ref_mapped) == len(set(ref_mapped)), f"{sid}: Non-injective speaker mapping: {mapping}"

            # All mapped speakers must exist in ground truth
            unique_gt_spk = set(t.speaker for t in ref_turns)
            for ref_s in ref_mapped:
                assert ref_s in unique_gt_spk, f"{sid}: Mapped speaker {ref_s} not in ground truth {unique_gt_spk}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
