#!/usr/bin/env python3
"""Zero-trust programmatic recalculation of hard 2-speaker benchmark metrics.

Verifies:
1. Predictions in `predictions_strategy_a_token0_bypass.json` recomputed against `metadata.json`.
2. Predictions in `predictions_strategy_b_pure_lite_twopass.json` recomputed against `metadata.json`.
3. Macro averages match `hard_2spk_comparative_summary.json` exactly.
4. Recovery metrics on key failure cases: hindi_062, hindi_064, hindi_085, hindi_067.
"""

import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import MetricsEngine
from src.models import Turn


def run_zero_trust_verification():
    metadata_path = PROJECT_ROOT / "data" / "hard_2spk_subset" / "metadata.json"
    summary_path = PROJECT_ROOT / "results" / "hard_2spk" / "hard_2spk_comparative_summary.json"
    pred_a_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_a_token0_bypass.json"
    pred_b_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_strategy_b_pure_lite_twopass.json"
    pred_base_lite_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_gemini_3_5_flash_lite_default.json"
    pred_base_25_path = PROJECT_ROOT / "results" / "hard_2spk" / "predictions_gemini_2_5_flash.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    ground_truth = {item["sample_id"]: item for item in metadata_list}

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    with open(pred_a_path, "r", encoding="utf-8") as f:
        pred_a_list = json.load(f)

    with open(pred_b_path, "r", encoding="utf-8") as f:
        pred_b_list = json.load(f)

    with open(pred_base_lite_path, "r", encoding="utf-8") as f:
        pred_base_lite_list = json.load(f)

    with open(pred_base_25_path, "r", encoding="utf-8") as f:
        pred_base_25_list = json.load(f)

    print(f"Total samples in metadata: {len(ground_truth)}")
    print(f"Total samples in pred_a: {len(pred_a_list)}")
    print(f"Total samples in pred_b: {len(pred_b_list)}")

    strategies = [
        ("strategy_a_token0_bypass", pred_a_list),
        ("strategy_b_pure_lite_twopass", pred_b_list),
    ]

    all_results = {}

    for strat_key, preds in strategies:
        print(f"\n==========================================")
        print(f"Verifying {strat_key} ({len(preds)} samples)")
        print(f"==========================================")

        recalc_records = []
        discrepancies = []

        for record in preds:
            sid = record["sample_id"]
            assert sid in ground_truth, f"Sample {sid} not found in metadata"
            gt = ground_truth[sid]

            ref_turns = [Turn(**t) for t in gt["ground_truth_turns"]]
            hyp_turns = [Turn(**t) for t in record["predicted_turns"]]

            # Compute independently
            eval_metrics = MetricsEngine.evaluate_sample(
                ref_turns=ref_turns,
                hyp_turns=hyp_turns,
                normalize=True,
            )

            rec_wer = round(eval_metrics.wer, 4)
            rec_cer = round(eval_metrics.cer, 4)
            rec_saa = round(eval_metrics.speaker_attribution_accuracy, 4)
            rec_cpwer = round(eval_metrics.cpwer, 4)
            rec_gap = round(eval_metrics.diarization_gap, 4)

            recorded_metrics = record["metrics"]

            diff_wer = abs(rec_wer - recorded_metrics["wer"])
            diff_cer = abs(rec_cer - recorded_metrics["cer"])
            diff_saa = abs(rec_saa - recorded_metrics["speaker_attribution_accuracy"])
            diff_cpwer = abs(rec_cpwer - recorded_metrics["cpwer"])
            diff_gap = abs(rec_gap - recorded_metrics["diarization_gap"])

            max_diff = max(diff_wer, diff_cer, diff_saa, diff_cpwer, diff_gap)
            if max_diff > 1e-4:
                discrepancies.append({
                    "sample_id": sid,
                    "recalculated": {
                        "wer": rec_wer, "cer": rec_cer, "saa": rec_saa, "cpwer": rec_cpwer, "gap": rec_gap
                    },
                    "recorded": recorded_metrics,
                    "max_diff": max_diff,
                })

            recalc_records.append({
                "sample_id": sid,
                "latency_seconds": record["latency_seconds"],
                "metrics": {
                    "wer": rec_wer,
                    "cer": rec_cer,
                    "speaker_attribution_accuracy": rec_saa,
                    "cpwer": rec_cpwer,
                    "diarization_gap": rec_gap,
                    "speaker_mapping": eval_metrics.speaker_mapping,
                    "format_valid": eval_metrics.format_valid,
                }
            })

        print(f"Sample-level metric check: {len(discrepancies)} discrepancies found.")
        if discrepancies:
            for d in discrepancies:
                print(f"  DISCREPANCY in {d['sample_id']}: {d}")

        # Compute independent macro averages
        wers = [r["metrics"]["wer"] for r in recalc_records]
        cers = [r["metrics"]["cer"] for r in recalc_records]
        saas = [r["metrics"]["speaker_attribution_accuracy"] for r in recalc_records]
        cpwers = [r["metrics"]["cpwer"] for r in recalc_records]
        gaps = [r["metrics"]["diarization_gap"] for r in recalc_records]
        lats = [r["latency_seconds"] for r in recalc_records]

        macro_wer = round(float(np.mean(wers)), 4)
        macro_cer = round(float(np.mean(cers)), 4)
        macro_saa = round(float(np.mean(saas)), 4)
        macro_cpwer = round(float(np.mean(cpwers)), 4)
        macro_gap = round(float(np.mean(gaps)), 4)
        macro_lat = round(float(np.mean(lats)), 3)

        # Outlier filtering (WER > 2.0)
        norm_wers = [w for w in wers if w <= 2.0]
        norm_cers = [c for c in cers if c <= 2.0]
        norm_cpwers = [cp for cp in cpwers if cp <= 2.0]
        macro_wer_norm = round(float(np.mean(norm_wers)), 4)
        macro_cer_norm = round(float(np.mean(norm_cers)), 4)
        macro_cpwer_norm = round(float(np.mean(norm_cpwers)), 4)
        outlier_count = len(wers) - len(norm_wers)

        independent_macro = {
            "count": len(recalc_records),
            "mean_wer": macro_wer,
            "mean_wer_normalized": macro_wer_norm,
            "mean_cer": macro_cer,
            "mean_cer_normalized": macro_cer_norm,
            "mean_speaker_attribution_accuracy": macro_saa,
            "mean_cpwer": macro_cpwer,
            "mean_cpwer_normalized": macro_cpwer_norm,
            "mean_diarization_gap": macro_gap,
            "mean_latency_seconds": macro_lat,
            "outlier_count": outlier_count,
        }

        reported_macro = summary_data["macro_summary"][strat_key]

        print("\nMacro Metrics Comparison:")
        print(f"{'Metric':<35} | {'Independent':<12} | {'Reported':<12} | {'Match?':<6}")
        print("-" * 75)
        mismatches = []
        for k in ["count", "mean_wer", "mean_wer_normalized", "mean_cer", "mean_cer_normalized",
                  "mean_speaker_attribution_accuracy", "mean_cpwer", "mean_cpwer_normalized",
                  "mean_diarization_gap", "mean_latency_seconds", "outlier_count"]:
            ind_val = independent_macro[k]
            rep_val = reported_macro.get(k)
            match = (ind_val == rep_val) if not isinstance(ind_val, float) else abs(ind_val - rep_val) < 1e-4
            if not match:
                mismatches.append((k, ind_val, rep_val))
            print(f"{k:<35} | {str(ind_val):<12} | {str(rep_val):<12} | {('PASS' if match else 'FAIL'):<6}")

        all_results[strat_key] = {
            "independent_macro": independent_macro,
            "reported_macro": reported_macro,
            "discrepancies": discrepancies,
            "mismatches": mismatches,
            "recalc_records": recalc_records,
        }

    # Verify recovery on failure cases: hindi_062, hindi_064, hindi_085, hindi_067
    print("\n=======================================================")
    print("Verification of Recovery on Extreme Overlap Failure Cases")
    print("=======================================================")

    target_samples = ["hindi_062", "hindi_064", "hindi_085", "hindi_067"]

    base_lite_dict = {r["sample_id"]: r for r in pred_base_lite_list}
    base_25_dict = {r["sample_id"]: r for r in pred_base_25_list}
    strat_a_dict = {r["sample_id"]: r for r in pred_a_list}
    strat_b_dict = {r["sample_id"]: r for r in pred_b_list}

    recovery_analysis = {}

    for sid in target_samples:
        gt = ground_truth[sid]
        b_lite = base_lite_dict[sid]["metrics"]
        b_25 = base_25_dict[sid]["metrics"]
        s_a = strat_a_dict[sid]["metrics"]
        s_b = strat_b_dict[sid]["metrics"]

        overlap = gt["overlap_ratio"]
        dur = gt["duration_seconds"]

        print(f"\nSample: {sid} (Duration: {dur}s, Overlap: {overlap}%)")
        print(f"  Model/Strategy                 | SAA       | WER       | cpWER     | DiarGap")
        print(f"  --------------------------------------------------------------------------")
        print(f"  Gemini 3.5 Flash Lite Default  | {b_lite['speaker_attribution_accuracy']*100:6.2f}%   | {b_lite['wer']*100:6.2f}%   | {b_lite['cpwer']*100:6.2f}%   | {b_lite['diarization_gap']*100:6.2f}%")
        print(f"  Strategy A: Token-0 Bypass     | {s_a['speaker_attribution_accuracy']*100:6.2f}%   | {s_a['wer']*100:6.2f}%   | {s_a['cpwer']*100:6.2f}%   | {s_a['diarization_gap']*100:6.2f}%")
        print(f"  Strategy B: Pure Lite Two-Pass | {s_b['speaker_attribution_accuracy']*100:6.2f}%   | {s_b['wer']*100:6.2f}%   | {s_b['cpwer']*100:6.2f}%   | {s_b['diarization_gap']*100:6.2f}%")
        print(f"  Gemini 2.5 Flash Baseline      | {b_25['speaker_attribution_accuracy']*100:6.2f}%   | {b_25['wer']*100:6.2f}%   | {b_25['cpwer']*100:6.2f}%   | {b_25['diarization_gap']*100:6.2f}%")

        saa_gain_a = (s_a["speaker_attribution_accuracy"] - b_lite["speaker_attribution_accuracy"]) * 100
        saa_gain_b = (s_b["speaker_attribution_accuracy"] - b_lite["speaker_attribution_accuracy"]) * 100
        gap_red_a = (b_lite["diarization_gap"] - s_a["diarization_gap"]) * 100
        gap_red_b = (b_lite["diarization_gap"] - s_b["diarization_gap"]) * 100

        print(f"  -> Strategy A recovery: SAA delta = {saa_gain_a:+.2f}%p, DiarGap delta = {-gap_red_a:+.2f}%p")
        print(f"  -> Strategy B recovery: SAA delta = {saa_gain_b:+.2f}%p, DiarGap delta = {-gap_red_b:+.2f}%p")

        recovery_analysis[sid] = {
            "duration": dur,
            "overlap_ratio": overlap,
            "base_lite": b_lite,
            "base_25": b_25,
            "strategy_a": s_a,
            "strategy_b": s_b,
            "gain_a": {"saa_delta_pct": saa_gain_a, "gap_reduction_pct": gap_red_a},
            "gain_b": {"saa_delta_pct": saa_gain_b, "gap_reduction_pct": gap_red_b},
        }

    return all_results, recovery_analysis


def test_zero_trust_verification():
    results, recovery = run_zero_trust_verification()
    # Assert exact match for Strategy A
    strat_a = results["strategy_a_token0_bypass"]
    assert len(strat_a["discrepancies"]) == 0, f"Discrepancies found in Strategy A: {strat_a['discrepancies']}"
    assert len(strat_a["mismatches"]) == 0, f"Macro mismatches in Strategy A: {strat_a['mismatches']}"
    assert strat_a["independent_macro"]["mean_speaker_attribution_accuracy"] == 0.8111
    assert strat_a["independent_macro"]["mean_diarization_gap"] == 0.1780
    assert strat_a["independent_macro"]["mean_wer"] == 0.2509
    assert strat_a["independent_macro"]["mean_cpwer"] == 0.4250

    # Assert exact match for Strategy B
    strat_b = results["strategy_b_pure_lite_twopass"]
    assert len(strat_b["discrepancies"]) == 0, f"Discrepancies found in Strategy B: {strat_b['discrepancies']}"
    assert len(strat_b["mismatches"]) == 0, f"Macro mismatches in Strategy B: {strat_b['mismatches']}"
    assert strat_b["independent_macro"]["mean_speaker_attribution_accuracy"] == 0.7433
    assert strat_b["independent_macro"]["mean_diarization_gap"] == 0.2660
    assert strat_b["independent_macro"]["mean_wer"] == 0.2529
    assert strat_b["independent_macro"]["mean_cpwer"] == 0.5187

    # Assert target recovery criteria
    # hindi_062
    assert recovery["hindi_062"]["strategy_a"]["speaker_attribution_accuracy"] > recovery["hindi_062"]["base_lite"]["speaker_attribution_accuracy"]
    assert recovery["hindi_062"]["strategy_b"]["speaker_attribution_accuracy"] > recovery["hindi_062"]["base_lite"]["speaker_attribution_accuracy"]

    # hindi_085
    assert recovery["hindi_085"]["strategy_a"]["speaker_attribution_accuracy"] > 0.95
    assert recovery["hindi_085"]["strategy_a"]["diarization_gap"] == 0.0

    # hindi_067
    assert recovery["hindi_067"]["strategy_a"]["speaker_attribution_accuracy"] > recovery["hindi_067"]["base_25"]["speaker_attribution_accuracy"]


if __name__ == "__main__":
    results, recovery = run_zero_trust_verification()
    test_zero_trust_verification()
    print("\nALL ZERO-TRUST PYTEST ASSERTIONS PASSED SUCCESSFULLY!")
