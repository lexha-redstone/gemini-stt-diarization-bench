"""Executes live baseline evaluation comparing Gemini 2.5 Flash vs Gemini 3.5 Flash Lite.

Runs Approach 1 (Single-Step Audio-to-Diarized Transcript) across all 20 benchmark
audio samples from data/benchmark_subset/.

Outputs:
- results/baseline/predictions_2.5_flash.json
- results/baseline/predictions_3.5_flash_lite.json
- results/baseline/baseline_comparison.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASELINE_MODELS,
    BASELINE_RESULTS_DIR,
    BENCHMARK_SUBSET_DIR,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_2_5_FLASH,
    MODEL_3_5_FLASH_LITE,
)
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import EvaluationMetrics, PipelinePrediction, SampleData
from src.pipelines.single_step import SingleStepPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_baseline")


def model_slug(model_name: str) -> str:
    """Converts a model name to a clean filesystem slug."""
    return model_name.replace("gemini-", "").replace("-", "_")


def run_baseline_evaluation(
    models: Optional[List[str]] = None,
    subset_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    sample_ids: Optional[List[str]] = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """Runs live baseline evaluation on all benchmark samples.

    Args:
        models: List of model strings to evaluate.
        subset_dir: Directory containing benchmark subset.
        output_dir: Directory to save results.
        sample_ids: Optional subset of sample IDs to evaluate.
        thinking_budget: Thinking budget token count (default 0).
        temperature: Temperature setting (default 0.0).

    Returns:
        Dictionary containing comparative evaluation summary.
    """
    eval_models = models or BASELINE_MODELS
    base_dir = Path(subset_dir) if subset_dir else BENCHMARK_SUBSET_DIR
    out_dir = Path(output_dir) if output_dir else BASELINE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark samples
    all_samples = DatasetLoader.load_benchmark_subset(subset_dir=base_dir)
    if sample_ids:
        samples = [s for s in all_samples if s.sample_id in sample_ids]
    else:
        samples = all_samples

    logger.info(
        f"Starting live baseline evaluation on {len(samples)} samples across models: {eval_models}"
    )

    pipeline = SingleStepPipeline(
        thinking_budget=thinking_budget,
        temperature=temperature,
    )

    # Store predictions and metrics per model
    # model -> list of prediction dicts
    model_predictions: Dict[str, List[Dict[str, Any]]] = {m: [] for m in eval_models}
    # model -> sample_id -> metrics dict
    model_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in eval_models}

    for model in eval_models:
        logger.info(f"\n{'='*30} Evaluating Model: {model} {'='*30}")
        slug = model_slug(model)
        predictions_file = out_dir / f"predictions_{slug}.json"

        for idx, sample in enumerate(samples, 1):
            logger.info(
                f"[{idx}/{len(samples)}] Running sample {sample.sample_id} "
                f"({sample.duration_seconds:.1f}s, {sample.num_speakers} spks, {sample.overlap_ratio:.1f}% ovl) on {model}..."
            )

            pred: PipelinePrediction = pipeline.run_sample(sample, model_id=model)
            eval_metrics: EvaluationMetrics = MetricsEngine.evaluate_sample(
                sample.ground_truth_turns, pred.predicted_turns
            )

            pred_record = {
                "sample_id": sample.sample_id,
                "model_id": model,
                "latency_seconds": round(pred.latency_seconds, 3),
                "duration_seconds": sample.duration_seconds,
                "num_speakers": sample.num_speakers,
                "overlap_ratio": sample.overlap_ratio,
                "num_turns_predicted": len(pred.predicted_turns),
                "num_turns_ground_truth": len(sample.ground_truth_turns),
                "metrics": {
                    "wer": round(eval_metrics.wer, 4),
                    "cer": round(eval_metrics.cer, 4),
                    "speaker_attribution_accuracy": round(
                        eval_metrics.speaker_attribution_accuracy, 4
                    ),
                    "cpwer": round(eval_metrics.cpwer, 4),
                    "diarization_gap": round(eval_metrics.diarization_gap, 4),
                    "speaker_mapping": eval_metrics.speaker_mapping,
                    "format_valid": eval_metrics.format_valid,
                },
                "predicted_turns": [
                    {"speaker": t.speaker, "text": t.text} for t in pred.predicted_turns
                ],
                "raw_response": pred.raw_response,
            }

            model_predictions[model].append(pred_record)
            model_metrics[model][sample.sample_id] = pred_record["metrics"]
            model_metrics[model][sample.sample_id]["latency_seconds"] = pred.latency_seconds

            logger.info(
                f"  -> Sample {sample.sample_id} | Latency: {pred.latency_seconds:.2f}s | "
                f"WER: {eval_metrics.wer:.4f} | SAA: {eval_metrics.speaker_attribution_accuracy:.4f} | "
                f"cpWER: {eval_metrics.cpwer:.4f} | DiarGap: {eval_metrics.diarization_gap:.4f}"
            )

        # Save model predictions to file
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(model_predictions[model], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {model} predictions to {predictions_file}")

    # 2. Compute Aggregate Metrics & Comparative Diagnostics
    aggregate_summary: Dict[str, Any] = {}
    for model in eval_models:
        metrics_list = list(model_metrics[model].values())
        n = len(metrics_list)
        avg_wer = sum(m["wer"] for m in metrics_list) / n
        avg_cer = sum(m["cer"] for m in metrics_list) / n
        avg_saa = sum(m["speaker_attribution_accuracy"] for m in metrics_list) / n
        avg_cpwer = sum(m["cpwer"] for m in metrics_list) / n
        avg_diar_gap = sum(m["diarization_gap"] for m in metrics_list) / n
        avg_latency = sum(m["latency_seconds"] for m in metrics_list) / n

        aggregate_summary[model] = {
            "mean_wer": round(avg_wer, 4),
            "mean_cer": round(avg_cer, 4),
            "mean_speaker_attribution_accuracy": round(avg_saa, 4),
            "mean_cpwer": round(avg_cpwer, 4),
            "mean_diarization_gap": round(avg_diar_gap, 4),
            "mean_latency_seconds": round(avg_latency, 3),
            "total_samples": n,
        }

    # 3. Sample-by-sample comparison between 2.5-flash and 3.5-flash-lite
    sample_comparisons: Dict[str, Any] = {}
    failure_cases: List[Dict[str, Any]] = []

    m25 = MODEL_2_5_FLASH
    m35 = MODEL_3_5_FLASH_LITE

    if m25 in eval_models and m35 in eval_models:
        for sample in samples:
            sid = sample.sample_id
            met25 = model_metrics[m25][sid]
            met35 = model_metrics[m35][sid]

            diff_wer = round(met35["wer"] - met25["wer"], 4)
            diff_saa = round(
                met35["speaker_attribution_accuracy"] - met25["speaker_attribution_accuracy"],
                4,
            )
            diff_cpwer = round(met35["cpwer"] - met25["cpwer"], 4)
            diff_diar_gap = round(met35["diarization_gap"] - met25["diarization_gap"], 4)

            is_saa_regression = diff_saa < -0.02
            is_wer_regression = diff_wer > 0.03
            is_cpwer_regression = diff_cpwer > 0.03

            comparison_entry = {
                "sample_id": sid,
                "duration_seconds": sample.duration_seconds,
                "num_speakers": sample.num_speakers,
                "overlap_ratio": sample.overlap_ratio,
                m25: met25,
                m35: met35,
                "delta_3_5_minus_2_5": {
                    "wer_delta": diff_wer,
                    "saa_delta": diff_saa,
                    "cpwer_delta": diff_cpwer,
                    "diarization_gap_delta": diff_diar_gap,
                    "latency_ratio": round(
                        met35["latency_seconds"] / max(0.001, met25["latency_seconds"]), 3
                    ),
                },
                "is_regression": is_saa_regression or is_wer_regression or is_cpwer_regression,
            }
            sample_comparisons[sid] = comparison_entry

            if comparison_entry["is_regression"]:
                failure_cases.append(
                    {
                        "sample_id": sid,
                        "num_speakers": sample.num_speakers,
                        "overlap_ratio": sample.overlap_ratio,
                        "duration_seconds": sample.duration_seconds,
                        "saa_2_5": met25["speaker_attribution_accuracy"],
                        "saa_3_5": met35["speaker_attribution_accuracy"],
                        "saa_delta": diff_saa,
                        "wer_2_5": met25["wer"],
                        "wer_3_5": met35["wer"],
                        "wer_delta": diff_wer,
                        "cpwer_2_5": met25["cpwer"],
                        "cpwer_3_5": met35["cpwer"],
                        "cpwer_delta": diff_cpwer,
                        "diar_gap_2_5": met25["diarization_gap"],
                        "diar_gap_3_5": met35["diarization_gap"],
                    }
                )

    # Calculate overall delta
    overall_delta = {}
    if m25 in aggregate_summary and m35 in aggregate_summary:
        agg25 = aggregate_summary[m25]
        agg35 = aggregate_summary[m35]
        overall_delta = {
            "wer_delta": round(agg35["mean_wer"] - agg25["mean_wer"], 4),
            "cer_delta": round(agg35["mean_cer"] - agg25["mean_cer"], 4),
            "saa_delta": round(
                agg35["mean_speaker_attribution_accuracy"]
                - agg25["mean_speaker_attribution_accuracy"],
                4,
            ),
            "cpwer_delta": round(agg35["mean_cpwer"] - agg25["mean_cpwer"], 4),
            "diarization_gap_delta": round(
                agg35["mean_diarization_gap"] - agg25["mean_diarization_gap"], 4
            ),
            "latency_speedup": round(
                agg25["mean_latency_seconds"] / max(0.001, agg35["mean_latency_seconds"]), 3
            ),
        }

    final_comparison = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_evaluated": eval_models,
        "total_samples": len(samples),
        "aggregate_summary": aggregate_summary,
        "overall_delta_3_5_vs_2_5": overall_delta,
        "total_failure_samples_count": len(failure_cases),
        "failure_samples": failure_cases,
        "sample_level_comparisons": sample_comparisons,
    }

    comparison_path = out_dir / "baseline_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(final_comparison, f, ensure_ascii=False, indent=2)

    logger.info(f"\nBaseline comparison successfully saved to {comparison_path}")

    # 4. Print Markdown Table to stdout
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION RESULTS (Approach 1: Single-Step Multimodal)")
    print("=" * 80)
    print(
        f"| Model | Mean WER | Mean CER | Mean SAA (Hungarian) | Mean cpWER | Diarization Gap | Latency (s) |"
    )
    print(
        f"|---|---|---|---|---|---|---|"
    )
    for model in eval_models:
        agg = aggregate_summary[model]
        print(
            f"| **{model}** | {agg['mean_wer']:.4f} | {agg['mean_cer']:.4f} | "
            f"{agg['mean_speaker_attribution_accuracy']:.4f} | {agg['mean_cpwer']:.4f} | "
            f"{agg['mean_diarization_gap']:.4f} | {agg['mean_latency_seconds']:.2f}s |"
        )
    if overall_delta:
        print(
            f"| **Delta (3.5 - 2.5)** | {overall_delta['wer_delta']:+.4f} | {overall_delta['cer_delta']:+.4f} | "
            f"{overall_delta['saa_delta']:+.4f} | {overall_delta['cpwer_delta']:+.4f} | "
            f"{overall_delta['diarization_gap_delta']:+.4f} | {overall_delta['latency_speedup']:.2f}x speedup |"
        )
    print("=" * 80)
    print(f"Identified Failure Cases (3.5 regression): {len(failure_cases)} samples")
    for fc in failure_cases:
        print(
            f"  - {fc['sample_id']}: SAA 2.5={fc['saa_2_5']:.3f} -> 3.5={fc['saa_3_5']:.3f} (delta: {fc['saa_delta']:+.3f}) | "
            f"WER 2.5={fc['wer_2_5']:.3f} -> 3.5={fc['wer_3_5']:.3f} | Overlap: {fc['overlap_ratio']:.1f}% | Spks: {fc['num_speakers']}"
        )
    print("=" * 80 + "\n")

    return final_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run live baseline evaluation on Indic diarization benchmark."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[MODEL_2_5_FLASH, MODEL_3_5_FLASH_LITE],
        help="Model identifiers to evaluate.",
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=BENCHMARK_SUBSET_DIR,
        help="Path to benchmark subset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_RESULTS_DIR,
        help="Path to results directory.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Optional list of specific sample IDs to run.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=DEFAULT_THINKING_BUDGET,
        help="Thinking budget (default: 0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature (default: 0.0).",
    )
    args = parser.parse_args()

    run_baseline_evaluation(
        models=args.models,
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
        sample_ids=args.sample_ids,
        thinking_budget=args.thinking_budget,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
