#!/usr/bin/env python3
"""Milestone 4 Phase 3: Generalization Certification Runner for Candidate C5.

Executes Candidate C5 (AdaptiveAcousticChampionPipeline) across all clips in
data/indic_diarbench_subset/ on live Vertex AI (my-argolis-prj).
Computes objective metrics, multi-speaker slices (2spk, 3spk, 4spk), and saves predictions.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.client import GeminiClient
from src.config import MODEL_3_5_FLASH_LITE
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import SampleData, Turn
from src.pipelines.adaptive_acoustic_champion import AdaptiveAcousticChampionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_generalization_c5")

DEFAULT_SUBSET_DIR = PROJECT_ROOT / "data" / "indic_diarbench_subset"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "results" / "optimized" / "predictions_strategy_c5_generalization.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "optimized" / "generalization_summary.json"


def compute_metrics_slice(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes macro statistics across a list of prediction records."""
    if not records:
        return {
            "count": 0,
            "mean_wer": 0.0,
            "mean_wer_normalized": 0.0,
            "mean_cer": 0.0,
            "mean_cer_normalized": 0.0,
            "mean_speaker_attribution_accuracy": 0.0,
            "mean_cpwer": 0.0,
            "mean_cpwer_normalized": 0.0,
            "mean_diarization_gap": 0.0,
            "mean_latency_seconds": 0.0,
            "outlier_count": 0,
        }

    wers = [r["metrics"]["wer"] for r in records]
    cers = [r["metrics"]["cer"] for r in records]
    saas = [r["metrics"]["speaker_attribution_accuracy"] for r in records]
    cpwers = [r["metrics"]["cpwer"] for r in records]
    gaps = [r["metrics"]["diarization_gap"] for r in records]
    latencies = [r["latency_seconds"] for r in records]

    n = len(records)
    normal_wers = [w for w in wers if w <= 2.0]
    normal_cers = [c for c in cers if c <= 2.0]
    normal_cpwers = [cp for cp in cpwers if cp <= 2.0]

    norm_wer = sum(normal_wers) / len(normal_wers) if normal_wers else sum(wers) / n
    norm_cer = sum(normal_cers) / len(normal_cers) if normal_cers else sum(cers) / n
    norm_cpwer = sum(normal_cpwers) / len(normal_cpwers) if normal_cpwers else sum(cpwers) / n

    return {
        "count": n,
        "mean_wer": round(sum(wers) / n, 4),
        "mean_wer_normalized": round(norm_wer, 4),
        "mean_cer": round(sum(cers) / n, 4),
        "mean_cer_normalized": round(norm_cer, 4),
        "mean_speaker_attribution_accuracy": round(sum(saas) / n, 4),
        "mean_cpwer": round(sum(cpwers) / n, 4),
        "mean_cpwer_normalized": round(norm_cpwer, 4),
        "mean_diarization_gap": round(sum(gaps) / n, 4),
        "mean_latency_seconds": round(sum(latencies) / n, 3),
        "outlier_count": len(wers) - len(normal_wers),
    }


def run_c5_generalization(
    project_id: str = "my-argolis-prj",
    location: str = "global",
    subset_dir: Path = DEFAULT_SUBSET_DIR,
    output_predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    output_summary_path: Path = DEFAULT_SUMMARY_PATH,
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs Candidate C5 live benchmark on the indic_diarbench subset."""
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to Vertex AI (project='{project_id}', location='{location}')...")
    client = GeminiClient(project_id=project_id, location=location)
    if not client.test_connection(MODEL_3_5_FLASH_LITE):
        raise RuntimeError(f"Failed to authenticate with Vertex AI project '{project_id}'")
    logger.info("Vertex AI connection verified successfully!")

    samples = DatasetLoader.load_benchmark_subset(subset_dir=subset_dir, project_root=PROJECT_ROOT)
    if sample_ids:
        samples = [s for s in samples if s.sample_id in sample_ids]

    logger.info(f"Evaluating {len(samples)} clips using Candidate C5 (AdaptiveAcousticChampionPipeline)...")

    pipeline = AdaptiveAcousticChampionPipeline(client=client)

    # Load existing predictions if resuming or merging
    existing_records: Dict[str, Dict[str, Any]] = {}
    if output_predictions_path.exists():
        try:
            with open(output_predictions_path, "r", encoding="utf-8") as f:
                old_list = json.load(f)
                existing_records = {r["sample_id"]: r for r in old_list}
        except Exception as e:
            logger.warning(f"Could not load existing predictions: {e}")

    partial_path = output_predictions_path.parent / (output_predictions_path.stem + "_partial.json")

    completed_records: List[Dict[str, Any]] = []

    for idx, sample in enumerate(samples, 1):
        logger.info(
            f"[{idx}/{len(samples)}] Executing Candidate C5 on sample {sample.sample_id} "
            f"(dur={sample.duration_seconds:.1f}s, spk={sample.num_speakers}, overlap={sample.overlap_ratio:.1f}%)..."
        )

        pred = pipeline.run_sample(sample, model_id=MODEL_3_5_FLASH_LITE)

        # Compute objective metrics via MetricsEngine
        eval_metrics = MetricsEngine.evaluate_sample(
            sample.ground_truth_turns,
            pred.predicted_turns,
            normalize=True,
        )

        record = {
            "sample_id": sample.sample_id,
            "model_id": pred.model_id,
            "approach": pred.approach,
            "duration_seconds": round(sample.duration_seconds, 2),
            "num_speakers": sample.num_speakers,
            "overlap_ratio": round(sample.overlap_ratio, 2),
            "latency_seconds": round(pred.latency_seconds, 3),
            "metrics": {
                "wer": round(eval_metrics.wer, 4),
                "cer": round(eval_metrics.cer, 4),
                "speaker_attribution_accuracy": round(eval_metrics.speaker_attribution_accuracy, 4),
                "cpwer": round(eval_metrics.cpwer, 4),
                "diarization_gap": round(eval_metrics.diarization_gap, 4),
                "speaker_mapping": eval_metrics.speaker_mapping,
                "format_valid": eval_metrics.format_valid,
            },
            "predicted_turns": [t.model_dump() for t in pred.predicted_turns],
            "raw_response": pred.raw_response,
        }

        completed_records.append(record)
        existing_records[sample.sample_id] = record

        logger.info(
            f"  -> {sample.sample_id} result: SAA={eval_metrics.speaker_attribution_accuracy:.1%}, "
            f"Gap={eval_metrics.diarization_gap:.1%}, WER={eval_metrics.wer:.1%}, "
            f"Latency={pred.latency_seconds:.2f}s ({len(pred.predicted_turns)} turns)"
        )

        # Save partial incremental progress
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(list(existing_records.values()), f, indent=2, ensure_ascii=False)

    # Save final complete predictions file
    final_records = list(existing_records.values())
    with open(output_predictions_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(final_records)} predictions to {output_predictions_path}")

    # Compute macro and slices
    macro_stats = compute_metrics_slice(final_records)

    slice_2spk = compute_metrics_slice([r for r in final_records if r["num_speakers"] == 2])
    slice_3spk = compute_metrics_slice([r for r in final_records if r["num_speakers"] == 3])
    slice_4spk = compute_metrics_slice([r for r in final_records if r["num_speakers"] == 4])

    summary_data = {
        "benchmark_title": "Candidate C5 Generalization Certification Benchmark",
        "dataset_path": str(subset_dir),
        "evaluated_model": MODEL_3_5_FLASH_LITE,
        "approach": "strategy_c5_adaptive_champion",
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(final_records),
        "macro_metrics": macro_stats,
        "slices": {
            "2_speakers": slice_2spk,
            "3_speakers": slice_3spk,
            "4_speakers": slice_4spk,
        },
        "baseline_comparison": {
            "default_3_5_flash_lite_baseline": {
                "mean_wer": 0.2475,
                "mean_cer": 0.1414,
                "mean_speaker_attribution_accuracy": 0.6904,
                "mean_cpwer": 0.5816,
                "mean_diarization_gap": 0.3341,
                "mean_latency_seconds": 3.092,
            },
            "gemini_2_5_flash_baseline": {
                "mean_wer": 3.6293,
                "mean_wer_normalized": 0.2418,
                "mean_cer": 2.8537,
                "mean_cer_normalized": 0.1437,
                "mean_speaker_attribution_accuracy": 0.7942,
                "mean_cpwer": 3.8778,
                "mean_cpwer_normalized": 0.3878,
                "mean_diarization_gap": 0.2497,
                "mean_latency_seconds": 8.205,
            },
            "candidate_c5_delta_vs_3_5_default": {
                "saa_gain": round(macro_stats["mean_speaker_attribution_accuracy"] - 0.6904, 4),
                "diarization_gap_reduction": round(0.3341 - macro_stats["mean_diarization_gap"], 4),
                "wer_delta": round(macro_stats["mean_wer"] - 0.2475, 4),
            },
            "candidate_c5_delta_vs_2_5_flash": {
                "saa_gain": round(macro_stats["mean_speaker_attribution_accuracy"] - 0.7942, 4),
                "diarization_gap_reduction": round(0.2497 - macro_stats["mean_diarization_gap"], 4),
                "speedup_vs_2_5": round(8.205 / macro_stats["mean_latency_seconds"], 2) if macro_stats["mean_latency_seconds"] > 0 else 0.0,
            },
        },
    }

    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved generalization summary to {output_summary_path}")

    # Remove partial file if completed cleanly
    if partial_path.exists():
        partial_path.unlink()

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Candidate C5 Generalization Benchmark Runner")
    parser.add_argument("--project-id", type=str, default="my-argolis-prj")
    parser.add_argument("--location", type=str, default="global")
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--sample-ids", nargs="+", default=None)

    args = parser.parse_args()

    summary = run_c5_generalization(
        project_id=args.project_id,
        location=args.location,
        subset_dir=args.subset_dir,
        output_predictions_path=args.output_predictions,
        output_summary_path=args.output_summary,
        sample_ids=args.sample_ids,
    )

    print("\n" + "=" * 80)
    print("      CANDIDATE C5 GENERALIZATION CERTIFICATION BENCHMARK SUMMARY")
    print("=" * 80)
    macro = summary["macro_metrics"]
    print(f"Total Evaluated Samples: {summary['total_samples']}")
    print(f"Macro Hungarian SAA:     {macro['mean_speaker_attribution_accuracy']:.2%}")
    print(f"Macro Diarization Gap:   {macro['mean_diarization_gap']:.2%}")
    print(f"Macro WER:               {macro['mean_wer']:.2%}")
    print(f"Macro CER:               {macro['mean_cer']:.2%}")
    print(f"Macro cpWER:             {macro['mean_cpwer']:.2%}")
    print(f"Macro Mean Latency:      {macro['mean_latency_seconds']:.2f}s")
    print(f"Outlier Count:           {macro['outlier_count']}")
    print("-" * 80)
    slices = summary["slices"]
    print(f"2-Speaker SAA (n={slices['2_speakers']['count']}): {slices['2_speakers']['mean_speaker_attribution_accuracy']:.2%}")
    print(f"3-Speaker SAA (n={slices['3_speakers']['count']}): {slices['3_speakers']['mean_speaker_attribution_accuracy']:.2%}")
    print(f"4-Speaker SAA (n={slices['4_speakers']['count']}): {slices['4_speakers']['mean_speaker_attribution_accuracy']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
