#!/usr/bin/env python3
"""Creative Iteration Benchmark Runner for Hard 2-Speaker High-Overlap Dialogue.

Evaluates:
- Candidate C3: Acoustic-Anchored Token-0 Bypass (`src/pipelines/acoustic_token0.py`)
- Candidate C4: Acoustic Multi-Stage Verifier (`src/pipelines/acoustic_multistage.py`)

Execution constraints:
- 100% live Vertex AI requests on project 'my-argolis-prj' (ADC authenticated).
- Pure `gemini-3.5-flash-lite` only (temperature=0.0, thinking_budget=0). Zero secondary models.
- Programmatic metric evaluation via `src/metrics.py`.
- Saves structured predictions to:
  - `results/hard_2spk/predictions_strategy_c3_acoustic_token0.json`
  - `results/hard_2spk/predictions_strategy_c4_multistage.json`
- Updates `results/hard_2spk/hard_2spk_comparative_summary.json` with macro summaries.
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
from src.config import MODEL_3_5_FLASH_LITE, get_location, get_project_id
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.pipelines.acoustic_multistage import AcousticMultiStagePipeline
from src.pipelines.acoustic_token0 import AcousticAnchorToken0Pipeline
from src.pipelines.adaptive_acoustic_champion import AdaptiveAcousticChampionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_hard_2spk_creative")

DEFAULT_SUBSET_DIR = PROJECT_ROOT / "data" / "hard_2spk_subset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hard_2spk"


def compute_macro_averages(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes macro average metrics across a list of prediction records."""
    if not records:
        return {}

    wers = [r["metrics"]["wer"] for r in records]
    cers = [r["metrics"]["cer"] for r in records]
    saas = [r["metrics"]["speaker_attribution_accuracy"] for r in records]
    cpwers = [r["metrics"]["cpwer"] for r in records]
    gaps = [r["metrics"]["diarization_gap"] for r in records]
    latencies = [r["latency_seconds"] for r in records]

    mean_wer = sum(wers) / len(wers) if wers else 0.0
    mean_cer = sum(cers) / len(cers) if cers else 0.0
    mean_saa = sum(saas) / len(saas) if saas else 0.0
    mean_cpwer = sum(cpwers) / len(cpwers) if cpwers else 0.0
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Normalized metrics excluding pathological outliers (WER > 2.0)
    normal_wers = [w for w in wers if w <= 2.0]
    normal_cers = [c for c in cers if c <= 2.0]
    normal_cpwers = [cp for cp in cpwers if cp <= 2.0]
    norm_wer = sum(normal_wers) / len(normal_wers) if normal_wers else mean_wer
    norm_cer = sum(normal_cers) / len(normal_cers) if normal_cers else mean_cer
    norm_cpwer = sum(normal_cpwers) / len(normal_cpwers) if normal_cpwers else mean_cpwer

    return {
        "count": len(records),
        "mean_wer": round(mean_wer, 4),
        "mean_wer_normalized": round(norm_wer, 4),
        "mean_cer": round(mean_cer, 4),
        "mean_cer_normalized": round(norm_cer, 4),
        "mean_speaker_attribution_accuracy": round(mean_saa, 4),
        "mean_cpwer": round(mean_cpwer, 4),
        "mean_cpwer_normalized": round(norm_cpwer, 4),
        "mean_diarization_gap": round(mean_gap, 4),
        "mean_latency_seconds": round(mean_latency, 3),
        "outlier_count": len(wers) - len(normal_wers),
    }


def run_creative_benchmark(
    project_id: str = "my-argolis-prj",
    location: str = "global",
    subset_dir: Path = DEFAULT_SUBSET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_ids: Optional[List[str]] = None,
    strategies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Executes live benchmark evaluating Candidate C3 and Candidate C4 on hard 2-speaker dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing GeminiClient (project={project_id}, location={location})...")
    client = GeminiClient(project_id=project_id, location=location)

    if not client.test_connection(MODEL_3_5_FLASH_LITE):
        logger.error(f"Failed to connect to Vertex AI using project_id='{project_id}'.")
        raise RuntimeError(f"Vertex AI authentication failed for project '{project_id}'.")
    logger.info(f"Vertex AI connection verified successfully on project '{project_id}'!")

    # Load samples
    all_dataset_samples = DatasetLoader.load_benchmark_subset(subset_dir=subset_dir, project_root=PROJECT_ROOT)
    samples = all_dataset_samples

    is_partial_run = bool(sample_ids)
    if sample_ids:
        samples = [s for s in samples if s.sample_id in sample_ids]
    logger.info(f"Loaded {len(samples)} 2-speaker hard samples for evaluation (partial_run={is_partial_run}).")

    # Initialize candidate pipelines
    pipeline_c3 = AcousticAnchorToken0Pipeline(client=client)
    pipeline_c4 = AcousticMultiStagePipeline(client=client)
    pipeline_c5 = AdaptiveAcousticChampionPipeline(client=client)

    available_strategies = {
        "strategy_c3_acoustic_token0": (
            "Strategy C3: Acoustic-Anchored Token-0",
            lambda s: pipeline_c3.run_sample(s, model_id=MODEL_3_5_FLASH_LITE),
        ),
        "strategy_c4_multistage": (
            "Strategy C4: Acoustic Multi-Stage Verifier",
            lambda s: pipeline_c4.run_sample(s, model_id=MODEL_3_5_FLASH_LITE),
        ),
        "strategy_c5_adaptive_champion": (
            "Strategy C5: Adaptive Acoustic Champion",
            lambda s: pipeline_c5.run_sample(s, model_id=MODEL_3_5_FLASH_LITE),
        ),
    }

    selected_strategy_keys = strategies or list(available_strategies.keys())
    active_pipelines = [
        (key, available_strategies[key][0], available_strategies[key][1])
        for key in selected_strategy_keys
        if key in available_strategies
    ]

    all_records: Dict[str, List[Dict[str, Any]]] = {key: [] for key, _, _ in active_pipelines}

    for idx, sample in enumerate(samples, 1):
        logger.info(
            f"[{idx}/{len(samples)}] Evaluating sample: {sample.sample_id} "
            f"(dur={sample.duration_seconds}s, overlap={sample.overlap_ratio}%)"
        )
        for key, name, runner_fn in active_pipelines:
            try:
                pred = runner_fn(sample)
                eval_metrics = MetricsEngine.evaluate_sample(
                    ref_turns=sample.ground_truth_turns,
                    hyp_turns=pred.predicted_turns,
                    normalize=True,
                )
                spk_count = sample.num_speakers or len(set(t.speaker for t in sample.ground_truth_turns)) or 2
                record = {
                    "sample_id": sample.sample_id,
                    "model_id": pred.model_id,
                    "approach": pred.approach,
                    "duration_seconds": sample.duration_seconds,
                    "num_speakers": spk_count,
                    "overlap_ratio": sample.overlap_ratio,
                    "latency_seconds": pred.latency_seconds,
                    "metrics": {
                        "wer": round(eval_metrics.wer, 4),
                        "cer": round(eval_metrics.cer, 4),
                        "speaker_attribution_accuracy": round(eval_metrics.speaker_attribution_accuracy, 4),
                        "cpwer": round(eval_metrics.cpwer, 4),
                        "diarization_gap": round(eval_metrics.diarization_gap, 4),
                        "speaker_mapping": eval_metrics.speaker_mapping,
                        "format_valid": eval_metrics.format_valid,
                    },
                    "predicted_turns": [{"speaker": t.speaker, "text": t.text} for t in pred.predicted_turns],
                    "raw_response": pred.raw_response,
                }
                all_records[key].append(record)
                logger.info(
                    f"  -> {name} on {sample.sample_id}: "
                    f"SAA={eval_metrics.speaker_attribution_accuracy:.2%}, "
                    f"WER={eval_metrics.wer:.2%}, Gap={eval_metrics.diarization_gap:.2%}, "
                    f"Turns={len(pred.predicted_turns)}, Latency={pred.latency_seconds:.2f}s"
                )
            except Exception as e:
                logger.error(f"Error evaluating {sample.sample_id} on {name}: {e}", exc_info=True)

    # Save raw predictions per pipeline
    for key, _, _ in active_pipelines:
        preds_path = output_dir / f"predictions_{key}.json"

        if is_partial_run:
            partial_preds_path = output_dir / f"predictions_{key}_partial.json"
            with open(partial_preds_path, "w", encoding="utf-8") as f:
                json.dump(all_records[key], f, ensure_ascii=False, indent=2)
            logger.info(f"Saved partial predictions for {key} ({len(all_records[key])} samples) to {partial_preds_path}")

            if preds_path.exists():
                try:
                    with open(preds_path, "r", encoding="utf-8") as f:
                        existing_records = json.load(f)
                    records_map = {r["sample_id"]: r for r in existing_records}
                    for r in all_records[key]:
                        records_map[r["sample_id"]] = r
                    merged_records = list(records_map.values())
                    with open(preds_path, "w", encoding="utf-8") as f:
                        json.dump(merged_records, f, ensure_ascii=False, indent=2)
                    logger.info(f"Merged {len(all_records[key])} samples into existing {preds_path} (now {len(merged_records)} samples)")
                except Exception as exc:
                    logger.warning(f"Could not merge records into {preds_path} ({exc}); keeping existing file.")
            else:
                with open(preds_path, "w", encoding="utf-8") as f:
                    json.dump(all_records[key], f, ensure_ascii=False, indent=2)
        else:
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(all_records[key], f, ensure_ascii=False, indent=2)
            logger.info(f"Saved predictions for {key} ({len(all_records[key])} samples) to {preds_path}")

    # Load and update comparative summary
    summary_path = output_dir / "hard_2spk_comparative_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing summary ({e}); creating fresh.")
            existing_summary = {}
    else:
        existing_summary = {}

    ref_samples = all_dataset_samples if all_dataset_samples else samples
    summary = {
        "benchmark_title": "Gemini STT & Speaker Diarization: Hard 2-Speaker High-Overlap Benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "total_samples": existing_summary.get("total_samples", len(ref_samples)),
        "mean_sample_duration_seconds": existing_summary.get(
            "mean_sample_duration_seconds",
            round(sum(s.duration_seconds for s in ref_samples) / len(ref_samples), 2) if ref_samples else 0.0,
        ),
        "mean_overlap_ratio": existing_summary.get(
            "mean_overlap_ratio",
            round(sum(s.overlap_ratio for s in ref_samples) / len(ref_samples), 2) if ref_samples else 0.0,
        ),
        "macro_summary": existing_summary.get("macro_summary", {}),
    }

    base_latency = 7.033
    if "gemini_2_5_flash" in summary["macro_summary"]:
        base_latency = summary["macro_summary"]["gemini_2_5_flash"].get("mean_latency_seconds", 7.033)

    for key, name, _ in active_pipelines:
        if is_partial_run:
            preds_path = output_dir / f"predictions_{key}.json"
            if preds_path.exists():
                try:
                    with open(preds_path, "r", encoding="utf-8") as f:
                        current_all = json.load(f)
                    if len(current_all) == len(ref_samples):
                        stats = compute_macro_averages(current_all)
                        stats["name"] = name
                        stats["speedup_vs_2_5"] = (
                            round(base_latency / stats["mean_latency_seconds"], 2)
                            if stats.get("mean_latency_seconds", 0) > 0 and base_latency > 0
                            else 0.0
                        )
                        summary["macro_summary"][key] = stats
                        logger.info(f"Updated full comparative summary for {key} from {len(current_all)} merged records.")
                except Exception as e:
                    logger.warning(f"Failed to read merged records for macro summary ({e}).")
        else:
            stats = compute_macro_averages(all_records[key])
            stats["name"] = name
            stats["speedup_vs_2_5"] = (
                round(base_latency / stats["mean_latency_seconds"], 2)
                if stats.get("mean_latency_seconds", 0) > 0 and base_latency > 0
                else 0.0
            )
            summary["macro_summary"][key] = stats
            logger.info(f"Updated comparative summary for {key}: SAA={stats['mean_speaker_attribution_accuracy']:.2%}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved updated comparative summary to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run creative benchmark for Candidate C3 and C4 on hard 2-speaker clips.")
    parser.add_argument("--project-id", default=get_project_id(), help="Vertex AI GCP Project ID")
    parser.add_argument("--location", default=get_location(), help="Vertex AI location")
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR, help="Hard subset directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Results output directory")
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Specific sample IDs to evaluate")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["strategy_c3_acoustic_token0", "strategy_c4_multistage", "strategy_c5_adaptive_champion"],
        help="Strategies to evaluate (strategy_c3_acoustic_token0, strategy_c4_multistage, strategy_c5_adaptive_champion)",
    )
    args = parser.parse_args()

    run_creative_benchmark(
        project_id=args.project_id,
        location=args.location,
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
        sample_ids=args.sample_ids,
        strategies=args.strategies,
    )


if __name__ == "__main__":
    main()
