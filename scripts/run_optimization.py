"""Executes candidate optimization strategies on Indic diarization benchmark samples.

Candidate Strategies:
1. Strategy 1 (Anchor Prompting): Conversation Anchor Cues with gemini-3.5-flash-lite.
2. Strategy 2 (Two-Step Decoupled): Step 1 Acoustic STT (3.5-flash-lite) + Step 2 LLM Role Attribution (3.5-flash).
3. Strategy 3 (GEPA Optimization): Automated Genetic Evolutionary Prompt Optimization.

Outputs:
- results/optimized/predictions_anchor_3.5.json
- results/optimized/predictions_decoupled_3.5.json
- results/optimized/optimization_comparison.json
- results/optimized/gepa_evolution_history.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASELINE_RESULTS_DIR,
    BENCHMARK_SUBSET_DIR,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_2_5_FLASH,
    MODEL_3_5_FLASH,
    MODEL_3_5_FLASH_LITE,
    OPTIMIZED_RESULTS_DIR,
)
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import EvaluationMetrics, PipelinePrediction, SampleData
from src.pipelines.anchor_prompting import AnchorPromptPipeline
from src.pipelines.decoupled_step import TwoStepDecoupledPipeline
from src.pipelines.gepa_optimizer import GEPALoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_optimization")


def load_baseline_data() -> Optional[Dict[str, Any]]:
    """Loads existing baseline evaluation results if available."""
    baseline_path = BASELINE_RESULTS_DIR / "baseline_comparison.json"
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load baseline comparison from {baseline_path}: {e}")
    return None


def run_strategy_anchor(
    samples: List[SampleData],
    model_id: str = MODEL_3_5_FLASH_LITE,
    out_dir: Path = OPTIMIZED_RESULTS_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs Strategy 1: Prompt Optimization with Conversation Anchor Cues."""
    logger.info(f"\n{'='*30} Running Strategy 1: Anchor Prompting ({model_id}) {'='*30}")
    pipeline = AnchorPromptPipeline()
    predictions_file = out_dir / "predictions_anchor_3.5.json"

    records = []
    for idx, sample in enumerate(samples, 1):
        logger.info(
            f"[{idx}/{len(samples)}] Anchor Prompting on {sample.sample_id} "
            f"({sample.duration_seconds:.1f}s, {sample.num_speakers} spks, {sample.overlap_ratio:.1f}% ovl)..."
        )
        pred = pipeline.run_sample(sample, model_id=model_id)
        eval_metrics = MetricsEngine.evaluate_sample(
            sample.ground_truth_turns, pred.predicted_turns
        )

        record = {
            "sample_id": sample.sample_id,
            "strategy": "anchor_prompting",
            "model_id": model_id,
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
        records.append(record)
        logger.info(
            f"  -> Sample {sample.sample_id} | Latency: {pred.latency_seconds:.2f}s | "
            f"WER: {eval_metrics.wer:.4f} | SAA: {eval_metrics.speaker_attribution_accuracy:.4f} | "
            f"DiarGap: {eval_metrics.diarization_gap:.4f}"
        )

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved Strategy 1 predictions to {predictions_file}")

    # Compute summary
    n = len(records)
    summary = {
        "strategy": "anchor_prompting",
        "model_id": model_id,
        "mean_wer": round(sum(r["metrics"]["wer"] for r in records) / n, 4),
        "mean_cer": round(sum(r["metrics"]["cer"] for r in records) / n, 4),
        "mean_speaker_attribution_accuracy": round(
            sum(r["metrics"]["speaker_attribution_accuracy"] for r in records) / n, 4
        ),
        "mean_cpwer": round(sum(r["metrics"]["cpwer"] for r in records) / n, 4),
        "mean_diarization_gap": round(
            sum(r["metrics"]["diarization_gap"] for r in records) / n, 4
        ),
        "mean_latency_seconds": round(
            sum(r["latency_seconds"] for r in records) / n, 3
        ),
        "total_samples": n,
    }
    return records, summary


def run_strategy_decoupled(
    samples: List[SampleData],
    step1_model_id: str = MODEL_3_5_FLASH_LITE,
    step2_model_id: str = MODEL_3_5_FLASH,
    out_dir: Path = OPTIMIZED_RESULTS_DIR,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs Strategy 2: Two-Step Decoupled Architecture."""
    logger.info(
        f"\n{'='*30} Running Strategy 2: Two-Step Decoupled (Step 1: {step1_model_id}, Step 2: {step2_model_id}) {'='*30}"
    )
    pipeline = TwoStepDecoupledPipeline(
        step1_model_id=step1_model_id,
        step2_model_id=step2_model_id,
    )
    predictions_file = out_dir / "predictions_decoupled_3.5.json"

    records = []
    for idx, sample in enumerate(samples, 1):
        logger.info(
            f"[{idx}/{len(samples)}] Two-Step Decoupled on {sample.sample_id} "
            f"({sample.duration_seconds:.1f}s, {sample.num_speakers} spks, {sample.overlap_ratio:.1f}% ovl)..."
        )
        pred = pipeline.run_sample(sample)
        eval_metrics = MetricsEngine.evaluate_sample(
            sample.ground_truth_turns, pred.predicted_turns
        )

        record = {
            "sample_id": sample.sample_id,
            "strategy": "two_step_decoupled",
            "model_id": f"{step1_model_id}+{step2_model_id}",
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
        records.append(record)
        logger.info(
            f"  -> Sample {sample.sample_id} | Latency: {pred.latency_seconds:.2f}s | "
            f"WER: {eval_metrics.wer:.4f} | SAA: {eval_metrics.speaker_attribution_accuracy:.4f} | "
            f"DiarGap: {eval_metrics.diarization_gap:.4f}"
        )

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved Strategy 2 predictions to {predictions_file}")

    # Compute summary
    n = len(records)
    summary = {
        "strategy": "two_step_decoupled",
        "model_id": f"{step1_model_id}+{step2_model_id}",
        "mean_wer": round(sum(r["metrics"]["wer"] for r in records) / n, 4),
        "mean_cer": round(sum(r["metrics"]["cer"] for r in records) / n, 4),
        "mean_speaker_attribution_accuracy": round(
            sum(r["metrics"]["speaker_attribution_accuracy"] for r in records) / n, 4
        ),
        "mean_cpwer": round(sum(r["metrics"]["cpwer"] for r in records) / n, 4),
        "mean_diarization_gap": round(
            sum(r["metrics"]["diarization_gap"] for r in records) / n, 4
        ),
        "mean_latency_seconds": round(
            sum(r["latency_seconds"] for r in records) / n, 3
        ),
        "total_samples": n,
    }
    return records, summary


def run_strategy_gepa(
    samples: List[SampleData],
    num_rounds: int = 2,
    out_dir: Path = OPTIMIZED_RESULTS_DIR,
) -> Dict[str, Any]:
    """Runs Strategy 3: Automated GEPA Evolutionary Loop on diagnostic failure samples."""
    logger.info(f"\n{'='*30} Running Strategy 3: Automated GEPA Evolutionary Loop {'='*30}")
    loop = GEPALoop(
        eval_model=MODEL_3_5_FLASH,
        mutator_model=MODEL_3_5_FLASH,
        candidate_model=MODEL_3_5_FLASH_LITE,
    )

    # Diagnostic sample subset for prompt evolution
    diagnostic_ids = ["hindi_070", "hindi_087", "hindi_066"]
    diag_samples = [s for s in samples if s.sample_id in diagnostic_ids]
    if not diag_samples:
        diag_samples = samples[:3]

    history_path = out_dir / "gepa_evolution_history.json"
    best_prompt, best_metrics, history = loop.run_evolution(
        validation_samples=diag_samples,
        num_rounds=num_rounds,
        history_output_path=history_path,
    )

    gepa_summary = {
        "strategy": "gepa_optimizer",
        "num_rounds": num_rounds,
        "best_fitness": best_metrics.get("fitness", 0.0),
        "mean_saa": best_metrics.get("mean_saa", 0.0),
        "mean_wer": best_metrics.get("mean_wer", 0.0),
        "mean_diarization_gap": best_metrics.get("mean_diarization_gap", 0.0),
        "history_file": str(history_path),
    }
    return gepa_summary


def generate_comparative_analysis(
    baseline_data: Optional[Dict[str, Any]],
    anchor_records: Optional[List[Dict[str, Any]]],
    decoupled_records: Optional[List[Dict[str, Any]]],
    summaries: Dict[str, Any],
    out_dir: Path = OPTIMIZED_RESULTS_DIR,
) -> Dict[str, Any]:
    """Builds comprehensive comparative summary and failure recovery analysis."""
    analysis = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "strategies_evaluated": list(summaries.keys()),
        "aggregate_summaries": summaries,
    }

    # Baseline comparison references
    b25_agg = baseline_data.get("aggregate_summary", {}).get(MODEL_2_5_FLASH, {}) if baseline_data else {}
    b35_agg = baseline_data.get("aggregate_summary", {}).get(MODEL_3_5_FLASH_LITE, {}) if baseline_data else {}

    if b25_agg:
        analysis["baseline_reference_2_5"] = b25_agg
    if b35_agg:
        analysis["baseline_reference_3_5_default"] = b35_agg

    # Detailed sample-by-sample comparisons for failure recovery
    anchor_map = {r["sample_id"]: r for r in (anchor_records or [])}
    decoupled_map = {r["sample_id"]: r for r in (decoupled_records or [])}

    sample_comparisons = {}
    recovery_evidence = []

    baseline_samples = baseline_data.get("sample_level_comparisons", {}) if baseline_data else {}
    sample_ids = list(set(list(anchor_map.keys()) + list(decoupled_map.keys()) + list(baseline_samples.keys())))
    sample_ids.sort()

    for sid in sample_ids:
        b_entry = baseline_samples.get(sid, {})
        m25_met = b_entry.get(MODEL_2_5_FLASH, {})
        m35_met = b_entry.get(MODEL_3_5_FLASH_LITE, {})
        anc_entry = anchor_map.get(sid, {})
        dec_entry = decoupled_map.get(sid, {})

        sample_comp = {
            "sample_id": sid,
            "duration_seconds": b_entry.get("duration_seconds") or anc_entry.get("duration_seconds") or dec_entry.get("duration_seconds"),
            "num_speakers": b_entry.get("num_speakers") or anc_entry.get("num_speakers") or dec_entry.get("num_speakers"),
            "overlap_ratio": b_entry.get("overlap_ratio") or anc_entry.get("overlap_ratio") or dec_entry.get("overlap_ratio"),
            "baseline_2_5_flash": m25_met,
            "baseline_3_5_flash_lite": m35_met,
            "strategy1_anchor": anc_entry.get("metrics"),
            "strategy2_decoupled": dec_entry.get("metrics"),
        }
        sample_comparisons[sid] = sample_comp

        # Check recovery on diagnosed failure samples (where baseline 3.5 regressed or SAA < 2.5)
        saa_25 = m25_met.get("speaker_attribution_accuracy", 0.0)
        saa_35_base = m35_met.get("speaker_attribution_accuracy", 0.0)
        saa_dec = dec_entry.get("metrics", {}).get("speaker_attribution_accuracy", 0.0) if dec_entry else 0.0
        saa_anc = anc_entry.get("metrics", {}).get("speaker_attribution_accuracy", 0.0) if anc_entry else 0.0

        if (sid in anchor_map or sid in decoupled_map) and (saa_35_base < saa_25 or b_entry.get("is_regression", False)):
            best_opt_saa = max(saa_dec, saa_anc)
            recovered = best_opt_saa >= (saa_25 - 0.05) or (best_opt_saa - saa_35_base) > 0.15
            recovery_evidence.append(
                {
                    "sample_id": sid,
                    "baseline_2_5_saa": saa_25,
                    "baseline_3_5_saa": saa_35_base,
                    "anchor_saa": saa_anc,
                    "decoupled_saa": saa_dec,
                    "best_optimized_saa": best_opt_saa,
                    "saa_gain_over_3_5_baseline": round(best_opt_saa - saa_35_base, 4),
                    "recovered": recovered,
                }
            )

    analysis["sample_level_comparisons"] = sample_comparisons
    analysis["failure_sample_recovery_evidence"] = recovery_evidence
    analysis["total_failure_samples"] = len(recovery_evidence)
    analysis["total_recovered_samples"] = sum(1 for r in recovery_evidence if r["recovered"])

    comparison_path = out_dir / "optimization_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved comparative optimization analysis to {comparison_path}")

    # Print Summary Tables
    print("\n" + "=" * 90)
    print("CANDIDATE OPTIMIZATION BENCHMARK RESULTS")
    print("=" * 90)
    print(
        f"| Approach / Strategy | Mean WER | Mean CER | Mean SAA (Hungarian) | Mean cpWER | Diarization Gap | Latency |"
    )
    print(
        f"|---|---|---|---|---|---|---|"
    )
    if b25_agg:
        print(
            f"| **Gemini 2.5 Flash (Baseline)** | {b25_agg.get('mean_wer', 0.0):.4f} | {b25_agg.get('mean_cer', 0.0):.4f} | "
            f"**{b25_agg.get('mean_speaker_attribution_accuracy', 0.0):.4f}** | {b25_agg.get('mean_cpwer', 0.0):.4f} | "
            f"{b25_agg.get('mean_diarization_gap', 0.0):.4f} | {b25_agg.get('mean_latency_seconds', 0.0):.2f}s |"
        )
    if b35_agg:
        print(
            f"| **Gemini 3.5 Flash Lite (Default)** | {b35_agg.get('mean_wer', 0.0):.4f} | {b35_agg.get('mean_cer', 0.0):.4f} | "
            f"{b35_agg.get('mean_speaker_attribution_accuracy', 0.0):.4f} | {b35_agg.get('mean_cpwer', 0.0):.4f} | "
            f"{b35_agg.get('mean_diarization_gap', 0.0):.4f} | {b35_agg.get('mean_latency_seconds', 0.0):.2f}s |"
        )
    for strat, summary in summaries.items():
        print(
            f"| **{strat}** | {summary['mean_wer']:.4f} | {summary['mean_cer']:.4f} | "
            f"**{summary['mean_speaker_attribution_accuracy']:.4f}** | {summary['mean_cpwer']:.4f} | "
            f"{summary['mean_diarization_gap']:.4f} | {summary['mean_latency_seconds']:.2f}s |"
        )
    print("=" * 90)

    print("\n" + "=" * 90)
    print(f"FAILURE SAMPLE RECOVERY TRACE ({analysis['total_recovered_samples']}/{analysis['total_failure_samples']} Recovered)")
    print("=" * 90)
    for rec in recovery_evidence:
        status = "RECOVERED" if rec["recovered"] else "PERSISTENT"
        print(
            f"  - {rec['sample_id']}: 2.5 Flash SAA={rec['baseline_2_5_saa']:.3f} | "
            f"3.5 Default={rec['baseline_3_5_saa']:.3f} -> Optimized={rec['best_optimized_saa']:.3f} "
            f"(Delta: {rec['saa_gain_over_3_5_baseline']:+.3f}) [{status}]"
        )
    print("=" * 90 + "\n")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run live optimization benchmarking on Indic diarization benchmark."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["anchor", "decoupled"],
        choices=["anchor", "decoupled", "gepa", "all"],
        help="Strategies to evaluate (anchor, decoupled, gepa, all).",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Optional list of specific sample IDs to evaluate.",
    )
    parser.add_argument(
        "--step2-model",
        type=str,
        default=MODEL_3_5_FLASH,
        help=f"Model for Step 2 of decoupled pipeline (default: {MODEL_3_5_FLASH}).",
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
        default=OPTIMIZED_RESULTS_DIR,
        help="Path to output results directory.",
    )
    parser.add_argument(
        "--gepa-rounds",
        type=int,
        default=2,
        help="Number of rounds for GEPA evolutionary search.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = DatasetLoader.load_benchmark_subset(subset_dir=args.subset_dir)
    if args.sample_ids:
        samples = [s for s in all_samples if s.sample_id in args.sample_ids]
    else:
        samples = all_samples

    logger.info(f"Loaded {len(samples)} benchmark samples from {args.subset_dir}")

    eval_strategies = args.strategies
    if "all" in eval_strategies:
        eval_strategies = ["anchor", "decoupled", "gepa"]

    baseline_data = load_baseline_data()
    summaries = {}
    anchor_records = None
    decoupled_records = None

    if "anchor" in eval_strategies:
        anchor_records, anchor_summary = run_strategy_anchor(
            samples=samples,
            model_id=MODEL_3_5_FLASH_LITE,
            out_dir=args.output_dir,
        )
        summaries["Strategy 1: Anchor Prompting (3.5 Lite)"] = anchor_summary

    if "decoupled" in eval_strategies:
        decoupled_records, decoupled_summary = run_strategy_decoupled(
            samples=samples,
            step1_model_id=MODEL_3_5_FLASH_LITE,
            step2_model_id=args.step2_model,
            out_dir=args.output_dir,
        )
        summaries[f"Strategy 2: Decoupled (3.5 Lite + {args.step2_model})"] = decoupled_summary

    if "gepa" in eval_strategies:
        gepa_summary = run_strategy_gepa(
            samples=samples,
            num_rounds=args.gepa_rounds,
            out_dir=args.output_dir,
        )
        logger.info(f"GEPA Optimization Complete: {gepa_summary}")

    generate_comparative_analysis(
        baseline_data=baseline_data,
        anchor_records=anchor_records,
        decoupled_records=decoupled_records,
        summaries=summaries,
        out_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
