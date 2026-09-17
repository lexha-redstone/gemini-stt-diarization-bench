#!/usr/bin/env python3
"""Live Vertex AI Comparative Benchmark Runner for 50-Sample Hindi Debt Collection Dataset (Milestone 6).

Evaluates and compares across all 50 samples in data/debt_collection_subset/:
1. Baseline: Gemini 2.5 Flash (`gemini-2.5-flash` via `SingleStepPipeline`)
2. Proposed Champion: Candidate C5 (`gemini-3.5-flash-lite` via `AdaptiveAcousticChampionPipeline`)

Features:
- Pre-warms Vertex AI project fallback (`my-argolis-prj` -> `cloud-llm-preview1`) once at startup via `test_connection()`
- Parallel execution via `ThreadPoolExecutor(max_workers=4)` sorted by duration ascending
- Atomic per-sample JSON checkpointing (`predictions_gemini_2_5_flash.json` & `predictions_strategy_c5_adaptive_champion.json`)
- Computes exact macro metrics and stratified breakdowns across 4 duration buckets (`<70s`, `70s-180s`, `180s-300s`, `300s-600s`),
  conflict tiers, and 2D Duration x Overlap regimes via `src/metrics.py` (`MetricsEngine`).
"""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.client import GeminiClient
from src.config import MODEL_2_5_FLASH, MODEL_3_5_FLASH_LITE
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import SampleData, Turn
from src.pipelines.adaptive_acoustic_champion import (
    AdaptiveAcousticChampionPipeline,
    should_trigger_stage2,
)
from src.pipelines.single_step import SingleStepPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_debt_collection_benchmark")

DEFAULT_SUBSET_DIR = PROJECT_ROOT / "data" / "debt_collection_subset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "debt_collection"


def compute_macro_summary(
    records: List[Dict[str, Any]], baseline_latency: Optional[float] = None
) -> Dict[str, Any]:
    """Computes macro average metrics across a list of prediction records."""
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
            "speedup_vs_2_5": 1.0,
        }

    n = len(records)
    wers = [r["metrics"]["wer"] for r in records]
    cers = [r["metrics"]["cer"] for r in records]
    saas = [r["metrics"]["speaker_attribution_accuracy"] for r in records]
    cpwers = [r["metrics"]["cpwer"] for r in records]
    gaps = [r["metrics"]["diarization_gap"] for r in records]
    latencies = [r["latency_seconds"] for r in records]

    mean_wer = sum(wers) / n
    mean_cer = sum(cers) / n
    mean_saa = sum(saas) / n
    mean_cpwer = sum(cpwers) / n
    mean_gap = sum(gaps) / n
    mean_latency = sum(latencies) / n

    normal_wers = [w for w in wers if w <= 2.0]
    normal_cers = [c for c in cers if c <= 2.0]
    normal_cpwers = [cp for cp in cpwers if cp <= 2.0]

    norm_wer = sum(normal_wers) / len(normal_wers) if normal_wers else mean_wer
    norm_cer = sum(normal_cers) / len(normal_cers) if normal_cers else mean_cer
    norm_cpwer = sum(normal_cpwers) / len(normal_cpwers) if normal_cpwers else mean_cpwer

    ref_lat = baseline_latency if baseline_latency and baseline_latency > 0 else mean_latency
    speedup = round(ref_lat / mean_latency, 2) if mean_latency > 0 else 1.0

    return {
        "count": n,
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
        "speedup_vs_2_5": speedup,
    }


def get_duration_bucket_key(duration_sec: float) -> str:
    """Assigns sample duration to one of the 4 stratified duration buckets."""
    if duration_sec < 70.0:
        return "<70s"
    elif duration_sec < 180.0:
        return "70s-180s"
    elif duration_sec < 300.0:
        return "180s-300s"
    else:
        return "300s-600s"


def get_conflict_tier_key(overlap_ratio: float) -> str:
    """Assigns sample overlap ratio to a conflict intensity tier."""
    if overlap_ratio < 16.0:
        return "moderate_12_16_pct"
    elif overlap_ratio < 20.0:
        return "high_16_20_pct"
    else:
        return "extreme_over_20_pct"


def format_prediction_record(
    sample: SampleData,
    model_id: str,
    approach: str,
    predicted_turns: List[Turn],
    raw_response: str,
    latency_seconds: float,
    source: str = "live_vertex_ai",
    stage2_triggered: Optional[bool] = None,
) -> Dict[str, Any]:
    """Formats a single prediction record with verified MetricsEngine evaluation."""
    eval_metrics = MetricsEngine.evaluate_sample(
        sample.ground_truth_turns, predicted_turns, normalize=True
    )
    rec = {
        "sample_id": sample.sample_id,
        "model_id": model_id,
        "approach": approach,
        "source": source,
        "duration_seconds": round(sample.duration_seconds, 2),
        "duration_bucket": get_duration_bucket_key(sample.duration_seconds),
        "num_speakers": sample.num_speakers,
        "overlap_ratio": round(sample.overlap_ratio, 2),
        "conflict_tier": get_conflict_tier_key(sample.overlap_ratio),
        "latency_seconds": round(float(latency_seconds), 4),
        "metrics": {
            "wer": round(eval_metrics.wer, 4),
            "cer": round(eval_metrics.cer, 4),
            "speaker_attribution_accuracy": round(eval_metrics.speaker_attribution_accuracy, 4),
            "cpwer": round(eval_metrics.cpwer, 4),
            "diarization_gap": round(eval_metrics.diarization_gap, 4),
            "speaker_mapping": eval_metrics.speaker_mapping,
            "format_valid": eval_metrics.format_valid,
        },
        "predicted_turns": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "start_time": t.start_time,
                "end_time": t.end_time,
            }
            for t in predicted_turns
        ],
        "raw_response": raw_response,
    }
    if stage2_triggered is not None:
        rec["stage2_triggered"] = stage2_triggered
    return rec


def run_debt_collection_benchmark(
    project_id: str = "my-argolis-prj",
    location: str = "global",
    subset_dir: Path = DEFAULT_SUBSET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Executes live comparative benchmark across all 50 debt collection samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_25_path = output_dir / "predictions_gemini_2_5_flash.json"
    pred_c5_path = output_dir / "predictions_strategy_c5_adaptive_champion.json"
    summary_path = output_dir / "debt_collection_comparative_summary.json"

    samples = DatasetLoader.load_benchmark_subset(subset_dir=subset_dir, verify_audio=True)
    assert len(samples) == 50, f"Expected 50 samples in {subset_dir}, found {len(samples)}"

    # Pre-warm project fallback once at startup
    logger.info(f"Pre-warming Vertex AI connection (initial project={project_id}, location={location})...")
    root_client = GeminiClient(project_id=project_id, location=location)
    conn_ok = root_client.test_connection(model=MODEL_2_5_FLASH)
    resolved_project_id = root_client.project_id
    logger.info(f"Pre-flight connection verified={conn_ok}. Resolved project_id='{resolved_project_id}'.")

    # Load any existing checkpointed predictions
    saved_25: Dict[str, Dict[str, Any]] = {}
    saved_c5: Dict[str, Dict[str, Any]] = {}

    if pred_25_path.exists():
        try:
            with open(pred_25_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if item.get("predicted_turns"):
                        saved_25[item["sample_id"]] = item
        except Exception as e:
            logger.warning(f"Could not read existing {pred_25_path}: {e}")

    if pred_c5_path.exists():
        try:
            with open(pred_c5_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if item.get("predicted_turns"):
                        saved_c5[item["sample_id"]] = item
        except Exception as e:
            logger.warning(f"Could not read existing {pred_c5_path}: {e}")

    lock = threading.Lock()
    records_25: Dict[str, Dict[str, Any]] = {}
    records_c5: Dict[str, Dict[str, Any]] = {}
    live_todo_samples: List[SampleData] = []

    for sample in samples:
        sid = sample.sample_id
        has_25 = False
        has_c5 = False

        if sid in saved_25:
            src_rec = saved_25[sid]
            hyp_turns = [Turn(**t) for t in src_rec["predicted_turns"]]
            records_25[sid] = format_prediction_record(
                sample=sample,
                model_id=MODEL_2_5_FLASH,
                approach="single_step",
                predicted_turns=hyp_turns,
                raw_response=src_rec.get("raw_response", ""),
                latency_seconds=src_rec["latency_seconds"],
                source="live_vertex_ai",
            )
            has_25 = True

        if sid in saved_c5:
            src_rec = saved_c5[sid]
            hyp_turns = [Turn(**t) for t in src_rec["predicted_turns"]]
            records_c5[sid] = format_prediction_record(
                sample=sample,
                model_id=MODEL_3_5_FLASH_LITE,
                approach="strategy_c5_adaptive_champion",
                predicted_turns=hyp_turns,
                raw_response=src_rec.get("raw_response", ""),
                latency_seconds=src_rec["latency_seconds"],
                source="live_vertex_ai",
                stage2_triggered=src_rec.get("stage2_triggered"),
            )
            has_c5 = True

        if not (has_25 and has_c5):
            live_todo_samples.append(sample)

    logger.info(
        f"Loaded {len(records_25)}/50 for 2.5 Flash and {len(records_c5)}/50 for Candidate C5 from checkpoints. "
        f"Remaining samples requiring live Vertex AI inference: {len(live_todo_samples)}"
    )

    def _process_sample(sample: SampleData) -> None:
        sid = sample.sample_id
        # Instantiate worker client directly with resolved_project_id to avoid redundant 403 uploads
        worker_client = GeminiClient(project_id=resolved_project_id, location=location)

        if sid not in records_25:
            logger.info(f"[LIVE START] {sid} (2.5 Flash, dur={sample.duration_seconds}s)...")
            p25 = SingleStepPipeline(client=worker_client)
            pred25 = p25.run_sample(sample, model_id=MODEL_2_5_FLASH)
            rec25 = format_prediction_record(
                sample=sample,
                model_id=MODEL_2_5_FLASH,
                approach="single_step",
                predicted_turns=pred25.predicted_turns,
                raw_response=pred25.raw_response,
                latency_seconds=pred25.latency_seconds,
                source="live_vertex_ai",
            )
            with lock:
                records_25[sid] = rec25
                ordered_25 = [records_25[s.sample_id] for s in samples if s.sample_id in records_25]
                with open(pred_25_path, "w", encoding="utf-8") as f:
                    json.dump(ordered_25, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[LIVE DONE] {sid} (2.5 Flash) [{len(records_25)}/50]: lat={pred25.latency_seconds:.2f}s, "
                f"SAA={rec25['metrics']['speaker_attribution_accuracy']:.4f}, WER={rec25['metrics']['wer']:.4f}"
            )

        if sid not in records_c5:
            logger.info(f"[LIVE START] {sid} (Candidate C5, dur={sample.duration_seconds}s)...")
            pc5 = AdaptiveAcousticChampionPipeline(client=worker_client)
            predc5 = pc5.run_sample(sample, model_id=MODEL_3_5_FLASH_LITE)
            s2_trig, _ = should_trigger_stage2(sample.duration_seconds, predc5.predicted_turns, [])
            recc5 = format_prediction_record(
                sample=sample,
                model_id=MODEL_3_5_FLASH_LITE,
                approach="strategy_c5_adaptive_champion",
                predicted_turns=predc5.predicted_turns,
                raw_response=predc5.raw_response,
                latency_seconds=predc5.latency_seconds,
                source="live_vertex_ai",
                stage2_triggered=s2_trig,
            )
            with lock:
                records_c5[sid] = recc5
                ordered_c5 = [records_c5[s.sample_id] for s in samples if s.sample_id in records_c5]
                with open(pred_c5_path, "w", encoding="utf-8") as f:
                    json.dump(ordered_c5, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[LIVE DONE] {sid} (Candidate C5) [{len(records_c5)}/50]: lat={predc5.latency_seconds:.2f}s, "
                f"SAA={recc5['metrics']['speaker_attribution_accuracy']:.4f}, WER={recc5['metrics']['wer']:.4f}"
            )

    if live_todo_samples:
        sorted_todo = sorted(live_todo_samples, key=lambda s: s.duration_seconds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_sample, s) for s in sorted_todo]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    # Final ordered lists of all 50 records
    final_list_25 = [records_25[s.sample_id] for s in samples]
    final_list_c5 = [records_c5[s.sample_id] for s in samples]

    with open(pred_25_path, "w", encoding="utf-8") as f:
        json.dump(final_list_25, f, ensure_ascii=False, indent=2)
    with open(pred_c5_path, "w", encoding="utf-8") as f:
        json.dump(final_list_c5, f, ensure_ascii=False, indent=2)

    # Compute overall macro summaries
    macro_25 = compute_macro_summary(final_list_25)
    macro_25["name"] = "Gemini 2.5 Flash Baseline"
    macro_25["speedup_vs_2_5"] = 1.0

    macro_c5 = compute_macro_summary(final_list_c5, baseline_latency=macro_25["mean_latency_seconds"])
    macro_c5["name"] = "Strategy C5: Adaptive Acoustic Champion (Pure 3.5 Flash Lite)"

    # Compute stratified duration bucket breakdowns
    bucket_defs = [
        ("<70s", [30.0, 70.0]),
        ("70s-180s", [70.0, 180.0]),
        ("180s-300s", [180.0, 300.0]),
        ("300s-600s", [300.0, 600.0]),
    ]
    duration_buckets: Dict[str, Any] = {}
    for b_key, b_range in bucket_defs:
        b_samples = [s for s in samples if get_duration_bucket_key(s.duration_seconds) == b_key]
        b_ids = {s.sample_id for s in b_samples}
        b_25 = [r for r in final_list_25 if r["sample_id"] in b_ids]
        b_c5 = [r for r in final_list_c5 if r["sample_id"] in b_ids]

        sum_25 = compute_macro_summary(b_25)
        sum_25["name"] = "Gemini 2.5 Flash Baseline"
        sum_c5 = compute_macro_summary(b_c5, baseline_latency=sum_25["mean_latency_seconds"])
        sum_c5["name"] = "Strategy C5: Adaptive Acoustic Champion (Pure 3.5 Flash Lite)"

        mean_dur = round(sum(s.duration_seconds for s in b_samples) / len(b_samples), 2) if b_samples else 0.0
        mean_ov = round(sum(s.overlap_ratio for s in b_samples) / len(b_samples), 2) if b_samples else 0.0

        saa_delta = round(sum_c5["mean_speaker_attribution_accuracy"] - sum_25["mean_speaker_attribution_accuracy"], 4)
        cpwer_delta = round(sum_c5["mean_cpwer"] - sum_25["mean_cpwer"], 4)
        gap_delta = round(sum_c5["mean_diarization_gap"] - sum_25["mean_diarization_gap"], 4)
        wer_delta = round(sum_c5["mean_wer"] - sum_25["mean_wer"], 4)

        duration_buckets[b_key] = {
            "bucket_label": b_key,
            "duration_range_seconds": b_range,
            "sample_count": len(b_samples),
            "mean_duration_seconds": mean_dur,
            "mean_overlap_ratio": mean_ov,
            "gemini_2_5_flash": sum_25,
            "strategy_c5_adaptive_champion": sum_c5,
            "comparison": {
                "saa_delta_c5_vs_2_5": saa_delta,
                "cpwer_delta_c5_vs_2_5": cpwer_delta,
                "diarization_gap_delta_c5_vs_2_5": gap_delta,
                "wer_delta_c5_vs_2_5": wer_delta,
                "speedup_c5_vs_2_5": sum_c5["speedup_vs_2_5"],
            },
        }

    # Compute conflict tiers breakdown
    conflict_tiers: Dict[str, Any] = {}
    for c_key in ["moderate_12_16_pct", "high_16_20_pct", "extreme_over_20_pct"]:
        c_samples = [s for s in samples if get_conflict_tier_key(s.overlap_ratio) == c_key]
        c_ids = {s.sample_id for s in c_samples}
        c_25 = [r for r in final_list_25 if r["sample_id"] in c_ids]
        c_c5 = [r for r in final_list_c5 if r["sample_id"] in c_ids]
        sum_25 = compute_macro_summary(c_25)
        sum_c5 = compute_macro_summary(c_c5, baseline_latency=sum_25["mean_latency_seconds"])
        conflict_tiers[c_key] = {
            "tier_label": c_key,
            "sample_count": len(c_samples),
            "mean_overlap_ratio": round(sum(s.overlap_ratio for s in c_samples) / len(c_samples), 2) if c_samples else 0.0,
            "gemini_2_5_flash": sum_25,
            "strategy_c5_adaptive_champion": sum_c5,
        }

    total_dur = sum(s.duration_seconds for s in samples)
    mean_dur = total_dur / len(samples)
    mean_ov = sum(s.overlap_ratio for s in samples) / len(samples)

    comparative_summary = {
        "benchmark_title": "Gemini STT & Speaker Diarization: 50-Sample Hindi Debt Collection & Fierce Argument Benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": resolved_project_id,
        "total_samples": len(samples),
        "total_duration_seconds": round(total_dur, 2),
        "mean_sample_duration_seconds": round(mean_dur, 2),
        "mean_overlap_ratio": round(mean_ov, 2),
        "macro_summary": {
            "gemini_2_5_flash": macro_25,
            "strategy_c5_adaptive_champion": macro_c5,
        },
        "duration_buckets": duration_buckets,
        "conflict_tiers": conflict_tiers,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparative_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved complete 50-sample comparative summary to {summary_path}")
    logger.info(f"2.5 Flash Macro Summary: {json.dumps(macro_25)}")
    logger.info(f"Candidate C5 Macro Summary: {json.dumps(macro_c5)}")
    return comparative_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50-sample Hindi debt collection benchmark.")
    parser.add_argument("--project-id", type=str, default="my-argolis-prj")
    parser.add_argument("--location", type=str, default="global")
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    run_debt_collection_benchmark(
        project_id=args.project_id,
        location=args.location,
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
