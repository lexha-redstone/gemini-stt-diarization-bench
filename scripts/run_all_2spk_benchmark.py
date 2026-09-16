#!/usr/bin/env python3
"""Milestone 5: Complete 37-Sample 2-Speaker Hybrid Benchmark Runner.

Evaluates and compares:
1. Gemini 2.5 Flash Baseline (`gemini-2.5-flash` via `SingleStepPipeline`)
2. Adaptive Acoustic Champion (`Candidate C5`, pure `gemini-3.5-flash-lite` via `AdaptiveAcousticChampionPipeline`)

Across all 37 two-speaker (`num_speakers == 2`) audio samples in `sarvamai/indic-diarbench` (Hindi split):
- Reuses verified prediction outputs for the 21 samples already evaluated in `results/hard_2spk/` (20 samples)
  and `results/baseline/` + `results/optimized/` (`hindi_069`, 1 sample).
- Executes live Vertex AI inference strictly on the 16 remaining un-evaluated 2-speaker samples.
- Computes exact macro metrics via `src/metrics.py` and saves:
  - `results/all_2spk/predictions_gemini_2_5_flash.json`
  - `results/all_2spk/predictions_strategy_c5_adaptive_champion.json`
  - `results/all_2spk/all_2spk_comparative_summary.json`
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
from src.config import MODEL_2_5_FLASH, MODEL_3_5_FLASH_LITE
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import SampleData, Turn
from src.pipelines.adaptive_acoustic_champion import AdaptiveAcousticChampionPipeline
from src.pipelines.single_step import SingleStepPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_all_2spk_benchmark")

DEFAULT_SUBSET_DIR = PROJECT_ROOT / "data" / "all_2spk_subset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "all_2spk"


def compute_macro_summary(records: List[Dict[str, Any]], baseline_latency: Optional[float] = None) -> Dict[str, Any]:
    """Computes macro average metrics across a list of prediction records."""
    if not records:
        return {}

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

    # Normalized metrics excluding pathological outliers (WER > 2.0)
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


def load_existing_21_predictions(project_root: Path = PROJECT_ROOT) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Loads verified predictions for the 21 previously evaluated 2-speaker samples."""
    reused_25: Dict[str, Dict[str, Any]] = {}
    reused_c5: Dict[str, Dict[str, Any]] = {}

    # 1. Load 20 samples from hard_2spk
    hard_25_path = project_root / "results" / "hard_2spk" / "predictions_gemini_2_5_flash.json"
    hard_c5_path = project_root / "results" / "hard_2spk" / "predictions_strategy_c5_adaptive_champion.json"

    if hard_25_path.exists():
        with open(hard_25_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                reused_25[item["sample_id"]] = item

    if hard_c5_path.exists():
        with open(hard_c5_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                reused_c5[item["sample_id"]] = item

    # 2. Load additional 2-speaker sample (hindi_069) from baseline / optimized
    base_25_path = project_root / "results" / "baseline" / "predictions_2.5_flash.json"
    gen_c5_path = project_root / "results" / "optimized" / "predictions_strategy_c5_generalization.json"

    if base_25_path.exists():
        with open(base_25_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                if item.get("num_speakers") == 2 and item["sample_id"] not in reused_25:
                    reused_25[item["sample_id"]] = item

    if gen_c5_path.exists():
        with open(gen_c5_path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                if item.get("num_speakers") == 2 and item["sample_id"] not in reused_c5:
                    reused_c5[item["sample_id"]] = item

    return {"gemini_2_5_flash": reused_25, "strategy_c5_adaptive_champion": reused_c5}


def format_prediction_record(
    sample: SampleData,
    model_id: str,
    approach: str,
    predicted_turns: List[Turn],
    raw_response: str,
    latency_seconds: float,
    source: str = "live_vertex_ai",
) -> Dict[str, Any]:
    """Formats a single prediction record with verified MetricsEngine evaluation."""
    eval_metrics = MetricsEngine.evaluate_sample(
        sample.ground_truth_turns, predicted_turns, normalize=True
    )
    return {
        "sample_id": sample.sample_id,
        "model_id": model_id,
        "approach": approach,
        "source": source,
        "duration_seconds": round(sample.duration_seconds, 2),
        "num_speakers": sample.num_speakers,
        "overlap_ratio": round(sample.overlap_ratio, 2),
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


def run_all_2spk_benchmark(
    project_id: str = "my-argolis-prj",
    location: str = "global",
    subset_dir: Path = DEFAULT_SUBSET_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """Executes the hybrid 37-sample 2-speaker benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_25_path = output_dir / "predictions_gemini_2_5_flash.json"
    pred_c5_path = output_dir / "predictions_strategy_c5_adaptive_champion.json"
    summary_path = output_dir / "all_2spk_comparative_summary.json"

    samples = DatasetLoader.load_benchmark_subset(subset_dir)
    assert len(samples) == 37, f"Expected 37 samples in {subset_dir}, found {len(samples)}"
    samples_by_id = {s.sample_id: s for s in samples}

    existing = load_existing_21_predictions(PROJECT_ROOT)
    reused_25 = existing["gemini_2_5_flash"]
    reused_c5 = existing["strategy_c5_adaptive_champion"]

    logger.info(f"Loaded {len(reused_25)} existing 2.5 Flash predictions and {len(reused_c5)} existing C5 predictions.")

    # Load any already saved progress in output_dir
    saved_25: Dict[str, Dict[str, Any]] = {}
    saved_c5: Dict[str, Dict[str, Any]] = {}
    if pred_25_path.exists():
        try:
            with open(pred_25_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    saved_25[item["sample_id"]] = item
        except Exception as e:
            logger.warning(f"Could not read existing {pred_25_path}: {e}")

    if pred_c5_path.exists():
        try:
            with open(pred_c5_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    saved_c5[item["sample_id"]] = item
        except Exception as e:
            logger.warning(f"Could not read existing {pred_c5_path}: {e}")

    import concurrent.futures
    import threading

    lock = threading.Lock()
    records_25: Dict[str, Dict[str, Any]] = {}
    records_c5: Dict[str, Dict[str, Any]] = {}

    # 1. Pre-populate reused and already-saved predictions
    live_todo_samples: List[SampleData] = []
    for sample in samples:
        sid = sample.sample_id
        has_25 = False
        has_c5 = False

        if sid in reused_25:
            src_rec = reused_25[sid]
            hyp_turns = [Turn(**t) for t in src_rec["predicted_turns"]]
            src_label = "reused_hard_2spk" if sid != "hindi_069" else "reused_indic_subset"
            records_25[sid] = format_prediction_record(
                sample=sample,
                model_id=MODEL_2_5_FLASH,
                approach="single_step",
                predicted_turns=hyp_turns,
                raw_response=src_rec.get("raw_response", ""),
                latency_seconds=src_rec["latency_seconds"],
                source=src_label,
            )
            has_25 = True
        elif sid in saved_25 and saved_25[sid].get("predicted_turns"):
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

        if sid in reused_c5:
            src_rec = reused_c5[sid]
            hyp_turns = [Turn(**t) for t in src_rec["predicted_turns"]]
            src_label = "reused_hard_2spk" if sid != "hindi_069" else "reused_indic_subset"
            records_c5[sid] = format_prediction_record(
                sample=sample,
                model_id=MODEL_3_5_FLASH_LITE,
                approach="strategy_c5_adaptive_champion",
                predicted_turns=hyp_turns,
                raw_response=src_rec.get("raw_response", ""),
                latency_seconds=src_rec["latency_seconds"],
                source=src_label,
            )
            has_c5 = True
        elif sid in saved_c5 and saved_c5[sid].get("predicted_turns"):
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
            )
            has_c5 = True

        if not (has_25 and has_c5):
            live_todo_samples.append(sample)

    logger.info(
        f"Pre-populated {len(records_25)}/37 for 2.5 Flash and {len(records_c5)}/37 for Candidate C5. "
        f"Remaining samples requiring live inference: {len(live_todo_samples)}"
    )

    def _process_live_sample(sample: SampleData) -> None:
        sid = sample.sample_id
        worker_client = GeminiClient(project_id=project_id, location=location)

        if sid not in records_25:
            logger.info(f"[LIVE] {sid} (2.5 Flash): Starting inference (dur={sample.duration_seconds}s)...")
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
                with open(pred_25_path, "w", encoding="utf-8") as f:
                    json.dump(list(records_25.values()), f, ensure_ascii=False, indent=2)
            logger.info(
                f"[LIVE DONE] {sid} (2.5 Flash): latency={pred25.latency_seconds:.2f}s, "
                f"WER={rec25['metrics']['wer']:.4f}, SAA={rec25['metrics']['speaker_attribution_accuracy']:.4f}"
            )

        if sid not in records_c5:
            logger.info(f"[LIVE] {sid} (Candidate C5): Starting inference (dur={sample.duration_seconds}s)...")
            pc5 = AdaptiveAcousticChampionPipeline(client=worker_client)
            predc5 = pc5.run_sample(sample, model_id=MODEL_3_5_FLASH_LITE)
            recc5 = format_prediction_record(
                sample=sample,
                model_id=MODEL_3_5_FLASH_LITE,
                approach="strategy_c5_adaptive_champion",
                predicted_turns=predc5.predicted_turns,
                raw_response=predc5.raw_response,
                latency_seconds=predc5.latency_seconds,
                source="live_vertex_ai",
            )
            with lock:
                records_c5[sid] = recc5
                with open(pred_c5_path, "w", encoding="utf-8") as f:
                    json.dump(list(records_c5.values()), f, ensure_ascii=False, indent=2)
            logger.info(
                f"[LIVE DONE] {sid} (Candidate C5): latency={predc5.latency_seconds:.2f}s, "
                f"WER={recc5['metrics']['wer']:.4f}, SAA={recc5['metrics']['speaker_attribution_accuracy']:.4f}"
            )

    if live_todo_samples:
        # Sort shorter clips first so progress is visible immediately
        live_todo_sorted = sorted(live_todo_samples, key=lambda s: s.duration_seconds)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_process_live_sample, s) for s in live_todo_sorted]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    # Final sorted lists of 37 records
    final_list_25 = [records_25[s.sample_id] for s in samples]
    final_list_c5 = [records_c5[s.sample_id] for s in samples]

    with open(pred_25_path, "w", encoding="utf-8") as f:
        json.dump(final_list_25, f, ensure_ascii=False, indent=2)
    with open(pred_c5_path, "w", encoding="utf-8") as f:
        json.dump(final_list_c5, f, ensure_ascii=False, indent=2)

    # Compute macro summaries
    macro_25 = compute_macro_summary(final_list_25)
    macro_25["name"] = "Gemini 2.5 Flash Baseline"
    macro_25["speedup_vs_2_5"] = 1.0

    macro_c5 = compute_macro_summary(final_list_c5, baseline_latency=macro_25["mean_latency_seconds"])
    macro_c5["name"] = "Strategy C5: Adaptive Acoustic Champion (Pure 3.5 Flash Lite)"

    # Also compute slices: reused_21 vs live_16, and hard_20 vs standard_17
    hard_20_ids = set(reused_25.keys()) - {"hindi_069"}
    hard_25_slice = compute_macro_summary([r for r in final_list_25 if r["sample_id"] in hard_20_ids])
    hard_c5_slice = compute_macro_summary(
        [r for r in final_list_c5 if r["sample_id"] in hard_20_ids],
        baseline_latency=hard_25_slice["mean_latency_seconds"],
    )

    non_hard_17_25 = compute_macro_summary([r for r in final_list_25 if r["sample_id"] not in hard_20_ids])
    non_hard_17_c5 = compute_macro_summary(
        [r for r in final_list_c5 if r["sample_id"] not in hard_20_ids],
        baseline_latency=non_hard_17_25["mean_latency_seconds"],
    )

    live_16_25 = compute_macro_summary([r for r in final_list_25 if r["source"] == "live_vertex_ai"])
    live_16_c5 = compute_macro_summary(
        [r for r in final_list_c5 if r["source"] == "live_vertex_ai"],
        baseline_latency=live_16_25["mean_latency_seconds"],
    )

    total_dur = sum(s.duration_seconds for s in samples)
    mean_dur = total_dur / len(samples)
    mean_ov = sum(s.overlap_ratio for s in samples) / len(samples)

    comparative_summary = {
        "benchmark_title": "Gemini STT & Speaker Diarization: Complete 37-Sample 2-Speaker Benchmark (all_2spk_subset)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "total_samples": len(samples),
        "reused_samples_count": 21,
        "live_evaluated_samples_count": 16,
        "total_duration_seconds": round(total_dur, 2),
        "mean_sample_duration_seconds": round(mean_dur, 2),
        "mean_overlap_ratio": round(mean_ov, 2),
        "macro_summary": {
            "gemini_2_5_flash": macro_25,
            "strategy_c5_adaptive_champion": macro_c5,
        },
        "slices": {
            "hard_20_high_overlap": {
                "gemini_2_5_flash": hard_25_slice,
                "strategy_c5_adaptive_champion": hard_c5_slice,
            },
            "remaining_17_moderate_overlap": {
                "gemini_2_5_flash": non_hard_17_25,
                "strategy_c5_adaptive_champion": non_hard_17_c5,
            },
            "newly_evaluated_16_live": {
                "gemini_2_5_flash": live_16_25,
                "strategy_c5_adaptive_champion": live_16_c5,
            },
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparative_summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved complete 37-sample comparative summary to {summary_path}")
    logger.info(f"2.5 Flash Macro Summary: {json.dumps(macro_25)}")
    logger.info(f"Candidate C5 Macro Summary: {json.dumps(macro_c5)}")
    return comparative_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete 37-sample 2-speaker benchmark.")
    parser.add_argument("--project-id", type=str, default="my-argolis-prj")
    parser.add_argument("--location", type=str, default="global")
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_all_2spk_benchmark(
        project_id=args.project_id,
        location=args.location,
        subset_dir=args.subset_dir,
        output_dir=args.output_dir,
    )
