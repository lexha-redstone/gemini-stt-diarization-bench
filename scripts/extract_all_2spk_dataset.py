#!/usr/bin/env python3
"""Dataset extraction script for sarvamai/indic-diarbench complete 37-sample 2-Speaker subset.

Extracts:
1. All 37 two-speaker (num_speakers == 2) audio WAV files (16kHz, 16-bit mono) into data/all_2spk_subset/audio/
2. Ground-truth turns, timestamps, overlap metrics, and metadata into data/all_2spk_subset/metadata.json
"""

import argparse
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_dataset import compute_overlap_duration, compute_turn_switches
from src.normalizer import IndicTextNormalizer

DEFAULT_PARQUET_PATH = Path.home() / ".cache/huggingface/hub/datasets--sarvamai--indic-diarbench/snapshots/92877bad8aab6e598167d91c6ee02aa8ca6ede09/Hindi/test-00000-of-00001.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "all_2spk_subset"
AUDIO_DIR = OUTPUT_DIR / "audio"
METADATA_PATH = OUTPUT_DIR / "metadata.json"


def extract_all_2spk_dataset(parquet_path: Path = DEFAULT_PARQUET_PATH, output_dir: Path = OUTPUT_DIR) -> List[Dict[str, Any]]:
    """Extracts all 37 two-speaker samples from the Parquet cache."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet cache not found at {parquet_path}")

    audio_dir = output_dir / "audio"
    metadata_path = output_dir / "metadata.json"
    audio_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    sub_df = df[df["num_speakers"] == 2].copy()

    sample_ids = sorted(list(sub_df["sample_id"]))
    assert len(sample_ids) == 37, f"Expected 37 2-speaker samples, found {len(sample_ids)}"

    metadata: List[Dict[str, Any]] = []
    for sid in sample_ids:
        row = sub_df[sub_df["sample_id"] == sid].iloc[0]
        wav_filename = f"{sid}.wav"
        wav_path = audio_dir / wav_filename

        audio_bytes = row["audio"]["bytes"]
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected mono for {sid}"
            assert wf.getframerate() == 16000, f"Expected 16kHz for {sid}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit PCM for {sid}"

        raw_segs = list(row["annotated_transcript"])
        sorted_segs = sorted(raw_segs, key=lambda s: float(s["start_time"]))
        dur = float(row["duration_seconds"])
        ov_dur = compute_overlap_duration(sorted_segs)
        ov_ratio = (ov_dur / dur * 100.0) if dur > 0 else 0.0
        switches = compute_turn_switches(sorted_segs)

        ground_truth_turns = [
            {
                "speaker": str(s["speaker_id"]),
                "text": str(s["transcript"]),
                "start_time": float(s["start_time"]),
                "end_time": float(s["end_time"]),
            }
            for s in sorted_segs
        ]

        dialogue_raw = "\n".join(f"{t['speaker']}: {t['text']}" for t in ground_truth_turns)
        dialogue_clean = "\n".join(
            f"{t['speaker']}: {IndicTextNormalizer.normalize(t['text'])}"
            for t in ground_truth_turns
            if IndicTextNormalizer.normalize(t["text"])
        )

        meta_entry = {
            "sample_id": sid,
            "recording_id": str(row["recording_id"]),
            "language": str(row["language"]),
            "dataset_type": str(row["dataset_type"]),
            "duration_seconds": round(dur, 2),
            "num_speakers": 2,
            "num_segments": int(row["num_segments"]),
            "overlap_duration": round(float(ov_dur), 2),
            "overlap_ratio": round(ov_ratio, 2),
            "turn_switches": switches,
            "audio_path": f"data/all_2spk_subset/audio/{wav_filename}",
            "audio_file": f"audio/{wav_filename}",
            "speaker_labels": sorted(list(set(t["speaker"] for t in ground_truth_turns))),
            "ground_truth_turns": ground_truth_turns,
            "annotated_transcript": [
                {
                    "speaker_id": str(s["speaker_id"]),
                    "transcript": str(s["transcript"]),
                    "start_time": float(s["start_time"]),
                    "end_time": float(s["end_time"]),
                }
                for s in sorted_segs
            ],
            "reference_dialogue_raw": dialogue_raw,
            "reference_dialogue_indic_clean": dialogue_clean,
        }
        metadata.append(meta_entry)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    total_dur = sum(m["duration_seconds"] for m in metadata)
    mean_ov = sum(m["overlap_ratio"] for m in metadata) / len(metadata)
    print(f"Successfully extracted {len(metadata)} 2-speaker samples to {output_dir}")
    print(f"Total duration: {total_dur:.1f}s (~{total_dur/60:.1f} min), Mean overlap: {mean_ov:.2f}%")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all 37 2-speaker samples from sarvamai/indic-diarbench.")
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    extract_all_2spk_dataset(parquet_path=args.parquet_path, output_dir=args.output_dir)
