#!/usr/bin/env python3
"""Dataset extraction script for sarvamai/indic-diarbench 20-sample Hard 2-Speaker subset.

Extracts:
1. 20 audio WAV files (16kHz, 16-bit mono) into data/hard_2spk_subset/audio/
2. Ground-truth turns, timestamps, overlap metrics, and metadata into data/hard_2spk_subset/metadata.json
"""

import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_dataset import compute_overlap_duration, compute_turn_switches
from src.normalizer import IndicTextNormalizer

DEFAULT_PARQUET_PATH = Path.home() / ".cache/huggingface/hub/datasets--sarvamai--indic-diarbench/snapshots/92877bad8aab6e598167d91c6ee02aa8ca6ede09/Hindi/test-00000-of-00001.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "hard_2spk_subset"
AUDIO_DIR = OUTPUT_DIR / "audio"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

# Curated 20 2-speaker hard samples sorted by difficulty/overlap
HARD_2SPK_SAMPLE_IDS = [
    "hindi_062",  # 20.62% overlap, 300.3s (Top hard in dataset)
    "hindi_085",  # 15.15% overlap, 60.1s
    "hindi_064",  # 14.33% overlap, 60.2s
    "hindi_067",  # 13.52% overlap, 60.4s
    "hindi_087",  # 12.18% overlap, 60.2s
    "hindi_063",  # 11.45% overlap, 300.5s (34.4s overlap)
    "hindi_093",  # 10.72% overlap, 60.4s
    "hindi_066",  # 9.78% overlap, 60.5s
    "hindi_084",  # 7.42% overlap, 300.1s (22.3s overlap)
    "hindi_086",  # 5.77% overlap, 60.2s
    "hindi_029",  # 5.60% overlap, 73.6s
    "hindi_070",  # 4.60% overlap, 60.5s
    "hindi_083",  # 4.16% overlap, 300.2s
    "hindi_065",  # 3.95% overlap, 60.1s
    "hindi_089",  # 3.36% overlap, 60.4s
    "hindi_022",  # 3.32% overlap, 192.8s
    "hindi_042",  # 2.77% overlap, 60.4s
    "hindi_092",  # 2.57% overlap, 60.3s
    "hindi_090",  # 2.42% overlap, 60.1s
    "hindi_073",  # 2.24% overlap, 60.3s
]


def extract_hard_2spk_dataset(parquet_path: Path = DEFAULT_PARQUET_PATH):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()
    sub_df = df[df["sample_id"].isin(HARD_2SPK_SAMPLE_IDS)].copy()

    metadata = []
    for sid in HARD_2SPK_SAMPLE_IDS:
        row = sub_df[sub_df["sample_id"] == sid].iloc[0]
        wav_filename = f"{sid}.wav"
        wav_path = AUDIO_DIR / wav_filename
        
        audio_bytes = row["audio"]["bytes"]
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
            
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected mono for {sid}"
            assert wf.getframerate() == 16000, f"Expected 16kHz for {sid}"

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
            "audio_path": f"data/hard_2spk_subset/audio/{wav_filename}",
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

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    total_dur = sum(m["duration_seconds"] for m in metadata)
    mean_ov = sum(m["overlap_ratio"] for m in metadata) / len(metadata)
    print(f"Successfully extracted {len(metadata)} hard 2-speaker samples to {OUTPUT_DIR}")
    print(f"Total duration: {total_dur:.1f}s (~{total_dur/60:.1f} min), Mean overlap: {mean_ov:.2f}%")


if __name__ == "__main__":
    extract_hard_2spk_dataset()
