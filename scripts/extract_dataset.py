#!/usr/bin/env python3
"""Dataset extraction script for sarvamai/indic-diarbench 20-sample benchmark subset.

Extracts:
1. 20 audio WAV files (16kHz, 16-bit mono) into data/benchmark_subset/audio/
2. Ground-truth turns, timestamps, overlap metrics, and metadata into data/benchmark_subset/metadata.json
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

from src.normalizer import IndicTextNormalizer

DEFAULT_PARQUET_PATH = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--sarvamai--indic-diarbench/snapshots/"
    "92877bad8aab6e598167d91c6ee02aa8ca6ede09/Hindi/test-00000-of-00001.parquet"
))

OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark_subset"
AUDIO_DIR = OUTPUT_DIR / "audio"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

BENCHMARK_SAMPLE_IDS = [
    "hindi_064", "hindi_067", "hindi_066", "hindi_070", "hindi_065",
    "hindi_069", "hindi_073", "hindi_044", "hindi_045", "hindi_046",
    "hindi_042", "hindi_024", "hindi_085", "hindi_001", "hindi_087",
    "hindi_093", "hindi_086", "hindi_089", "hindi_092", "hindi_090",
]


def compute_overlap_duration(segments: List[Dict[str, Any]]) -> float:
    """Computes total duration where at least two different speakers speak simultaneously."""
    events = []
    for s in segments:
        events.append((float(s["start_time"]), 1, s["speaker_id"]))
        events.append((float(s["end_time"]), -1, s["speaker_id"]))

    events.sort(key=lambda x: x[0])

    current_speakers: Dict[str, int] = {}
    total_overlap_duration = 0.0
    last_time = 0.0

    for t, delta, spk in events:
        active_unique_spks = sum(1 for v in current_speakers.values() if v > 0)
        if active_unique_spks >= 2 and t > last_time:
            total_overlap_duration += (t - last_time)
        current_speakers[spk] = current_speakers.get(spk, 0) + delta
        last_time = t

    return total_overlap_duration


def compute_turn_switches(segments: List[Dict[str, Any]]) -> int:
    """Computes number of speaker switches in chronological order."""
    if not segments:
        return 0
    switches = 0
    prev_spk = segments[0]["speaker_id"]
    for s in segments[1:]:
        if s["speaker_id"] != prev_spk:
            switches += 1
            prev_spk = s["speaker_id"]
    return switches


def extract_benchmark_dataset(parquet_path: Path = DEFAULT_PARQUET_PATH) -> List[Dict[str, Any]]:
    """Extracts benchmark subset audio files and metadata.json."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet cache not found at {parquet_path}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading parquet archive from {parquet_path}...")
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    sub_df = df[df["sample_id"].isin(BENCHMARK_SAMPLE_IDS)].copy()
    if len(sub_df) != len(BENCHMARK_SAMPLE_IDS):
        found_ids = set(sub_df["sample_id"])
        missing = set(BENCHMARK_SAMPLE_IDS) - found_ids
        raise ValueError(f"Missing expected sample IDs in parquet file: {missing}")

    metadata: List[Dict[str, Any]] = []

    for sid in BENCHMARK_SAMPLE_IDS:
        row = sub_df[sub_df["sample_id"] == sid].iloc[0]
        wav_filename = f"{sid}.wav"
        wav_path = AUDIO_DIR / wav_filename

        # Write WAV audio bytes directly
        audio_bytes = row["audio"]["bytes"]
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        # Validate audio format with wave module
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            calc_duration = nframes / float(framerate)

            assert channels == 1, f"Expected mono audio for {sid}, got {channels} channels"
            assert framerate == 16000, f"Expected 16000Hz for {sid}, got {framerate}Hz"
            assert sampwidth == 2, f"Expected 16-bit PCM for {sid}, got {sampwidth} bytes/sample"

        # Process annotations
        raw_segs = list(row["annotated_transcript"])
        sorted_segs = sorted(raw_segs, key=lambda s: float(s["start_time"]))

        overlap_duration = compute_overlap_duration(sorted_segs)
        duration_sec = round(float(row["duration_seconds"]), 2)
        overlap_ratio = round((overlap_duration / duration_sec) * 100.0, 2) if duration_sec > 0 else 0.0
        turn_switches = compute_turn_switches(sorted_segs)

        ground_truth_turns = [
            {
                "speaker": str(s["speaker_id"]),
                "text": str(s["transcript"]),
                "start_time": float(s["start_time"]),
                "end_time": float(s["end_time"]),
            }
            for s in sorted_segs
        ]

        dialogue_raw = "\n".join(
            f"{t['speaker']}: {t['text']}" for t in ground_truth_turns
        )
        dialogue_clean = "\n".join(
            f"{t['speaker']}: {IndicTextNormalizer.normalize(t['text'])}"
            for t in ground_truth_turns
            if IndicTextNormalizer.normalize(t["text"])
        )

        speaker_labels = sorted(list(set(t["speaker"] for t in ground_truth_turns)))

        # Relative path from project root
        rel_audio_path = f"data/benchmark_subset/audio/{wav_filename}"

        meta_entry = {
            "sample_id": sid,
            "recording_id": str(row["recording_id"]),
            "language": str(row["language"]),
            "dataset_type": str(row["dataset_type"]),
            "duration_seconds": duration_sec,
            "num_speakers": int(row["num_speakers"]),
            "num_segments": int(row["num_segments"]),
            "overlap_duration": round(float(overlap_duration), 2),
            "overlap_ratio": overlap_ratio,
            "turn_switches": turn_switches,
            "audio_path": rel_audio_path,
            "audio_file": f"audio/{wav_filename}",
            "speaker_labels": speaker_labels,
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

    print(f"Successfully extracted {len(metadata)} samples to {OUTPUT_DIR}")
    print(f"Metadata written to {METADATA_PATH}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract curated benchmark subset from sarvamai/indic-diarbench."
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help="Path to local Parquet cache file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for extracted audio and metadata.",
    )
    args = parser.parse_args()
    extract_benchmark_dataset(parquet_path=args.parquet_path)

