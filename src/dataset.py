"""Dataset loader and benchmark management for indic-diarbench subset."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.models import SampleData, Turn

# 20 curated benchmark sample IDs from sarvamai/indic-diarbench Hindi split
BENCHMARK_SAMPLE_IDS = [
    "hindi_064", "hindi_067", "hindi_066", "hindi_070", "hindi_065",
    "hindi_069", "hindi_073", "hindi_044", "hindi_045", "hindi_046",
    "hindi_042", "hindi_024", "hindi_085", "hindi_001", "hindi_087",
    "hindi_093", "hindi_086", "hindi_089", "hindi_092", "hindi_090",
]


class DatasetLoader:
    """Loader for the cached 20-sample benchmark subset."""

    DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_SUBSET_DIR = DEFAULT_PROJECT_ROOT / "data" / "benchmark_subset"

    @classmethod
    def get_metadata_path(cls, subset_dir: Optional[Union[str, Path]] = None) -> Path:
        """Returns the path to metadata.json."""
        base_dir = Path(subset_dir) if subset_dir else cls.DEFAULT_SUBSET_DIR
        return base_dir / "metadata.json"

    @classmethod
    def load_benchmark_subset(
        cls,
        subset_dir: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
        verify_audio: bool = True,
    ) -> List[SampleData]:
        """Loads and validates the 20 benchmark samples.

        Args:
            subset_dir: Directory containing metadata.json and audio/ folder.
            project_root: Root workspace directory for resolving relative audio paths.
            verify_audio: Whether to verify that audio files exist on disk.

        Returns:
            List of validated SampleData Pydantic models.

        Raises:
            FileNotFoundError: If metadata.json or referenced audio files are missing.
            ValueError: If sample data fails Pydantic schema validation.
        """
        root = Path(project_root) if project_root else cls.DEFAULT_PROJECT_ROOT
        base_dir = Path(subset_dir) if subset_dir else cls.DEFAULT_SUBSET_DIR
        metadata_path = base_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Benchmark metadata not found at {metadata_path}. "
                f"Run `scripts/extract_dataset.py` to extract the dataset."
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        samples: List[SampleData] = []
        for item in raw_entries:
            # Resolve audio path
            audio_path_str = item.get("audio_path")
            if not audio_path_str:
                audio_file_str = item.get("audio_file", f"audio/{item['sample_id']}.wav")
                audio_path_str = str(base_dir / audio_file_str)

            resolved_path = Path(audio_path_str)
            if not resolved_path.is_absolute():
                # Try relative to root first, then relative to subset_dir
                if (root / resolved_path).exists():
                    resolved_path = root / resolved_path
                elif (base_dir / resolved_path).exists():
                    resolved_path = base_dir / resolved_path
                else:
                    resolved_path = root / resolved_path

            if verify_audio and not resolved_path.exists():
                raise FileNotFoundError(
                    f"Audio file for sample {item.get('sample_id')} not found at {resolved_path}"
                )

            # Build turns
            turns_data = item.get("ground_truth_turns", [])
            turns = [
                Turn(
                    speaker=t["speaker"],
                    text=t["text"],
                    start_time=t.get("start_time"),
                    end_time=t.get("end_time"),
                )
                for t in turns_data
            ]

            sample = SampleData(
                sample_id=str(item["sample_id"]),
                audio_path=str(resolved_path),
                duration_seconds=float(item["duration_seconds"]),
                num_speakers=int(item["num_speakers"]),
                overlap_ratio=float(item.get("overlap_ratio", 0.0)),
                ground_truth_turns=turns,
                recording_id=item.get("recording_id"),
                language=item.get("language"),
                dataset_type=item.get("dataset_type"),
                num_segments=item.get("num_segments"),
                overlap_duration=item.get("overlap_duration"),
                turn_switches=item.get("turn_switches"),
                audio_file=item.get("audio_file"),
                speaker_labels=item.get("speaker_labels"),
                annotated_transcript=item.get("annotated_transcript"),
                reference_dialogue_raw=item.get("reference_dialogue_raw"),
                reference_dialogue_indic_clean=item.get("reference_dialogue_indic_clean"),
            )
            samples.append(sample)

        return samples

    @classmethod
    def get_sample(
        cls,
        sample_id: str,
        subset_dir: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> SampleData:
        """Retrieves a single benchmark sample by its sample_id."""
        samples = cls.load_benchmark_subset(subset_dir=subset_dir, project_root=project_root)
        for s in samples:
            if s.sample_id == sample_id:
                return s
        raise KeyError(f"Sample ID '{sample_id}' not found in benchmark subset.")

    @classmethod
    def get_summary_stats(cls, samples: List[SampleData]) -> Dict[str, Any]:
        """Computes summary statistics for a collection of SampleData objects."""
        if not samples:
            return {}

        total_duration = sum(s.duration_seconds for s in samples)
        mean_duration = total_duration / len(samples)
        mean_overlap = sum(s.overlap_ratio for s in samples) / len(samples)
        speaker_counts: Dict[int, int] = {}
        conditions: Dict[str, int] = {}

        for s in samples:
            speaker_counts[s.num_speakers] = speaker_counts.get(s.num_speakers, 0) + 1
            if s.dataset_type:
                conditions[s.dataset_type] = conditions.get(s.dataset_type, 0) + 1

        return {
            "total_samples": len(samples),
            "total_duration_seconds": round(total_duration, 2),
            "mean_duration_seconds": round(mean_duration, 2),
            "mean_overlap_ratio": round(mean_overlap, 2),
            "speaker_distribution": speaker_counts,
            "acoustic_conditions": conditions,
        }
