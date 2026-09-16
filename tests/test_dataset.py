"""Unit tests for DatasetLoader and benchmark subset integrity."""

import wave
from pathlib import Path
import pytest
from src.dataset import BENCHMARK_SAMPLE_IDS, DatasetLoader
from src.models import SampleData, Turn


class TestDatasetLoader:
    """Test suite for benchmark dataset subset and audio validation."""

    def test_load_all_20_benchmark_samples(self):
        """Verifies exactly 20 curated benchmark samples are loaded."""
        samples = DatasetLoader.load_benchmark_subset()
        assert len(samples) == 20
        loaded_ids = [s.sample_id for s in samples]
        assert loaded_ids == BENCHMARK_SAMPLE_IDS

    def test_schema_validation(self):
        """Verifies that all samples and turns adhere to Pydantic models."""
        samples = DatasetLoader.load_benchmark_subset()
        for sample in samples:
            assert isinstance(sample, SampleData)
            assert isinstance(sample.sample_id, str)
            assert len(sample.sample_id) > 0
            assert sample.duration_seconds > 0
            assert sample.num_speakers in (2, 3)
            assert sample.overlap_ratio > 0.0
            assert len(sample.ground_truth_turns) > 0

            for turn in sample.ground_truth_turns:
                assert isinstance(turn, Turn)
                assert isinstance(turn.speaker, str)
                assert len(turn.speaker) > 0
                assert isinstance(turn.text, str)
                assert len(turn.text) > 0
                if turn.start_time is not None and turn.end_time is not None:
                    assert turn.end_time >= turn.start_time

    def test_audio_files_exist_and_format_valid(self):
        """Verifies each audio file exists and is valid 16kHz 16-bit mono WAV."""
        samples = DatasetLoader.load_benchmark_subset(verify_audio=True)
        for sample in samples:
            audio_path = Path(sample.audio_path)
            assert audio_path.exists(), f"Audio file does not exist: {audio_path}"
            assert audio_path.is_file()
            assert audio_path.suffix.lower() == ".wav"

            with wave.open(str(audio_path), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                duration = nframes / float(framerate)

                assert channels == 1, f"{sample.sample_id}: Expected mono audio, got {channels} channels"
                assert framerate == 16000, f"{sample.sample_id}: Expected 16kHz, got {framerate}Hz"
                assert sampwidth == 2, f"{sample.sample_id}: Expected 16-bit PCM, got {sampwidth * 8}-bit"
                # Duration should match metadata within 0.5s tolerance
                assert duration == pytest.approx(sample.duration_seconds, abs=0.5)

    def test_speaker_and_acoustic_distributions(self):
        """Verifies speaker balance and acoustic environment representation."""
        samples = DatasetLoader.load_benchmark_subset()
        stats = DatasetLoader.get_summary_stats(samples)

        # 15 2-speaker dialogues, 5 3-speaker dialogues
        assert stats["speaker_distribution"][2] == 15
        assert stats["speaker_distribution"][3] == 5

        # All 3 environments present
        assert stats["acoustic_conditions"]["Near field"] == 9
        assert stats["acoustic_conditions"]["Far field"] == 7
        assert stats["acoustic_conditions"]["In the wild"] == 4

        # High overlap stress test
        assert stats["mean_overlap_ratio"] > 5.0
        assert stats["total_duration_seconds"] > 1300.0

    def test_get_single_sample(self):
        """Verifies retrieval of a single sample by ID."""
        sample = DatasetLoader.get_sample("hindi_064")
        assert sample.sample_id == "hindi_064"
        assert sample.num_speakers == 2
        assert sample.dataset_type == "Far field"
        assert len(sample.ground_truth_turns) == 20

    def test_get_nonexistent_sample_raises(self):
        """Verifies requesting an invalid sample ID raises KeyError."""
        with pytest.raises(KeyError):
            DatasetLoader.get_sample("nonexistent_id_999")
