"""Unit and functional tests for Milestone 3 optimization strategies.

Covers:
1. AnchorPromptPipeline (Strategy 1)
2. TwoStepDecoupledPipeline (Strategy 2)
3. GEPA (Genetic Evolutionary Prompt Optimization) components (Strategy 3)
4. Live API end-to-end execution on benchmark samples
"""

import pytest
from src.client import GeminiClient
from src.config import (
    BENCHMARK_SUBSET_DIR,
    MODEL_3_5_FLASH,
    MODEL_3_5_FLASH_LITE,
)
from src.dataset import DatasetLoader
from src.models import PipelinePrediction, SampleData, Turn
from src.pipelines.anchor_prompting import (
    ANCHOR_SYSTEM_INSTRUCTION,
    AnchorPromptPipeline,
)
from src.pipelines.decoupled_step import (
    DEFAULT_STEP1_SYSTEM_INSTRUCTION,
    DEFAULT_STEP2_SYSTEM_INSTRUCTION,
    TwoStepDecoupledPipeline,
)
from src.pipelines.gepa_optimizer import (
    GEPAEvaluator,
    GEPALoop,
    GEPAMutator,
)


def test_anchor_prompt_pipeline_initialization():
    """Tests AnchorPromptPipeline initialization and default configuration."""
    pipeline = AnchorPromptPipeline()
    assert pipeline.system_instruction is not None
    assert "Conversation Anchor" in pipeline.system_instruction
    assert "Anti-Alternation" in pipeline.system_instruction
    assert "Backchannels" in pipeline.system_instruction
    assert pipeline.thinking_budget == 0
    assert pipeline.temperature == 0.0


def test_anchor_prompt_turn_parsing():
    """Tests turn parsing with AnchorPromptPipeline."""
    raw = (
        "Speaker 0: नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?\n"
        "Speaker 1: हाँ, मैं बोल रहा हूँ।\n"
        "Speaker 0: आपकी बकाया राशि के संबंध में बात करनी थी।"
    )
    turns = AnchorPromptPipeline.parse_turns(raw)
    assert len(turns) == 3
    assert turns[0].speaker == "Speaker 0"
    assert "नमस्कार" in turns[0].text
    assert turns[1].speaker == "Speaker 1"
    assert turns[2].speaker == "Speaker 0"


def test_decoupled_pipeline_initialization():
    """Tests TwoStepDecoupledPipeline initialization."""
    pipeline = TwoStepDecoupledPipeline()
    assert pipeline.step1_model_id == MODEL_3_5_FLASH_LITE
    assert pipeline.step2_model_id == MODEL_3_5_FLASH
    assert "Turn 1:" in pipeline.step1_system_instruction
    assert "Speaker <ID>:" in pipeline.step2_system_instruction


def test_decoupled_unattributed_turn_parsing():
    """Tests parsing of Step 1 un-attributed turns."""
    raw_step1 = (
        "Turn 1: नमस्कार, आपका स्वागत है।\n"
        "Turn 2: जी धन्यवाद।\n"
        "Turn 3: क्या आप अपना नाम बता सकते हैं?\n"
        "यह तीसरी बारी का अतिरिक्त वाक्य है।"
    )
    turns = TwoStepDecoupledPipeline.parse_unattributed_turns(raw_step1)
    assert len(turns) == 3
    assert turns[0] == "नमस्कार, आपका स्वागत है।"
    assert turns[1] == "जी धन्यवाद।"
    assert "क्या आप अपना नाम बता सकते हैं?" in turns[2]
    assert "यह तीसरी बारी का अतिरिक्त वाक्य है।" in turns[2]


def test_decoupled_attributed_turn_parsing():
    """Tests parsing of Step 2 attributed turns."""
    raw_step2 = (
        "Speaker 0: नमस्ते।\n"
        "Speaker 1: जी नमस्ते।\n"
        "Speaker 0: आपकी किस्त जमा हो गई है।"
    )
    turns = TwoStepDecoupledPipeline.parse_turns(raw_step2)
    assert len(turns) == 3
    assert turns[0].speaker == "Speaker 0"
    assert turns[1].speaker == "Speaker 1"
    assert turns[2].speaker == "Speaker 0"


def test_gepa_evaluator_predictions_scoring():
    """Tests GEPAEvaluator scoring computation on mock predictions."""
    sample = SampleData(
        sample_id="test_001",
        audio_path="dummy.wav",
        duration_seconds=30.0,
        num_speakers=2,
        ground_truth_turns=[
            Turn(speaker="Speaker 0", text="नमस्ते"),
            Turn(speaker="Speaker 1", text="हाँ जी नमस्ते"),
        ],
    )
    prediction = PipelinePrediction(
        sample_id="test_001",
        model_id="gemini-3.5-flash-lite",
        approach="anchor_prompting",
        predicted_turns=[
            Turn(speaker="Speaker 0", text="नमस्ते"),
            Turn(speaker="Speaker 1", text="हाँ जी नमस्ते"),
        ],
        latency_seconds=1.5,
    )

    evaluator = GEPAEvaluator()
    summary = evaluator.evaluate_predictions([sample], [prediction])
    assert summary["total_evaluated"] == 1
    assert summary["mean_saa"] == 1.0
    assert summary["mean_wer"] == 0.0
    assert summary["mean_diarization_gap"] == 0.0
    assert summary["fitness"] > 0.9


def test_gepa_mutator_initialization():
    """Tests GEPAMutator configuration."""
    mutator = GEPAMutator()
    assert mutator.mutator_model == MODEL_3_5_FLASH


def test_gepa_loop_initialization():
    """Tests GEPALoop setup with candidate and mutator models."""
    loop = GEPALoop()
    assert loop.candidate_model == MODEL_3_5_FLASH_LITE
    assert loop.evaluator is not None
    assert loop.mutator is not None


def test_live_anchor_prompting_sample():
    """Live functional test executing AnchorPromptPipeline on benchmark sample hindi_064."""
    samples = DatasetLoader.load_benchmark_subset(BENCHMARK_SUBSET_DIR)
    s64 = next(s for s in samples if s.sample_id == "hindi_064")

    pipeline = AnchorPromptPipeline()
    pred = pipeline.run_sample(s64, model_id=MODEL_3_5_FLASH_LITE)

    assert pred.sample_id == "hindi_064"
    assert pred.approach == "anchor_prompting"
    assert len(pred.predicted_turns) >= 2
    assert pred.latency_seconds > 0.0
    assert any("Speaker" in t.speaker for t in pred.predicted_turns)


def test_live_two_step_decoupled_sample():
    """Live functional test executing TwoStepDecoupledPipeline on benchmark sample hindi_064."""
    samples = DatasetLoader.load_benchmark_subset(BENCHMARK_SUBSET_DIR)
    s64 = next(s for s in samples if s.sample_id == "hindi_064")

    pipeline = TwoStepDecoupledPipeline(
        step1_model_id=MODEL_3_5_FLASH_LITE,
        step2_model_id=MODEL_3_5_FLASH_LITE,
    )
    pred = pipeline.run_sample(s64)

    assert pred.sample_id == "hindi_064"
    assert pred.approach == "two_step_decoupled"
    assert len(pred.predicted_turns) >= 2
    assert pred.latency_seconds > 0.0
    assert "STEP 1" in pred.raw_response
    assert "STEP 2" in pred.raw_response
