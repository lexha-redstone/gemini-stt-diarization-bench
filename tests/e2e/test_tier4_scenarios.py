"""Tier 4: Real-World Workload Scenarios E2E Tests (>=5 realistic scenarios).

Scenarios:
1. Scenario 1: Realistic 2-Speaker Debt Collection Call Pipeline
2. Scenario 2: High-Overlap Speech Collision Dialogue (15%+ Overlap Stress Test)
3. Scenario 3: 3-Party Customer Escalation Call with Supervisor Intervention
4. Scenario 4: Noisy Telephonic Dialogue with Code-Mixing, Loan Terms & Non-Speech Tokens
5. Scenario 5: Full 20-Sample Benchmark Subset Batch Validation & Self-Consistency
"""
import os
import sys
import json
import wave
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import Turn, SampleData, EvaluationMetrics
from src.normalizer import IndicTextNormalizer
from src.metrics import MetricsEngine
from src.dataset import DatasetLoader

load_benchmark_subset = DatasetLoader.load_benchmark_subset


def test_s1_debt_collection_two_speaker_dialogue_pipeline():
    """Scenario 1: Realistic 2-Speaker Debt Collection Call Pipeline.
    
    Models an end-to-end collections conversation:
    1. Agent Greeting & Verification: 'नमस्कार, क्या मेरी बात राजेश शर्मा जी से हो रही है?'
    2. Customer Confirmation: 'हाँ, मैं राजेश बोल रहा हूँ।'
    3. Agent Due Notice: 'सर, आपके एचडीएफसी पर्सनल लोन की ₹12,500 की ईएमआई 5 तारीख से ड्यू है।'
    4. Customer Financial Excuse: 'मेरी कंपनी में सैलरी अभी तक क्रेडिट नहीं हुई है।'
    5. Agent Urgent Demand: 'सर, आज शाम 5 बजे से पहले पेमेंट करना अनिवार्य है, वरना पेनल्टी लगेगी।'
    6. Customer Promise to Pay: 'ठीक है, मैं शाम 4 बजे तक लिंक से पेमेंट कर दूंगा।'
    7. Agent Closing: 'धन्यवाद सर, हमने पेमेंट लिंक एसएमएस पर भेज दिया है।'
    """
    engine = MetricsEngine()
    
    ground_truth = [
        Turn(speaker="Agent", text="नमस्कार, क्या मेरी बात राजेश शर्मा जी से हो रही है?", start_time=0.0, end_time=3.2),
        Turn(speaker="Customer", text="हाँ, मैं राजेश बोल रहा हूँ।", start_time=3.5, end_time=5.8),
        Turn(speaker="Agent", text="सर, आपके एचडीएफसी पर्सनल लोन की ₹12500 की ईएमआई 5 तारीख से ड्यू है।", start_time=6.0, end_time=11.4),
        Turn(speaker="Customer", text="मेरी कंपनी में सैलरी अभी तक क्रेडिट नहीं हुई है।", start_time=11.6, end_time=15.2),
        Turn(speaker="Agent", text="सर, आज शाम 5 बजे से पहले पेमेंट करना अनिवार्य है, वरना पेनल्टी लगेगी।", start_time=15.5, end_time=20.1),
        Turn(speaker="Customer", text="ठीक है, मैं शाम 4 बजे तक लिंक से पेमेंट कर दूंगा।", start_time=20.5, end_time=24.3),
        Turn(speaker="Agent", text="धन्यवाद सर, हमने पेमेंट लिंक एसएमएस पर भेज दिया है।", start_time=24.5, end_time=28.0)
    ]
    
    # Model output with normalized punctuation and Hungarian mapped speakers ('Speaker 1', 'Speaker 2')
    predicted = [
        Turn(speaker="Speaker 1", text="नमस्कार क्या मेरी बात राजेश शर्मा जी से हो रही है"),
        Turn(speaker="Speaker 2", text="हाँ मैं राजेश बोल रहा हूँ"),
        Turn(speaker="Speaker 1", text="सर आपके एचडीएफसी पर्सनल लोन की 12500 की ईएमआई 5 तारीख से ड्यू है"),
        Turn(speaker="Speaker 2", text="मेरी कंपनी में सैलरी अभी तक क्रेडिट नहीं हुई है"),
        Turn(speaker="Speaker 1", text="सर आज शाम 5 बजे से पहले पेमेंट करना अनिवार्य है वरना पेनल्टी लगेगी"),
        Turn(speaker="Speaker 2", text="ठीक है मैं शाम 4 बजे तक लिंक से पेमेंट कर दूंगा"),
        Turn(speaker="Speaker 1", text="धन्यवाद सर हमने पेमेंट लिंक एसएमएस पर भेज दिया है")
    ]
    
    eval_result = engine.evaluate_sample(ground_truth, predicted)
    
    assert eval_result.wer == 0.0, f"Expected 0.0 WER for normalized clean transcript, got {eval_result.wer}"
    assert eval_result.speaker_attribution_accuracy == 1.0, f"Expected 1.0 SAA, got {eval_result.speaker_attribution_accuracy}"
    assert eval_result.speaker_mapping == {"Speaker 1": "Agent", "Speaker 2": "Customer"}
    assert eval_result.cpwer == 0.0
    assert abs(eval_result.diarization_gap) < 1e-4
    assert eval_result.format_valid is True


def test_s2_high_overlap_dispute_speech_collision():
    """Scenario 2: High-Overlap Speech Collision Dialogue (15%+ Overlap Stress Test).
    
    Simulates shouting and rapid overlapping speech where borrower and agent talk simultaneously.
    Demonstrates that:
    - Verbatim transcription with mixed speaker labels yields low WER but HIGH Diarization Degradation Gap.
    """
    engine = MetricsEngine()
    
    # Overlapping intervals: both speaking between t=2.0 and t=4.5
    ground_truth = [
        Turn(speaker="Agent", text="आप बार-बार वादा करके पेमेंट क्यों टाल रहे हैं", start_time=0.0, end_time=4.5),
        Turn(speaker="Customer", text="आप मुझे परेशान करना बंद कीजिए मैं कोर्ट में शिकायत करूंगा", start_time=2.0, end_time=6.0),
        Turn(speaker="Agent", text="बैंक का पैसा वापस करना ही पड़ेगा", start_time=5.0, end_time=8.0)
    ]
    
    # Flawed model output during overlap: assigns customer's words to Agent
    flawed_prediction = [
        Turn(speaker="Agent", text="आप बार-बार वादा करके पेमेंट क्यों टाल रहे हैं आप मुझे परेशान करना बंद कीजिए"),  # Customer words merged into Agent
        Turn(speaker="Customer", text="मैं कोर्ट में शिकायत करूंगा"),
        Turn(speaker="Agent", text="बैंक का पैसा वापस करना ही पड़ेगा")
    ]
    
    eval_result = engine.evaluate_sample(ground_truth, flawed_prediction)
    
    # Words are verbatim, so basic WER is 0.0
    assert eval_result.wer == 0.0
    # But attribution accuracy is degraded due to merged overlap words
    assert eval_result.speaker_attribution_accuracy < 1.0
    # cpWER is significantly greater than 0.0
    assert eval_result.cpwer > 0.0
    # Diarization Gap detects the collision breakdown
    assert eval_result.diarization_gap > 0.0


def test_s3_three_party_supervisor_escalation():
    """Scenario 3: 3-Party Customer Escalation Call with Supervisor Intervention.
    
    Simulates multi-party call where a supervisor intervenes to de-escalate:
    - Customer disputes a charge.
    - Agent transfers to Supervisor.
    - Supervisor offers a settlement waiver.
    """
    engine = MetricsEngine()
    
    ground_truth = [
        Turn(speaker="Customer", text="मुझे यह लेट फीस बिल्कुल बर्दाश्त नहीं है"),
        Turn(speaker="Agent", text="सर मैं अपनी सीनियर टीम लीडर को कॉल कनेक्ट कर रही हूँ"),
        Turn(speaker="Supervisor", text="नमस्ते सर मैं मैनेजर हूँ मैं आपकी लेट फीस वेव कर देती हूँ"),
        Turn(speaker="Customer", text="ठीक है फिर मैं मूल राशि तुरंत ट्रांसफर कर दूंगा"),
        Turn(speaker="Supervisor", text="धन्यवाद सर मैंने सिस्टम में अपडेट कर दिया है")
    ]
    
    # Prediction has generic cluster labels (A, B, C)
    prediction = [
        Turn(speaker="C1", text="मुझे यह लेट फीस बिल्कुल बर्दाश्त नहीं है"),
        Turn(speaker="C2", text="सर मैं अपनी सीनियर टीम लीडर को कॉल कनेक्ट कर रही हूँ"),
        Turn(speaker="C3", text="नमस्ते सर मैं मैनेजर हूँ मैं आपकी लेट फीस वेव कर देती हूँ"),
        Turn(speaker="C1", text="ठीक है फिर मैं मूल राशि तुरंत ट्रांसफर कर दूंगा"),
        Turn(speaker="C3", text="धन्यवाद सर मैंने सिस्टम में अपडेट कर दिया है")
    ]
    
    eval_result = engine.evaluate_sample(ground_truth, prediction)
    assert eval_result.wer == 0.0
    assert eval_result.speaker_attribution_accuracy == 1.0
    assert len(eval_result.speaker_mapping) == 3
    assert eval_result.speaker_mapping.get("C1") == "Customer"
    assert eval_result.speaker_mapping.get("C2") == "Agent"
    assert eval_result.speaker_mapping.get("C3") == "Supervisor"
    assert eval_result.cpwer == 0.0


def test_s4_noisy_telephonic_codemixed_dialogue():
    """Scenario 4: Noisy Telephonic Dialogue with Code-Mixing, Loan Terms & Non-Speech Tokens.
    
    Validates the full normalizer and evaluation pipeline under real telephonic conditions:
    - English loanwords with Devanagari transliteration: 'क्रेडिट कार्ड (credit card)'
    - Non-speech noise tags: '[Coughing]', '<background speech>', '[Unintelligible]'
    - Danda punctuation: '।'
    - Arabic and Devanagari numerals: '५०००' and '5000'
    """
    engine = MetricsEngine()
    norm = IndicTextNormalizer()
    
    raw_reference = (
        "नमस्ते सर [Unintelligible] आपके क्रेडिट कार्ड (credit card) का बिल ₹५००० पेंडिंग है। "
        "<coughing> क्या आप आज ऑनलाइन नेट बैंकिंग (net banking) से ५००० रुपये ट्रांसफर करेंगे?"
    )
    
    # Normalized reference
    clean_ref = norm.normalize(raw_reference)
    assert "[Unintelligible]" not in clean_ref
    assert "<coughing>" not in clean_ref
    assert "credit card" not in clean_ref  # gloss removed
    assert "net banking" not in clean_ref  # gloss removed
    assert "।" not in clean_ref
    assert "क्रेडिट कार्ड" in clean_ref
    assert "नेट बैंकिंग" in clean_ref
    
    ref_turns = [Turn(speaker="Agent", text=raw_reference)]
    hyp_turns = [Turn(speaker="Agent", text=clean_ref)]
    
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_s5_full_benchmark_subset_metadata_and_reference_self_eval():
    """Scenario 5: Full 20-Sample Benchmark Subset Batch Validation & Self-Consistency.
    
    Iterates over all 20 benchmark samples in data/benchmark_subset/metadata.json:
    1. Validates each sample's WAV audio exists and is readable via wave module.
    2. Validates duration is within bounds [50s, 250s].
    3. Validates each sample's turns against Pydantic SampleData schema.
    4. Evaluates reference against itself: verifies WER=0.0, SAA=1.0, cpWER=0.0 across all 20 samples.
    """
    engine = MetricsEngine()
    samples = load_benchmark_subset()
    assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"
    
    for idx, sample in enumerate(samples):
        # 1. Check audio file
        audio_full_path = os.path.join(PROJECT_ROOT, sample.audio_path)
        assert os.path.exists(audio_full_path), f"Audio path {audio_full_path} does not exist"
        
        with wave.open(audio_full_path, "rb") as wf:
            assert wf.getnchannels() == 1, f"Sample {sample.sample_id} is not mono"
            assert wf.getframerate() == 16000, f"Sample {sample.sample_id} is not 16kHz"
            assert wf.getsampwidth() == 2, f"Sample {sample.sample_id} is not 16-bit"
            
        # 2. Check duration
        assert 50.0 <= sample.duration_seconds <= 250.0, f"Sample {sample.sample_id} duration out of bounds: {sample.duration_seconds}"
        
        # 3. Check ground truth turns exist
        assert len(sample.ground_truth_turns) >= 2, f"Sample {sample.sample_id} has fewer than 2 turns"
        
        # 4. Self-evaluation sanity check
        res = engine.evaluate_sample(sample.ground_truth_turns, sample.ground_truth_turns)
        assert res.wer == 0.0, f"Self-eval WER non-zero on {sample.sample_id}: {res.wer}"
        assert res.speaker_attribution_accuracy == 1.0, f"Self-eval SAA not 1.0 on {sample.sample_id}: {res.speaker_attribution_accuracy}"
        assert res.cpwer == 0.0, f"Self-eval cpWER non-zero on {sample.sample_id}: {res.cpwer}"
        assert abs(res.diarization_gap) < 1e-4
