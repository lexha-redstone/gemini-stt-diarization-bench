"""Tier 3: Cross-Feature Combinations E2E Tests.

Tests pairwise interactions between:
- Speaker Inversion x WER x Hungarian SAA
- Code-Mixed Devanagari/Roman Normalization x Word Alignment x SAA
- 3-Speaker Multi-Turn Overlap x Hungarian Matching x cpWER
- Over-Segmentation x Punctuation Stripping x Speaker Attribution
- Hallucinated Speaker Tag x Word Insertions x Rectangular Matrix Matching
- Speaker Drop x Deletion WER x Diarization Degradation Gap
"""
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import Turn, EvaluationMetrics
from src.normalizer import IndicTextNormalizer
from src.metrics import MetricsEngine


def test_c1_speaker_inversion_and_wer_degradation():
    """Interaction: Inverted Speaker Labels x Global WER x Hungarian SAA x cpWER.
    
    Verifies that when speaker labels are completely swapped:
    - Global WER remains 0.0 (all words are verbatim accurate).
    - Hungarian SAA correctly recovers 1.0 (100% accuracy) via permutation mapping.
    - cpWER with Hungarian permutation correctly matches speakers and yields 0.0 degradation gap.
    """
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Agent", text="नमस्ते सर मैं एचडीएफसी बैंक से बोल रहा हूँ"),
        Turn(speaker="Customer", text="हाँ बताइए क्या बात है"),
        Turn(speaker="Agent", text="आपकी ईएमआई पिछले महीने से बाकी है"),
        Turn(speaker="Customer", text="मेरी सैलरी लेट हो गई थी मैं कल जमा कर दूंगा")
    ]
    # Hypothesis: Exact same words, but labels inverted (Customer <-> Agent)
    hyp_turns = [
        Turn(speaker="Customer", text="नमस्ते सर मैं एचडीएफसी बैंक से बोल रहा हूँ"),
        Turn(speaker="Agent", text="हाँ बताइए क्या बात है"),
        Turn(speaker="Customer", text="आपकी ईएमआई पिछले महीने से बाकी है"),
        Turn(speaker="Agent", text="मेरी सैलरी लेट हो गई थी मैं कल जमा कर दूंगा")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    
    assert res.wer == 0.0, f"Expected 0.0 WER for verbatim text, got {res.wer}"
    assert res.speaker_attribution_accuracy == 1.0, f"Hungarian matching should recover swapped labels: {res.speaker_attribution_accuracy}"
    assert res.speaker_mapping == {"Customer": "Agent", "Agent": "Customer"}
    assert res.cpwer == 0.0, f"Expected 0.0 cpWER under optimal permutation, got {res.cpwer}"
    assert abs(res.diarization_gap) < 1e-4, f"Diarization gap should be 0.0, got {res.diarization_gap}"


def test_c2_codemixed_indic_roman_and_hungarian_saa():
    """Interaction: Code-Mixed Transliteration Normalization x Word Alignment x Hungarian SAA.
    
    Verifies that dual-script glosses (e.g. 'शॉपिंग (shopping)') are stripped cleanly
    and word-level Levenshtein alignment stays perfectly synchronized to attribute words to speakers.
    """
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Speaker 1", text="मैंने बहुत शॉपिंग (shopping) कर ली है।"),
        Turn(speaker="Speaker 2", text="यह तो बढ़िया बात है! शॉपिंग (shopping) तो थेरेपी (therapy) है।")
    ]
    # Hypothesis contains only the Devanagari representation without glosses
    hyp_turns = [
        Turn(speaker="Speaker 1", text="मैंने बहुत शॉपिंग कर ली है"),
        Turn(speaker="Speaker 2", text="यह तो बढ़िया बात है शॉपिंग तो थेरेपी है")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0, f"Expected 0.0 WER after normalizer cleans glosses and punctuation, got {res.wer}"
    assert res.speaker_attribution_accuracy == 1.0, f"Expected 1.0 SAA for code-mixed alignment, got {res.speaker_attribution_accuracy}"


def test_c3_three_speaker_dialogue_with_partial_overlap():
    """Interaction: 3-Speaker Multi-Turn Dialogue x Hungarian Permutation x cpWER.
    
    Verifies 3x3 Hungarian cost matrix computation on multi-turn dialogue with overlapping turns.
    """
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Customer", text="मेरा अकाउंट ब्लॉक हो गया है", start_time=0.0, end_time=2.5),
        Turn(speaker="Agent", text="मैं चेक करता हूँ सर लाइन पर रहिए", start_time=2.0, end_time=4.5),
        Turn(speaker="Supervisor", text="नमस्ते सर मैं सीनियर मैनेजर बात कर रहा हूँ", start_time=4.0, end_time=7.0),
        Turn(speaker="Customer", text="जल्दी अनब्लॉक कर दीजिए", start_time=6.5, end_time=8.5)
    ]
    # Hypothesis uses generic speaker labels SPK_A, SPK_B, SPK_C
    hyp_turns = [
        Turn(speaker="SPK_A", text="मेरा अकाउंट ब्लॉक हो गया है"),
        Turn(speaker="SPK_B", text="मैं चेक करता हूँ सर लाइन पर रहिए"),
        Turn(speaker="SPK_C", text="नमस्ते सर मैं सीनियर मैनेजर बात कर रहा हूँ"),
        Turn(speaker="SPK_A", text="जल्दी अनब्लॉक कर दीजिए")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0
    assert res.speaker_mapping.get("SPK_A") == "Customer"
    assert res.speaker_mapping.get("SPK_B") == "Agent"
    assert res.speaker_mapping.get("SPK_C") == "Supervisor"
    assert res.cpwer == 0.0


def test_c4_oversegmentation_with_indic_punctuation():
    """Interaction: Turn Over-segmentation x Devanagari Danda Stripping x Word SAA.
    
    Verifies that when reference contains sentences ended by danda ('।') and hypothesis splits
    turns mid-sentence, word-level alignment correctly attributes every word to the speaker.
    """
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Agent", text="आपका लोन अप्रूव हो गया है। आपको आज ही डॉक्यूमेंट्स जमा करने होंगे।")
    ]
    hyp_turns = [
        Turn(speaker="Agent", text="आपका लोन अप्रूव हो गया है"),
        Turn(speaker="Agent", text="आपको आज ही डॉक्यूमेंट्स जमा करने होंगे")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    assert res.wer == 0.0
    assert res.speaker_attribution_accuracy == 1.0


def test_c5_hallucinated_speaker_with_word_insertions():
    """Interaction: Rectangular Cost Matrix (2 Ref x 3 Hyp) x Word Insertion WER x SAA.
    
    Verifies Hungarian matching behavior when hypothesis introduces an ungrounded third speaker
    and spurious inserted words.
    """
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Agent", text="नमस्ते सर क्या आप सुन रहे हैं"),
        Turn(speaker="Customer", text="हाँ मैं सुन रहा हूँ")
    ]
    hyp_turns = [
        Turn(speaker="Agent", text="नमस्ते सर क्या आप सुन रहे हैं"),
        Turn(speaker="Customer", text="हाँ मैं सुन"),
        Turn(speaker="Ghost_Speaker", text="अनावश्यक शब्द रहा हूँ")  # Spurious speaker with inserted words
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    # WER should be positive due to inserted words
    assert res.wer > 0.0
    # SAA should be less than 1.0 due to words misattributed to Ghost_Speaker
    assert res.speaker_attribution_accuracy < 1.0
    # Diarization gap should be positive
    assert res.diarization_gap >= 0.0


def test_c6_speaker_drop_with_transcript_deletion():
    """Interaction: Incomplete Diarization (2 Ref x 1 Hyp) x Deletion WER x Diarization Gap.
    
    Verifies that when the model completely fails to identify Speaker 2 (dropping all turns):
    - WER captures deleted words.
    - SAA only gives credit for Speaker 1 words.
    - cpWER is heavily degraded.
    """
    engine = MetricsEngine()
    ref_turns = [
        Turn(speaker="Agent", text="नमस्ते सर एचडीएफसी बैंक से कॉल है"),  # 7 words
        Turn(speaker="Customer", text="मुझे आपकी कोई कॉल नहीं चाहिए")     # 6 words
    ]
    # Hypothesis drops the Customer turns entirely
    hyp_turns = [
        Turn(speaker="Agent", text="नमस्ते सर एचडीएफसी बैंक से कॉल है")
    ]
    res = engine.evaluate_sample(ref_turns, hyp_turns)
    # 6 words deleted out of 13 total words = ~0.46 WER
    assert res.wer > 0.4
    # SAA on aligned words: Agent words are 100% matched
    assert res.speaker_attribution_accuracy == 1.0
    # But cpWER reflects the missing customer document
    assert res.cpwer > 0.4
