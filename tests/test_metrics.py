"""Unit tests for MetricsEngine (WER, CER, Hungarian SAA, cpWER, Diarization Gap)."""

import pytest
from src.metrics import MetricsEngine
from src.models import EvaluationMetrics, Turn


class TestMetricsEngine:
    """Test suite for objective STT and Diarization metrics."""

    def test_wer_exact_match(self):
        """WER between identical strings should be 0.0."""
        ref = "नमस्ते क्या हाल है"
        hyp = "नमस्ते क्या हाल है"
        assert MetricsEngine.compute_wer(ref, hyp) == 0.0

    def test_wer_with_normalization(self):
        """WER should ignore danda, punctuation, and capitalization differences."""
        ref = "हाँ, मैं बात कर रहा हूँ।"
        hyp = "हाँ मैं बात कर रहा हूँ"
        assert MetricsEngine.compute_wer(ref, hyp, normalize=True) == 0.0

    def test_wer_substitutions_and_insertions(self):
        """WER should correctly calculate substitutions, insertions, deletions."""
        ref = "one two three four"
        hyp = "one twenty three four"  # 1 substitution out of 4
        assert MetricsEngine.compute_wer(ref, hyp) == 0.25

        hyp_insert = "one two extra three four"  # 1 insertion
        assert MetricsEngine.compute_wer(ref, hyp_insert) == 0.25

        hyp_delete = "one two three"  # 1 deletion
        assert MetricsEngine.compute_wer(ref, hyp_delete) == 0.25

    def test_wer_empty_cases(self):
        """WER edge cases with empty strings."""
        assert MetricsEngine.compute_wer("", "") == 0.0
        assert MetricsEngine.compute_wer("hello world", "") == 1.0
        assert MetricsEngine.compute_wer("", "hello world") == 1.0

    def test_cer_exact_and_partial(self):
        """CER should be 0.0 on identical strings and reflect character edits."""
        ref = "नमस्ते"
        hyp = "नमस्ते"
        assert MetricsEngine.compute_cer(ref, hyp) == 0.0

        ref = "abc"
        hyp = "abd"
        assert MetricsEngine.compute_cer(ref, hyp) == pytest.approx(1 / 3)

        assert MetricsEngine.compute_cer("", "") == 0.0
        assert MetricsEngine.compute_cer("abc", "") == 1.0

    def test_hungarian_saa_identical_speakers(self):
        """Hungarian SAA with identical speaker labels and transcripts."""
        ref = [
            Turn(speaker="spk_0", text="hello world"),
            Turn(speaker="spk_1", text="good morning"),
            Turn(speaker="spk_0", text="how are you"),
        ]
        hyp = [
            Turn(speaker="spk_0", text="hello world"),
            Turn(speaker="spk_1", text="good morning"),
            Turn(speaker="spk_0", text="how are you"),
        ]
        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == 1.0
        assert mapping == {"spk_0": "spk_0", "spk_1": "spk_1"}

    def test_hungarian_saa_inverted_speaker_labels(self):
        """MANDATORY TEST: Hungarian matching correctly resolves inverted speaker labels.

        Reference: ['spk_0', 'spk_1', 'spk_0']
        Hypothesis: ['B', 'A', 'B']
        Expected mapping: {'B': 'spk_0', 'A': 'spk_1'} with 1.0 accuracy.
        """
        ref = [
            Turn(speaker="spk_0", text="hello world"),
            Turn(speaker="spk_1", text="good morning"),
            Turn(speaker="spk_0", text="how are you"),
        ]
        hyp = [
            Turn(speaker="B", text="hello world"),
            Turn(speaker="A", text="good morning"),
            Turn(speaker="B", text="how are you"),
        ]
        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == 1.0
        assert mapping == {"B": "spk_0", "A": "spk_1"}

    def test_hungarian_saa_three_speakers(self):
        """Hungarian SAA correctly resolves 3-speaker permutations."""
        ref = [
            Turn(speaker="spk_1", text="alpha one"),
            Turn(speaker="spk_2", text="beta two"),
            Turn(speaker="spk_3", text="gamma three"),
        ]
        hyp = [
            Turn(speaker="X", text="alpha one"),
            Turn(speaker="Y", text="beta two"),
            Turn(speaker="Z", text="gamma three"),
        ]
        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == 1.0
        assert mapping == {"X": "spk_1", "Y": "spk_2", "Z": "spk_3"}

    def test_hungarian_saa_partial_attribution_error(self):
        """Verifies accuracy calculation when speaker attribution is partially swapped."""
        ref = [
            Turn(speaker="spk_0", text="word one word two word three"),  # 3 words
            Turn(speaker="spk_1", text="word four word five"),            # 2 words
        ]
        # Hypothesis incorrectly attributes all words to Speaker A
        hyp = [
            Turn(speaker="spk_A", text="word one word two word three word four word five"),
        ]
        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        # Optimal Hungarian matching maps spk_A -> spk_0 (3 correct words out of 5)
        assert acc == pytest.approx(3 / 5)
        assert mapping == {"spk_A": "spk_0"}

    def test_cpwer_exact_match(self):
        """cpWER should be 0.0 when transcripts and speakers match perfectly."""
        ref = [
            Turn(speaker="spk_0", text="first speaker words"),
            Turn(speaker="spk_1", text="second speaker words"),
        ]
        hyp = [
            Turn(speaker="spk_0", text="first speaker words"),
            Turn(speaker="spk_1", text="second speaker words"),
        ]
        cpwer, mapping = MetricsEngine.compute_cpwer(ref, hyp)
        assert cpwer == 0.0
        assert mapping == {"spk_0": "spk_0", "spk_1": "spk_1"}

    def test_cpwer_with_inverted_speaker_labels(self):
        """cpWER should be 0.0 when speakers are permuted but speech is attributed correctly."""
        ref = [
            Turn(speaker="Agent", text="main bank se bol raha hoon"),
            Turn(speaker="Customer", text="haan bolo bhai"),
        ]
        hyp = [
            Turn(speaker="Speaker 2", text="main bank se bol raha hoon"),
            Turn(speaker="Speaker 1", text="haan bolo bhai"),
        ]
        cpwer, mapping = MetricsEngine.compute_cpwer(ref, hyp)
        assert cpwer == 0.0
        assert mapping == {"Speaker 2": "Agent", "Speaker 1": "Customer"}

    def test_cpwer_diarization_degradation_gap(self):
        """Tests that turn confusion produces cpWER > WER, yielding a positive Diarization Gap."""
        ref = [
            Turn(speaker="spk_0", text="alpha beta"),
            Turn(speaker="spk_1", text="gamma delta"),
            Turn(speaker="spk_0", text="epsilon zeta"),
            Turn(speaker="spk_1", text="eta theta"),
        ]
        # Hypothesis transcribes all text correctly, but attributes spk_0's second turn to spk_1:
        hyp = [
            Turn(speaker="spk_0", text="alpha beta"),
            Turn(speaker="spk_1", text="gamma delta"),
            Turn(speaker="spk_1", text="epsilon zeta"),
            Turn(speaker="spk_1", text="eta theta"),
        ]
        eval_result = MetricsEngine.evaluate_sample(ref, hyp)
        assert isinstance(eval_result, EvaluationMetrics)

        # Standard pooled WER is 0.0 because all words were recognized in order
        assert eval_result.wer == 0.0
        # But per-speaker edit distance (cpWER) is high due to speaker misattribution
        assert eval_result.cpwer > 0.0
        assert eval_result.diarization_gap > 0.0
        assert eval_result.speaker_attribution_accuracy < 1.0

    def test_evaluate_sample_empty_hyp(self):
        """Verifies evaluate_sample handles empty hypothesis without crashing."""
        ref = [Turn(speaker="spk_0", text="hello world")]
        hyp: list[Turn] = []
        result = MetricsEngine.evaluate_sample(ref, hyp)
        assert result.wer == 1.0
        assert result.format_valid is False
        assert result.speaker_attribution_accuracy == 0.0
