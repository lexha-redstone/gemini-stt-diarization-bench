"""Adversarial stress tests for Hungarian Bipartite Matching and Speaker Attribution Accuracy (SAA).

Milestone 1 Stress Test Matrix:
1. N-speaker cyclic permutations (N=2, 3, 4, 5, 8)
2. Exhaustive permutations (all 24 for N=4, all 120 for N=5)
3. Rectangular confusion matrices (K_ref > K_hyp: 3->1 equal, 3->1 unequal, 5->2)
4. Rectangular confusion matrices (K_ref < K_hyp: 1->4, 2->5, 2->8)
5. Highly unbalanced word distributions (1,000 vs 1 word, 10,000 vs 1 word, collapse to dominant)
6. Empty, whitespace-only, and punctuation-only turns
7. Completely disjoint vocabulary across speakers
8. Padded K x K cost matrix behavior in cpWER
9. Non-negativity invariant of Diarization Degradation Gap (cpWER - WER >= 0.0)
10. Unicode and special character speaker label robustness
11. Scrambled chronological turn ordering
"""

import itertools
import pytest
from src.metrics import MetricsEngine
from src.models import EvaluationMetrics, Turn


class TestHungarianCyclicPermutations:
    """Stress-test N-speaker cyclic permutations."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8])
    def test_n_speaker_cyclic_shifts(self, n: int):
        """Tests cyclic shift by 1 across N speakers."""
        speakers = [f"spk_{i}" for i in range(n)]
        ref = [Turn(speaker=speakers[i], text=f"turn text for speaker {i}") for i in range(n)]

        # Cyclic shift by 1: spk_i -> spk_{(i+1)%n}
        hyp_speakers = [speakers[(i + 1) % n] for i in range(n)]
        hyp = [Turn(speaker=hyp_speakers[i], text=f"turn text for speaker {i}") for i in range(n)]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

        assert acc == 1.0, f"Failed for N={n}: acc={acc}"
        assert cpwer == 0.0, f"Failed for N={n}: cpwer={cpwer}"
        assert len(mapping) == n
        assert len(cpwer_map) == n

        for i in range(n):
            assert mapping[hyp_speakers[i]] == speakers[i]
            assert cpwer_map[hyp_speakers[i]] == speakers[i]

    def test_all_24_permutations_n4(self):
        """Exhaustively verifies all 4! = 24 permutations for 4 speakers."""
        speakers = ["A", "B", "C", "D"]
        ref = [Turn(speaker=s, text=f"speech block for speaker {s}") for s in speakers]

        count = 0
        for p in itertools.permutations(speakers):
            hyp = [Turn(speaker=p[i], text=f"speech block for speaker {speakers[i]}") for i in range(4)]
            acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
            cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

            assert acc == 1.0
            assert cpwer == 0.0
            for i in range(4):
                assert mapping[p[i]] == speakers[i]
                assert cpwer_map[p[i]] == speakers[i]
            count += 1

        assert count == 24

    def test_all_120_permutations_n5(self):
        """Exhaustively verifies all 5! = 120 permutations for 5 speakers."""
        speakers = ["S1", "S2", "S3", "S4", "S5"]
        ref = [Turn(speaker=s, text=f"speech utterance of speaker {s}") for s in speakers]

        count = 0
        for p in itertools.permutations(speakers):
            hyp = [Turn(speaker=p[i], text=f"speech utterance of speaker {speakers[i]}") for i in range(5)]
            acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
            cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

            assert acc == 1.0
            assert cpwer == 0.0
            for i in range(5):
                assert mapping[p[i]] == speakers[i]
                assert cpwer_map[p[i]] == speakers[i]
            count += 1

        assert count == 120


class TestRectangularConfusionMatrices:
    """Stress-test rectangular cost and confusion matrices (K_ref != K_hyp)."""

    def test_undersegmentation_equal_word_counts(self):
        """3 reference speakers collapsed into 1 hypothesis speaker (equal words)."""
        ref = [
            Turn(speaker="spk_0", text="apple banana cherry date"),  # 4 words
            Turn(speaker="spk_1", text="fig grape honeydew kiwi"),    # 4 words
            Turn(speaker="spk_2", text="lemon mango orange pear"),    # 4 words
        ]
        hyp = [
            Turn(speaker="H0", text="apple banana cherry date fig grape honeydew kiwi lemon mango orange pear")
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

        assert acc == pytest.approx(4 / 12)
        assert len(mapping) == 1
        assert mapping["H0"] in ["spk_0", "spk_1", "spk_2"]
        assert cpwer == pytest.approx(16 / 12)  # 8 insertions on matched + 4+4 deletions on unmatched

    def test_undersegmentation_unequal_word_counts_picks_argmax(self):
        """3 reference speakers (20, 10, 5 words) collapsed to 1 hyp speaker."""
        ref = [
            Turn(speaker="spk_major", text=" ".join([f"w0_{j}" for j in range(20)])),
            Turn(speaker="spk_med", text=" ".join([f"w1_{j}" for j in range(10)])),
            Turn(speaker="spk_minor", text=" ".join([f"w2_{j}" for j in range(5)])),
        ]
        hyp = [
            Turn(
                speaker="H0",
                text=" ".join(
                    [f"w0_{j}" for j in range(20)]
                    + [f"w1_{j}" for j in range(10)]
                    + [f"w2_{j}" for j in range(5)]
                ),
            )
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == pytest.approx(20 / 35)
        assert mapping == {"H0": "spk_major"}

    def test_undersegmentation_5_to_2_speakers(self):
        """5 reference speakers collapsed into 2 hypothesis speakers."""
        ref = [Turn(speaker=f"spk_{i}", text=f"words for speaker {i} repeated") for i in range(5)]
        h0_text = " ".join(ref[i].text for i in [0, 1, 2])
        h1_text = " ".join(ref[i].text for i in [3, 4])
        hyp = [Turn(speaker="H0", text=h0_text), Turn(speaker="H1", text=h1_text)]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

        assert len(mapping) == 2
        assert 0.0 < acc < 1.0
        assert cpwer > 0.0

    def test_oversegmentation_2_to_5_speakers(self):
        """2 reference speakers over-segmented into 5 hypothesis speakers."""
        ref = [
            Turn(speaker="spk_0", text=" ".join([f"alpha_{i}" for i in range(50)])),
            Turn(speaker="spk_1", text=" ".join([f"beta_{i}" for i in range(50)])),
        ]
        hyp = [
            Turn(speaker="H0", text=" ".join([f"alpha_{i}" for i in range(0, 30)])),
            Turn(speaker="H1", text=" ".join([f"alpha_{i}" for i in range(30, 50)])),
            Turn(speaker="H2", text=" ".join([f"beta_{i}" for i in range(0, 25)])),
            Turn(speaker="H3", text=" ".join([f"beta_{i}" for i in range(25, 40)])),
            Turn(speaker="H4", text=" ".join([f"beta_{i}" for i in range(40, 50)])),
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

        # Optimal matching: H0 -> spk_0 (30 words) and H2 -> spk_1 (25 words) = 55 / 100
        assert acc == pytest.approx(55 / 100)
        assert mapping == {"H0": "spk_0", "H2": "spk_1"}
        assert cpwer == pytest.approx(90 / 100)

    def test_oversegmentation_1_to_4_speakers(self):
        """1 reference speaker split across 4 hypothesis speakers."""
        ref = [Turn(speaker="spk_0", text=" ".join([f"word_{i}" for i in range(100)]))]
        hyp = [
            Turn(speaker=f"H{i}", text=" ".join([f"word_{j}" for j in range(i * 25, (i + 1) * 25)]))
            for i in range(4)
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

        assert acc == pytest.approx(25 / 100)
        assert len(mapping) == 1
        assert cpwer == pytest.approx(150 / 100)


class TestHighlyUnbalancedDistributions:
    """Stress-test highly unbalanced speaker word distributions."""

    def test_1000_words_vs_1_word_inverted_labels(self):
        """Speaker 1 has 1,000 words, Speaker 2 has 1 word. Labels are inverted."""
        spk1_text = " ".join([f"dominant_{i}" for i in range(1000)])
        spk2_text = "rare_word"

        ref = [
            Turn(speaker="spk_dom", text=spk1_text),
            Turn(speaker="spk_rare", text=spk2_text),
        ]
        hyp = [
            Turn(speaker="Speaker_B", text=spk1_text),
            Turn(speaker="Speaker_A", text=spk2_text),
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)

        assert acc == 1.0
        assert mapping == {"Speaker_B": "spk_dom", "Speaker_A": "spk_rare"}
        assert cpwer == 0.0

    def test_1000_words_vs_1_word_collapsed(self):
        """Speaker 1 has 1,000 words, Speaker 2 has 1 word. Model collapses both to one speaker."""
        spk1_text = " ".join([f"dominant_{i}" for i in range(1000)])
        spk2_text = "rare_word"

        ref = [
            Turn(speaker="spk_dom", text=spk1_text),
            Turn(speaker="spk_rare", text=spk2_text),
        ]
        hyp = [
            Turn(speaker="Speaker_B", text=spk1_text + " " + spk2_text),
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == pytest.approx(1000 / 1001)
        assert mapping == {"Speaker_B": "spk_dom"}

    def test_10000_words_scale_stability(self):
        """10,000 words vs 1 word to verify lack of numeric overflow or latency issues."""
        dom_text = " ".join(["namaste" for _ in range(10000)])
        ref = [
            Turn(speaker="spk_0", text=dom_text),
            Turn(speaker="spk_1", text="alvida"),
        ]
        hyp = [
            Turn(speaker="H_0", text=dom_text),
            Turn(speaker="H_1", text="alvida"),
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == 1.0
        assert mapping == {"H_0": "spk_0", "H_1": "spk_1"}


class TestDegenerateAndDisjointTurns:
    """Stress-test empty, disjoint, punctuation-only, and degenerate turns."""

    def test_empty_turns(self):
        """Empty turn texts."""
        r = [Turn(speaker="A", text="")]
        h = [Turn(speaker="B", text="")]
        acc, m = MetricsEngine.compute_speaker_attribution_accuracy(r, h)
        cpwer, cm = MetricsEngine.compute_cpwer(r, h)
        res = MetricsEngine.evaluate_sample(r, h)

        assert acc == 1.0
        assert cpwer == 0.0
        assert res.wer == 0.0
        assert res.diarization_gap == 0.0

    def test_whitespace_only_turns(self):
        """Whitespace-only turns."""
        r = [Turn(speaker="A", text="   \t  \n ")]
        h = [Turn(speaker="B", text="   ")]
        acc, m = MetricsEngine.compute_speaker_attribution_accuracy(r, h)
        cpwer, cm = MetricsEngine.compute_cpwer(r, h)

        assert acc == 1.0
        assert cpwer == 0.0

    def test_punctuation_only_turns_normalize_cleanly(self):
        """Indic danda and punctuation only turns."""
        r = [Turn(speaker="A", text="। ॥ , . ! ? -")]
        h = [Turn(speaker="B", text="? ! ॥ ।")]
        acc, m = MetricsEngine.compute_speaker_attribution_accuracy(r, h)
        cpwer, cm = MetricsEngine.compute_cpwer(r, h)

        assert acc == 1.0
        assert cpwer == 0.0

    def test_completely_disjoint_vocabulary(self):
        """Completely disjoint vocabulary between reference and hypothesis."""
        r = [
            Turn(speaker="A", text="alpha beta gamma"),
            Turn(speaker="B", text="delta epsilon zeta"),
        ]
        h = [
            Turn(speaker="X", text="one two three"),
            Turn(speaker="Y", text="four five six"),
        ]

        res = MetricsEngine.evaluate_sample(r, h)
        assert res.wer == 1.0
        assert res.cpwer == 1.0
        assert res.speaker_attribution_accuracy == 1.0  # speaker boundaries match perfectly
        assert res.diarization_gap == 0.0

    def test_unicode_speaker_identifiers(self):
        """Verifies Hungarian matching with Devanagari and special character speaker labels."""
        ref = [
            Turn(speaker="वक्ता_१", text="पहला वाक्य यहाँ है"),
            Turn(speaker="वक्ता_२", text="दूसरा वाक्य यहाँ है"),
        ]
        hyp = [
            Turn(speaker="Speaker #B", text="पहला वाक्य यहाँ है"),
            Turn(speaker="Speaker #A", text="दूसरा वाक्य यहाँ है"),
        ]

        acc, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert acc == 1.0
        assert mapping == {"Speaker #B": "वक्ता_१", "Speaker #A": "वक्ता_२"}


class TestCpwerPaddedMatrixAndDegradationGap:
    """Stress-test cpWER padded cost matrix and Diarization Degradation Gap non-negativity."""

    def test_scrambled_chronology_clips_negative_gap_to_zero(self):
        """When turn order is scrambled, cpWER < WER can occur. Diarization gap must clip to 0.0."""
        ref = [
            Turn(speaker="A", text="cat dog fish"),
            Turn(speaker="B", text="red blue green"),
            Turn(speaker="A", text="bird horse sheep"),
            Turn(speaker="B", text="yellow purple black"),
        ]
        # Hypothesis groups turns by speaker
        hyp = [
            Turn(speaker="A", text="cat dog fish bird horse sheep"),
            Turn(speaker="B", text="red blue green yellow purple black"),
        ]

        res = MetricsEngine.evaluate_sample(ref, hyp)
        assert res.wer > 0.0
        assert res.cpwer == 0.0
        assert res.diarization_gap == 0.0

    def test_randomized_scenarios_gap_always_non_negative(self):
        """Randomized property test verifying diarization_gap >= 0.0 across varied conditions."""
        vocab = ["namaste", "aap", "kaise", "hain", "main", "theek", "hoon", "shukriya"]

        for seed in range(20):
            import random
            rng = random.Random(seed)

            ref_turns = []
            for i in range(rng.randint(2, 6)):
                spk = f"spk_{rng.randint(0, 2)}"
                words = rng.choices(vocab, k=rng.randint(2, 5))
                ref_turns.append(Turn(speaker=spk, text=" ".join(words)))

            hyp_turns = []
            for i in range(rng.randint(1, 6)):
                spk = f"hyp_{rng.randint(0, 3)}"
                words = rng.choices(vocab, k=rng.randint(2, 5))
                hyp_turns.append(Turn(speaker=spk, text=" ".join(words)))

            res = MetricsEngine.evaluate_sample(ref_turns, hyp_turns)
            assert res.diarization_gap >= 0.0, f"Failed at seed {seed}: gap={res.diarization_gap}"
            assert 0.0 <= res.speaker_attribution_accuracy <= 1.0
            assert res.wer >= 0.0
            assert res.cpwer >= 0.0
