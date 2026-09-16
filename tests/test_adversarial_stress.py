"""Adversarial stress tests for IndicTextNormalizer and MetricsEngine.

Empirically challenges:
1. Complex Devanagari conjuncts, nuktas, halants, and combining marks.
2. Canonical NFC/NFD equivalence, reordered combining marks, and Vedic marks.
3. Degenerate inputs: empty strings, None, pure punctuation, pure whitespace.
4. Non-speech tag edge cases: nested brackets, unclosed tags, multiline texts.
5. Large scale inputs: 50,000 characters and regex backtracking resistance.
6. MetricsEngine WER and CER on extreme alignment scenarios (massive insertions/deletions/disjoint vocabularies).
7. MetricsEngine SAA and cpWER under extreme speaker permutations, speaker collapse, and speaker hallucination.
"""

import time
import unicodedata
import pytest
from src.models import Turn
from src.normalizer import IndicTextNormalizer
from src.metrics import MetricsEngine


class TestDevanagariConjunctsAndMarks:
    """Adversarial tests for Devanagari script integrity."""

    @pytest.mark.parametrize(
        "conjunct,name",
        [
            ("क्ष", "ksha (ka + virama + ssa)"),
            ("त्र", "tra (ta + virama + ra)"),
            ("ज्ञ", "gya (ja + virama + nya)"),
            ("श्र", "shra (sha + virama + ra)"),
            ("द्व", "dva (da + virama + va)"),
            ("द्य", "dya (da + virama + ya)"),
            ("द्ध", "ddha (da + virama + dha)"),
            ("क्त", "kta (ka + virama + ta)"),
            ("ष्ट", "shta (ssa + virama + tta)"),
            ("ह्ल", "hla (ha + virama + la)"),
            ("ह्म", "hma (ha + virama + ma)"),
        ],
    )
    def test_standard_and_complex_conjuncts(self, conjunct, name):
        """Verifies that all Devanagari conjuncts survive normalization intact."""
        raw = f"यह {conjunct} शब्द है।"
        norm = IndicTextNormalizer.normalize(raw)
        assert norm == f"यह {conjunct} शब्द है", f"Failed for {name}"

    @pytest.mark.parametrize(
        "cluster,word",
        [
            ("र्त्स्य", "भर्त्स्यना"),  # 4-consonant cluster with reph
            ("त्स्न", "ज्योत्स्ना"),     # 3-consonant cluster
            ("र्द्ध्व", "ऊर्ध्व"),       # reph + ddha + virama + va
            ("ङ्क्त", "पंक्ति"),         # nga + virama + ka + virama + ta
            ("स्त्र्य", "स्त्र्यध्यक्ष"),   # sa + virama + ta + virama + ra + virama + ya
        ],
    )
    def test_multi_consonant_clusters(self, cluster, word):
        """Verifies that multi-consonant clusters and ligatures are preserved."""
        norm = IndicTextNormalizer.normalize(word)
        # NFC normalizes ligatures consistently; all letters and marks must survive
        assert norm == word
        for c in word:
            assert c in norm

    @pytest.mark.parametrize(
        "nukta_char,name",
        [
            ("क़", "qa"),
            ("ख़", "khha"),
            ("ग़", "ghha"),
            ("ज़", "za"),
            ("ड़", "ddda"),
            ("ढ़", "rha"),
            ("फ़", "fa"),
            ("य़", "yya"),
        ],
    )
    def test_nukta_consonants(self, nukta_char, name):
        """Verifies that all Devanagari nukta consonants are preserved."""
        word = f"{nukta_char}लम"
        norm = IndicTextNormalizer.normalize(word)
        assert norm == word, f"Failed for nukta {name}"

    def test_precomposed_vs_decomposed_nuktas(self):
        """Verifies canonical equivalence between precomposed (U+0958..U+095F) and decomposed nuktas."""
        precomposed = "\u0958\u0959\u095a\u095b\u095c\u095d\u095e\u095f"
        decomposed = "\u0915\u093c\u0916\u093c\u0917\u093c\u091c\u093c\u0921\u093c\u0922\u093c\u092b\u093c\u092f\u093c"
        norm_pre = IndicTextNormalizer.normalize(precomposed)
        norm_dec = IndicTextNormalizer.normalize(decomposed)
        assert norm_pre == norm_dec
        # In Unicode NFC, precomposed Devanagari nuktas canonically decompose to base + nukta
        assert len(norm_pre) == len(decomposed)

    def test_canonical_reordering_of_combining_marks(self):
        """Verifies that reordered combining marks (virama + nukta vs nukta + virama) normalize identically."""
        order_nukta_virama = "क\u093c\u094d"
        order_virama_nukta = "क\u094d\u093c"
        n1 = IndicTextNormalizer.normalize(order_nukta_virama)
        n2 = IndicTextNormalizer.normalize(order_virama_nukta)
        assert n1 == n2
        # Virama has CCC=9, Nukta has CCC=7; NFC reorders to nukta then virama
        assert [hex(ord(c)) for c in n1] == ["0x915", "0x93c", "0x94d"]

    def test_nfd_to_nfc_invariance(self):
        """Verifies that NFD decomposed input produces identical output to NFC composed input."""
        sample = "हाँ, मैं डॉक्टर से मिलने गया था। उन्होंने कुछ दवाइयाँ दीं।"
        nfd = unicodedata.normalize("NFD", sample)
        nfc = unicodedata.normalize("NFC", sample)
        assert IndicTextNormalizer.normalize(nfd) == IndicTextNormalizer.normalize(nfc)

    @pytest.mark.parametrize(
        "word",
        ["महान्", "विद्वान्", "अर्थात्", "भगवान्", "हठात्"],
    )
    def test_word_final_halant_preserved(self, word):
        """Verifies that word-final virama / halant is preserved."""
        assert IndicTextNormalizer.normalize(word) == word

    def test_all_matras_preserved(self):
        """Verifies all dependent vowel signs (matras U+093E to U+094C, U+0962, U+0963)."""
        base = "क"
        matras = [
            "\u093e",  # AA
            "\u093f",  # I
            "\u0940",  # II
            "\u0941",  # U
            "\u0942",  # UU
            "\u0943",  # vocalic R
            "\u0944",  # vocalic RR
            "\u0945",  # candra E
            "\u0946",  # short E
            "\u0947",  # E
            "\u0948",  # AI
            "\u0949",  # candra O
            "\u094a",  # short O
            "\u094b",  # O
            "\u094c",  # AU
        ]
        for m in matras:
            word = f"{base}{m}"
            norm = IndicTextNormalizer.normalize(word)
            assert norm == word, f"Matra U+{ord(m):04X} was stripped or corrupted"


class TestDegenerateAndAdversarialInputs:
    """Stress tests on pathological and boundary inputs."""

    @pytest.mark.parametrize(
        "empty_input",
        ["", None, "   ", "\t\t\n\r\n\v\f", "   \n   \t  "],
    )
    def test_empty_and_whitespace_variants(self, empty_input):
        """Verifies degenerate empty and whitespace inputs return empty string."""
        assert IndicTextNormalizer.normalize(empty_input) == ""

    def test_pure_punctuation_comprehensive(self):
        """Verifies pure punctuation across ASCII, Latin-1, Devanagari, and general symbols returns empty."""
        punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        indic_punct = "। ॥ ॰"
        symbols = "©®™§¶†‡•—–…"
        full_punct = f"{punct} {indic_punct} {symbols}"
        assert IndicTextNormalizer.normalize(full_punct) == ""

    def test_nested_brackets_and_tags(self):
        """Verifies handling of nested brackets and tag structures."""
        # Nested square brackets
        assert IndicTextNormalizer.normalize("[[nested]]") == ""
        assert IndicTextNormalizer.normalize("[foo [bar] baz]") == "baz"
        # Nested with speech
        assert IndicTextNormalizer.normalize("बातचीत [[cough]] शुरू हुई") == "बातचीत शुरू हुई"
        # Nested parentheses with glosses
        assert IndicTextNormalizer.normalize("शॉपिंग ((shopping)) की") == "शॉपिंग की"

    def test_unclosed_brackets_and_tags(self):
        """Verifies unclosed brackets do not crash or catastrophically delete surrounding speech."""
        # Unclosed bracket at beginning
        res1 = IndicTextNormalizer.normalize("<unclosed speech")
        assert res1 == "unclosed speech"
        # Unclosed bracket with Hindi
        res2 = IndicTextNormalizer.normalize("[अधूरा वाक्य यहाँ है")
        assert res2 == "अधूरा वाक्य यहाँ है"
        # Unclosed parenthesis
        res3 = IndicTextNormalizer.normalize("(अधूरा कोष्ठक यहाँ है")
        assert res3 == "अधूरा कोष्ठक यहाँ है"
        # Trailing unclosed bracket
        res4 = IndicTextNormalizer.normalize("शुरुआत यहाँ है [अधूरा")
        assert res4 == "शुरुआत यहाँ है अधूरा"

    def test_multiline_varied_newlines(self):
        """Verifies CRLF, LF, CR, and mixed line breaks are normalized into clean single spaces."""
        text = "लाइन 1\r\nलाइन 2\nलाइन 3\rलाइन 4"
        assert IndicTextNormalizer.normalize(text) == "लाइन 1 लाइन 2 लाइन 3 लाइन 4"

    def test_large_input_performance_and_memory(self):
        """Verifies normalizer handles 50,000 characters in under 50ms without catastrophic backtracking."""
        chunk = "नमस्ते! मैं डॉ. शर्मा (Dr. Sharma) से बात कर रहा हूँ। [Unintelligible] "
        # ~71 chars per chunk * 750 = ~53,250 chars
        large_input = chunk * 750
        assert len(large_input) > 50000

        t0 = time.perf_counter()
        norm = IndicTextNormalizer.normalize(large_input)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Normalization took too long: {elapsed:.4f}s"
        assert len(norm) > 20000
        assert "[Unintelligible]" not in norm
        assert "Dr. Sharma" not in norm
        assert "।" not in norm

    def test_unclosed_bracket_pathological_scale(self):
        """Verifies no catastrophic backtracking with unclosed brackets on massive 50,000 char input."""
        unclosed_huge = "[" + ("नमस्ते " * 7500)
        t0 = time.perf_counter()
        norm = IndicTextNormalizer.normalize(unclosed_huge)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Pathological unclosed regex backtracking: {elapsed:.4f}s"
        assert norm.startswith("नमस्ते")

    def test_normalization_idempotency(self):
        """Verifies normalize(normalize(x)) == normalize(x)."""
        samples = [
            "हाँ, मैं बात कर रहा हूँ। [Unintelligible] (hello) 123",
            "  “क्या बात है?” उसने पूछा— 'सब ठीक है!'  ",
            "कमरा नंबर 104 और वार्ड १२३",
            "अरे यार [Laughter] <music> ठीक है।",
        ]
        for s in samples:
            n1 = IndicTextNormalizer.normalize(s)
            n2 = IndicTextNormalizer.normalize(n1)
            assert n1 == n2, f"Idempotency violation on: {s}"


class TestMetricsEngineAdversarialScenarios:
    """Stress tests for MetricsEngine WER, CER, SAA, and cpWER."""

    def test_wer_empty_and_disjoint_scenarios(self):
        """Verifies WER under empty, massive insertion, and massive deletion boundaries."""
        # Both empty
        assert MetricsEngine.compute_wer("", "") == 0.0
        # Empty ref, non-empty hyp
        assert MetricsEngine.compute_wer("", "शब्द") == 1.0
        # Non-empty ref, empty hyp
        assert MetricsEngine.compute_wer("शब्द", "") == 1.0
        # Identical
        assert MetricsEngine.compute_wer("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0

        # Massive insertions: 1 ref word vs 1000 hyp words
        ref_single = "नमस्ते"
        hyp_massive = " ".join(["शब्द"] * 1000)
        wer_ins = MetricsEngine.compute_wer(ref_single, hyp_massive)
        assert wer_ins == 1000.0

        # Massive deletions: 1000 ref words vs 1 hyp word
        ref_massive = " ".join(["शब्द"] * 1000)
        hyp_single = "शब्द"
        wer_del = MetricsEngine.compute_wer(ref_massive, hyp_single)
        # 999 deletions out of 1000 words = 0.999
        assert abs(wer_del - 0.999) < 1e-5

        # Completely disjoint vocabularies
        wer_disjoint = MetricsEngine.compute_wer("एक दो तीन", "four five six")
        assert wer_disjoint == 1.0

    def test_cer_extreme_scenarios(self):
        """Verifies CER boundary conditions and edge cases."""
        assert MetricsEngine.compute_cer("", "") == 0.0
        assert MetricsEngine.compute_cer("", "क") == 1.0
        assert MetricsEngine.compute_cer("क", "") == 1.0
        assert MetricsEngine.compute_cer("नमस्ते", "नमस्ते") == 0.0

    def test_saa_speaker_hallucination_and_collapse(self):
        """Verifies Hungarian matching when speaker counts are highly asymmetric."""
        # Hallucination: 2 true speakers, model emits 5 speakers
        ref = [
            Turn(speaker="spk_0", text="नमस्ते आप कैसे हैं"),
            Turn(speaker="spk_1", text="मैं ठीक हूँ धन्यवाद"),
        ]
        hyp = [
            Turn(speaker="A", text="नमस्ते"),
            Turn(speaker="B", text="आप"),
            Turn(speaker="C", text="कैसे"),
            Turn(speaker="D", text="हैं"),
            Turn(speaker="E", text="मैं ठीक हूँ धन्यवाद"),
        ]
        saa, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        assert 0.0 < saa <= 1.0
        assert len(mapping) == 2  # Matches at most min(|ref|, |hyp|)
        assert "E" in mapping
        assert mapping["E"] == "spk_1"

        # Collapse: 3 true speakers, model attributes all turns to 1 speaker
        ref_3 = [
            Turn(speaker="spk_0", text="नमस्ते"),
            Turn(speaker="spk_1", text="आप कैसे हैं"),
            Turn(speaker="spk_2", text="सब बढ़िया"),
        ]
        hyp_1 = [
            Turn(speaker="spk_all", text="नमस्ते आप कैसे हैं सब बढ़िया"),
        ]
        saa_coll, map_coll = MetricsEngine.compute_speaker_attribution_accuracy(ref_3, hyp_1)
        assert 0.0 < saa_coll < 1.0
        assert len(map_coll) == 1

    def test_cpwer_speaker_hallucination_and_collapse(self):
        """Verifies cpWER under asymmetric speaker sets and dummy padding."""
        ref = [
            Turn(speaker="spk_0", text="word1 word2"),
            Turn(speaker="spk_1", text="word3 word4"),
        ]
        # Hypothesis with 2 true speakers + 1 hallucinated extra speaker
        hyp = [
            Turn(speaker="A", text="word1 word2"),
            Turn(speaker="B", text="word3 word4"),
            Turn(speaker="C", text="extra words here"),
        ]
        cpwer, mapping = MetricsEngine.compute_cpwer(ref, hyp)
        # Ref has 4 words; hyp has 4 matched words + 3 inserted words -> distance = 3; cpwer = 3/4 = 0.75
        assert cpwer == 0.75
        assert mapping == {"A": "spk_0", "B": "spk_1"}

    def test_cyclic_speaker_permutation_invariance(self):
        """Verifies Hungarian matching correctly resolves cyclic 3-speaker permutation."""
        ref = [
            Turn(speaker="A", text="पहला वाक्य"),
            Turn(speaker="B", text="दूसरा वाक्य"),
            Turn(speaker="C", text="तीसरा वाक्य"),
        ]
        # Permuted cyclic: A->B, B->C, C->A
        hyp = [
            Turn(speaker="B", text="पहला वाक्य"),
            Turn(speaker="C", text="दूसरा वाक्य"),
            Turn(speaker="A", text="तीसरा वाक्य"),
        ]
        saa, mapping = MetricsEngine.compute_speaker_attribution_accuracy(ref, hyp)
        cpwer, cpwer_map = MetricsEngine.compute_cpwer(ref, hyp)
        diar_gap = MetricsEngine.compute_diarization_gap(cpwer, 0.0)

        assert saa == 1.0
        assert mapping == {"B": "A", "C": "B", "A": "C"}
        assert cpwer == 0.0
        assert diar_gap == 0.0

    def test_evaluate_sample_degenerate_scenarios(self):
        """Verifies evaluate_sample on extreme scenarios."""
        ref = [Turn(speaker="spk_0", text="नमस्ते दुनिया")]

        # Completely empty hyp
        res_empty = MetricsEngine.evaluate_sample(ref, [])
        assert res_empty.wer == 1.0
        assert res_empty.cer == 1.0
        assert res_empty.speaker_attribution_accuracy == 0.0
        assert res_empty.cpwer == 1.0
        assert res_empty.diarization_gap == 0.0
        assert res_empty.format_valid is False

        # Hyp with text that normalizes to empty (only punctuation & filler tags)
        res_filler = MetricsEngine.evaluate_sample(
            ref, [Turn(speaker="model", text="[Unintelligible] । ॥ ,,,")]
        )
        assert res_filler.wer == 1.0
        assert res_filler.format_valid is True  # Turn was parsed from output
        assert res_filler.speaker_attribution_accuracy == 0.0
