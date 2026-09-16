"""Unit tests for IndicTextNormalizer.

Verifies:
1. Devanagari vowel signs (matras), virama/halant, candrabindu, anusvara, and nukta preservation.
2. Indic punctuation (Danda U+0964, Double Danda U+0965) and Latin punctuation removal.
3. Acoustic filler token removal ([Unintelligible], <laughter>, etc.).
4. Parenthetical English gloss removal ((shopping), (hello)).
5. Zero-width character stripping (ZWJ, ZWNJ, BOM).
6. Number preservation (both Devanagari and Latin digits).
7. Turn-level normalization and empty turn pruning.
"""

import pytest
from src.models import Turn
from src.normalizer import IndicTextNormalizer


class TestIndicTextNormalizer:
    """Test suite for IndicTextNormalizer."""

    def test_matras_and_combining_marks_preserved(self):
        """Crucial test: verifies that Devanagari matras are not stripped."""
        text = "हाँ, मैं बात कर रहा हूँ।"
        result = IndicTextNormalizer.normalize(text)
        expected = "हाँ मैं बात कर रहा हूँ"
        assert result == expected

        # Check individual vowel marks
        vowel_samples = {
            "किताब": "किताब",      # Chhoti I matra (ि)
            "कीमत": "कीमत",        # Badi EE matra (ी)
            "दुकान": "दुकान",      # Chhota U matra (ु)
            "दूर": "दूर",          # Bada OO matra (ू)
            "केला": "केला",        # E matra (े)
            "पैसा": "पैसा",        # AI matra (ै)
            "सोना": "सोना",        # O matra (ो)
            "पौधा": "पौधा",        # AU matra (ौ)
            "कष्ट": "कष्ट",        # Virama / Halant (्)
            "ज़िंदगी": "ज़िंदगी",  # Nukta (़) and Anusvara (ं)
            "अतः": "अतः",          # Visarga (ः)
        }
        for raw, exp in vowel_samples.items():
            assert IndicTextNormalizer.normalize(raw) == exp

    def test_danda_and_double_danda_removed(self):
        """Verifies removal of Devanagari Danda (U+0964) and Double Danda (U+0965)."""
        text = "पहला वाक्य। दूसरा वाक्य॥"
        result = IndicTextNormalizer.normalize(text)
        assert result == "पहला वाक्य दूसरा वाक्य"
        assert "।" not in result
        assert "॥" not in result

    def test_latin_punctuation_removed(self):
        """Verifies stripping of commas, periods, quotes, dashes, question marks."""
        text = '“क्या बात है?” उसने पूछा— "सब ठीक है!"'
        result = IndicTextNormalizer.normalize(text)
        assert result == "क्या बात है उसने पूछा सब ठीक है"

    def test_filler_tags_removed(self):
        """Verifies removal of bracketed and angle-bracketed acoustic tags."""
        text = "अरे यार [Unintelligible] कोई काम नहीं <laughter> सब बेकार है [Noise] <background speech>"
        result = IndicTextNormalizer.normalize(text)
        assert result == "अरे यार कोई काम नहीं सब बेकार है"

    def test_parenthetical_glosses_removed_by_default(self):
        """Verifies removal of English transliteration glosses in parentheses."""
        text = "यार आज ना... हेलो....(hello) पता है मैंने कितनी शॉपिंग (shopping) कर ली है।"
        result = IndicTextNormalizer.normalize(text, remove_glosses=True)
        assert result == "यार आज ना हेलो पता है मैंने कितनी शॉपिंग कर ली है"
        assert "hello" not in result
        assert "shopping" not in result

    def test_parenthetical_glosses_retained_when_flag_false(self):
        """Verifies gloss retention when remove_glosses=False."""
        text = "शॉपिंग (shopping) कर ली"
        result = IndicTextNormalizer.normalize(text, remove_glosses=False)
        assert result == "शॉपिंग shopping कर ली"

    def test_zero_width_characters_removed(self):
        """Verifies stripping of ZWJ, ZWNJ, and BOM."""
        text = "नमस्ते\u200cदुनिया\u200dपरीक्षण\ufeff"
        result = IndicTextNormalizer.normalize(text)
        assert result == "नमस्तेदुनियापरीक्षण"

    def test_numbers_preserved(self):
        """Verifies both Latin (0-9) and Devanagari (०-९) digits are preserved."""
        text = "कमरा नंबर 104 और वार्ड १२३"
        result = IndicTextNormalizer.normalize(text)
        assert result == "कमरा नंबर 104 और वार्ड १२३"

    def test_code_mixed_latin_casing(self):
        """Verifies Latin text is lowercased for case-insensitive WER comparison."""
        text = "Main HDFC Bank se bol raha hoon. EMI Due date kal hai."
        result = IndicTextNormalizer.normalize(text)
        assert result == "main hdfc bank se bol raha hoon emi due date kal hai"

    def test_empty_and_whitespace_inputs(self):
        """Verifies edge cases with empty strings, None, and whitespace."""
        assert IndicTextNormalizer.normalize("") == ""
        assert IndicTextNormalizer.normalize(None) == ""
        assert IndicTextNormalizer.normalize("   \n\t   ") == ""
        assert IndicTextNormalizer.normalize("[Unintelligible]") == ""
        assert IndicTextNormalizer.normalize("() [] <>") == ""

    def test_normalize_turns(self):
        """Verifies Turn-level batch normalization and pruning of empty turns."""
        turns = [
            Turn(speaker="spk_0", text="हाँ, मैं बात कर रहा हूँ।", start_time=0.0, end_time=1.5),
            Turn(speaker="spk_1", text="[Unintelligible]", start_time=1.6, end_time=2.0),
            Turn(speaker="spk_1", text="नमस्ते (hello) sir!", start_time=2.1, end_time=3.5),
        ]
        norm_turns = IndicTextNormalizer.normalize_turns(turns)
        assert len(norm_turns) == 2
        assert norm_turns[0].speaker == "spk_0"
        assert norm_turns[0].text == "हाँ मैं बात कर रहा हूँ"
        assert norm_turns[0].start_time == 0.0
        assert norm_turns[1].speaker == "spk_1"
        assert norm_turns[1].text == "नमस्ते sir"
