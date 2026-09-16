"""Indic-safe text normalizer for Devanagari / Hindi STT and evaluation."""

import re
import unicodedata
from typing import List, Optional
from src.models import Turn


class IndicTextNormalizer:
    """Normalizes Indic (Devanagari) and code-mixed text while preserving vowel signs/matras.

    Key considerations:
    - Standard regex `re.sub(r'[^\\w\\s]', ...)` strips Devanagari vowel signs (`Mc` / `Mn`),
      turning 'हाँ' into 'ह'.
    - This normalizer uses Unicode categories: preserves L* (letters), M* (combining marks /
      matras / halant / nukta / candrabindu / anusvara), N* (digits), and Z* (spaces).
    - Strips P* (punctuation, including Danda U+0964 and Double Danda U+0965) and S* (symbols).
    - Strips zero-width and invisible formatting characters (ZWJ, ZWNJ, BOM).
    - Strips non-speech tags ([Unintelligible], <laughter>, etc.).
    - Optionally strips parenthetical English transliteration glosses e.g. '(shopping)'.
    """

    # Non-speech annotation markers (bracketed and angle-bracketed)
    FILLER_TAGS_PATTERN = re.compile(r"\[.*?\]|<.*?>", re.DOTALL)

    # Parenthetical transliteration glosses e.g. '(shopping)', '(hello)'
    GLOSS_PATTERN = re.compile(r"\(.*?\)", re.DOTALL)

    # Zero-width / invisible formatting characters
    ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")

    # Consecutive whitespace
    WHITESPACE_PATTERN = re.compile(r"\s+")

    @classmethod
    def normalize(cls, text: Optional[str], remove_glosses: bool = True) -> str:
        """Normalizes a single text string.

        Args:
            text: Raw input text string.
            remove_glosses: Whether to remove English parenthetical glosses (e.g., '(shopping)').
                           Defaults to True for standard Indic STT reference evaluation.

        Returns:
            Normalized, whitespace-collapsed string.
        """
        if not text:
            return ""

        # 1. Canonical Unicode Decomposition/Composition (NFC)
        t = unicodedata.normalize("NFC", text)

        # 2. Strip bracketed non-speech acoustic tags
        t = cls.FILLER_TAGS_PATTERN.sub(" ", t)

        # 3. Strip parenthetical glosses if requested
        if remove_glosses:
            t = cls.GLOSS_PATTERN.sub(" ", t)

        # 4. Remove zero-width characters (ZWJ, ZWNJ, etc.)
        t = cls.ZERO_WIDTH_PATTERN.sub("", t)

        # 5. Lowercase Latin characters
        t = t.lower()

        # 6. Unicode category filtering:
        # Keep L* (letters), M* (marks/matras), N* (numbers), Z* (separators)
        # Strip P* (punctuation), S* (symbols), C* (control)
        cleaned_chars: List[str] = []
        for c in t:
            cat = unicodedata.category(c)
            if cat.startswith(("L", "M", "N", "Z")):
                cleaned_chars.append(c)
            else:
                cleaned_chars.append(" ")

        # 7. Collapse whitespace and strip
        return cls.WHITESPACE_PATTERN.sub(" ", "".join(cleaned_chars)).strip()

    @classmethod
    def normalize_turns(cls, turns: List[Turn], remove_glosses: bool = True) -> List[Turn]:
        """Normalizes the text of each turn, filtering out turns that become empty.

        Args:
            turns: List of Turn objects.
            remove_glosses: Whether to remove parenthetical glosses.

        Returns:
            New list of Turn objects with normalized text.
        """
        normalized_turns: List[Turn] = []
        for turn in turns:
            norm_text = cls.normalize(turn.text, remove_glosses=remove_glosses)
            if norm_text:
                normalized_turns.append(
                    Turn(
                        speaker=turn.speaker,
                        text=norm_text,
                        start_time=turn.start_time,
                        end_time=turn.end_time,
                    )
                )
        return normalized_turns
