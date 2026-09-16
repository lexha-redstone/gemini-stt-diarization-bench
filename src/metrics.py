"""Metrics Engine for STT and Speaker Diarization evaluation.

Implements:
- Word Error Rate (WER) via jiwer 4.0.0
- Character Error Rate (CER) via jiwer 4.0.0
- Word-Level Speaker Attribution Accuracy (SAA) via Levenshtein word alignment
  and Hungarian permutation matching (scipy.optimize.linear_sum_assignment)
- Concatenated Permutation Word Error Rate (cpWER) following CHiME / LibriCSS standards
- Diarization Degradation Gap (cpWER - WER)
- Full sample evaluation returning EvaluationMetrics Pydantic model
"""

from typing import Dict, List, Optional, Tuple
import jiwer
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models import EvaluationMetrics, Turn
from src.normalizer import IndicTextNormalizer


class MetricsEngine:
    """Calculates objective STT and Speaker Diarization metrics."""

    @staticmethod
    def _word_edit_distance(ref: str, hyp: str) -> int:
        """Calculates Levenshtein word edit distance between two strings."""
        ref_words = ref.split() if ref else []
        hyp_words = hyp.split() if hyp else []

        if not ref_words and not hyp_words:
            return 0
        if not ref_words:
            return len(hyp_words)
        if not hyp_words:
            return len(ref_words)

        out = jiwer.process_words(ref, hyp)
        return int(out.substitutions + out.deletions + out.insertions)

    @classmethod
    def compute_wer(cls, ref: str, hyp: str, normalize: bool = True) -> float:
        """Computes Word Error Rate (WER) using jiwer.

        Args:
            ref: Reference transcript string.
            hyp: Hypothesis transcript string.
            normalize: Whether to apply IndicTextNormalizer first.

        Returns:
            WER as a float (0.0 to 1.0+).
        """
        if normalize:
            ref = IndicTextNormalizer.normalize(ref)
            hyp = IndicTextNormalizer.normalize(hyp)

        ref_clean = ref.strip()
        hyp_clean = hyp.strip()

        if not ref_clean and not hyp_clean:
            return 0.0
        if not ref_clean and hyp_clean:
            return 1.0
        if ref_clean and not hyp_clean:
            return 1.0
        if ref_clean == hyp_clean:
            return 0.0

        return float(jiwer.wer(ref_clean, hyp_clean))

    @classmethod
    def compute_cer(cls, ref: str, hyp: str, normalize: bool = True) -> float:
        """Computes Character Error Rate (CER) using jiwer.

        Args:
            ref: Reference transcript string.
            hyp: Hypothesis transcript string.
            normalize: Whether to apply IndicTextNormalizer first.

        Returns:
            CER as a float (0.0 to 1.0+).
        """
        if normalize:
            ref = IndicTextNormalizer.normalize(ref)
            hyp = IndicTextNormalizer.normalize(hyp)

        ref_clean = ref.strip()
        hyp_clean = hyp.strip()

        if not ref_clean and not hyp_clean:
            return 0.0
        if not ref_clean and hyp_clean:
            return 1.0
        if ref_clean and not hyp_clean:
            return 1.0
        if ref_clean == hyp_clean:
            return 0.0

        return float(jiwer.cer(ref_clean, hyp_clean))

    @classmethod
    def compute_speaker_attribution_accuracy(
        cls,
        ref_turns: List[Turn],
        hyp_turns: List[Turn],
        normalize: bool = True,
    ) -> Tuple[float, Dict[str, str]]:
        """Computes Word-Level Speaker Attribution Accuracy (SAA) via Hungarian matching.

        Algorithm:
        1. Extract word sequence and speaker tags for both reference and hypothesis.
        2. Align reference and hypothesis words using Levenshtein alignment (jiwer.process_words).
        3. Build speaker confusion matrix C where C[i, j] is the count of words spoken
           by reference speaker i and predicted as hypothesis speaker j.
        4. Solve maximum weight bipartite matching using scipy.optimize.linear_sum_assignment(-C).
        5. Return (Attribution Accuracy, mapping from hypothesis speaker to reference speaker).

        Args:
            ref_turns: List of ground-truth Turn objects.
            hyp_turns: List of predicted Turn objects.
            normalize: Whether to normalize turn text.

        Returns:
            Tuple of (accuracy float between 0.0 and 1.0, mapping dict: {hyp_speaker: ref_speaker}).
        """
        if normalize:
            ref_turns = IndicTextNormalizer.normalize_turns(ref_turns)
            hyp_turns = IndicTextNormalizer.normalize_turns(hyp_turns)

        ref_words: List[str] = []
        ref_speakers: List[str] = []
        for turn in ref_turns:
            words = turn.text.split()
            for w in words:
                ref_words.append(w)
                ref_speakers.append(turn.speaker)

        hyp_words: List[str] = []
        hyp_speakers: List[str] = []
        for turn in hyp_turns:
            words = turn.text.split()
            for w in words:
                hyp_words.append(w)
                hyp_speakers.append(turn.speaker)

        # Edge cases
        if not ref_words and not hyp_words:
            return 1.0, {}
        if not ref_words or not hyp_words:
            return 0.0, {}

        unique_ref = sorted(list(set(ref_speakers)))
        unique_hyp = sorted(list(set(hyp_speakers)))

        # Word-level alignment
        ref_str = " ".join(ref_words)
        hyp_str = " ".join(hyp_words)
        out = jiwer.process_words(ref_str, hyp_str)

        # Construct speaker confusion matrix C: shape (|unique_ref|, |unique_hyp|)
        conf = np.zeros((len(unique_ref), len(unique_hyp)), dtype=np.int64)

        for chunk in out.alignments[0]:
            if chunk.type in ("equal", "substitute"):
                r_indices = range(chunk.ref_start_idx, chunk.ref_end_idx)
                h_indices = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
                for r_idx, h_idx in zip(r_indices, h_indices):
                    r_spk = ref_speakers[r_idx]
                    h_spk = hyp_speakers[h_idx]
                    conf[unique_ref.index(r_spk), unique_hyp.index(h_spk)] += 1

        total_aligned_words = int(conf.sum())
        if total_aligned_words == 0:
            return 0.0, {}

        # Hungarian maximum-weight matching on -conf
        r_ind, c_ind = linear_sum_assignment(-conf)

        speaker_mapping = {
            unique_hyp[c]: unique_ref[r]
            for r, c in zip(r_ind, c_ind)
        }

        correct_attributed_words = int(sum(conf[r, c] for r, c in zip(r_ind, c_ind)))
        accuracy = float(correct_attributed_words / total_aligned_words)

        return accuracy, speaker_mapping

    @classmethod
    def compute_cpwer(
        cls,
        ref_turns: List[Turn],
        hyp_turns: List[Turn],
        normalize: bool = True,
    ) -> Tuple[float, Dict[str, str]]:
        """Computes Concatenated Permutation Word Error Rate (cpWER).

        Algorithm (CHiME / LibriCSS standard):
        1. Group and concatenate utterances per speaker into single documents D^R(s) and D^H(s).
        2. Construct a square cost matrix E of size K x K (K = max(|S^R|, |S^H|)),
           where E[i, j] is the Levenshtein word edit distance between D^R(s_i) and D^H(s_j).
           Empty speakers are padded with insertion/deletion cost.
        3. Solve minimum weight bipartite matching using scipy.optimize.linear_sum_assignment(E).
        4. Total distance = sum of matched costs; Total reference words = sum of reference words.
        5. cpWER = total_distance / total_reference_words.

        Args:
            ref_turns: Ground-truth turns.
            hyp_turns: Predicted turns.
            normalize: Whether to normalize turn text.

        Returns:
            Tuple of (cpWER float, mapping dict: {hyp_speaker: ref_speaker}).
        """
        if normalize:
            ref_turns = IndicTextNormalizer.normalize_turns(ref_turns)
            hyp_turns = IndicTextNormalizer.normalize_turns(hyp_turns)

        # Concatenate utterances per speaker
        ref_docs: Dict[str, List[str]] = {}
        for turn in ref_turns:
            ref_docs.setdefault(turn.speaker, []).append(turn.text)

        hyp_docs: Dict[str, List[str]] = {}
        for turn in hyp_turns:
            hyp_docs.setdefault(turn.speaker, []).append(turn.text)

        ref_speakers = sorted(list(ref_docs.keys()))
        hyp_speakers = sorted(list(hyp_docs.keys()))

        ref_doc_strings = {spk: " ".join(ref_docs[spk]) for spk in ref_speakers}
        hyp_doc_strings = {spk: " ".join(hyp_docs[spk]) for spk in hyp_speakers}

        total_ref_words = sum(len(doc.split()) for doc in ref_doc_strings.values())
        total_hyp_words = sum(len(doc.split()) for doc in hyp_doc_strings.values())

        if total_ref_words == 0 and total_hyp_words == 0:
            return 0.0, {}
        if total_ref_words == 0:
            return 1.0, {}
        if total_hyp_words == 0:
            return 1.0, {}

        m = len(ref_speakers)
        n = len(hyp_speakers)
        k = max(m, n)

        # Build padded square cost matrix
        cost = np.zeros((k, k), dtype=np.int64)
        for i in range(k):
            for j in range(k):
                r_text = ref_doc_strings[ref_speakers[i]] if i < m else ""
                h_text = hyp_doc_strings[hyp_speakers[j]] if j < n else ""
                cost[i, j] = cls._word_edit_distance(r_text, h_text)

        r_ind, c_ind = linear_sum_assignment(cost)
        total_distance = int(sum(cost[r, c] for r, c in zip(r_ind, c_ind)))

        cpwer = float(total_distance / total_ref_words)

        mapping = {
            hyp_speakers[c]: ref_speakers[r]
            for r, c in zip(r_ind, c_ind)
            if r < m and c < n
        }

        return cpwer, mapping

    @classmethod
    def compute_diarization_gap(cls, cpwer: float, wer: float) -> float:
        """Calculates Diarization Degradation Gap: cpWER - WER.

        A gap close to 0 proves high diarization fidelity. A large positive gap
        diagnoses turn swapping or speaker misattribution despite accurate transcription.
        """
        return max(0.0, float(cpwer - wer))

    @classmethod
    def evaluate_sample(
        cls,
        ref_turns: List[Turn],
        hyp_turns: List[Turn],
        normalize: bool = True,
    ) -> EvaluationMetrics:
        """Evaluates a complete sample, returning structured EvaluationMetrics.

        Args:
            ref_turns: Ground-truth turns.
            hyp_turns: Model-predicted turns.
            normalize: Whether to apply Indic normalization.

        Returns:
            EvaluationMetrics instance populated with all objective metrics.
        """
        format_valid = len(hyp_turns) > 0

        if normalize:
            norm_ref_turns = IndicTextNormalizer.normalize_turns(ref_turns)
            norm_hyp_turns = IndicTextNormalizer.normalize_turns(hyp_turns)
        else:
            norm_ref_turns = ref_turns
            norm_hyp_turns = hyp_turns

        ref_full_text = " ".join(t.text for t in norm_ref_turns)
        hyp_full_text = " ".join(t.text for t in norm_hyp_turns)

        # Standard STT metrics
        wer = cls.compute_wer(ref_full_text, hyp_full_text, normalize=False)
        cer = cls.compute_cer(ref_full_text, hyp_full_text, normalize=False)

        # Diarization & speaker attribution metrics
        saa, saa_mapping = cls.compute_speaker_attribution_accuracy(
            norm_ref_turns, norm_hyp_turns, normalize=False
        )
        cpwer, cpwer_mapping = cls.compute_cpwer(
            norm_ref_turns, norm_hyp_turns, normalize=False
        )

        diarization_gap = cls.compute_diarization_gap(cpwer, wer)

        # Prefer SAA mapping, fall back to cpWER mapping
        speaker_mapping = saa_mapping if saa_mapping else cpwer_mapping

        return EvaluationMetrics(
            wer=wer,
            cer=cer,
            speaker_attribution_accuracy=saa,
            cpwer=cpwer,
            diarization_gap=diarization_gap,
            speaker_mapping=speaker_mapping,
            format_valid=format_valid,
        )
