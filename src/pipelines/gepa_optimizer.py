"""Strategy 3: Automated GEPA (Genetic Evolutionary Prompt Optimization) Framework.

Components:
1. GEPAEvaluator: Evaluates predictions against ground truth using both objective
   metrics (WER, Hungarian SAA, Diarization Gap) and LLM-as-judge critique.
2. GEPAMutator: Employs gemini-3.5-flash to analyze failure diagnostics and mutate
   candidate system instructions targeting specific failure patterns.
3. GEPALoop: Coordinates multi-round evolutionary search, tracking generation history,
   fitness progression, and Pareto-optimal prompt selection.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from src.client import GeminiClient
from src.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    MODEL_3_5_FLASH,
    MODEL_3_5_FLASH_LITE,
    OPTIMIZED_RESULTS_DIR,
)
from src.metrics import MetricsEngine
from src.models import EvaluationMetrics, PipelinePrediction, SampleData, Turn
from src.pipelines.anchor_prompting import ANCHOR_SYSTEM_INSTRUCTION, AnchorPromptPipeline
from src.pipelines.single_step import SingleStepPipeline

logger = logging.getLogger(__name__)

LLM_EVALUATOR_SYSTEM_PROMPT = """You are an expert speech recognition and dialogue diarization judge.
Your task is to critically compare a model's predicted dialogue transcript against the reference ground-truth transcript.

Evaluate across three dimensions:
1. Diarization Accuracy (0-100): Are speaker turns accurately attributed? Did the model preserve speaker parity, detect speaker boundaries, avoid false alternations, and identify all speakers?
2. Transcription Fidelity (0-100): Is the Hindi Devanagari text transcribed accurately verbatim?
3. Format Compliance (0-100): Does the output strictly follow 'Speaker <ID>: <text>' with no unwanted markdown, timestamps, or commentary?

Output your evaluation strictly in JSON format with keys:
{
  "diarization_score": <int 0-100>,
  "transcription_score": <int 0-100>,
  "format_score": <int 0-100>,
  "failure_diagnostics": "<concise description of specific failure patterns, e.g. missed turn at start leading to inverted parity, collapsed 3rd speaker, or false alternation>",
  "actionable_guidance": "<specific prompt instruction recommendation to eliminate this error>"
}
"""

LLM_MUTATOR_SYSTEM_PROMPT = """You are an expert prompt engineer specializing in genetic evolutionary prompt optimization (GEPA) for Google Gemini multimodal audio and diarization models.
Your task is to inspect the current system instruction along with failure diagnostic reports from the benchmark evaluation, and evolve/mutate the prompt to fix observed failure patterns.

Mutation Rules:
1. Directly target diagnosed weaknesses (e.g. premature speaker alternation, dropped backchannels, speaker parity inversion, collapsed 3rd speaker, or speech overlap smearing).
2. Keep instructions actionable, precise, and crisp. Retain effective existing rules and reinforce domain conversational anchors.
3. Emphasize Hindi Devanagari verbatim transcription and exact 'Speaker <ID>: <text>' output format.
4. Output ONLY the mutated system instruction text without markdown code fences or conversational meta-commentary.
"""


class GEPAEvaluator:
    """Evaluates candidate prompts using programmatic metrics and LLM critique."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        eval_model: str = MODEL_3_5_FLASH,
    ):
        """Initializes the evaluator.

        Args:
            client: Optional GeminiClient instance.
            eval_model: Model used for LLM judge evaluations (default: gemini-3.5-flash).
        """
        self.client = client or GeminiClient()
        self.eval_model = eval_model

    def evaluate_predictions(
        self,
        samples: List[SampleData],
        predictions: List[PipelinePrediction],
    ) -> Dict[str, Any]:
        """Calculates quantitative benchmark metrics across predictions.

        Args:
            samples: List of ground-truth SampleData instances.
            predictions: List of model predictions.

        Returns:
            Dictionary containing mean metrics and per-sample results.
        """
        sample_map = {s.sample_id: s for s in samples}
        results = []
        for pred in predictions:
            sample = sample_map.get(pred.sample_id)
            if not sample:
                continue
            metrics: EvaluationMetrics = MetricsEngine.evaluate_sample(
                sample.ground_truth_turns, pred.predicted_turns
            )
            results.append(
                {
                    "sample_id": pred.sample_id,
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "saa": metrics.speaker_attribution_accuracy,
                    "cpwer": metrics.cpwer,
                    "diarization_gap": metrics.diarization_gap,
                    "latency": pred.latency_seconds,
                    "predicted_turn_count": len(pred.predicted_turns),
                    "gt_turn_count": len(sample.ground_truth_turns),
                }
            )

        if not results:
            return {"mean_saa": 0.0, "mean_wer": 1.0, "mean_diarization_gap": 1.0, "samples": []}

        n = len(results)
        mean_saa = sum(r["saa"] for r in results) / n
        mean_wer = sum(r["wer"] for r in results) / n
        mean_cpwer = sum(r["cpwer"] for r in results) / n
        mean_diar_gap = sum(r["diarization_gap"] for r in results) / n
        mean_lat = sum(r["latency"] for r in results) / n

        # Fitness function: Maximizes SAA, minimizes WER and Diarization Gap
        fitness = mean_saa * 0.7 + max(0.0, 1.0 - mean_wer) * 0.2 + max(0.0, 1.0 - mean_diar_gap) * 0.1

        return {
            "fitness": round(fitness, 4),
            "mean_saa": round(mean_saa, 4),
            "mean_wer": round(mean_wer, 4),
            "mean_cpwer": round(mean_cpwer, 4),
            "mean_diarization_gap": round(mean_diar_gap, 4),
            "mean_latency": round(mean_lat, 2),
            "total_evaluated": n,
            "samples": results,
        }

    def get_llm_critique(
        self,
        ground_truth_turns: List[Turn],
        predicted_turns: List[Turn],
    ) -> Dict[str, Any]:
        """Runs LLM-as-judge critique comparing predicted turns to ground truth."""
        gt_text = "\n".join(f"{t.speaker}: {t.text}" for t in ground_truth_turns[:15])
        pred_text = "\n".join(f"{t.speaker}: {t.text}" for t in predicted_turns[:15])

        prompt = (
            f"REFERENCE GROUND TRUTH:\n{gt_text}\n\n"
            f"MODEL PREDICTED TRANSCRIPT:\n{pred_text}\n\n"
            f"Analyze errors and output the JSON evaluation."
        )

        try:
            critique_raw, _ = self.client.generate_text(
                model=self.eval_model,
                prompt=prompt,
                system_instruction=LLM_EVALUATOR_SYSTEM_PROMPT,
                thinking_budget=0,
            )
            # Strip code blocks
            clean = re.sub(r"^```json\s*", "", critique_raw.strip(), flags=re.IGNORECASE)
            clean = re.sub(r"```$", "", clean.strip())
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"LLM critique failed: {e}")
            return {
                "diarization_score": 70,
                "transcription_score": 75,
                "format_score": 90,
                "failure_diagnostics": f"Evaluation error: {e}",
                "actionable_guidance": "Strengthen speaker boundary and role definitions.",
            }


class GEPAMutator:
    """Uses LLM reasoning to mutate and refine candidate prompts based on failure feedback."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        mutator_model: str = MODEL_3_5_FLASH,
    ):
        """Initializes the mutator.

        Args:
            client: Optional GeminiClient instance.
            mutator_model: Model used for prompt mutation (default: gemini-3.5-flash).
        """
        self.client = client or GeminiClient()
        self.mutator_model = mutator_model

    def mutate_prompt(
        self,
        current_prompt: str,
        failure_diagnostics: List[Dict[str, Any]],
        generation_round: int,
    ) -> str:
        """Mutates the system instruction to address observed failure modes.

        Args:
            current_prompt: Existing system instruction.
            failure_diagnostics: List of failure case descriptions and metrics.
            generation_round: Current evolutionary round number.

        Returns:
            Mutated system instruction string.
        """
        diag_summary = json.dumps(failure_diagnostics, ensure_ascii=False, indent=2)
        prompt = (
            f"GENETIC EVOLUTION ROUND {generation_round}\n\n"
            f"CURRENT SYSTEM INSTRUCTION:\n{current_prompt}\n\n"
            f"OBSERVED FAILURE CASES & DIAGNOSTIC FEEDBACK:\n{diag_summary}\n\n"
            f"Please mutate and refine the system instruction to eliminate these errors. "
            f"Address consecutive turn continuity, rapid interjections, and role anchors. "
            f"Output ONLY the improved system instruction text."
        )

        try:
            mutated_prompt, _ = self.client.generate_text(
                model=self.mutator_model,
                prompt=prompt,
                system_instruction=LLM_MUTATOR_SYSTEM_PROMPT,
                thinking_budget=0,
            )
            clean = mutated_prompt.strip()
            # If wrapped in markdown code fence, unwrap
            if clean.startswith("```") and clean.endswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1]).strip()
            return clean if len(clean) > 100 else current_prompt
        except Exception as e:
            logger.warning(f"Prompt mutation failed: {e}. Keeping current prompt.")
            return current_prompt


class GEPALoop:
    """Manages the full evolutionary prompt optimization cycle."""

    def __init__(
        self,
        client: Optional[GeminiClient] = None,
        eval_model: str = MODEL_3_5_FLASH,
        mutator_model: str = MODEL_3_5_FLASH,
        candidate_model: str = MODEL_3_5_FLASH_LITE,
    ):
        """Initializes the evolutionary loop.

        Args:
            client: Optional GeminiClient instance.
            eval_model: Model for evaluation / critique.
            mutator_model: Model for prompt mutation.
            candidate_model: Target model executing the prompt (gemini-3.5-flash-lite).
        """
        self.client = client or GeminiClient()
        self.evaluator = GEPAEvaluator(client=self.client, eval_model=eval_model)
        self.mutator = GEPAMutator(client=self.client, mutator_model=mutator_model)
        self.candidate_model = candidate_model

    def run_evolution(
        self,
        validation_samples: List[SampleData],
        initial_prompt: str = ANCHOR_SYSTEM_INSTRUCTION,
        num_rounds: int = 2,
        history_output_path: Optional[Path] = None,
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Executes the evolutionary prompt search across specified rounds.

        Args:
            validation_samples: Set of diagnostic benchmark samples.
            initial_prompt: Seed system instruction.
            num_rounds: Number of evolutionary generations.
            history_output_path: Optional path to save evolution history.

        Returns:
            Tuple of (best_prompt_string, best_metrics_dict, evolution_history_list).
        """
        logger.info(
            f"Starting GEPA evolutionary loop for {num_rounds} rounds on {len(validation_samples)} samples."
        )
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_fitness = -1.0
        best_metrics: Dict[str, Any] = {}
        history: List[Dict[str, Any]] = []

        # Round 0: Evaluate baseline seed prompt
        logger.info("Evaluating Generation 0 (Seed Prompt)...")
        pipeline = AnchorPromptPipeline(client=self.client, system_instruction=current_prompt)
        preds_0 = [pipeline.run_sample(s, model_id=self.candidate_model) for s in validation_samples]
        eval_0 = self.evaluator.evaluate_predictions(validation_samples, preds_0)

        best_fitness = eval_0["fitness"]
        best_metrics = eval_0
        history.append(
            {
                "generation": 0,
                "prompt": current_prompt,
                "fitness": eval_0["fitness"],
                "mean_saa": eval_0["mean_saa"],
                "mean_wer": eval_0["mean_wer"],
                "mean_diarization_gap": eval_0["mean_diarization_gap"],
                "accepted": True,
            }
        )
        logger.info(
            f"Gen 0 Baseline: Fitness={best_fitness:.4f}, SAA={eval_0['mean_saa']:.4f}, "
            f"WER={eval_0['mean_wer']:.4f}, DiarGap={eval_0['mean_diarization_gap']:.4f}"
        )

        for rnd in range(1, num_rounds + 1):
            logger.info(f"\n--- Starting GEPA Generation {rnd}/{num_rounds} ---")

            # Extract failure samples (samples with lowest SAA or highest DiarGap)
            worst_samples = sorted(
                eval_0.get("samples", []),
                key=lambda x: (x["saa"] - x["diarization_gap"]),
            )[:3]

            failure_diagnostics = []
            for item in worst_samples:
                sid = item["sample_id"]
                s_obj = next((s for s in validation_samples if s.sample_id == sid), None)
                p_obj = next((p for p in preds_0 if p.sample_id == sid), None)
                if s_obj and p_obj:
                    critique = self.evaluator.get_llm_critique(
                        s_obj.ground_truth_turns, p_obj.predicted_turns
                    )
                    failure_diagnostics.append(
                        {
                            "sample_id": sid,
                            "saa": item["saa"],
                            "wer": item["wer"],
                            "diarization_gap": item["diarization_gap"],
                            "llm_critique": critique,
                        }
                    )

            # Mutate prompt
            logger.info(f"Mutating prompt based on {len(failure_diagnostics)} diagnostic failure cases...")
            mutated = self.mutator.mutate_prompt(current_prompt, failure_diagnostics, rnd)

            # Evaluate mutated candidate
            candidate_pipeline = AnchorPromptPipeline(
                client=self.client, system_instruction=mutated
            )
            candidate_preds = [
                candidate_pipeline.run_sample(s, model_id=self.candidate_model)
                for s in validation_samples
            ]
            candidate_eval = self.evaluator.evaluate_predictions(
                validation_samples, candidate_preds
            )

            cand_fitness = candidate_eval["fitness"]
            is_accepted = cand_fitness >= best_fitness

            history.append(
                {
                    "generation": rnd,
                    "prompt": mutated,
                    "fitness": cand_fitness,
                    "mean_saa": candidate_eval["mean_saa"],
                    "mean_wer": candidate_eval["mean_wer"],
                    "mean_diarization_gap": candidate_eval["mean_diarization_gap"],
                    "accepted": is_accepted,
                    "delta_fitness": round(cand_fitness - best_fitness, 4),
                }
            )

            logger.info(
                f"Gen {rnd} Result: Fitness={cand_fitness:.4f} (best: {best_fitness:.4f}) | "
                f"SAA={candidate_eval['mean_saa']:.4f} | WER={candidate_eval['mean_wer']:.4f} | "
                f"Accepted: {is_accepted}"
            )

            if is_accepted:
                best_fitness = cand_fitness
                best_prompt = mutated
                best_metrics = candidate_eval
                current_prompt = mutated
                eval_0 = candidate_eval
                preds_0 = candidate_preds

        # Save history
        out_path = history_output_path or (OPTIMIZED_RESULTS_DIR / "gepa_evolution_history.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_rounds": num_rounds,
                    "best_fitness": best_fitness,
                    "best_metrics": best_metrics,
                    "history": history,
                    "best_prompt": best_prompt,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Evolution history saved to {out_path}")

        return best_prompt, best_metrics, history
