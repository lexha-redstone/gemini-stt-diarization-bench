# Original User Request

## Follow-up — 2026-09-16T02:03:26Z

This is a single self-contained fix; keep it small and focused.

Evaluate and compare `gemini-2.5-flash` (Baseline) against the `Adaptive Acoustic Champion Pipeline` (`Candidate C5`, pure `gemini-3.5-flash-lite`) across **all 37 two-speaker (`num_speakers == 2`) audio samples** in the `sarvamai/indic-diarbench` dataset (Hindi split), and update `benchmark_report.md`, `one-pager.md`, and `README.md` with the complete 37-sample 2-speaker benchmark results.

Working directory: `/Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final`
Integrity mode: development

## Requirements

### R1. Complete 37-Sample 2-Speaker Dataset Assembly (`data/all_2spk_subset/`)
Extract and assemble all 37 two-speaker (`num_speakers == 2`) samples from the local HuggingFace Parquet cache (`~/.cache/huggingface/hub/datasets--sarvamai--indic-diarbench/snapshots/92877bad8aab6e598167d91c6ee02aa8ca6ede09/Hindi/test-00000-of-00001.parquet`) into `data/all_2spk_subset/` (16kHz mono WAV files + `metadata.json`).

### R2. Hybrid Benchmark Execution (Reuse 21 Existing + Run 16 Remaining Live)
For both **Gemini 2.5 Flash Baseline (`gemini-2.5-flash`)** and **Adaptive Acoustic Champion (`Candidate C5`, pure `gemini-3.5-flash-lite`)**:
1. Reuse the verified prediction outputs for the **21 samples** already evaluated in `results/hard_2spk/` (20 samples) and `results/optimized/` (`indic_diarbench_subset` 2-speaker slice, 1 additional sample).
2. Execute live Vertex AI inference (`project_id="my-argolis-prj"`, `location="global"`) strictly on the **16 remaining un-evaluated 2-speaker samples**.
3. Combine all 37 predictions into unified result files (`results/all_2spk/predictions_gemini_2_5_flash.json` and `results/all_2spk/predictions_strategy_c5_adaptive_champion.json`) and compute exact macro metrics using `src/metrics.py` (Hungarian SAA, Diarization Gap $\Delta_{\text{diar}}$, WER, CER, cpWER, Mean Latency, Speedup, and Outliers).

### R3. Comprehensive Report & Documentation Updates
Update the following documents in the working directory with the complete 37-sample 2-speaker comparative results:
1. `benchmark_report.md`: Add a dedicated section and comparative summary table for the **Full 37-Sample 2-Speaker Benchmark (`all_2spk_subset`)**.
2. `one-pager.md`: Update the 2-speaker comparative metrics and narrative to reflect the full 37-sample benchmark alongside the hard-overlap subset.
3. `README.md`: Update the overview, key results table, and reproduction commands to include the 37-sample 2-speaker evaluation.

## Acceptance Criteria

### Quantitative & Programmatic Verification
- [ ] `data/all_2spk_subset/metadata.json` contains all 37 samples where `num_speakers == 2`, with verified 16kHz mono WAV files in `data/all_2spk_subset/audio/`.
- [ ] `results/all_2spk/predictions_gemini_2_5_flash.json` and `results/all_2spk/predictions_strategy_c5_adaptive_champion.json` each contain valid predictions for all 37 sample IDs with zero mocks.
- [ ] Macro metrics across all 37 samples are computed via `src/metrics.py` and saved to `results/all_2spk/all_2spk_comparative_summary.json`.
- [ ] `benchmark_report.md`, `one-pager.md`, and `README.md` are cleanly updated with the 37-sample metrics table and reproduction instructions.
- [ ] Existing automated test suite (`pytest tests/`) continues to pass 100%.
