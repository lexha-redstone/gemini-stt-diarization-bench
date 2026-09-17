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

## 2026-09-17T07:07:29Z

Build a 50-sample synthetic Hindi debt collection & fierce argument audio benchmark dataset with dynamic call lengths (30 seconds to 10 minutes), high speaker overlap, and realistic background noise, and evaluate Gemini 3.5 Flash Lite (Candidate C5) vs. Gemini Flash to empirically determine the audio length and conflict degradation cutoff.

Working directory: /Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final
Integrity mode: development

## Requirements

### R1. 50-Sample Hindi Debt Collection & Fierce Argument Audio Dataset (`data/debt_collection_subset/`)
Generate 50 single-channel WAV audio files (16kHz, 16-bit mono) simulating realistic Hindi debt collection calls featuring fierce arguments between a debt collector (`Speaker 0`) and a debtor (`Speaker 1`).
- **Dynamic Length Distribution**: Calls must span durations from 30 seconds to 10 minutes (600 seconds), distributed across stratified duration buckets (e.g., Short: 30s–70s, Medium: 70s–180s, Long: 180s–300s, Very Long: 300s–600s).
- **High Acoustic Overlap**: Inject realistic interruptions, rapid turn-taking, and simultaneous speech resulting in high overlap ratios (target mean overlap >= 15%, range 10%–25%+).
- **Telephony & Background Noise**: Mix subtle random background noise (e.g., call center babble, room ambience, street noise) and telephony acoustic characteristics so the audio realistically reflects single-channel call recordings.

### R2. Standardized Ground-Truth Diarization Metadata (`data/debt_collection_subset/metadata.json`)
Create an exact ground-truth `metadata.json` strictly adhering to the repository's existing schema (`sample_id`, `duration_seconds`, `num_speakers: 2`, `overlap_duration`, `overlap_ratio`, `turn_switches`, `audio_path`, `ground_truth_turns` with exact `start_time`, `end_time`, `speaker`, and verbatim Devanagari script `text`).

### R3. Live Comparative Benchmark Execution (`results/debt_collection/`)
Execute live Vertex AI inference (`project_id="my-argolis-prj"`, `location="global"`) across all 50 generated debt collection samples comparing:
1. **Baseline**: `gemini-2.5-flash` (or `gemini-3.5-flash`)
2. **Proposed Champion**: `Adaptive Acoustic Champion Pipeline` (`Candidate C5`, pure `gemini-3.5-flash-lite`)
Compute exact macro and per-bucket metrics via `src/metrics.py` (Hungarian SAA, Diarization Gap, WER, CER, cpWER, Mean Latency, and Outliers).

### R4. Audio Length Cutoff & Conflict Analysis Report
Analyze the relationship between audio duration (30s to 600s), overlap intensity, and speaker diarization accuracy across both pipelines. Determine the exact empirical duration cutoff above which `gemini-3.5-flash-lite` degrades on argumentative debt collection calls, and formulate concrete model routing recommendations. Publish the complete analysis in `results/debt_collection/debt_collection_benchmark_report.md` and update `benchmark_report.md` and `one-pager.md`.

## Acceptance Criteria

### Dataset Verification
- [ ] `data/debt_collection_subset/audio/` contains exactly 50 valid, playable 16kHz mono WAV files spanning ~30s to ~600s across all four duration buckets.
- [ ] `data/debt_collection_subset/metadata.json` contains 50 complete ground-truth entries with Devanagari transcripts, exact timestamps, and verified mean overlap ratio >= 12.0%.
- [ ] Programmatic audio verification confirms background noise mixing and zero corrupted or silent audio files.

### Benchmark & Analytical Verification
- [ ] Live Vertex AI predictions saved in `results/debt_collection/` for all 50 samples for both models with zero mocks or placeholders.
- [ ] `results/debt_collection/debt_collection_comparative_summary.json` contains overall macro metrics plus stratified breakdown by duration buckets (<70s, 70s–180s, 180s–300s, 300s–600s).
- [ ] `results/debt_collection/debt_collection_benchmark_report.md` documents the length-vs-accuracy degradation curve, pinpoints the exact routing cutoff threshold, and provides per-bucket comparison tables.
- [ ] Existing automated test suite (`pytest tests/`) passes 100%.

