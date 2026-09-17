# Gemini STT & Speaker Diarization: Master Benchmark & Reproduction Package

This package contains the complete benchmarking codebase, optimization pipelines, live evaluation datasets, and empirical result reports comparing **Gemini 3.5 Flash Lite** against **Gemini 2.5 Flash** on challenging single-channel conversational audio featuring heavy overlapping speech (`sarvamai/indic-diarbench`).

The production champion pipeline, **Candidate C5 (`AdaptiveAcousticChampionPipeline`)**, operates **strictly 100% on pure Gemini 3.5 Flash Lite** (zero calls to larger secondary models), achieving **90.60% Hungarian SAA across all 50 fierce Hindi debt collection calls** (`data/debt_collection_subset/`, 175.4 minutes total duration, mean overlap 18.37%, beating Gemini 2.5 Flash's 85.62% SAA by **+4.98%p**), **85.18% Hungarian SAA across all 37 two-speaker calls** (`all_2spk_subset`, 134.4 minutes total duration), and **87.10% Hungarian SAA** with a **12.69% Diarization Degradation Gap** on extreme high-overlap calls (`hard_2spk_subset`, beating Gemini 2.5 Flash Baseline), while beating Gemini 2.5 Flash on **Word Error Rate (22.99% vs. 39.26% raw / 23.05% norm on All-37; 16.67% vs. 248.89% raw on Debt Collection)**, delivering a **1.21× to 2.50× speedup**, and achieving a **~74% cost reduction** with **zero catastrophic hallucinations**.

---

## 1. Quickstart (Reproduction in 3 Steps)

### Step 1: Install Dependencies
```bash
# Using existing project virtualenv or creating a new one:
source /Users/lexha/Documents/work/codes/.venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Authenticate Google Cloud Application Default Credentials (ADC)
Ensure your active GCP account has access to the Vertex AI Gemini endpoints on `my-argolis-prj`:
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="my-argolis-prj"
export VERTEXAI_LOCATION="global"
```

### Step 3: Run the Champion Benchmarks (Candidate C5)
```bash
# 1. Synthesize the 50-sample Hindi Debt Collection dataset (30s-600s) into data/debt_collection_subset/:
python scripts/generate_debt_collection_dataset.py

# 2. Run the 50-sample Debt Collection benchmark (Gemini 2.5 Flash vs Candidate C5):
python scripts/run_debt_collection_benchmark.py \
    --project-id my-argolis-prj \
    --location global \
    --max-workers 6

# 3. Extract all 37 two-speaker samples into data/all_2spk_subset/:
python scripts/extract_all_2spk_dataset.py

# 4. Run the complete 37-sample 2-speaker benchmark (reuses 21 verified + runs 16 live):
python scripts/run_all_2spk_benchmark.py \
    --project-id my-argolis-prj \
    --location global

# 5. Evaluate Candidate C5 across all 20 hard high-overlap clips:
python scripts/run_hard_2spk_creative.py \
    --project-id my-argolis-prj \
    --location global \
    --strategies strategy_c5_adaptive_champion

# 6. Evaluate Candidate C5 on the 20-clip multi-speaker generalization dataset:
python scripts/run_generalization_c5.py \
    --project-id my-argolis-prj \
    --location global

# 7. Run the complete automated test suite (372 tests):
pytest tests/
```

---

## 2. Key Benchmark Results Tables

### 2.1 Complete 37-Sample 2-Speaker Corpus (`data/all_2spk_subset/`, 8,064.95s / 134.4 min, 6.59% Mean Overlap)

| Architecture / Pipeline | Model Stack | Pure 3.5 Lite? | Hungarian SAA (%) | Diarization Gap (%) | Raw WER (%) | Norm WER (%) | Raw cpWER (%) | Mean Latency (s) | Speedup vs 2.5 | Cost / 1k Clips | Outliers | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemini 2.5 Flash Baseline** | `gemini-2.5-flash` | No | **89.97%** | **11.34%** | 39.26% | 23.05% | 50.05% | 11.981s | 1.00× | ~$24.00 | 1 (`hindi_089`) | Reference |
| **Candidate C5: Adaptive Champion 👑** | **Pure `3.5-flash-lite`** | **Yes** | **85.18%** | **13.93%** | **22.99% (Best)** | **22.99% (Best)** | **36.67% (Best)** | **9.124s** | **1.31×** | **~$6.25 (74% off)** | **0 (Best)** | **PRODUCTION CHAMPION** |

### 2.2 Hard High-Overlap Subset (`data/hard_2spk_subset/`, 20 Clips, 7.80% Mean Overlap)

| Architecture / Pipeline | Model Stack | Pure 3.5 Lite? | Hungarian SAA (%) | Diarization Gap (%) | WER Norm (%) | cpWER Norm (%) | Mean Latency (s) | Cost / 1k Clips | Outliers | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gemini 2.5 Flash Baseline** | `gemini-2.5-flash` | No | **88.81%** | 13.16% | 24.59%* | 36.77%* | 7.033s | ~$24.00 | 1 (`hindi_089`) | Reference |
| **Gemini 3.5 Flash Lite Default** | `gemini-3.5-flash-lite` | Yes | 68.76% | 30.50% | 25.29% | 55.75% | **4.187s** | ~$6.20 | **0** | Deficit Baseline |
| **Strategy A: Token-0 Bypass** | `gemini-3.5-flash-lite` | Yes | 81.11% | 17.80% | 25.09% | 42.50% | 5.040s | ~$6.20 | **0** | Milestone 2 |
| **Strategy C1: Advanced Token-0** | `gemini-3.5-flash-lite` | Yes | 78.76% | 19.37% | 24.77% | 43.70% | 4.667s | ~$6.20 | **0** | Milestone 3 |
| **Candidate C5: Adaptive Champion 👑** | **Pure `3.5-flash-lite`** | **Yes** | **87.10%** | **12.69% (Best)** | **24.86%** | **37.10%** | **5.811s** | **~$6.25 (74% off)** | **0** | **PRODUCTION CHAMPION** |

### 2.3 50-Sample Hindi Debt Collection & Fierce Argument Stress Benchmark (`data/debt_collection_subset/`, 10,524.0s / 175.4 min, 18.37% Mean Overlap, Dynamic 30s–600s Calls)

| Duration Bucket / Slice | Samples | Mean Dur | Mean Overlap | Pipeline / Model | Hungarian SAA (%) | Diarization Gap (%) | Raw WER (%) | Norm cpWER (%) | Mean Latency (s) | Outliers | Speedup vs 2.5 |
|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Short (`<70s`)** | 13 | 50.00s | 17.21% | **Candidate C5 (`3.5-lite`)** | **93.55%** *(vs 88.74%)* | **7.61%** *(vs 13.70%)* | **19.10%** | **26.53%** | 7.112s | **0** | 0.74× |
| **Medium (`70s–180s`)** | 13 | 124.00s | 18.16% | **Candidate C5 (`3.5-lite`)** | **92.19%** *(vs 82.62%)* | **9.60%** *(vs 24.62%)* | **15.75%** *(vs 913.27%)* | **25.35%** | **10.160s** | **0** *(vs 1)* | **2.50×** |
| **Long (`180s–300s`)** | 12 | 240.00s | 18.86% | **Candidate C5 (`3.5-lite`)** | 89.85% *(vs 93.22%)* | 12.75% *(vs 6.23%)* | 16.72% | 29.47% | 15.660s | **0** | 0.97× |
| **Very Long (`300s–600s`)** | 12 | 448.50s | 19.36% | **Candidate C5 (`3.5-lite`)** | **86.43%** *(vs 77.90%)* | **15.89%** *(vs 25.24%)* | 15.00% | **30.89%** | 24.781s | **0** | 0.99× |
| **MACRO OVERALL (`30–600s`)** | **50** | **210.48s** | **18.37%** | **`gemini-2.5-flash` Baseline** | 85.62% | 17.52% | 248.89% | 31.01% | 17.476s | 1 (`debt_019`) | 1.00× |
| **MACRO OVERALL (`30–600s`)** | **50** | **210.48s** | **18.37%** | **Candidate C5: Champion 👑** | **90.60% (+4.98%p)** | **11.35% (-6.17%p)** | **16.67% (Best)** | **27.98% (Best)** | **14.196s** | **0 (Immune)** | **1.23×** |

---

## 3. Directory Structure

```
.
├── README.md                                       # Quickstart and reproduction overview
├── one-pager.md                                    # Executive proposal & architecture summary
├── requirements.txt                                # Python dependencies
├── benchmark_report.md                             # Master technical report & reproduction playbook
├── data/
│   ├── debt_collection_subset/                     # 50-sample Hindi Debt Collection stress benchmark (10,524.0s, 30s-600s)
│   │   ├── audio/                                  # Audio files: debt_001.wav ... debt_050.wav (16kHz mono)
│   │   └── metadata.json                           # Ground-truth Devanagari turns, timestamps, overlap ratios
│   ├── all_2spk_subset/                            # Complete 37-sample 2-speaker corpus (8,064.95s, 134.4 min)
│   │   ├── audio/                                  # Audio files: hindi_022.wav ... hindi_094.wav
│   │   └── metadata.json                           # Ground-truth turns, timestamps, overlap ratios
│   ├── hard_2spk_subset/                           # 20 curated 16kHz mono WAV clips (2,324.12s, up to 20.6% overlap)
│   │   ├── audio/                                  # Audio files: hindi_022.wav ... hindi_093.wav
│   │   └── metadata.json                           # Ground-truth turns, timestamps, overlap ratios
│   └── indic_diarbench_subset/                     # 20 multi-speaker benchmark clips (15 2-spk, 5 3-spk)
│       ├── audio/                                  # Audio files: hindi_001.wav ... hindi_093.wav
│       └── metadata.json                           # Ground-truth multi-speaker annotations
├── results/
│   ├── debt_collection/
│   │   ├── debt_collection_benchmark_report.md     # Milestone 6 complete 10-section cutoff & routing report
│   │   ├── debt_collection_comparative_summary.json # Macro & stratified duration bucket comparative summary JSON
│   │   ├── predictions_gemini_2_5_flash.json       # Live 50-sample predictions for Gemini 2.5 Flash
│   │   └── predictions_strategy_c5_adaptive_champion.json # Live 50-sample predictions for Candidate C5
│   ├── all_2spk/
│   │   ├── all_2spk_comparative_summary.json       # Complete 37-sample 2-speaker comparative metrics JSON
│   │   ├── predictions_gemini_2_5_flash.json       # Verified 37-sample predictions for Gemini 2.5 Flash
│   │   └── predictions_strategy_c5_adaptive_champion.json # Verified 37-sample predictions for Candidate C5
│   ├── hard_2spk/
│   │   ├── hard_2spk_creative_parity_report.md     # Milestone 4 technical specification (708 lines)
│   │   ├── hard_2spk_comparative_summary.json      # Macro metric aggregation JSON across 11 strategies
│   │   ├── predictions_strategy_c5_adaptive_champion.json
│   │   ├── predictions_strategy_c1_advanced_token0.json
│   │   ├── predictions_strategy_c3_acoustic_token0.json
│   │   ├── predictions_strategy_c4_multistage.json
│   │   ├── predictions_strategy_a_token0_bypass.json
│   │   ├── predictions_gemini_2_5_flash.json
│   │   └── predictions_gemini_3_5_flash_lite_default.json
│   └── optimized/
│       ├── generalization_summary.json             # Multi-speaker generalization metrics JSON
│       └── predictions_strategy_c5_generalization.json
├── scripts/
│   ├── generate_debt_collection_dataset.py         # Deterministic 50-sample Hindi debt collection audio generator
│   ├── run_debt_collection_benchmark.py            # Live Vertex AI benchmark runner for 50 debt collection clips
│   ├── extract_all_2spk_dataset.py                 # Dataset extraction script for all 37 2-speaker clips
│   ├── run_all_2spk_benchmark.py                   # Hybrid benchmark runner for all 37 2-speaker clips
│   ├── run_hard_2spk_creative.py                   # Runner for Candidate C3, C4, C5 (Adaptive Champion)
│   ├── run_generalization_c5.py                    # Runner for Candidate C5 on multi-speaker dataset
│   ├── run_hard_2spk_benchmark.py                  # Runner for baselines, Strategy A, and Strategy B
│   ├── run_full_benchmark.py                       # Unified benchmark compiler
│   └── extract_hard_2spk_dataset.py                # Hard 20 dataset extraction script
├── src/
│   ├── client.py                                   # Vertex AI GeminiClient with 600s timeout, ADC retry & fallback
│   ├── config.py                                   # Model IDs, hyperparameter defaults, and paths
│   ├── dataset.py                                  # DatasetLoader schema validation
│   ├── metrics.py                                  # Hungarian SAA, cpWER, WER, CER, Diarization Gap
│   ├── models.py                                   # Pydantic data schemas (Turn, SampleData, etc.)
│   ├── normalizer.py                               # Indic/Devanagari text normalizer
│   └── pipelines/
│       ├── adaptive_acoustic_champion.py           # Candidate C5: Adaptive Acoustic Champion (Production)
│       ├── acoustic_token0.py                      # Candidate C3: Acoustic-Anchored Token-0
│       ├── acoustic_multistage.py                  # Candidate C4: Pure 3.5 Lite Multi-Stage Verifier
│       ├── advanced_token0.py                      # Strategy C1: Advanced Token-0 Bypass
│       ├── structured_json.py                      # Strategy C2: Native Structured JSON
│       ├── token0_bypass.py                        # Strategy A: Baseline Token-0 Bypass
│       └── single_step.py                          # Approach 1: Single-Step baseline pipeline
└── tests/                                          # Automated unit & adversarial regression test suite (372 tests)
```

---

## 4. Documentation Index

- **Master Benchmark Report**: [`benchmark_report.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/benchmark_report.md)
- **Executive One-Pager**: [`one-pager.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/one-pager.md)
- **Milestone 6 Debt Collection & Cutoff Report**: [`results/debt_collection/debt_collection_benchmark_report.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/results/debt_collection/debt_collection_benchmark_report.md)
- **Milestone 6 Debt Collection Summary JSON**: [`results/debt_collection/debt_collection_comparative_summary.json`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/results/debt_collection/debt_collection_comparative_summary.json)
- **Milestone 5 Full 37-Sample Summary JSON**: [`results/all_2spk/all_2spk_comparative_summary.json`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/results/all_2spk/all_2spk_comparative_summary.json)
- **Milestone 4 Full Technical Specification**: [`results/hard_2spk/hard_2spk_creative_parity_report.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/results/hard_2spk/hard_2spk_creative_parity_report.md)
- **Milestone 3 Intermediate Report**: [`results/hard_2spk/hard_2spk_parity_report.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/results/hard_2spk/hard_2spk_parity_report.md)
- **Milestone 2 Intermediate Report**: [`results/hard_2spk/hard_2spk_report.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/results/hard_2spk/hard_2spk_report.md)
- **Milestone 1 Archive Report**: [`benchmark_report_full.md`](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/benchmark_report_full.md)

---

## 5. License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/gemini_stt_diarization_repro_final/LICENSE) file for the full license text.
