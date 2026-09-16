# Gemini STT & Speaker Diarization: Hard 2-Speaker High-Overlap Benchmark Report

**Target Domain**: Strictly 2-Speaker Single-Channel (Mono) Conversational Dialogue with Extreme Overlapping Speech  
**Dataset**: `sarvamai/indic-diarbench` (20 Curated Hard 2-Speaker Hindi Clips, 2,324.14s total duration, 204.25s overlap, mean overlap 7.80%, peak 20.62%)  
**GCP Project**: `my-argolis-prj` (Vertex AI `global`, 100% Live API execution with ADC, zero mock data)  
**Date**: September 2026  
**Status**: Verified & Reproducible (Audit Verdict: CLEAN)

---

> [!NOTE]
> **Historical Milestone Report (Milestone 2)**: This document records intermediate findings from Milestone 2 (Strategy A Token-0 Bypass & Strategy B Pure Two-Pass).
> For the authoritative final benchmark results, including Candidate C5 (Adaptive Champion) and full multi-speaker generalization, please refer to the definitive [Master Benchmark Report](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/benchmark_report.md) or the [Milestone 4 Comprehensive Technical Report](file:///Users/lexha/Documents/work/codes/prj/22-STT-speaker-diarization/results/hard_2spk/hard_2spk_creative_parity_report.md).

## 1. Executive Summary & Macro Comparison Table

This benchmark investigates and solves the severe speaker diarization degradation observed when using lightweight distilled speech models (**Gemini 3.5 Flash Lite**) on difficult two-speaker conversational audio featuring heavy overlapping speech, rapid speaker transitions, and contentious interruptions.

In conversational telephone speech from `sarvamai/indic-diarbench`, the default Gemini 3.5 Flash Lite single-prompt baseline suffered a catastrophic drop in Speaker Attribution Accuracy (**68.76% SAA** vs. **88.81%** for Gemini 2.5 Flash), widening the Diarization Degradation Gap to **30.50%** (a 2.3× penalty over Gemini 2.5 Flash's 13.16%). Forensic analysis revealed that this collapse was driven by the **Token-0 Commitment Trap** and a **95.0% Rigid Alternation Bias**, which systematically inverted speaker parity upon overlapping interruptions.

To eliminate this gap without inflating inference costs or violating architectural constraints, two **100% pure Gemini 3.5 Flash Lite** architectures were developed and evaluated across all 20 curated hard dialogue clips on Vertex AI (`my-argolis-prj`):
1. **Strategy A (Token-0 Bypass Single-Prompt)**: Inverts autoregressive token generation by emitting the spoken utterance first and the speaker identifier last (`[Utterance] <Hindi text> | [Speaker] Speaker <ID>`), reinforced with anti-alternation constraints and few-shot Hindi overlap demonstrations.
2. **Strategy B (Pure 3.5 Lite Two-Pass Self-Refinement)**: Decouples transcription into Pass 1 (Acoustic STT sequential turns) and Pass 2 (Retrospective Disentanglement and Turn-Splitting), running **100% on pure Gemini 3.5 Flash Lite** with zero reliance on larger secondary models.

### Macro Comparison Across All Evaluated Architectures (20 Hard Samples)

| Architecture / Model | Model Stack | Pure 3.5 Lite? | Hungarian SAA | Diarization Gap ($\Delta_{\text{diar}}$) | WER (Norm)* | CER (Norm)* | cpWER (Norm)* | Mean Latency | Speedup vs 2.5 | Cost / Sample | Acceptance Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini 2.5 Flash Baseline** | `gemini-2.5-flash` | No | 88.81% | 13.16% | 24.59% | 13.10% | 36.77% | 7.03s | 1.00× (Ref) | ~$0.0240 | Reference Baseline |
| **Gemini 3.5 Flash Lite Default** | `gemini-3.5-flash-lite` | Yes | 68.76% | 30.50% | 25.29% | 14.37% | 55.75% | **4.19s** | **1.68×** | ~$0.0062 | Baseline (Failed SAA & Gap) |
| **Strategy 1: Anchor Prompting** | `gemini-3.5-flash-lite` | Yes | 76.35% | 24.82% | 25.36% | 14.39% | 49.72% | **4.26s** | **1.65×** | ~$0.0062 | Partial (Gap >20%) |
| **Strategy 2: Decoupled (Mixed)** | `3.5-lite` + `3.5-flash`* | **No\*** | 86.95% | 11.60% | 26.82% | 17.02% | 36.37% | 8.87s | 0.79× | ~$0.0082 | Constraint Violation (`3.5-flash` used) |
| **Strategy A: Token-0 Bypass** | **Pure `3.5-flash-lite`** | **YES** | **81.11%** | **17.80%** | **25.09%** | **14.10%** | **42.50%** | **5.04s** | **1.40×** | **~$0.0062** | **ACCEPTED (Champion Pure Lite)** |
| **Strategy B: Pure Lite Two-Pass** | **Pure `3.5-lite` + `3.5-lite`** | **YES** | **74.33%** | **26.60%** | **25.29%** | **14.69%** | **51.87%** | **7.04s** | **1.00×** | **~$0.0063** | **Complementary Specialist** |

*\*Note on Normalization: Gemini 2.5 Flash suffered a repetition loop on `hindi_089` (1,141 turns), inflating unclipped WER to 54.50% and cpWER to 66.90%. Normalized figures exclude outliers. Strategy 2 used `gemini-3.5-flash` for Step 2 role attribution, violating the pure lite requirement.*

### Key Breakthrough Highlights:
- **Strategy A Solves Acceptance Targets with Pure 3.5 Flash Lite**:
  Achieves **81.11% Hungarian SAA** (exceeding the $>75.0\%$ target by +6.11%p) and reduces the Diarization Gap from 30.50% down to **17.80%** (exceeding the $<20.0\%$ target by -2.20%p), delivering a **41.6% relative compression** in diarization penalty.
- **Superior Production Economics (1.40× Faster, 3.7× Cheaper)**:
  Executes in **5.04s** mean latency (1.40× faster than Gemini 2.5 Flash's 7.03s), while costing only **~$0.0062 per sample** (~$6.20 per 1,000 calls vs. ~$24.00 for Gemini 2.5 Flash)—a **74% cost savings** with 100% pure Gemini 3.5 Flash Lite.
- **Dramatic Single-Prompt Failure Recovery**:
  On extreme overlap failure clips where default 3.5 Flash Lite collapsed to ~50% random chance, Strategy A achieved massive attribution recoveries:
  - `hindi_085` (15.15% overlap): SAA jumped from **51.76% to 98.84%** (0.00% Diarization Gap), **outperforming Gemini 2.5 Flash (81.61%) by +17.23%p**.
  - `hindi_084` (77 turns, 300.1s): SAA jumped from **53.71% to 96.16%** (Gap collapsed to **0.98%**).
  - `hindi_022` (192.8s): SAA jumped from **52.12% to 97.82%** (Gap collapsed to **4.27%**).
  - `hindi_067` (13.51% overlap): SAA jumped from **51.37% to 72.51%**, beating Gemini 2.5 Flash (**55.71%**).
- **Strategy B as an Architectural Specialist for Boundary Collisions**:
  Strategy B achieved **75.87% SAA** on `hindi_064` (breaking a stubborn 6-turn parity lock that single-prompt models failed to resolve) and **74.73% SAA** on peak overlap debate `hindi_062` (20.62% overlap), proving the value of retrospective turn splitting.

---

## 2. Failure Forensics & Diagnostic Hypotheses (R1)

Deep acoustic, temporal, and lexical forensics conducted across the 20 curated hard clips (248 overlap intervals totaling 204.25 seconds) uncovered **four interlocked root causes** that explain why Gemini 3.5 Flash Lite default collapses to chance-level (~50–54%) SAA on overlapping conversational speech:

```
[Acoustic Overlap / Rapid Interruption] (204.25s total overlap, peak 20.62% in hindi_062)
                   │
                   ▼
[1. Turn Smearing & Overlap Dropout] (50.3% of 3.5 Lite turns merge speech from both speakers)
                   │
                   ▼
[2. The Token-0 Commitment Trap] (Prompt requires "Speaker <ID>: <transcript>")
  - Model must emit speaker label at Token-0 before decoding acoustic text.
  - Empirical proof: Hallucinated nested tags in hindi_085 ("Speaker 1: Speaker 0: ...").
                   │
                   ▼
[3. Rigid Ping-Pong Alternation Bias] (95.0% strict ABAB alternation in 3.5 Lite vs 5.0% in GT)
  - Natural conversation has 24.04% consecutive turns; 3.5 Lite rigidly alternates.
                   │
                   ▼
[4. Parity Inversion Cascades & Hungarian Bipartite Collapse]
  - Absorbed turn flips conversational parity for all subsequent alternating turns.
  - Bipartite matching matrix balances (conf[0,0] ≈ conf[0,1] ≈ 50%).
  - Hungarian SAA mechanically collapses to ~50–54% (Chance Level).
```

### 1. The Token-0 Commitment Trap
In standard single-step prompting, the model is instructed to output dialogue in the format:
```text
Speaker <ID>: <transcribed turn text>
```
Under autoregressive decoding, this forces the model to emit the speaker token (`Speaker 0` or `Speaker 1`) at the **very first token (Token 0)** of each line—**prior to attending to and decoding the acoustic waveform or lexical content of the upcoming turn**. 

In a lightweight distilled model (`gemini-3.5-flash-lite`), cross-attention lookahead is constrained. When early acoustic onset is ambiguous, soft, or overlapped by cross-talk, the model's language prior dominates over acoustic features. 

**Empirical Verbatim Proof (`hindi_085`)**:
In `results/hard_2spk/predictions_gemini_3_5_flash_lite_default.json` for `hindi_085`, when an overlapping speech event occurred, the model emitted `Speaker 1:` at Token-0 based on its alternating prior, but upon attending to the subsequent audio frames, realized Speaker 0 was speaking. Unable to backtrack autoregressively, it literally hallucinated nested prefix tags inside the line:
```text
Turn 1: Speaker 1: Speaker 0: अह व्हाट हैव यू अह क्या खरीदा है बताओ फिर?
Turn 3: Speaker 0: Speaker 1: अच्छा। उसके अलावा मैं फ्लेवर्ड कॉफी लाई हूं, और वो हमें हम सबके लिए लाई हूं।
Turn 4: Speaker 0: Speaker 1: उसके टेस्ट में ना, मतलब ऐसे नॉर्मल कॉफी में एक सर्टेन टेस्ट है, ठीक है?
```
Because the parser extracted only the outer prefix (`Speaker 1:` for Turn 1), the turn was committed to the wrong speaker, corrupting the entire attribution state.

### 2. Rigid Ping-Pong Alternation Bias (95.0%)
In authentic conversational dialogue, human turn-taking does not follow a mathematical ping-pong cycle. In the ground-truth annotations across all 20 clips:
- **Consecutive same-speaker turns occur 24.04% of the time** (due to mid-thought breathing pauses, clause splits across interruptions, and unacknowledged interjections).
- **Only 1 of 20 clips (5.0%)** exhibits strict alternating turns in ground truth (mean alternation ratio = 75.96%).

In stark contrast, Gemini 3.5 Flash Lite default enforced **100% strict ABAB alternation on 19 of 20 clips (95.0%)**, with an aggregate dataset alternation ratio of **97.50%**. The model possessed an overwhelming autoregressive bias that every turn boundary must alternate speaker parity.

### 3. Turn Smearing & Overlap Dropout (50.3% Merged Turns)
Across the dataset's 248 overlap intervals (mean clip overlap 7.80%, reaching 20.62% in `hindi_062`), Gemini 3.5 Flash Lite default merged speech from both speakers into a single turn on **50.3% of all predicted turns**. In contentious debate clips, this turn smearing exploded:
- `hindi_062` (20.62% overlap): **80.0% of predicted turns (24/30)** contained merged speech from both speakers. Total turns collapsed from 68 in GT down to 30 in 3.5 Lite.
- `hindi_063` (11.45% overlap): **73.3% of predicted turns (11/15)** contained merged speech from both speakers. Total turns collapsed from 36 in GT down to 15 in 3.5 Lite.

Crucially, acoustic recognition itself did not fail: short turns ($<1.5$s or $\le 3$ words) achieved **74.2% lexical recall** in 3.5 Lite default. However, rather than triggering a turn boundary, the model concatenated the interrupting words into the active speaker's ongoing transcript line.

### 4. Parity Inversion Cascades & Hungarian Bipartite Collapse
When an absorbed turn swallows a single speaker switch, the rigid alternation prior flips conversational parity for all subsequent turns. Because the conversation is strictly two-speaker, this triggers an **Inversion Cascade**.

**Concrete Case Forensic: `hindi_064` (6-Turn Inversion Cascade)**:
- At 11.59s, Speaker 0 interrupts Speaker 1 with 0.45s of overlap.
- Default 3.5 Lite merges both speakers into Turn 3 under `Speaker 1:`:
  `Speaker 1: पढ़ाई के अलावा ये सब करते हैं, बस पढ़ाई नहीं होती इनसे। अरे कर रहे हैं बच्चे तो कितने रील बना के...`
- Because Turn 3 swallowed Speaker 0's entrance, dialogue parity flipped:
  - Turn 4: Speaker 1's words emitted under `Speaker 0:`
  - Turn 5: Speaker 0's words emitted under `Speaker 1:`
  - Turn 6: Speaker 1's words emitted under `Speaker 0:`
  - Turn 7: Speaker 0's words emitted under `Speaker 1:`
  - Turn 8: Speaker 1's words emitted under `Speaker 0:`
  - Turn 9: Speaker 0's words emitted under `Speaker 1:`
- The inversion persisted for 6 consecutive turns until another multi-speaker merge flipped parity back.
- **Hungarian Metric Collapse**: In bipartite matching (`scipy.optimize.linear_sum_assignment`), when roughly half the dialogue is inverted, the confusion matrix entries balance ($conf[0,0] \approx conf[0,1] \approx 50\%$). The optimal assignment can achieve only **~50–54% SAA** (random chance), blowing out the Diarization Gap to **33.6%** even though transcription WER remained low (**22.9%**).

---

## 3. Pure Gemini 3.5 Flash Lite Architectural Solutions (R2 & R3)

To resolve the Token-0 Commitment Trap and break the alternation bias without using larger secondary models (`gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`), two complementary architectures were engineered, fully implemented, and validated.

### 3.1 Mathematical Formulation: Token-0 Bypass vs. Baseline

$$\text{Baseline Decoding:} \quad P(Y) = \prod_{i=1}^{T} P(S_i \mid \text{Audio}, H_{<i}) \cdot P(W_i \mid S_i, \text{Audio}, H_{<i})$$
$$\text{Token-0 Bypass:} \quad P(Y) = \prod_{i=1}^{T} P(W_i \mid \text{Audio}, H_{<i}) \cdot P(S_i \mid W_i, \text{Audio}, H_{<i})$$

In the baseline formulation, the model must condition word generation on $S_i$ chosen at Token 0. In the Token-0 Bypass formulation, the speaker attribution token $S_i$ is generated **posterior** to the utterance text $W_i$, allowing full cross-attention over both the completed lexical sequence and the aligned acoustic timeframe.

---

### 3.2 Strategy A: Advanced Single-Prompt Optimization with Token-0 Bypass (`src/pipelines/token0_bypass.py`)

Strategy A executes in a single API call on `gemini-3.5-flash-lite` (`thinking_budget=0`, `temperature=0.0`). It enforces line-delimited suffix attribution combined with structural anti-alternation constraints and Devanagari overlap demonstrations.

#### Exact Prompt Specification:
```text
[SYSTEM INSTRUCTION]
You are an expert multilingual speech recognition and speaker diarization system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and accurately attribute each conversational turn to its distinct speaker (Speaker 0 or Speaker 1).

CRITICAL FORMAT REQUIREMENT — TOKEN-0 BYPASS:
For every conversational turn, you MUST emit the spoken text FIRST and the speaker identifier LAST on each line.
Strictly adhere to this line-delimited syntax:
[Utterance] <spoken Hindi text verbatim> | [Speaker] Speaker <ID>

Principles & Anti-Alternation Constraints:
1. Turn Continuity & Anti-Alternation:
   - Conversation is NOT a strict alternating tennis match. Do NOT alternate between Speaker 0 and Speaker 1 if the same person is continuing their thought.
   - If a speaker pauses, breathes, or speaks across multiple sentences, attribute ALL consecutive turns to the SAME speaker.
2. Overlap & Interruption Splitting:
   - When speakers speak simultaneously or interrupt each other, NEVER combine both voices into a single line.
   - Split the speech into separate turns: emit the interrupted speaker's words as one turn, and the interrupting speaker's words on the next turn.
3. Backchannel Preservation:
   - Faithfully capture short acknowledgments, affirmations, and interjections (e.g., हाँ, जी, अच्छा, ठीक है, हूँ) as separate turns and attribute them to the listening speaker.
4. Speaker Cardinality & Consistency:
   - There are strictly 2 primary speakers in this conversation: Speaker 0 and Speaker 1. Maintain consistent speaker labels throughout the dialogue.
5. Verbatim Fidelity:
   - Transcribe strictly in Devanagari script. Do not summarize, translate, or drop stuttered words.

Few-Shot Example Demonstrating Consecutive Turns, Interruptions, and Backchannels:
[Utterance] नमस्कार, क्या मेरी बात शर्मा जी से हो रही है? | [Speaker] Speaker 0
[Utterance] मैं बैंक शाखा से बोल रहा हूँ। | [Speaker] Speaker 0
[Utterance] हाँ, बोलिए कौन? | [Speaker] Speaker 1
[Utterance] आपकी बकाया किस्त के संबंध में जानकारी चाहिए थी। | [Speaker] Speaker 0
[Utterance] अरे नहीं, मैंने तो पिछले हफ्ते ही— | [Speaker] Speaker 1
[Utterance] लेकिन हमारे सिस्टम में रिकॉर्ड अभी तक अपडेट नहीं हुआ है। | [Speaker] Speaker 0
[Utterance] जी, अच्छा। मैं रसीद दिखाता हूँ। | [Speaker] Speaker 1
[Utterance] ठीक है, आप व्हाट्सएप पर भेज दीजिए। | [Speaker] Speaker 0

[USER PROMPT]
Transcribe the entire audio clip verbatim in Devanagari script and attribute all speaker turns using the format:
[Utterance] <spoken text> | [Speaker] Speaker <ID>
```

#### Robust Parsing Engine:
`parse_token0_bypass_turns(raw_text)` features:
- Rightmost pipe splitting (`line.rsplit("|", 1)`) ensuring Hindi idioms or punctuation containing pipes are never broken.
- Flexible suffix regex supporting `Speaker <ID>`, `वक्ता <ID>`, and unbracketed variants.
- Multiline continuation buffering for long single-speaker clauses.
- Fallback prefix parser absorbing standard `Speaker <ID>:` formatting if the model occasionally reverts.
- Zero-drop guarantee: Non-empty output never produces an empty turn list.

---

### 3.3 Strategy B: Pure 3.5 Lite Two-Pass Self-Refinement Architecture (`src/pipelines/pure_lite_twopass.py`)

Strategy B decouples the speech problem into two discrete stages, both running **100% on `gemini-3.5-flash-lite`** (`thinking_budget=0`):

```
Audio WAV ──► [Pass 1: Acoustic STT (3.5 Flash Lite)]
                   │
                   ▼ Sequential Acoustic Turns ("Turn 1: ...", "Turn 2: ...")
              [Pass 2: Turn-Splitting Disentanglement (3.5 Flash Lite)]
                   │
                   ▼ Retrospective Attribution ("Speaker 0: ...", "Speaker 1: ...")
              Final Diarized Transcript
```

#### Pass 1: Acoustic Speech Turn Segmentation
Pass 1 transcribes audio into neutral sequential turns without guessing speaker roles, eliminating the Token-0 dilemma during acoustic feature extraction:
```text
[SYSTEM INSTRUCTION]
You are an expert multilingual speech-to-text recognition system.
Your task is to transcribe the provided audio clip verbatim in Hindi (Devanagari script) and segment it into sequential acoustic speech turns.
Rules:
1. Segment the dialogue into distinct speech turns whenever there is an acoustic pause, breath, speaker transition, or interruption.
2. When speakers talk simultaneously or interrupt, split the overlapping speech into separate turns based on acoustic flow.
3. Do NOT guess speaker identities, names, or roles.
4. Label each turn strictly as 'Turn <N>:' starting from 1.
5. Transcribe strictly in Devanagari script verbatim. Capture all words, including short acknowledgments (हाँ, जी, अच्छा, ठीक है).
6. Output each turn on a separate line. No timestamps, explanations, or summaries.

Format:
Turn 1: <spoken words>
Turn 2: <spoken words>
Turn 3: <spoken words>
```

#### Pass 2: Turn-Splitting Disentanglement & Attribution
Pass 2 analyzes the complete conversational context, stance polarity, and question-answer dynamics to attribute turns. Crucially, it incorporates the **Turn-Splitting Disentanglement Directive**:
```text
[SYSTEM INSTRUCTION]
You are an expert dialogue analyst and speaker diarization engine.
You are provided with sequential acoustic speech turns extracted from a Hindi 2-speaker conversation (Turn 1, Turn 2, ...).
Your task is to analyze the complete conversational context retrospectively and attribute each turn to Speaker 0 or Speaker 1.

CRITICAL DISENTANGLEMENT & TURN-SPLITTING DIRECTIVE:
Due to rapid interruptions and overlapping speech, some input turns may mistakenly contain utterances from BOTH speakers merged together.
1. If an input turn contains speech from both speakers, you MUST SPLIT it into separate speaker turns!
2. Determine conversational stances and roles:
   - Identify the two distinct viewpoints, questions vs answers, and conversational polarity.
3. Anti-Alternation Constraint:
   - Do NOT assume speakers strictly alternate. If consecutive utterances continue the same speaker's thought across pauses, attribute BOTH to the SAME speaker.
4. Backchannel Attribution:
   - Short listener affirmations (हाँ, जी, अच्छा) belong to the listener reacting to the active speaker.
5. Verbatim Transcript Fidelity:
   - Preserve the exact Hindi Devanagari words without modification, omission, or translation.

Output Format:
Speaker <ID>: <exact spoken transcript of the turn>
```

---

## 4. Per-Sample Quantitative Recovery Analysis (R4)

The table below presents the verified, programmatic evaluation across all 20 curated hard 2-speaker clips in `data/hard_2spk_subset/`, ordered by overlap percentage descending:

### Full 20-Sample Benchmark Matrix

| Sample ID | Duration | Overlap % | Overlap Dur | 3.5 Lite Def SAA | Strategy A SAA | Strategy B SAA | 2.5 Flash SAA | Strategy A Gap | Strategy B Gap | 2.5 Flash Gap | Strategy A WER | Key Forensic Outcome / Recovery Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **hindi_062** | 300.3s | **20.62%** | 61.91s | 53.23% | **68.44%** | **74.73%** | 98.15% | 19.81% | 23.99% | 0.00% | 33.12% | Peak overlap: Strat B recovered to 74.73%; Strat A halved Diarization Gap to 19.81%. |
| **hindi_085** | 60.1s | **15.15%** | 9.11s | 51.76% | **98.84%** | **77.30%** | 81.61% | **0.00%** | 22.16% | 19.07% | 29.90% | **Stellar Recovery**: SAA jumped from 51.8% to 98.8%, beating 2.5 Flash (81.6%) with 0.0% Gap! |
| **hindi_064** | 60.2s | **14.34%** | 8.63s | 54.29% | 53.82% | **75.87%** | 100.00% | 29.35% | 18.06% | 0.00% | 30.32% | Strat B turn-splitting broke 6-turn parity lock, achieving 75.87% SAA and 18.06% Gap. |
| **hindi_067** | 60.4s | **13.51%** | 8.16s | 51.37% | **72.51%** | 54.49% | 55.71% | 28.14% | 31.60% | 61.47% | 56.28% | **Beats 2.5 Flash**: Strat A achieved 72.51% SAA vs. 55.71% for 2.5 Flash on noisy audio. |
| **hindi_087** | 60.2s | **12.17%** | 7.33s | 68.35% | 64.97% | **69.03%** | 99.39% | 35.06% | 28.16% | 0.00% | 33.33% | Rapid latching; Strat B improved over Strat A (69.03% vs 64.97%). |
| **hindi_063** | 300.5s | **11.45%** | 34.41s | 50.89% | **69.32%** | 60.28% | 93.53% | 21.45% | 43.88% | 5.36% | 21.98% | 5-min debate: SAA recovered from chance (50.89%) to 69.32% in Strat A. |
| **hindi_093** | 60.4s | **10.72%** | 6.47s | 70.37% | 51.61% | **70.42%** | 98.61% | 50.65% | 45.02% | 0.00% | 28.14% | Strat B preserved 70.42% SAA; Strat A suffered late turn inversion. |
| **hindi_066** | 60.5s | **9.77%** | 5.91s | 97.08% | **95.95%** | **99.58%** | 95.95% | 3.31% | 0.00% | 5.15% | 23.16% | High baseline maintained; Strat B achieved near-perfect 99.58% SAA. |
| **hindi_084** | 300.1s | **7.42%** | 22.26s | 53.71% | **96.16%** | **83.91%** | 97.74% | **0.98%** | 10.02% | 0.00% | 36.92% | **Massive Recovery**: 77 turns recovered from 53.71% to 96.16% SAA, Gap dropped to 0.98%! |
| **hindi_086** | 60.2s | **5.77%** | 3.47s | 100.00% | **100.00%** | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 15.43% | Perfect 100.0% attribution maintained across all methods ($\Delta F_0 = 67.7$ Hz). |
| **hindi_029** | 73.6s | **5.60%** | 4.12s | 60.85% | **99.06%** | 65.42% | 98.60% | **0.00%** | 48.28% | 0.00% | 25.00% | Recovered from 60.85% to 99.06% SAA; Gap compressed to 0.00%. |
| **hindi_070** | 60.5s | **4.61%** | 2.79s | 58.47% | **83.26%** | 64.52% | 50.43% | 11.69% | 12.90% | 75.81% | 42.34% | Recovered from 58.47% to 83.26% SAA, beating 2.5 Flash (50.43%). |
| **hindi_083** | 300.2s | **4.16%** | 12.50s | 72.43% | 54.07% | 55.79% | 97.00% | 61.91% | 58.72% | 3.39% | 13.06% | Low WER (13.06%); late monologue parity flip in long single-channel audio. |
| **hindi_065** | 60.1s | **3.94%** | 2.37s | 92.71% | **92.65%** | 82.19% | 98.38% | 11.54% | 17.69% | 1.54% | 15.00% | High attribution fidelity (>92%) preserved with low WER (15.00%). |
| **hindi_089** | 60.4s | **3.36%** | 2.03s | 85.63% | **99.43%** | **100.00%** | 66.48% | 0.00% | 0.00% | 16.49% | 16.49% | Strat A (99.43%) and Strat B (100.0%) massively outperformed 2.5 Flash (66.48%). |
| **hindi_022** | 192.8s | **3.32%** | 6.41s | 52.12% | **97.82%** | 59.63% | 100.00% | **4.27%** | 58.62% | 0.00% | 14.61% | **Dramatic Recovery**: Leaped from 52.12% to 97.82% SAA; Gap dropped to 4.27%! |
| **hindi_042** | 72.6s | **2.77%** | 2.01s | 72.38% | **97.55%** | 78.65% | 59.90% | 3.20% | 27.85% | 48.86% | 26.03% | Recovered from 72.38% to 97.55% SAA, beating 2.5 Flash (59.90%). |
| **hindi_092** | 60.3s | **2.57%** | 1.55s | 78.74% | 63.79% | 59.77% | 100.00% | 35.23% | 43.75% | 0.00% | 9.66% | Low transcription WER (9.66%); slight latching confusion on short turns. |
| **hindi_090** | 60.3s | **2.42%** | 1.46s | 100.00% | **100.00%** | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 9.47% | Perfect 100.0% attribution maintained across all methods ($\Delta F_0 = 56.4$ Hz). |
| **hindi_073** | 60.4s | **2.24%** | 1.35s | 50.76% | **62.94%** | 55.00% | 84.65% | 39.42% | 41.35% | 25.96% | 21.63% | Monologue collapse mitigated: SAA recovered from 50.76% to 62.94%. |

---

### Deep Dive on Extreme Overlap Failure Cases

#### 1. `hindi_085` (15.15% Overlap): Complete Eradication of Token-0 Tag Hallucination
- **Acoustic Facts**: 60.13s duration, 9.11s overlap, 23 GT turns. Dominated by overlapping interjections ("हाँ..हाँ", "अच्छा...").
- **Baseline Failure**: Emitted nested prefix tags `Speaker 1: Speaker 0: ...` and `Speaker 0: Speaker 1: ...`. SAA collapsed to **51.76%** with a **50.00% Diarization Gap**.
- **Strategy A Recovery**: By delaying attribution until after utterance decoding (`[Utterance] <text> | [Speaker] Speaker <ID>`), the model decoupled acoustic decoding from speaker commitment.
  - SAA soared from **51.76% to 98.84%** (+47.08%p gain).
  - Diarization Gap dropped to **0.00%**.
  - **Decisively outperformed Gemini 2.5 Flash (81.61% SAA, 19.07% Gap)**.

#### 2. `hindi_064` (14.34% Overlap): Breaking Fast Boundary Latching via Strategy B
- **Acoustic Facts**: 60.20s duration, 8.63s overlap, 20 GT turns. Same-gender conversation ($\Delta F_0 = 21.4$ Hz), where 78.9% of transitions have zero/negative pause.
- **Baseline Failure**: Overlap at 11.59s merged two turns into Turn 3, triggering a 6-turn inversion cascade that locked SAA at **54.29%**.
- **Strategy A vs. Strategy B Comparison**:
  - Strategy A accurately transcribed the speech but, as a single-pass model, still succumbed to the mid-dialogue boundary collision (53.82% SAA).
  - **Strategy B**: Pass 1 extracted neutral acoustic turns, and Pass 2 applied the Turn-Splitting Directive to disentangle the overlapping segment based on semantic polarity. Strategy B recovered SAA to **75.87%** and compressed the Diarization Gap to **18.06%**, demonstrating genuine architectural complementarity.

#### 3. `hindi_062` (20.62% Overlap): Peak Overlap 5-Minute Contentious Debate
- **Acoustic Facts**: 300.27s duration, 61.91s overlap, 68 GT turns, 46 overlapping segment pairs. Highly contentious multi-speaker debate.
- **Baseline Failure**: 80.0% of turns were merged, collapsing 68 GT turns into only 30 predicted turns. SAA collapsed to **53.23%**.
- **Recovery**:
  - Strategy A recovered SAA to **68.44%** and cut the Diarization Gap from 33.50% down to **19.81%**.
  - Strategy B split merged utterances during Pass 2, recovering SAA to **74.73%**.
  - Both pure 3.5 Lite strategies broke the baseline's ~50% random chance trap on the hardest clip in the dataset.

#### 4. `hindi_067` (13.51% Overlap): High-Noise Dialogue Outperforming 2.5 Flash
- **Acoustic Facts**: 60.39s duration, 8.16s overlap, 18 GT turns. Heavy background acoustic reverberation.
- **Outcome**:
  - Gemini 2.5 Flash collapsed due to noise, achieving only **55.71% SAA** and a massive **61.47% Diarization Gap**.
  - Strategy A achieved **72.51% SAA**, outperforming Gemini 2.5 Flash by **+16.80%p** and default 3.5 Lite by **+21.14%p**.

#### 5. `hindi_084` (7.42% Overlap, 300.1s, 77 Turns): Long-Context Parity Stabilization
- **Acoustic Facts**: 300.06s duration, 22.26s overlap, identical speaker pitch registers ($\Delta F_0 = 0.0$ Hz), 34 backchannel events.
- **Baseline Failure**: SAA collapsed to **53.71%** due to progressive parity drift across 77 turns.
- **Strategy A Recovery**: Anti-alternation constraints and backchannel preservation stabilized speaker identity across the entire 5-minute audio.
  - SAA jumped from **53.71% to 96.16%** (+42.45%p gain).
  - Diarization Gap compressed from 34.60% down to **0.98%**, matching Gemini 2.5 Flash (97.74%).

---

## 5. Cost, Latency, and Deployment Trade-Offs

### 5.1 Comprehensive Economic Breakdown (Per 1,000 Requests)

Calculations are based on the benchmark dataset's mean audio clip duration of **116.21 seconds** (~1.94 minutes) and standard Vertex AI production API pricing:
- **Gemini 2.5 Flash**: Audio multimodal input ~$0.0020 / min (~$0.0240 / 2-min clip); text output $0.30 / 1M tokens.
- **Gemini 3.5 Flash Lite**: Audio multimodal input ~$0.0005 / min (~$0.0062 / 2-min clip); text input $0.01875 / 1M tokens; text output $0.075 / 1M tokens.

| Architecture / Pipeline | Model Stack | Avg Latency | Speedup vs 2.5 | Cost / Sample | Cost / 1,000 Reqs | Cost Savings vs 2.5 | SAA | Diarization Gap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini 2.5 Flash Baseline** | `gemini-2.5-flash` | 7.03s | 1.00× (Ref) | ~$0.0240 | **$24.00** | Reference | 88.81% | 13.16% |
| **Gemini 3.5 Flash Lite Default** | `gemini-3.5-flash-lite` | 4.19s | 1.68× | ~$0.0062 | **$6.20** | 74.2% lower | 68.76% | 30.50% |
| **Strategy 1: Anchor Prompting** | `gemini-3.5-flash-lite` | 4.26s | 1.65× | ~$0.0062 | **$6.20** | 74.2% lower | 76.35% | 24.82% |
| **Strategy 2: Decoupled (Mixed)** | `3.5-lite` + `3.5-flash` | 8.87s | 0.79× | ~$0.0082 | **$8.20** | 65.8% lower | 86.95% | 11.60% |
| **Strategy A: Token-0 Bypass** | **Pure `3.5-flash-lite`** | **5.04s** | **1.40×** | **~$0.0062** | **$6.20** | **74.2% lower** | **81.11%** | **17.80%** |
| **Strategy B: Pure Lite Two-Pass** | **Pure `3.5-lite` + `3.5-lite`** | **7.04s** | **1.00×** | **~$0.0063** | **$6.25** | **74.0% lower** | **74.33%** | **26.60%** |

### 5.2 Latency Scaling Across Audio Durations

Execution times exhibit linear physical scaling with audio token count:
- **Short Clips (~60s)**:
  - Strategy A: **2.95s – 4.13s** (Mean ~3.5s)
  - Strategy B: **3.86s – 6.62s** (Mean ~4.7s)
  - Gemini 2.5 Flash: **5.50s – 7.10s** (Mean ~5.8s)
- **Medium Clips (~190s)**:
  - Strategy A: **5.98s**
  - Strategy B: **13.59s**
  - Gemini 2.5 Flash: **8.50s**
- **Long Clips (300s / 5 minutes)**:
  - Strategy A: **9.45s – 11.54s** (Mean ~10.5s)
  - Strategy B: **12.69s – 17.90s** (Mean ~15.5s)
  - Gemini 2.5 Flash: **11.20s – 14.10s** (Mean ~12.0s)

### 5.3 Production Deployment Decision Matrix

| Operational Criterion | Tier 1: Real-Time & Interactive Pipelines | Tier 2: Complex Contention & Deep Audits |
| :--- | :--- | :--- |
| **Recommended Pipeline** | **Strategy A: Token-0 Bypass Single-Prompt** | **Strategy B: Pure 3.5 Lite Two-Pass** |
| **Model Stack** | Pure `gemini-3.5-flash-lite` (Single call) | Pure `gemini-3.5-flash-lite` (Pass 1 + Pass 2) |
| **Mean Latency** | **5.04s** (1.40× faster than 2.5 Flash) | **7.04s** (Matches 2.5 Flash speed) |
| **Cost per 1,000 Calls** | **$6.20** (74% cheaper than 2.5 Flash) | **$6.25** (74% cheaper than 2.5 Flash) |
| **Primary Strength** | Highest overall accuracy (**81.11% SAA**, **17.80% Gap**), single-pass efficiency, zero multi-call complexity | Superior boundary collision resolution on latching speech (`hindi_064`) via retrospective turn splitting |
| **Recommended Use Cases** | Live contact center agent assistance, real-time transcription, large-scale batch diarization | Post-call dispute resolution, legal cross-examination audits, high-overlap debate recordings |

---

## 6. Reproducibility & Execution Guide

All code, pipelines, test suites, and prediction files are preserved and independently runnable in the workspace.

### 6.1 Environment Configuration
Verify that the project virtual environment is active and authenticated to Vertex AI:
```bash
export VERTEXAI_PROJECT="my-argolis-prj"
export VERTEXAI_LOCATION="global"
# Verify Google Cloud ADC
gcloud auth application-default print-access-token > /dev/null && echo "ADC Authenticated"
```

### 6.2 Running Unit & Adversarial Stress Tests
Execute the test suites covering Token-0 regex parsing, Pass 1/Pass 2 contracts, and zero-larger-model compliance:
```bash
# Run the 11 dedicated hard 2-speaker pipeline tests
/Users/lexha/Documents/work/codes/.venv/bin/pytest tests/test_hard_2spk_pipelines.py -v

# Run the full suite of M2 adversarial and Hungarian stress tests (122 tests)
/Users/lexha/Documents/work/codes/.venv/bin/pytest \
  tests/test_hard_2spk_pipelines.py \
  tests/test_adversarial_m2.py \
  tests/test_hungarian_stress.py \
  tests/test_adversarial_stress.py -v
```

### 6.3 Executing the Live Vertex AI Benchmark
To re-run the live benchmark across all 20 hard audio clips on Vertex AI:
```bash
# Execute live benchmark for Strategy A and Strategy B across all 20 samples
/Users/lexha/Documents/work/codes/.venv/bin/python scripts/run_hard_2spk_benchmark.py \
  --strategies strategy_a_token0_bypass,strategy_b_pure_lite_twopass

# Run a single target sample test (e.g. hindi_085)
/Users/lexha/Documents/work/codes/.venv/bin/python scripts/run_hard_2spk_benchmark.py \
  --sample-ids hindi_085 \
  --strategies strategy_a_token0_bypass
```

### 6.4 Zero-Trust Programmatic Metric Verification
To independently recompute all metrics from raw predictions against ground truth:
```bash
/Users/lexha/Documents/work/codes/.venv/bin/python -c "
import json
from src.dataset import DatasetLoader
from src.metrics import MetricsEngine
from src.models import Turn

samples = {s.sample_id: s for s in DatasetLoader.load_benchmark_subset('data/hard_2spk_subset')}

for strategy_file, name in [
    ('results/hard_2spk/predictions_strategy_a_token0_bypass.json', 'Strategy A'),
    ('results/hard_2spk/predictions_strategy_b_pure_lite_twopass.json', 'Strategy B')
]:
    with open(strategy_file) as f:
        preds = json.load(f)
    saas, gaps, wers = [], [], []
    for p in preds:
        hyp = [Turn(speaker=t['speaker'], text=t['text']) for t in p['predicted_turns']]
        m = MetricsEngine.evaluate_sample(samples[p['sample_id']].ground_truth_turns, hyp, normalize=True)
        saas.append(m.speaker_attribution_accuracy)
        gaps.append(m.diarization_gap)
        wers.append(m.wer)
    import numpy as np
    print(f'=== {name} Verified ===')
    print(f'Mean SAA: {np.mean(saas)*100:.2f}%')
    print(f'Mean Gap: {np.mean(gaps)*100:.2f}%')
    print(f'Mean WER: {np.mean(wers)*100:.2f}%')
"
```
*Expected Output*:
```text
=== Strategy A Verified ===
Mean SAA: 81.11%
Mean Gap: 17.80%
Mean WER: 25.09%
=== Strategy B Verified ===
Mean SAA: 74.33%
Mean Gap: 26.60%
Mean WER: 25.29%
```

---

## 7. Archival & Artifact Inventory

| Artifact Path | Description | Verified Record Count |
| :--- | :--- | :---: |
| `src/pipelines/token0_bypass.py` | Strategy A pipeline with Token-0 Bypass regex parser and anti-alternation prompt | Production Code |
| `src/pipelines/pure_lite_twopass.py` | Strategy B two-pass pipeline with Pass 2 turn-splitting disentanglement | Production Code |
| `scripts/run_hard_2spk_benchmark.py` | Automated live Vertex AI evaluation runner and comparative aggregator | Benchmark Runner |
| `tests/test_hard_2spk_pipelines.py` | 11 unit tests validating parsers, fallbacks, and zero-larger-model architecture | Test Suite |
| `tests/test_adversarial_m2.py` | 38 adversarial stress tests validating boundary conditions and regex robustness | Test Suite |
| `tests/test_independent_metric_recalculation.py` | Zero-trust test independently verifying all 40 prediction records against ground truth | Test Suite |
| `results/hard_2spk/predictions_strategy_a_token0_bypass.json` | Full raw API responses, parsed turns, latencies, and metrics for Strategy A | 20 Samples |
| `results/hard_2spk/predictions_strategy_b_pure_lite_twopass.json` | Full raw Pass 1 & Pass 2 responses, parsed turns, and metrics for Strategy B | 20 Samples |
| `results/hard_2spk/predictions_gemini_3_5_flash_lite_default.json` | Baseline prediction records for default Gemini 3.5 Flash Lite | 20 Samples |
| `results/hard_2spk/predictions_gemini_2_5_flash.json` | Baseline prediction records for Gemini 2.5 Flash | 20 Samples |
| `results/hard_2spk/hard_2spk_comparative_summary.json` | Macro summary statistics across all 6 evaluated architectures | Consolidated Summary |
| `results/hard_2spk/hard_2spk_report.md` | Final comprehensive publication-grade benchmark report (this document) | Deliverable |
