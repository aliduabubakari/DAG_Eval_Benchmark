# `plan_v3.md` — Implementation-Faithful Research Plan (Methodology-First)

## 0) One-Sentence Summary

We benchmark **evaluation architectures (judge systems)** for workflow orchestration code (Airflow/Prefect/Dagster) using **objective runtime oracle gates (PCT)** and **mutation-based injected defect labels (GT1)**, measuring **accuracy, localization, cost, and stability** at 1000+ dataset scale.

---

## 1) Motivation (What Problem We Solve)

Generated or human-written orchestration code (DAGs/flows/jobs) must often pass practical “gates” before deployment:

- Does it parse?
- Do imports resolve in the evaluation environment?
- Does the orchestrator discover an executable object?
- Does it contain meaningful structure (tasks/dependencies)?

At the same time, teams want automated issue detection (security, undefined vars, poor error handling, etc.) to triage risk.

**The core research objective is not to rank LLMs.**  
It is to determine how to build **reliable evaluation systems (judges)** with objective ground truth under real constraints (cost, latency, stability).

---

## 2) What We Benchmark (Evaluator Architectures = Leaderboard Units)

We evaluate **five architectures**:

1) **`deterministic_tools_v1`**  
   Runs static tools (pylint/flake8/bandit/radon) + heuristic gate prediction.

2) **`deterministic_heuristic_v1`**  
   No external tools. AST + regex heuristics for issues and gates. Fast, stable, non-circular vs GT2.

3) **`llm_single_<model>`**  
   One LLM produces issues + gates in a single structured output.

4) **`llm_multi_agent_<model>`**  
   Security agent + quality agent + orchestrator agent; aggregate issues; gates from orchestrator agent.

5) **`hybrid_<model>`** (heuristic proposer → LLM judge)  
   Deterministic proposer suggests candidate issues; LLM judge filters/adjusts/adds issues + predicts gates.

**Future work** (explicitly not in this paper): verifier, debate, self-consistency ensembles.

---

## 3) Benchmark Tasks / Tracks (What We Measure)

### Track A (Primary): PCT Gate Prediction
**Goal:** Predict objective runtime gate outcomes.

**Ground truth:** stored runtime oracle outputs (no recomputation during benchmarking), under standardized evaluation environments.

Gates:
- `syntax_valid`
- `imports_resolve`
- `instantiates`
- `has_structure`

**Metrics:**
- overall gate accuracy
- all-gates-correct accuracy
- per-gate accuracy
- (recommended) critical-gate FN rate: false negatives on `imports_resolve` and `instantiates`

---

### Track B (Primary): GT1 Injected Defect Detection + Localization
**Goal:** Detect injected defects with known labels (objective, no humans).

**Ground truth:** mutation label JSON (category + injected_line).

**Metrics:**
- injection recall (did evaluator output at least one issue in injected category?)
- localization rate (did evaluator locate it within ±K lines of injected_line?)
- per-mutation-id breakdown (recall + localization)

Notes:
- Localization is meaningful only if the evaluator emits consistent integer line numbers.  
  We enforce this via line-numbered prompts + postprocessing.

---

### Track C (Secondary / Appendix): GT2 Tool Alignment
**Goal:** Measure agreement with static tool ensemble outputs.

**Ground truth:** tool ensemble JSON loaded from disk (no recomputation).

**Metrics:**
- strict F1 (category + line-sensitive)
- relaxed F1 (category-only or line-insensitive)
- per-category breakdown

**Interpretation rule:** Track C is “agreement with tools,” not “truth,” and is not used as the headline leaderboard ranking.

---

## 4) Dataset and Scale (1000+)

### 4.1 Dataset composition goals
We will benchmark at 1000+ scale with balanced representation across:
- orchestrators: Airflow / Prefect / Dagster
- mutation types (GT1): syntax/import/undefined/unused/security/error_handling/orchestrator_structure/orchestrator_config

### 4.2 Recommended “1000+” target sizes (practical)
To keep runtime manageable while achieving statistical stability:

**Option A (recommended):**
- 500 originals
- 500 mutated (GT1)
Total items evaluated per run: **1000**

**Option B (larger):**
- 1000 originals
- 1000 mutated
Total: **2000**

Because LLM evaluators are expensive, Option A is usually the best “paper-scale” baseline; Option B becomes a scalability study or appendix.

### 4.3 Stratified sampling (important for methodology claims)
We should not sample “randomly” without constraints because mutation distribution may drift.

We will stratify:
- equal number of items per orchestrator
- equal number of mutated items per mutation_id (or as close as possible)

This ensures the per-mutation results reflect methodology differences, not dataset imbalance.

---

## 5) Oracle Environment Profile (Reproducibility)

We use a **heavy/high coverage environment profile** so the oracle can evaluate more files:

- pandas
- redis
- psycopg2-binary
- snowflake-connector-python
- plus dataset-derived extras and airflow providers (installed under airflow constraints)

We store:
- pip freeze snapshots per venv
- oracle ground truth caches on disk

The benchmark runner **loads oracle GT from disk** (no oracle recomputation), so results are reproducible once GT artifacts are produced.

---

## 6) Robustness Engineering (Why This Benchmark is Leaderboard-Ready)

LLM evaluators can fail structurally (invalid JSON, missing fields, non-integer lines). To prevent “benchmark fragility,” our evaluators include:

- **line-numbered prompts** (issues reference exact integer lines)
- **strict evidence constraints** (`evidence=null` unless single-line safe)
- **robust JSON parsing**:
  - strip trailing commas
  - one repair call to the same model if JSON parse fails
  - repair tokens included in total tokens
- consistent token accounting through the shared `LLMProvider`

This makes architecture comparisons meaningful at 1000+ scale.

---

## 7) Research Questions (Three RQs, Methodology-Focused)

### RQ1 — Gate prediction reliability (Track A)
**Which evaluation architecture most reliably predicts objective runtime gates for orchestration code?**

Report:
- overall gate accuracy
- all-gates-correct accuracy
- per-gate accuracy
- critical-gate FN rate (`imports_resolve`, `instantiates`)

Methodology angle:
- Do LLM judges help with orchestration semantics, or do deterministic heuristics dominate?

---

### RQ2 — Injected defect detection and localization (Track B)
**Which architecture best detects injected defects and localizes them reliably?**

Report:
- injection recall
- localization rate at ±K (K=2 default; optionally show K=0,2,5)
- per mutation_id recall/localization table

Methodology angle:
- Do agentic/hybrid architectures improve localization?
- Where do LLMs help beyond deterministic heuristics?

---

### RQ3 — Cost, latency, and stability (cross-track)
**What is the cost–quality–stability trade-off across judge architectures?**

Report:
- tokens/item (mean/std)
- latency/item (mean/std)
- stability across runs (std dev for Track A + Track B)
- Pareto frontiers:
  - gate accuracy vs tokens/item
  - GT1 recall/localization vs tokens/item

Methodology angle:
- When does extra complexity (multi-agent/hybrid) justify itself?

---

## 8) Experimental Protocol (What We Will Run)

### 8.1 Main experiments (single model, methodology-first)
Use one representative model (e.g., `Qwen3-Coder`) across all LLM-based architectures.

Evaluators:
- deterministic_tools_v1
- deterministic_heuristic_v1
- llm_single_Qwen3-Coder
- llm_multi_agent_Qwen3-Coder
- hybrid_heuristic_proposer_Qwen3-Coder

Runs per file:
- deterministic: 1 (variance ≈ 0)
- LLM/hybrid: 3 (stability measurement)

### 8.2 Reported statistics
Because 1000+ items makes tiny differences statistically significant, we must report uncertainty:

- For gate accuracy: Wilson interval or bootstrap CI
- For GT1 recall/localization: bootstrap CI across mutation items
- For stability: std dev across repeated runs + distribution plots

(We can implement bootstrap as a post-processing script on the benchmark JSON.)

### 8.3 Secondary (optional) robustness study across models
Not required for the main paper, but can be an appendix:
- repeat with 1–2 additional DeepInfra models
- show architecture ranking correlation (Spearman) across models

---

## 9) Tables/Figures (Directly Supported by Current Output JSON)

### Table 1 (Main): Primary leaderboard (Tracks A + B + Cost/Stability)
Rows = architectures  
Columns:
- Track A overall gate accuracy (mean ± std)
- Track A all-gates-correct (mean ± std)
- Track B injection recall (mean ± std)
- Track B localization rate (mean ± std)
- tokens/item (mean ± std)
- latency/item (mean ± std)

### Table 2 (Main): Per-gate breakdown (Track A)
- syntax_valid
- imports_resolve
- instantiates
- has_structure

### Table 3 (Main): Mutation-type breakdown (Track B)
Per mutation_id:
- recall
- localization rate
This is the most diagnostic methodology table.

### Figure 1 (Main): Pareto frontier
- Gate accuracy vs tokens/item
- Injection recall vs tokens/item
Optionally overlay latency.

### Appendix Table A1 (Track C): Tool alignment
- strict F1, relaxed F1
- per-category breakdown
Interpretation statement included: “agreement, not truth.”

---

## 10) Paper Outline (Aligned to Implementation)

1. **Introduction**
   - why orchestration judges matter
   - why evaluator architectures vs model leaderboard

2. **Benchmark Definition**
   - tracks A/B/C
   - evaluator output schema
   - dataset artifacts

3. **Ground Truth Construction**
   - runtime oracle (stored)
   - mutation GT1 (objective)
   - tool ensemble GT2 (secondary)

4. **Evaluator Architectures**
   - deterministic_tools_v1
   - deterministic_heuristic_v1
   - llm_single
   - llm_multi_agent
   - hybrid propose→judge
   - robustness engineering (JSON repair, line-numbering)

5. **Experimental Results**
   - RQ1 Track A
   - RQ2 Track B
   - RQ3 cost/stability + Pareto

6. **Discussion**
   - when heuristics dominate
   - where LLMs add value or fail
   - limitations: oracle environment profile, GT2 non-truth

7. **Open-source Release**
   - reproducibility
   - how to submit evaluators
   - leaderboard design

8. **Conclusion + Future Work**
   - verifier/debate/self-consistency as extensions