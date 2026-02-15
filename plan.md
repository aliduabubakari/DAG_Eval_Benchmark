# plan.md — Vision & Research Plan: Benchmarking Deterministic vs LLM-Based *Evaluation Architectures* for Workflow Orchestration Code

## 1) Vision

Build a **publishable**, **model-agnostic**, and **open-source** benchmark + leaderboard that evaluates **code evaluation systems (judges)** for workflow orchestration code (Airflow / Prefect / Dagster).  

The key idea is to benchmark **evaluation architectures** (deterministic, LLM-based, agentic, hybrid) against **objective ground truth**, rather than benchmarking individual LLMs.

This produces a durable research contribution: LLMs change, but the best ways to build reliable judges (evaluation architectures) remain broadly meaningful.

---

## 2) Publishable Positioning Statement

> We introduce an objective benchmark and open-source leaderboard for **automatic evaluation systems (judges)** of workflow orchestration code, using **runtime oracles and mutation-based labels** to compare deterministic, LLM, and agentic evaluation architectures in terms of accuracy, stability, and cost.

---

## 3) Why This is Novel (and Durable)

### Not the goal
- Not “LLM A beats LLM B at code review.”
- Not “state-of-the-art model leaderboard.”

### The goal
Benchmark and rank **judge systems** for orchestration code:
- Deterministic evaluators (tools + heuristics)
- LLM judges (single model)
- Agentic LLM judges (multi-agent, debate, verifier)
- Hybrid judges (tools propose → LLM judge filters)

### Durable contributions
- **Evaluator architectures** are the leaderboard unit (not the model).
- Ground truth is automated and objective:
  - Runtime oracle for platform compliance gates (PCT ground truth)
  - Mutation injection labels for static/orchestrator issue detection (SAT+orchestrator GT)
  - Tool-ensemble ground truth retained as secondary

---

## 4) The Benchmark Tasks (What We Measure)

### Task A — Gate Prediction (PCT-style)
**Objective:** Predict whether code will pass objective platform gates.

**Ground truth:** runtime oracle that actually executes checks:
- `syntax_valid` (AST parse)
- `imports_resolve` (import roots resolvable)
- `instantiates` (DAG/Flow/Job discoverable)
- `has_structure` (tasks exist; dependencies where applicable)

**Outputs from evaluators:** boolean gate predictions.

**Metrics:**
- Accuracy / F1 overall and per-gate
- Critical failures (false negatives on “instantiates” / “imports_resolve”) emphasized

---

### Task B — Issue Detection (SAT-like + Orchestrator-Specific)
**Objective:** Detect issues in a standardized taxonomy:
- `syntax`, `import`, `security`, `complexity`, `style`, `naming`, `documentation`, `error_handling`, `unused`, `undefined`, `best_practice`
- plus orchestration-specific:
  - `orchestrator_structure` (missing DAG/Flow/Job, missing tasks, missing deps)
  - `orchestrator_config` (invalid schedule, missing task_ids, etc.)

**Ground truth (primary):** mutation injection labels (objective)  
**Ground truth (secondary):** tool ensemble union (pylint, flake8, bandit, radon)

**Metrics:**
- Precision, Recall, F1 (overall and per-category)
- Severity-aware metrics (critical/major/minor)
- Calibration for confidence (LLM outputs), when available

---

## 5) Core Research Questions (Paper RQs)

### RQ1 — Gate prediction reliability
**Question:**  
How accurately can automated evaluators predict whether orchestration code will pass objective runtime compliance gates (imports, instantiation, structure)?

**Ground truth:** runtime oracle  
**Metrics:** per-gate accuracy/F1, false negative rate

---

### RQ2 — Architecture effectiveness at issue detection under objective labels
**Question:**  
Among deterministic, single-LLM, multi-agent, and hybrid architectures, which judge architecture best detects known defects while minimizing false positives?

**Ground truth:** mutation labels (primary), tool ensemble (secondary)  
**Metrics:** precision/recall/F1 by category/severity

---

### RQ3 — Cost–quality–stability trade-offs
**Question:**  
How do judge architectures compare on the Pareto frontier of accuracy vs cost vs stability (variance across runs)?

**Metrics:**
- Variance of F1 / gate predictions across repeated runs
- Token usage
- Runtime latency

---

## 6) Key Gap in Current Benchmark (and How to Fix It)

### Current gap / vulnerability
The SAT “ground truth” is based on static tools (pylint, flake8, bandit, radon).  
If the deterministic baseline uses the same tools, reviewers may argue:

> “The benchmark rewards imitation of tool outputs; it’s circular.”

Even if the baseline uses a subset or different configs, this is still a methodological weakness.

### Fix (high priority): Mutation Injection Ground Truth
Add a **mutation engine** that injects known defects into pipeline files.  
Because we generate the defect, we know the label without human annotation.

Examples of mutation types:
- Syntax errors (missing colon, missing parenthesis)
- Import failures (rename import module to nonexistent)
- Undefined variable insertion
- Hardcoded secret insertion (non-placeholder)
- Bare `except:`
- Remove DAG/Flow/Job decorator/context
- Remove all task definitions
- Remove dependencies between tasks
- Invalid schedule patterns (orchestrator-specific)
- Division-by-zero or obvious runtime hazards

This provides:
- Objective and label-perfect ground truth for many categories
- Ground truth for orchestrator-specific issues that tools won’t cover

**Outcome:** eliminates “tool imitation” criticism and strengthens novelty.

---

## 7) Evaluation Architectures to Compare (Leaderboard Units)

### Baselines
- **D0: Deterministic tool baseline**  
  Runs tools and heuristic gate checks.

### LLM-based
- **J1: Single LLM judge**  
  One prompt generates issues + gates.

### Agentic / system-level
- **J2: Multi-agent judge**  
  Specialized agents:
  - Security agent
  - Quality/style agent
  - Orchestrator expert agent (also outputs gates)
  Aggregation and deduplication.

### Hybrid
- **J3: Tools propose → LLM judge filters**
  Tools generate candidate issues; a strong judge LLM:
  - filters false positives
  - adjusts severity/category
  - adds conservative “missed obvious” issues
  - predicts gates

---

## 8) Optional Additional “Agentic” Architectures (for stronger novelty)

Pick 1–2 to include if time permits:

### A) Critic–Defender / Debate Judge
- Agent A proposes issues + gates  
- Agent B challenges each item (“prove using evidence from code”)  
- Final adjudicator consolidates

Expected benefit: reduced hallucinations; improved precision.

### B) Verifier Judge (LLM + deterministic proof)
- LLM proposes issues
- A verifier checks them using:
  - AST checks
  - importlib spec checks
  - string/regex validation
  - minimal runtime-safe checks
- Only verified issues are kept

Expected benefit: strong precision, more “engineering-grade” judge.

### C) Self-consistency within budget
- Run the same judge multiple times with slight prompt variations
- majority vote for gates; dedupe issues
- compare marginal F1 gains vs token cost

---

## 9) Leaderboard Design (Open Source)

### What is ranked
Rank **evaluator submissions**, i.e. judge systems:
- deterministic
- LLM single
- multi-agent
- hybrid
- verifier/debate variants (optional)

Avoid ranking “LLMs” directly as the primary headline.

### What the leaderboard reports
- Gate accuracy (overall + per-gate)
- Issue detection precision/recall/F1 (overall + per-category)
- Stability score (variance across runs)
- Token usage and latency
- Pareto frontier plots (accuracy vs cost vs stability)

### Submission format
At minimum:
- Evaluator outputs JSON in a standard schema (`EvaluatorOutput`)

Potential future enhancement:
- Containerized submissions for reproducibility

---

## 10) Metrics Summary

### For gates
- Overall gate accuracy
- Per-gate accuracy/F1
- False negative rate on critical gates (imports, instantiation)

### For issue detection
- Precision / recall / F1 overall
- Per-category metrics
- Severity-aware breakdown
- Calibration (confidence vs correctness) for LLM-based evaluators

### For operational constraints
- Execution time (ms)
- Tokens (input/output)
- Estimated cost ($), if available
- Stability across repeated trials (variance)

---

## 11) Experimental Protocol (Recommended)

### Primary comparisons
Compare evaluation architectures:
- Deterministic vs Single LLM vs Multi-agent vs Hybrid
- Optionally include Verifier or Debate judge

### Secondary comparisons
Run each architecture on a small set of representative models:
- 2–4 DeepInfra models (e.g., Qwen3-Coder, DeepSeek-V3, Llama-3.3-70B, etc.)
Report:
- Architecture performance averaged across models
- Architecture variance across models

This makes the study model-agnostic and future-proof.

---

## 12) Immediate Next Steps (Implementation Roadmap)

### Step 1 — Lock the benchmark tasks + schemas
- Confirm gate set for runtime oracle (PCT)
- Confirm issue taxonomy + formatting (SAT-like + orchestrator-specific)
- Ensure all evaluators emit the same schema

### Step 2 — Add mutation engine + mutation ground truth
- Implement mutation types and label generation
- Store:
  - original file
  - mutated file
  - mutation metadata (category, severity, expected gate effects)
- This is the key novelty enabler

### Step 3 — Expand benchmarking metrics & reporting
- Add per-category and per-orchestrator breakdown
- Add stability measurement (repeat LLM-based runs N times)
- Add cost/time aggregation

### Step 4 — Run first methodology study
- Compare: deterministic, single LLM, multi-agent, hybrid
- Use 2–4 models to show architecture ranking stability

### Step 5 — Release as open-source leaderboard
- CLI for running benchmark locally
- Output: JSON + CSV + plots
- Publish docs + reproducibility guide
- Optional GitHub Pages leaderboard

---

## 13) Intended Paper / Report Outline (High-Level)

1. Introduction: why evaluating orchestration code judges matters  
2. Related work: code benchmarks vs evaluator benchmarks  
3. Benchmark definition: tasks, schema, dataset  
4. Ground truth construction:
   - runtime oracle gates
   - mutation-based issue labels (primary)
   - tool ensemble issues (secondary)
5. Evaluator architectures:
   - deterministic baseline
   - single LLM judge
   - multi-agent judge
   - hybrid propose→judge
   - optional verifier/debate
6. Experimental results:
   - RQ1 gate prediction
   - RQ2 issue detection
   - RQ3 cost/stability
7. Discussion:
   - hallucination patterns
   - what tools miss vs what LLMs miss
   - design recommendations for judge systems
8. Open-source leaderboard release

---

## 14) Success Criteria

The project is successful if:
- Methodology-level conclusions are stable across models
- Hybrid/agentic architectures demonstrably improve reliability (precision/stability) under cost constraints
- The benchmark and leaderboard are reproducible and useful for the community
- The paper can claim novelty via objective ground truth (runtime oracle + mutation labels)

---