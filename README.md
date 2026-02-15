# DAG Eval Benchmark

![status](https://img.shields.io/badge/status-research--in--progress-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.10+-orange)

**Benchmarking deterministic, LLM, agentic, and hybrid evaluation architectures for workflow orchestration code**

A model-agnostic, objective benchmark and open leaderboard for automatic code evaluation systems (judges) for Airflow, Prefect, and Dagster pipelines.

---

## ✨ Motivation

Modern LLMs can review code — but:
- LLMs change rapidly
- Model leaderboards become obsolete
- Reliability matters more than raw model capability

This project benchmarks **evaluation architectures**, not models.

We answer: *What is the most reliable way to build an automated judge for orchestration code?*

---

## 🧠 Core Idea

We evaluate **judge systems**, not LLMs.

Each submission is an evaluation architecture:
- **Deterministic** (tools + rules)
- **Single-LLM judge**
- **Multi-agent judge**
- **Hybrid** (tools → LLM filter)
- **Verifier / debate systems** (optional)

All are measured against objective ground truth.

---

## 🏗 Benchmark Tasks

### Task A — Gate Prediction (Runtime Compliance)

Predict whether code passes platform-critical checks:
- `syntax_valid`
- `imports_resolve`
- `instantiates`
- `has_structure`

**Ground truth:** Runtime oracle that actually executes these checks.

*Why it matters:* These gates determine whether a DAG/Flow/Job is deployable.

### Task B — Issue Detection

Detect standardized issue categories:

**General code quality**
- syntax
- imports
- security
- complexity
- style
- naming
- documentation
- error_handling
- unused
- undefined
- best_practice

**Orchestrator-specific**
- orchestrator_structure
- orchestrator_config

**Ground truth:** 
- Primary → mutation injection labels
- Secondary → static tool ensemble

---

## 🧬 Ground Truth: Mutation Engine

We automatically inject known defects into pipelines.

This gives:
- Objective labels
- Perfect category mapping
- Orchestrator-specific truth
- No human annotation
- No "tool imitation" criticism

**Example mutations:**
- Remove DAG/Flow/Job definition
- Break imports
- Insert undefined variables
- Hardcode secrets
- Remove task dependencies
- Invalid schedules
- Bare except

---

## 📊 Metrics

### Gate prediction
- Accuracy / F1 (overall & per-gate)
- Critical false negative rate

### Issue detection
- Precision / Recall / F1
- Per-category performance
- Severity-aware scoring

### Operational metrics
- Cost (tokens / $)
- Latency
- Stability (variance across runs)

---

## 🏆 Leaderboard Philosophy

We rank:

**Evaluator architectures** — not models.

This makes results:
- Durable
- Model-agnostic
- Scientifically meaningful

---

## 🧪 Research Questions

**RQ1 — Gate reliability**  
How accurately can evaluators predict runtime compliance?

**RQ2 — Defect detection under objective labels**  
Which architecture best detects real defects with minimal false positives?

**RQ3 — Cost–quality–stability trade-offs**  
What sits on the Pareto frontier?

---

## 🏛 Supported Orchestrators

- Apache Airflow
- Prefect
- Dagster

---

## 📂 Project Structure

```
DAG_Eval_Benchmark/
│
├── benchmark/
│   ├── runtime_oracle/        # Ground truth execution engine
│   ├── mutation_engine/        # Defect injection system
│   ├── datasets/               # Benchmark datasets
│
├── evaluators/
│   ├── deterministic/          # Rule-based evaluators
│   ├── single_llm/             # Single LLM judges
│   ├── multi_agent/            # Multi-agent systems
│   ├── hybrid/                  # Hybrid approaches
│
├── schemas/
│   └── evaluator_output.json   # Output schema specification
│
├── scripts/
│   └── run_benchmark.py        # Main entry point
│
├── results/                     # Benchmark results
└── leaderboard/                 # Leaderboard data
```

---

## ⚙️ Evaluator Output Schema

All evaluators must emit:

```json
{
  "gates": {
    "syntax_valid": true,
    "imports_resolve": true,
    "instantiates": false,
    "has_structure": true
  },
  "issues": [
    {
      "category": "security",
      "severity": "critical",
      "message": "Hardcoded password detected",
      "location": "line 42",
      "confidence": 0.91
    }
  ]
}
```

---

## 🚀 Quick Start

### 1️⃣ Install

```bash
git clone https://github.com/<your-username>/DAG_Eval_Benchmark.git
cd DAG_Eval_Benchmark

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Run the benchmark

```bash
python scripts/run_benchmark.py \
  --evaluator deterministic \
  --orchestrator airflow
```

### 3️⃣ View results

Outputs include:
- JSON results
- CSV summaries
- Performance plots
- Leaderboard entry

---

## 🧩 Implementing a Custom Evaluator

1. Create a new directory: `evaluators/my_evaluator/`
2. Implement the evaluation function:

```python
def evaluate(file_path: str) -> EvaluatorOutput:
    # Your evaluation logic here
    ...
    return EvaluatorOutput(...)
```

3. Run your evaluator:

```bash
python scripts/run_benchmark.py --evaluator my_evaluator
```

---

## 📈 Leaderboard Metrics

Each submission reports:
- Gate accuracy
- Issue detection F1
- Stability score (variance across runs)
- Token usage
- Runtime latency

---

## 🔬 Reproducibility

Planned features:
- Fixed datasets with versioning
- Mutation metadata preservation
- Deterministic runtime oracle
- Containerized submissions for isolation

---

## 🗺 Roadmap

- [x] Runtime oracle (PCT gates)
- [x] Mutation engine
- [ ] Deterministic baseline
- [ ] Single-LLM judge
- [ ] Multi-agent judge
- [ ] Hybrid judge
- [ ] Stability runner
- [ ] Cost tracking
- [ ] Public leaderboard

---

## 📜 Intended Contribution

This benchmark enables:
- Research on automated code judges
- Reliable LLM evaluation systems
- Orchestration-specific code quality analysis

---

## 🤝 Contributing

We welcome contributions in:
- New evaluator architectures
- New mutation types
- Additional orchestrator support
- Reproducibility improvements
- Documentation and examples

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Citation

```bibtex
@misc{dag_eval_benchmark,
  title={Benchmarking Evaluation Architectures for Workflow Orchestration Code},
  author={DAG Eval Benchmark Contributors},
  year={2026},
  publisher={GitHub},
  url={https://github.com/<your-username>/DAG_Eval_Benchmark}
}
```

---

## 🙏 Acknowledgments

Built with inspiration from the LLM-as-a-judge research community and workflow orchestration tooling ecosystems.