# T-05 — Retrieval-Augmented Generation and the Study of Hallucination

## Phase 1 Report — YadYar Lite (Lightweight ML Learning Assistant)

**Topic:** T-05 — Retrieval-Augmented Generation and the Study of Hallucination
**Phase:** Phase 1 only
**Deliverable:** This report + the notebook `T05_Phase1_LowCode_RAG.ipynb`
**Companion notebook:** `T05_Phase1_LowCode_RAG.ipynb`

---

## 0. Project Overview

This project sets up a **lightweight Retrieval-Augmented Generation (RAG)**
pipeline on a small subset of **SQuAD 2.0** and prepares the ground for
studying **hallucination** behaviour in Phase 2. The pipeline has three
small, off-the-shelf parts: a dense retriever
(`sentence-transformers/all-MiniLM-L6-v2` + FAISS flat L2), a small
instruction-tuned generator (`google/flan-t5-small`), and a No-RAG baseline
that uses the same generator without retrieved context.

This is **Phase 1 only**. We deliver a narrowed question, a documented
dataset, a reproducible baseline, an evaluation plan, and a simple analysis
plan. We do **not** deliver full evaluation results, final error analysis, a
demo, or future-integration notes — those belong to Phase 2. We also do
**not** train any model from scratch, perform fine-tuning, build a UI, or
run formal statistical testing; all of these are explicitly out of scope
for Phase 1 according to the project specification.

The goal of this phase is to make sure the baseline is correctly wired,
reproducible, and easy to defend orally, so that Phase 2 can focus on
actually measuring hallucination behaviour rather than on debugging
infrastructure.

---

## 1. Narrow Project Question

> **Can a lightweight RAG pipeline using dense retrieval reduce unsupported or
> hallucinated answers compared with a no-retrieval baseline on a small subset
> of SQuAD 2.0?**

This question is deliberately narrow. It compares two modes of the same
generator model (`google/flan-t5-small`) on the same small subset of SQuAD 2.0:

- **RAG mode** — the generator is conditioned on a retrieved context.
- **No-RAG baseline** — the generator sees only the question.

The comparison is therefore caused only by the presence or absence of the
retrieved context, not by a different model, different prompt template family,
or different training procedure. The dependent variable is the rate of
unsupported or hallucinated answers, which in Phase 2 will be approximated by
the **Abstention rate** on unanswerable questions and by a small **manual
hallucination rate** on a sample of RAG answers.

The question is feasible within the course scope for three reasons. First, no
model is trained from scratch and no fine-tuning is performed; only pretrained
off-the-shelf models are used. Second, the dataset is small (~300 contexts,
~80 questions), so the full evaluation in Phase 2 will run in a few minutes on
a free Colab runtime. Third, the two failure regimes that the question targets
(absent-evidence hallucination when the retriever misses, and ignored-evidence
hallucination when the retriever hits but the generator still writes past the
context) are exactly the two regimes named in the T-05 catalogue entry, so the
question is well aligned with the topic.

---

## 2. Dataset Description

| Field | Value |
|---|---|
| Dataset | **SQuAD 2.0** (Stanford Question Answering Dataset v2) |
| Source | Hugging Face Datasets: `rajpurkar/squad_v2` |
| Split used | `train` (a small deterministic subset of it) |
| Subset size | ~300 unique contexts, ~80 questions (mixed answerable / unanswerable) |

### Important fields

| Field | Meaning |
|---|---|
| `context` | The passage (paragraph) from which the answer should be extracted |
| `question` | The student-style question |
| `answers["text"]` | A list of gold answer strings; may be **empty** → unanswerable |
| `answers["answer_start"]` | Character offsets of each gold answer in the context |

### Detecting unanswerable questions

In the Hugging Face version of `squad_v2`, the `is_impossible` column is **not**
guaranteed to exist. The robust way to detect an unanswerable question is:

```python
is_unanswerable = len(example["answers"]["text"]) == 0
```

This convention is used everywhere in the notebook (subset construction,
sanity check, and the planned Phase 2 metrics).

### Why SQuAD 2.0 fits T-05

SQuAD 2.0 is public, small, well-documented, and one of the most widely used
benchmarks for QA and RAG studies. It contains both answerable and
unanswerable questions, which lets us study hallucination under two failure
regimes that the T-05 catalogue names explicitly: (a) the answer is in the
context but the model misses it (ignored-evidence hallucination), and (b) the
answer is **not** in the context (absent-evidence hallucination). Each context
is a single short paragraph, so both the embedding model (`all-MiniLM-L6-v2`)
and the generator (`flan-t5-small`) fit comfortably in a free Colab runtime
with no GPU required. SQuAD is also listed in the project catalogue as one of
the suggested starting points for T-05.

### Subset construction (deterministic)

1. The `train` split is shuffled with a fixed seed (`SEED = 42`).
2. The first ~300 unique contexts are collected.
3. From the rows whose context is in that set, ~80 questions are selected with
   at least ~25% unanswerable, so Phase 2 can measure abstention rate on a
   non-trivial number of unanswerable questions.

---

## 3. Baseline Setup

### 3.1 Components

| Role | Tool / Model | Notes |
|---|---|---|
| Dataset loader | `datasets` (Hugging Face) | Loads `rajpurkar/squad_v2` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, fast, CPU-friendly |
| Vector search | `faiss-cpu` (flat L2 index) | Simplest explainable ANN baseline |
| Generator (RAG) | `google/flan-t5-small` | ~80M params, runs on free Colab |
| No-RAG baseline | Same `flan-t5-small` **without** retrieved context | Apples-to-apples comparison |

Both RAG and No-RAG use the same generator, so any difference in answer
quality can be attributed to the retrieved context rather than to a different
model. Retrieval uses a flat L2 index (no IVF, no PQ) which is exact and easy
to explain in an oral defence: given a query vector, FAISS computes the L2
distance to every stored context vector and returns the k smallest.

### 3.2 Reproducibility

- Random seed is fixed (`SEED = 42`) for both `random` and `numpy.random`.
- Subset selection is deterministic given the seed.
- Models are pinned by their Hugging Face IDs (`all-MiniLM-L6-v2`,
  `flan-t5-small`); the FAISS index type is pinned (`IndexFlatL2`).
- Generation is greedy (`do_sample=False, num_beams=1`) so the same prompt
  always produces the same output.
- The whole pipeline runs end-to-end on a free Colab CPU runtime in a few
  minutes (one-time download of the SQuAD subset, the embedding model, and the
  FLAN-T5-small weights).

### 3.3 Pipeline (cell-by-cell summary)

The notebook contains these code cells (markdown headings explain each one):

1. Install dependencies (`datasets`, `sentence-transformers`, `faiss-cpu`,
   `transformers`).
2. Imports + fixed seed.
3. Load `rajpurkar/squad_v2`, split `train`.
4. Build the small balanced subset (~300 contexts, ~80 questions, ~25%
   unanswerable).
5. Embed the contexts with `all-MiniLM-L6-v2` (384-dim, L2-normalised).
6. Build a FAISS `IndexFlatL2` and add the context vectors.
7. `retrieve(question, k)` — embeds the question, searches the index, returns
   the top-k contexts.
8. Load `flan-t5-small` tokenizer and model.
9. `generate_with_rag(question)` — retrieves top-1 context, prompts FLAN-T5
   with `Context: ... Question: ... Answer:`, allows abstention.
10. `generate_without_rag(question)` — prompts FLAN-T5 with only the question.
11. Sanity check on 3 sample questions (1 answerable, 1 unanswerable,
    1 answerable).

### 3.4 Sanity check (Phase 1 only)

The 3-question smoke test confirms that the pipeline runs end-to-end, that
both generation functions return a string, and that the No-RAG baseline
behaves qualitatively differently from the RAG mode (e.g. it tends to invent
short answers when the question asks for a fact it does not know). This is
**not** evaluation — it is just evidence that the baseline is correctly wired
and ready for Phase 2.

---

## 4. Evaluation Plan (for Phase 2)

Phase 1 only requires a plan. The following metrics will be computed in
Phase 2 on the same subset.

### 4.1 Retrieval metrics

| Metric | What it measures |
|---|---|
| **Recall@5** | Of the gold context, how often is it inside the top-5 retrieved contexts? |
| **MRR** (Mean Reciprocal Rank) | How high is the gold context in the retrieved list, on average (1/rank)? |

A context is considered "gold" if its string equals the row's `context` field
in SQuAD. Both metrics are standard for IR evaluation and are simple to
compute on a subset of ~80 questions.

### 4.2 Generation metrics

| Metric | What it measures |
|---|---|
| **Exact Match (EM)** | Fraction of answers that exactly match the gold (after normalisation: lowercase, strip articles and punctuation) |
| **Token F1** | Token-level F1 between predicted and gold answer — gives partial credit for partially correct answers |
| **ROUGE-L** | Longest common subsequence overlap between prediction and gold |

EM and Token F1 are the canonical SQuAD metrics; ROUGE-L is included as a
softer measure that is informative when EM is harsh (e.g. paraphrased answers).

### 4.3 Hallucination-specific metrics

| Metric | What it measures |
|---|---|
| **Abstention rate** | On unanswerable questions, the fraction where the model says "I don't know" (substring rule on the lower-cased answer) instead of inventing an answer |
| **Hallucination rate** (manual, small sample) | On a sample of ~30 RAG answers, the fraction judged by a human as unsupported by the retrieved context |

The abstention rate is the primary quantitative signal for hallucination in
Phase 2: a model that abstains on unanswerable questions is, by construction,
not hallucinating. The manual hallucination rate is a small qualitative check
on the answerable subset, where a hallucinated answer is one that is not
supported by the retrieved context even if it sounds plausible.

### 4.4 Two-mode comparison

Every metric will be reported in two columns: **RAG** and **No-RAG**. The
difference between the two columns is the answer to the narrow project
question.

---

## 5. Simple Analysis Plan (for Phase 2)

We will group errors into **four categories**. The first three come directly
from the T-05 catalogue entry; the fourth is a data slice that is easy to
compute on SQuAD.

### 5.1 Error categories

| # | Error type | Definition |
|---|---|---|
| 1 | **Retrieval miss / absent-evidence hallucination** | The gold context was not in top-k, so the generator either hallucinates or abstains. |
| 2 | **Ignored-evidence hallucination** | The gold context *was* retrieved, but the generator still produced an answer unsupported by it. |
| 3 | **Unanswerable failure** | The question is unanswerable, but the model produced a confident (hallucinated) answer instead of abstaining. |
| 4 | **Short question vs long question** | Performance breakdown by question length (≤ 8 tokens vs > 8 tokens) — short questions may retrieve more weakly. |

### 5.2 Error-analysis template

We will fill in the following small table for ~20 representative failures in
Phase 2. Each row will be short (one sentence per cell).

| Question | Gold answer | Retrieved context correct? | RAG answer | No-RAG answer | Error type | Explanation |
|---|---|---|---|---|---|---|
| _(filled in Phase 2)_ | | | | | | |

The point of this table is qualitative understanding of *where* the baseline
breaks, not formal statistics. The Phase 2 report will also include a small
breakdown of EM and Token F1 by question length (≤ 8 tokens vs > 8 tokens)
and by answerable vs unanswerable.

---

## 6. Scope and Limitations

- This is **Phase 1 only**. We deliver a baseline and an evaluation plan, not
  final numbers.
- The subset is **very small** (~300 contexts, ~80 questions). Numbers from
  Phase 2 will be indicative, not statistically rigorous. No formal hypothesis
  testing is planned.
- The generator is **`flan-t5-small`** (~80M params). It is intentionally weak
  so that hallucination cases are visible and easy to discuss. A stronger
  model would hallucinate less and make the comparison less legible at the
  course-project scale.
- The prompt is **a single, simple template**. There is no prompt engineering,
  no chain-of-thought, no self-consistency, no reranker, and no hybrid
  (keyword + dense) retrieval. These are all valid Future Work items.
- Retrieval is **flat L2 with no reranker**. There is no approximate nearest
  neighbour search, which is fine for ~300 vectors but would not scale.
- **Hallucination detection in Phase 1 is rule-based and preliminary** — we
  detect abstention with a substring rule (`"i don't know"` in the lower-cased
  answer). A proper human-judged hallucination rate is part of Phase 2.
- **Unanswerable detection** relies on `len(answers["text"]) == 0` because the
  Hugging Face version of `squad_v2` does not reliably expose `is_impossible`.

These limitations are by design. The project catalogue explicitly states that
heavy fine-tuning, training from scratch, custom objectives, formal
statistical testing, and production deployment are **not required**, and that
a smaller clean project is preferred over an unfinished one that attempts too
many advanced extensions.

---

## 7. External AI Tools Acknowledgement

In line with Section 8.3 of the project specification, the team acknowledges
the following use of external AI tools during Phase 1:

- An external LLM assistant was used to draft the initial structure of this
  notebook, the helper functions (`retrieve`, `generate_with_rag`,
  `generate_without_rag`), and the Phase 1 report. All generated code was
  read, understood, and adjusted by the team before submission.
- The team is responsible for every cell in the notebook and can defend the
  choice of every model, parameter, and metric in an oral examination.
- No external AI tool was used to invent data, results, or citations. The
  dataset (`rajpurkar/squad_v2`) and the models
  (`sentence-transformers/all-MiniLM-L6-v2`, `google/flan-t5-small`) are
  public, named, and pinned by their Hugging Face identifiers.

---

## 8. Phase 1 Deliverables Checklist

| Deliverable (from §5.1 of the project spec) | Status | Where |
|---|---|---|
| Narrow project question | Done | §1 of this report; §1 of the notebook |
| Dataset description | Done | §2 of this report; §2 of the notebook |
| Baseline setup | Done | §3 of this report; §3 of the notebook (cells 3.3 – 3.12) |
| Evaluation plan | Done | §4 of this report; §4 of the notebook |
| Simple analysis plan | Done | §5 of this report; §5 of the notebook |
| Phase 1 report (≤ 6 pages) | Done | This document |

### Phase 1 Grading Rubric — self-check

| Criterion | Max | Self-check |
|---|---|---|
| Clarity and feasibility of the narrowed question | 20 | Question is small, single-model, single-dataset, two-mode comparison — feasible in a semester. |
| Appropriateness and documentation of the dataset | 20 | SQuAD 2.0 is public, documented, has both answerable/unanswerable rows; fields and source listed. |
| Correctness and reproducibility of the baseline | 25 | Seed fixed; models pinned by HF ID; flat L2 + FLAN-T5-small; runs end-to-end on free Colab. |
| Suitability of the planned metrics | 15 | Retrieval (Recall@5, MRR) + generation (EM, Token F1, ROUGE-L) + hallucination (Abstention rate). |
| Realistic scope and simplicity of the plan | 10 | No fine-tuning, no training, no UI; only a sanity check is run. |
| Clarity and structure of the Phase 1 report | 10 | Report follows the 7 sections required by the spec. |
| **Total** | **100** | |

---

_End of Phase 1 report. Phase 2 (full evaluation + error analysis + demo) is
out of scope for this submission._
