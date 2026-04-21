# CARMEN-I Metrics — Medical Anonymisation Evaluation System

Evaluation system for measuring LLM performance on the medical text anonymisation task from the **CARMEN-I** benchmark (1942 documents in Spanish, Catalan, and bilingual).

---

## Table of contents

1. [Problem context](#1-problem-context)
2. [Architecture of the evaluated LLM system](#2-architecture-of-the-evaluated-llm-system)
3. [Repository structure](#3-repository-structure)
4. [Input data format](#4-input-data-format)
5. [The three metrics: M1, M2, M3](#5-the-three-metrics-m1-m2-m3)
6. [Evaluation pipeline step by step](#6-evaluation-pipeline-step-by-step)
7. [Ground truth extraction](#7-ground-truth-extraction)
8. [Prediction extraction](#8-prediction-extraction)
9. [Text normalisation](#9-text-normalisation)
10. [Two-pass multiset matching](#10-two-pass-multiset-matching)
11. [Single-document evaluation: evaluate_document()](#11-single-document-evaluation-evaluate_document)
12. [Handling span mismatches between LLM calls](#12-handling-span-mismatches-between-llm-calls)
13. [Malformed markers: nesting](#13-malformed-markers-nesting)
14. [Special documents](#14-special-documents)
15. [Metric aggregation](#15-metric-aggregation)
16. [Output files](#16-output-files)
17. [M3 error attribution](#17-m3-error-attribution)
18. [Faithfulness score](#18-faithfulness-score)
19. [Auxiliary scripts](#19-auxiliary-scripts)
20. [Running the pipeline](#20-running-the-pipeline)
21. [Precision, Recall and F1 formulae](#21-precision-recall-and-f1-formulae)
22. [Design decisions and rationale](#22-design-decisions-and-rationale)

---

## 1. Problem context

**Clinical text anonymisation** consists of detecting and categorising protected health information (PHI) entities in real medical documents. Examples of PHI: patient names, dates, hospitals, addresses, identification numbers, etc.

The **CARMEN-I** benchmark contains 1942 clinical documents (discharge summaries, progress notes, medical histories…) manually annotated by domain experts. Each document has a list of PHI entities with their exact text and type/category.

This system evaluates how well each LLM performs this task, measuring three independent aspects: identification, categorisation, and end-to-end performance.

---

## 2. Architecture of the evaluated LLM system

The anonymisation system uses a pipeline of **two sequential LLM calls** per document:

```
                       Original clinical text
                              │
                              ▼
            ┌─────────────────────────────────┐
            │     CALL 1: IDENTIFICATION      │
            │                                 │
            │   Input:  raw text              │
            │   Output: text with entities    │
            │           marked as             │
            │           [**entity**]          │
            └─────────────────────────────────┘
                              │
                    identified_text
                              │
                              ▼
            ┌─────────────────────────────────┐
            │     CALL 2: CATEGORISATION      │
            │                                 │
            │   Input:  identified_text       │
            │   Output: text with entities    │
            │           replaced by           │
            │           [**CATEGORY**]        │
            └─────────────────────────────────┘
                              │
                    anonymized_text
```

**Concrete example:**

- Original text: `Paciente atendido en Clinica COVID el 08/03/2021 por Dr. García.`
- `identified_text`: `Paciente atendido en [**Clinica COVID**] el [**08/03/2021**] por [**Dr. García**].`
- `anonymized_text`: `Paciente atendido en [**HOSPITAL**] el [**FECHAS**] por [**NOMBRE_PERSONAL_SANITARIO**].`

Positional correspondence is guaranteed: the N-th `[**...**]` in `identified_text` corresponds to the N-th `[**...**]` in `anonymized_text`. That is, `id_spans[i]` is the detected entity and `anon_spans[i]` is its assigned category.

---

## 3. Repository structure

```
carmina_metrics/
│
├── run_evaluation.py                  ← Main entry point
├── CARMEN1_mappings.tsv               ← Per-document language metadata
├── README.md                          ← Quick-start guide
├── EVALUATION_DOCUMENTATION.md        ← This document
│
├── scripts/
│   ├── evaluate_model.py              ← Core: M1/M2/M3 metrics, matching, extraction
│   ├── analyze_errors.py              ← Post-hoc FN/FP/type-error analysis
│   ├── identify_worst_docs.py         ← Top-N worst-performing documents
│   ├── plots.py                       ← Confusion matrices, model comparison, summary table
│   ├── audit_outputs.py               ← Pre-evaluation diagnostic of model output files
│   └── problematic_docs.json          ← Documents to exclude (annotation errors)
│
├── models/                            ← Output JSONs from each model
│   ├── output_claude-opus-4-6.json
│   ├── output_claude-sonnet-4-6.json
│   └── ...
│
└── results/                           ← Generated automatically
    ├── models_summary.json
    ├── <model>_metrics.json
    ├── <model>_by_language.json
    ├── <model>_by_entity_type.json
    ├── <model>_document_results.json
    ├── <model>_partial_pairs.json
    └── plots/
        ├── <model>_m2_confusion.png
        ├── models_comparison.png
        └── models_summary_table.png
```

---

## 4. Input data format

### 4.1. Model JSON (`models/output_<name>.json`)

A list of documents. Each document has this structure:

```json
{
  "namefile": "CARMEN-I_CC_1.txt",
  "labels": [
    {"text": "Clinica COVID", "type": "HOSPITAL"},
    {"text": "08/03/2021",    "type": "FECHAS"},
    {"text": "Dr. García",    "type": "NOMBRE_PERSONAL_SANITARIO"}
  ],
  "chunks": {
    "identified_text": "Paciente de [**Clinica COVID**] ingresado el [**08/03/2021**] por [**Dr. García**].",
    "anonymized_text": "Paciente de [**HOSPITAL**] ingresado el [**FECHAS**] por [**NOMBRE_PERSONAL_SANITARIO**].",
    "entities_identified": ["Clinica COVID", "08/03/2021", "Dr. García"],
    "entities_anonymized": ["HOSPITAL", "FECHAS", "NOMBRE_PERSONAL_SANITARIO"]
  }
}
```

| Field | Description |
|---|---|
| `namefile` | Document identifier in the benchmark |
| `labels[]` | **Ground truth**: expert-annotated entities. `.text` = what to detect, `.type` = correct category |
| `chunks.identified_text` | Call 1 output (identification). Entities enclosed in `[**...**]` |
| `chunks.anonymized_text` | Call 2 output (categorisation). Entities replaced by `[**TYPE**]` |
| `chunks.entities_identified` | Entity list (fallback if `identified_text` is empty) |
| `chunks.entities_anonymized` | Type list (fallback if `anonymized_text` is empty) |

### 4.2. Language metadata TSV (`CARMEN1_mappings.tsv`)

```
filename	language	ner_annotations
CARMEN-I_CC_1	es	False
CARMEN-I_CC_2	ca	False
CARMEN-I_CC_3	ca-es	False
```

Used for per-language metric breakdown.

### 4.3. Problematic documents (`scripts/problematic_docs.json`)

```json
[
  {
    "namefile": "CARMEN-I_IA_EVOL_94.txt",
    "total_labels": 49,
    "issues_count": 37,
    "issues": ["FRAGMENT: '2'", "MASKING: '01/0[**3'"]
  }
]
```

Documents with known annotation errors (entity fragments, broken markers in GT) that are excluded from evaluation to avoid contaminating the metrics.

---

## 5. The three metrics: M1, M2, M3

### M1 — Identification

> *Did the model detect the correct entities in the text, regardless of category?*

| Aspect | Value |
|---|---|
| **Evaluates** | LLM call 1 (identification) |
| **GT source** | `labels[].text` |
| **Prediction source** | `[**...**]` markers extracted from `chunks.identified_text` |
| **Matching** | Multiset, two passes (exact + partial), on normalised text |
| **TP** | Predicted entity that matches a GT entity |
| **FP** | Predicted entity that matches no GT (model hallucinated something) |
| **FN** | GT entity that matches no prediction (model missed it) |

### M2 — Categorisation (conditioned on M1)

> *Of the correctly identified entities, was the right category assigned?*

| Aspect | Value |
|---|---|
| **Evaluates** | LLM call 2 (categorisation), only on M1 true-positives |
| **Base** | Only the matched pairs from M1 (identification TPs) |
| **GT source** | `labels[gt_idx].type` for matched pairs |
| **Prediction source** | `anon_spans[pred_idx]` from `anonymized_text`, positionally aligned |
| **TP** | Predicted type matches GT type |
| **FP** | Predicted type differs from GT type (categorisation error) |
| **FN** | = FP (each type error is simultaneously FP and FN in M2) |
| **Does not penalise** | Entities missed by M1 (those errors belong to M1) |

The denominator for M2 precision and recall is the same: `len(m1_matches)`. This is correct because M2 measures categorisation quality *given that the entity was detected*.

### M3 — End-to-end

> *Did the system identify **and** categorise each entity correctly?*

| Aspect | Value |
|---|---|
| **Evaluates** | The full pipeline (call 1 + call 2) |
| **Matching** | On `(normalised_text, type)` pairs. Both must match |
| **TP** | Text matches **and** type matches |
| **FP** | Predicted pair with no match (wrong text, or wrong type, or both) |
| **FN** | GT pair with no match |

M3 performs its own independent matching (it does not reuse M1's). An M3 TP requires exactly `text_pred == text_gt AND type_pred == type_gt`.

**Relationship between metrics**: M3 F1 ≤ M1 F1 always (M3 is strictly more demanding).

### Summary table

```
M1 (Identification)  → Did it find the entities?            → text vs text
M2 (Categorisation)  → Did it assign the right types?       → type vs type (M1 TPs only)
M3 (End-to-end)      → Did it find AND type them correctly?  → (text, type) vs (text, type)
```

---

## 6. Evaluation pipeline step by step

```
run_evaluation.py
│
├── 1. Load metadata (once)
│   ├── problematic_docs.json  → set of namefiles to exclude
│   ├── CARMEN1_mappings.tsv   → dict {namefile: language}
│   └── NormalizationConfig    → normalize_case=True, partial_match=flag
│
└── 2. For each models/output_*.json:
    │
    └── evaluate_all(data, config, excluded_docs, language_map)
        │
        └── For each document in the JSON:
            │
            ├─── [EXCLUDED]
            │    namefile ∈ excluded_docs
            │    → counts['excluded'] += 1
            │    → SKIP (does not contribute to any metric)
            │
            ├─── [NO GT]
            │    labels = [] (document with no real PHI)
            │    → counts['no_gt'] += 1
            │    → Extract model predictions
            │    → If predictions exist: they are FP in M1 and M3
            │    → counts['evaluated'] += 1
            │    → CONTINUE (does not reach evaluate_document)
            │
            ├─── [ANON TEXT MISSING]
            │    identified_text has spans but anonymized_text is empty
            │    → M1 is computed normally
            │    → M2 and M3 are SKIPPED (pipeline failure, no categorisation attempted)
            │    → counts['anon_text_missing'] += 1
            │
            └─── [NORMAL]
                 labels ≠ [] (standard document)
                 → counts['evaluated'] += 1
                 │
                 └── evaluate_document(doc, config)
                     │
                     ├── extract_ground_truth(doc)
                     │   → gt_entities: ["Clinica COVID", "08/03/2021"]
                     │   → gt_types:    ["HOSPITAL",      "FECHAS"]
                     │
                     ├── extract_predictions(doc)
                     │   → pred_entities: ["Clinica COVID", "08/03/2021"]
                     │   → pred_types:    ["HOSPITAL",      "FECHAS"]
                     │   → meta: {length_mismatch, mismatch_detail, ...}
                     │
                     ├── normalize_entity_list() on both lists
                     │   → gt_norm, pred_norm
                     │
                     ├── M1: match_entities(pred_norm, gt_norm, config)
                     │   → tp, fp, fn, partial, matches[(pred_idx, gt_idx)]
                     │
                     ├── M2: for each (pred_idx, gt_idx) in matches
                     │   → does type_pred == type_gt?
                     │
                     ├── M3: independent matching on (text, type) pairs
                     │   → tp, fp, fn
                     │
                     ├── Per-entity-type breakdown (M1 and M3)
                     │
                     └── Faithfulness scores (cosine similarity)
```

---

## 7. Ground truth extraction

**Function**: `extract_ground_truth(doc)`

Reads the `labels[]` field from the document and returns two parallel lists:

```python
gt_entities = ["Clinica COVID", "08/03/2021", "Dr. García"]
gt_types    = ["HOSPITAL",      "FECHAS",     "NOMBRE_PERSONAL_SANITARIO"]
```

**Edge cases handled**:

| Case | Action |
|---|---|
| Label without `text` field | Skipped |
| Label with empty or whitespace-only text | Skipped |
| Label without `type` field | Assigned `'UNKNOWN'` |

---

## 8. Prediction extraction

**Function**: `extract_predictions(doc)`

Uses a cascade of strategies, from most reliable to least reliable:

### 8.1. Primary strategy: parsing from full texts

Markers `[**...**]` are extracted from `identified_text` and `anonymized_text` using the regex:

```python
_PAT = re.compile(r'\[\*\*((?:(?!\[\*\*|\*\*\]).)*?)\*\*\]', re.DOTALL)
```

This regex has a negative lookahead: the captured content cannot contain `[**` or `**]`, which prevents a malformed unclosed marker from consuming adjacent markers. Before parsing, both texts pass through `_flatten_nested_markers()` to resolve nesting (see [section 13](#13-malformed-markers-nesting)) and `_strip_prompt_leakage()` to remove any echoed system prompt.

```
identified_text → id_raw   = ["Clinica COVID", "08/03/2021", "Dr. García"]
anonymized_text → anon_raw = ["HOSPITAL",      "FECHAS",     "NOMBRE_PERSONAL_SANITARIO"]
```

**Sub-case A — equal lengths** (`len(id_raw) == len(anon_raw)`):

Direct positional pairing: `entities[i] = id_raw[i]`, `types[i] = anon_raw[i]`. Pairs where `id_raw[i]` is empty are discarded.

**Sub-case B — different lengths** (`len(id_raw) ≠ len(anon_raw)`):

The context-anchor alignment mechanism is activated (see [section 12](#12-handling-span-mismatches-between-llm-calls)). `meta['length_mismatch'] = True` is set.

**Sub-case C — anonymized_text missing**:

When `identified_text` has spans but `anonymized_text` is empty or has no spans, `meta['anon_text_missing'] = True`. Entities are extracted from call 1, but types are all set to `'UNKNOWN'`. M2 and M3 are skipped for this document during aggregation.

### 8.2. Fallback strategy: explicit lists

If `identified_text` is empty, the JSON lists are used:

```python
entities = chunks.get('entities_identified', [])
types    = chunks.get('entities_anonymized', [])
```

Edge cases: if they come as `dict` instead of `list` (LLM format error), they are converted. If lengths differ, `types` is padded with `'UNKNOWN'` or truncated.

### 8.3. Type normalisation

All predicted types pass through `_normalize_pred_type()`, which uppercases the string and returns it unchanged if it belongs to `_VALID_TYPES` (the 27 categories from the labelling schema + `UNKNOWN`). Any type not in this set is mapped to `'INVALID_CATEGORY'` to prevent model-invented labels from polluting the confusion matrix.

### 8.4. Return value

```python
return entities, types, meta
# meta = {
#     'length_mismatch': bool,
#     'mismatch_detail': str,
#     'anon_text_missing': bool,
#     'aligned_by_context': int,
#     'align_phase0': int,
#     'align_phase1': int,
#     'align_phase2': int,
#     'align_phase3_unknown': int,
# }
```

---

## 9. Text normalisation

**Function**: `_normalize_token(s) → str`

Applied to all entities (GT and predictions) before matching. Three steps in order:

```
1. _normalize_whitespace(s)   →  strip() + collapse internal spaces
   "  22- 11-2018  "  →  "22- 11-2018"

2. .lower()                   →  "clinica covid"

3. _remove_accents(s)         →  NFD decomposition + diacritic stripping
   "García"  →  "garcia"
```

**Rationale**:
- **Whitespace**: the GT sometimes has extra spaces, or the model introduces minor variations.
- **Lowercase**: case-insensitive comparison to avoid penalising capitalisation differences.
- **Accents**: the benchmark GT has occasional accentuation typos. Without this normalisation, a model that writes the correct form would be penalised.

---

## 10. Two-pass multiset matching

**Function**: `match_entities(pred_entities, gt_entities, config)`

Matching respects **multiset** semantics: if an entity appears N times in the GT, it requires N predictions to produce N TPs.

### Pass 1 — Exact match

```python
for i, pred in enumerate(pred_entities):
    for j, gt in enumerate(gt_entities):
        if pred == gt and not matched:
            # → TP
```

Greedy: iterates predictions in order; for each one, finds the first unconsumed GT that matches exactly. Once matched, both are marked as used.

**Why exact first?** To prevent a partial match from "stealing" what could have been an exact match. Example:

```
GT:   ["08/03/2021", "08/03"]
Pred: ["08/03/2021", "08/03"]
```

If partial ran first, `"08/03"` could match `"08/03/2021"` (substring), leaving `"08/03/2021"` as an orphaned prediction. With exact first, both match correctly.

### Pass 2 — Partial match (only if `config.partial_match = True`)

For elements not matched in Pass 1, two strategies are available:

**Strategy `'substring'`** (default):
```python
if pred in gt or gt in pred:  # substring in either direction
    # → partial TP
```

**Strategy `'sequence_matcher'`**:
```python
score = SequenceMatcher(None, pred, gt).ratio()
if score >= config.partial_match_threshold:  # default 0.6
    # → partial TP
```

**Minimum length constraint**: both strings must have ≥ `config.partial_match_min_length` (default 3) characters. Without this, very short GT entities like `"2"` or `"10"` would spuriously match any prediction containing those digits as a substring (e.g. `"20/03/2021"` contains `"20"`).

### Result

```python
return tp, fp, fn, partial, matches, partial_pairs
# tp            = exact matches (Pass 1)
# fp            = predictions with no match
# fn            = GT with no match
# partial       = partial matches (added to TP for M1)
# matches       = [(pred_idx, gt_idx), ...] — used for M2
# partial_pairs = [{pred, gt, score, method}, ...] — for logging
```

---

## 11. Single-document evaluation: `evaluate_document()`

### 11.1. M1 calculation

```python
m1 = MetricsResult(
    tp = exact_tp + partial_tp,  # partial counts as TP for identification
    fp = pred_without_match,
    fn = gt_without_match,
    partial = partial_count
)
```

### 11.2. M2 calculation

Operates only on the pairs `(pred_idx, gt_idx)` matched in M1:

```python
for pred_idx, gt_idx in m1_matches:
    gt_type   = gt_types[gt_idx].upper()
    pred_type = _normalize_pred_type(pred_types[pred_idx])

    if gt_type == pred_type:
        m2_tp += 1
    else:
        m2_fp += 1
        # (entity, gt_type, pred_type) is recorded for later analysis

m2_fn = m2_fp  # each type error is simultaneously FP and FN
```

**Key**: `pred_types[pred_idx]` comes from `anonymized_text` (marker position), not from a separate list. This guarantees that the type corresponds exactly to the identified entity.

### 11.3. M3 calculation

**Independent** matching on pairs:

```python
gt_pairs   = [(gt_norm[j],   gt_types[j].upper())   for j in range(len(gt))]
pred_pairs = [(pred_norm[i], pred_types[i].upper()) for i in range(len(pred))]
```

Same two-pass structure:
- **Pass 1**: `p_text == g_text AND p_type == g_type`
- **Pass 2** (if partial_match): `(p_text in g_text or g_text in p_text) AND p_type == g_type`

Type match is required in both passes.

### 11.4. Per-entity-type breakdown

For each GT category, independent TP/FP/FN are computed:

**M1 by type**:
- TPs: matched GT → `m1_by_type[gt_type].tp += 1`
- FNs: unmatched GT → `m1_by_type[gt_type].fn += 1`
- FPs: unmatched predictions → `m1_by_type[pred_type].fp += 1`

**M3 by type**: same logic but using M3's `(text, type)` pairs.

### 11.5. Faithfulness scores

Two cosine-similarity scores are computed per document (see [section 18](#18-faithfulness-score)):

- **Call 1 faithfulness**: how well identified_text preserves the original text (markers stripped).
- **Call 2 faithfulness**: how well the inter-span text is preserved between identified_text and anonymized_text.

---

## 12. Handling span mismatches between LLM calls

### 12.1. The problem

Call 2 (categorisation) receives `identified_text` from call 1, but since it is an independent LLM call, it can introduce differences:
- Add markers that did not exist in `identified_text`
- Omit markers that did exist
- Slightly modify surrounding text

When this happens, `len(id_raw) ≠ len(anon_raw)`, and direct positional pairing `types[i] = anon_raw[i]` produces incorrect types or missing types.

### 12.2. The context-anchor alignment mechanism

**Function**: `_align_markers_by_context(identified_text, anonymized_text)`

**Core idea**: although the LLM may modify markers, **it does not modify the surrounding text**. Thus the text segment following each marker serves as a positional anchor for pairing the correct markers between both versions.

**Step 1 — Decomposition**:

Using `re.split` with a capturing group, alternating segments are obtained:

```
"Text [**ENT1**] between [**ENT2**] end"
→ segments: ["Text ", " between ", " end"]
→ entities: ["ENT1", "ENT2"]
```

**Step 2 — Anchor computation**:

For each marker `i`, the anchor is a normalised representation of the text segment following it:

```python
def _anchor(seg, tail_words=10, head_words=3):
    words = re.sub(r'[^\w\s]', ' ', seg.lower()).split()
    if len(words) < 1:
        return None   # unreliable anchor
    return ' '.join(words[-tail_words:] + words[:head_words])
```

The last 10 + first 3 words are extracted (no punctuation, lowercased). This makes anchors robust to minor LLM edits (loose brackets, extra spaces, typo corrections).

**Step 3 — Greedy matching with a cosine-similarity threshold (≥ 0.75)**:

The algorithm maintains a pointer `i` for `identified_text` and a pointer `j` for `anonymized_text`:

| Phase | Condition | Action |
|---|---|---|
| **Phase 0 — Short anchor** | `_anchor()` returns `None` (< 1 word) | Order-based pairing: `types[i] = anon[j]`. Best-effort — works in practice because short-anchor zones are typically consecutive markers of the same type |
| **Phase 1 — Direct alignment** | `cosine(anchor_id, anchor_anon[j]) >= 0.75` | Anchors match: `types[i] = anon[j]`, both pointers advance. Most frequent case |
| **Phase 2 — Skip-ahead** | Anchor matches at `anon[j + skip]` for `skip ∈ 1..5` | Call 2 inserted `skip` extra markers. They are skipped and the correct marker is paired |
| **Phase 3 — No match** | Anchor matches none of `anon[j..j+5]` | Call 2 dropped this marker. Type is set to `'UNKNOWN'`. The `j` pointer does not advance |

### 12.3. Phase distribution

Empirical breakdown across all evaluated models:

```
Phase       %       Description
Phase 1     ~45%    Direct anchor alignment — perfect resolution
Phase 0     ~39%    Short anchor, order-based — works in practice
Phase 3     ~11%    No match → UNKNOWN
Phase 2     ~5%     Skip-ahead over extra anon markers
```

---

## 13. Malformed markers: nesting

**Function**: `_flatten_nested_markers(text) → str`

Some models (especially Llama) produce nested markers:

```
Input:   [**02/[**07**]/2024**]
Output:  [**02/07/2024**]
```

The algorithm is an O(n) stack automaton:
- Maintains a `depth` counter for nesting level
- Only emits `[**` and `**]` when `depth` transitions from 0→1 and 1→0, respectively
- Content of inner markers is preserved as plain text
- Orphan `**]` (depth=0) are ignored
- Unclosed `[**` produce text without `**]`, which the `_PAT` regex safely ignores

---

## 14. Special documents

### 14.1. Excluded documents

**Source**: `scripts/problematic_docs.json`

Documents with benchmark annotation errors (entity fragments like `"2"`, broken markers like `"01/0[**3"`). They are excluded **before** any evaluation:

```python
if namefile in excluded_docs:
    counts['excluded'] += 1
    continue  # does not contribute to any metric
```

### 14.2. No-PHI documents (`labels: []`)

`labels: []` means the document was annotated and **confirmed to contain no PHI entities**. It is clean clinical text.

```python
if not gt_entities:
    counts['no_gt'] += 1
    pred_entities, pred_types, _ = extract_predictions(doc)
    if pred_entities:
        n = len(pred_entities)
        global_m1.fp += n    # anonymising clean text is a real error
        global_m3.fp += n
        lang_m1[lang].fp += n
        lang_m3[lang].fp += n
    counts['evaluated'] += 1
    continue  # does not reach evaluate_document()
```

**Behaviour**:
- If the model **marked nothing**: correct, no impact on metrics.
- If the model **marked N entities**: those are N real false positives in M1 and M3, and in the per-language breakdown. M2 is not affected (no M1 TPs to evaluate type on).
- **Does not appear** in `document_results` (does not go through `evaluate_document()`).

### 14.3. Missing anonymized_text

When `identified_text` has spans but `anonymized_text` is completely empty (pipeline failure where call 2 did not produce output), M1 metrics are computed normally but **M2 and M3 are entirely skipped** to avoid double-penalising a categorisation that was never attempted.

---

## 15. Metric aggregation

### 15.1. Global metrics

After processing each document (normal case with GT), counters are added to the global accumulators:

```python
global_m1.tp += result['m1'].tp
global_m1.fp += result['m1'].fp
global_m1.fn += result['m1'].fn
global_m1.partial += result['m1'].partial
# same for global_m2 and global_m3
```

Final metrics (P, R, F1) are computed over the aggregated totals (micro-averaging):

```
Precision_global = TP_total / (TP_total + FP_total)
Recall_global    = TP_total / (TP_total + FN_total)
F1_global        = 2 × P × R / (P + R)
```

### 15.2. Per-language metrics

Same mechanism as global metrics but accumulating in separate dictionaries per language (`es`, `ca`, `ca-es`):

```python
lang_m1[lang].tp += result['m1'].tp
# etc.
```

### 15.3. Per-entity-type metrics

`m1_by_type` and `m3_by_type` from each document are accumulated:

```python
for etype, mr in result['m1_by_type'].items():
    global_m1_by_type[etype].tp += mr.tp
    global_m1_by_type[etype].fp += mr.fp
    global_m1_by_type[etype].fn += mr.fn
```

### 15.4. Confusion matrix

M2 `(gt_type, pred_type)` pairs are collected across all documents and passed to `_compute_m2_confusion()`, which builds an N×N confusion matrix and computes per-class, micro, macro, and weighted metrics. Uses `sklearn.metrics.multilabel_confusion_matrix` when available, with a manual one-vs-rest fallback otherwise. The matrix is saved inside the `_metrics.json` file and rendered as a PNG plot.

---

## 16. Output files

Generated in `results/` for each model:

### `<model>_metrics.json`

```json
{
  "model_name": "claude-opus-4-6",
  "matching_config": {
    "normalize_case": true,
    "partial_match": true
  },
  "counts": {
    "total": 1981,
    "excluded": 39,
    "no_gt": 3,
    "evaluated": 1942,
    "length_mismatches": 15,
    "anon_text_missing": 0
  },
  "global": {
    "m1": {"tp": 7284, "fp": 416, "fn": 338, "partial": 152, "precision": 0.9456, "recall": 0.9557, "f1": 0.9506},
    "m2": {"tp": 6993, "fp": 291, "fn": 291, "partial": 0, "precision": 0.9601, "recall": 0.9601, "f1": 0.9601},
    "m3": {"tp": 6906, "fp": 700, "fn": 612, "partial": 0, "precision": 0.9081, "recall": 0.9178, "f1": 0.9130},
    "m2_confusion": { "classes": [...], "matrix": {...}, "per_class": {...}, "micro": {...}, "macro": {...}, "weighted": {...} },
    "align_phases": {"phase0_empty_anchor": 10, "phase1_direct": 20, "phase2_skip": 3, "phase3_unknown": 5, "total_resolved": 33},
    "faithfulness_identify_mean": 0.9812,
    "faithfulness_anonymize_mean": 0.9945
  }
}
```

### `<model>_by_language.json`

```json
{
  "ca": {
    "m1": {"tp": ..., "fp": ..., "fn": ..., "precision": ..., "recall": ..., "f1": ...},
    "m2": {...},
    "m3": {...}
  },
  "ca-es": {...},
  "es": {...}
}
```

### `<model>_by_entity_type.json`

```json
{
  "m1": {
    "FECHAS":    {"tp": 2300, "fp": 50, "fn": 30, "precision": 0.9787, "recall": 0.9871, "f1": 0.9829},
    "HOSPITAL":  {"tp": 120,  "fp": 10, "fn": 25, "precision": 0.9231, "recall": 0.8276, "f1": 0.8727}
  },
  "m3": {
    "FECHAS":    {"tp": 2280, "fp": 70, "fn": 50},
    "HOSPITAL":  {"tp": 110,  "fp": 20, "fn": 35}
  }
}
```

### `<model>_document_results.json`

One entry per evaluated document (those with non-empty GT):

```json
{
  "namefile": "CARMEN-I_CC_1.txt",
  "language": "es",
  "gt_count": 5,
  "pred_count": 5,
  "m1": {"tp": 5, "fp": 0, "fn": 0, "partial": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
  "m2": {"tp": 4, "fp": 1, "fn": 1, "partial": 0, "precision": 0.8, "recall": 0.8, "f1": 0.8},
  "m3": {"tp": 4, "fp": 1, "fn": 1, "partial": 0, "precision": 0.8, "recall": 0.8, "f1": 0.8},
  "length_mismatch": false,
  "mismatch_detail": "",
  "anon_text_missing": false,
  "faithfulness_identify": 0.9987,
  "faithfulness_anonymize": 0.9998,
  "gt_entities": ["Clinica COVID", "08/03/2021", "..."],
  "gt_types": ["HOSPITAL", "FECHAS", "..."],
  "pred_entities": ["Clinica COVID", "08/03/2021", "..."],
  "pred_types": ["HOSPITAL", "FECHAS", "..."],
  "m2_type_errors": [{"entity": "Dr. García", "gt_type": "NOMBRE_PERSONAL_SANITARIO", "pred_type": "NOMBRE_SUJETO_ASISTENCIA"}],
  "m1_partial_pairs": [{"pred": "08/03", "gt": "08/03/2021", "score": 1.0, "method": "substring"}]
}
```

### `<model>_partial_pairs.json`

All partial (substring/sequence) matches across all documents for manual review:

```json
[
  {"pred": "08/03", "gt": "08/03/2021", "pred_idx": 2, "gt_idx": 3, "score": 1.0, "method": "substring"}
]
```

### `models_summary.json`

Comparative table across all models:

```json
[
  {
    "model": "claude-opus-4-6",
    "docs_m1": 1942,
    "docs_m2_m3": 1942,
    "m1_precision": 0.9456,
    "m1_recall": 0.9557,
    "m1_f1": 0.9506,
    "m2_f1": 0.9601,
    "m3_precision": 0.9081,
    "m3_recall": 0.9178,
    "m3_f1": 0.9130,
    "m3_errors_total": 1312,
    "m3_errors_identification": 744,
    "m3_errors_categorization": 572,
    "m3_pct_identification": 56.7,
    "m3_pct_categorization": 43.6,
    "faithfulness_identify_mean": 0.9812,
    "faithfulness_anonymize_mean": 0.9945
  }
]
```

### `plots/`

| File | Content |
|---|---|
| `<model>_m2_confusion.png` | M2 confusion matrix (GT type vs predicted type) |
| `models_comparison.png` | Bar chart comparing M1/M2/M3 F1 across all models |
| `models_summary_table.png` | Tabular summary rendered as an image |

---

## 17. M3 error attribution

End-to-end (M3) errors are decomposed into two sources:

```
errors_identification = M1_FP + M1_FN     (entities missed or hallucinated)
errors_categorization = M2_FP + M2_FN     (entities found but incorrectly typed)
m3_errors_total       = M3_FP + M3_FN
```

```
pct_identification = errors_identification / m3_errors_total × 100
pct_categorization = errors_categorization / m3_errors_total × 100
```

**Note**: the two percentages may sum to slightly more or less than 100% because M1, M2, and M3 match independently over different sets (text-only vs text+type pairs). The discrepancy is typically < 1%.

**Interpretation**:
- `pct_identification` high (>70%) → the bottleneck is entity detection.
- `pct_categorization` high (>40%) → the bottleneck is type assignment.

---

## 18. Faithfulness score

Two character-bigram cosine similarity scores are computed per document to measure how well the model preserves the original text:

### Call 1 faithfulness (identify)

Compares `identified_text` (with markers stripped, keeping their content) against the `original_text` field. Measures whether the model preserved the source text while adding entity annotations.

```
faithfulness_identify = cosine_sim(strip_markers(identified_text), original_text)
```

### Call 2 faithfulness (anonymize)

Compares the **inter-span text** (text between markers) of `identified_text` and `anonymized_text`. Measures whether the model preserved non-annotated text during categorisation.

```
faithfulness_anonymize = cosine_sim(interspan(identified_text), interspan(anonymized_text))
```

Both scores are `None` when either text is empty. Global means are computed excluding `None` values.

---

## 19. Auxiliary scripts

### `scripts/analyze_errors.py`

Detailed error analysis from `_document_results.json`. Produces:

1. **M1 false negatives by type**: which entities the model missed, grouped by category, with text examples and frequency.
2. **M1 false positives by predicted type**: which entities the model hallucinated.
3. **M2 categorisation errors**: patterns `GT_TYPE → PRED_TYPE` with examples.
4. **Documents with span mismatches**: list of documents with `length_mismatch=True`.

```bash
python scripts/analyze_errors.py results/claude-opus-4-6_document_results.json
```

### `scripts/identify_worst_docs.py`

Shows the N documents with worst M3 F1:

1. Distribution of documents by F1 range (=0, <0.5, 0.5–0.8, ≥0.8).
2. Top-N table with namefile, language, GT count, pred count, M1/M2/M3 F1.
3. Detail of the 10 worst: GT vs predicted entities, TP/FP/FN per metric, mismatch warnings.

```bash
python scripts/identify_worst_docs.py results/claude-opus-4-6_document_results.json 20
```

### `scripts/audit_outputs.py`

Pre-evaluation diagnostic tool that scans model output files for structural problems:
- Prompt leakage, thinking traces, empty texts, span count mismatches, category-as-entity errors, placeholder residuals, empty/unclosed/nested markers, entity text in anonymized spans.

```bash
python scripts/audit_outputs.py
python scripts/audit_outputs.py --model claude --verbose
python scripts/audit_outputs.py --check thinking_leak
```

### `scripts/plots.py`

Plotting utilities used by `run_evaluation.py`:
- `plot_confusion()` — M2 confusion matrix heatmap per model
- `plot_comparison()` — bar chart comparing M1/M2/M3 F1 across all models
- `plot_summary_table()` — tabular summary rendered as an image

---

## 20. Running the pipeline

### Standard evaluation (with partial match, default)

```bash
python run_evaluation.py
```

### Strict evaluation (exact match only)

```bash
python run_evaluation.py --no-partial-match
```

### With detailed per-language and per-type output

```bash
python run_evaluation.py --verbose
```

### All available arguments

| Argument | Default | Description |
|---|---|---|
| `--models-dir` | `models` | Directory containing model JSON files |
| `--results-dir` | `results` | Output directory |
| `--excluded-docs` | `scripts/problematic_docs.json` | JSON of documents to exclude |
| `--language-file` | `CARMEN1_mappings.tsv` | TSV with per-document language metadata |
| `--no-partial-match` | (off, i.e. partial match enabled by default) | Disable substring/partial matching |
| `--partial-match-threshold` | `0.6` | SequenceMatcher threshold for partial matching [0–1] |
| `--verbose` | `False` | Show per-language and per-type breakdown in console |

---

## 21. Precision, Recall and F1 formulae

```
             TP                          TP                      2 × P × R
P = ─────────────────        R = ─────────────────        F1 = ─────────────
       TP + FP                      TP + FN                       P + R
```

- **Precision**: of everything the model marked, how much is correct?
- **Recall**: of everything that should have been marked, how much did the model find?
- **F1**: harmonic mean of P and R. Penalises imbalance between the two.

Metrics are aggregated by **micro-averaging**: all TP, FP, FN from all documents are summed before computing P/R/F1. This weights documents with more entities more heavily.

---

## 22. Design decisions and rationale

### Why use `labels[]` as ground truth instead of the `.txt` files?

The JSON labels already contain the text and type of each benchmark entity. Parsing the original `.txt` files would introduce encoding issues, file path dependencies, and format variations.

### Why extract predictions from the full text (`identified_text`) instead of the lists (`entities_identified`)?

Positional alignment of the full text is guaranteed by construction: `anonymized_text` is generated by replacing each `[**entity**]` in `identified_text` with `[**CATEGORY**]` in order. The lists `entities_identified` / `entities_anonymized` are fragile because the second LLM call can add or remove entities, shifting all subsequent types.

### Why exact match first and partial second?

To prevent a partial match from "stealing" a future exact match. If we have GT=["08/03/2021", "08/03"] and pred=["08/03/2021", "08/03"], with partial first `"08/03"` (pred) could match `"08/03/2021"` (gt), leaving both `"08/03/2021"` without an exact match.

### Why M2_FN = M2_FP?

M2 only operates on M1 TPs (already-identified entities). If a matched entity has the wrong type, that is simultaneously:
- An FP: the predicted type is incorrect (should not have predicted that type).
- An FN: the correct type was not predicted (that type is missing).

There are no "extra" or "missing" entities in M2 — only pairs with correct or incorrect type.

### Why does M3 do its own matching instead of combining M1+M2?

Because M1 and M3 operate over different spaces: M1 matches texts, M3 matches `(text, type)` pairs. A combined matching could produce inconsistent results. Independent M3 matching guarantees that "correct text + correct type" is evaluated as a single atomic operation.

### Why normalise accents?

The benchmark GT has accentuation inconsistencies (some annotators omit diacritics). Without this normalisation, a model that writes "García" correctly would be penalised if the GT says "Garcia".

### Why minimum 3 characters for partial match?

Without this threshold, very short GT entities like `"2"`, `"10"`, `"20"` would spuriously match any prediction containing those digits as a substring (e.g. `"20/03/2021"` contains `"20"`).

### Why do predictions on no-GT documents count as FP?

`labels: []` means the document was annotated and confirmed as PHI-free. If a model marks entities in a clean document, that is a real over-anonymisation error that should penalise precision.

### Why map invalid predicted types to INVALID_CATEGORY?

Models sometimes invent category names not present in the labelling schema. Without normalisation, these would create spurious columns in the M2 confusion matrix and make per-class metrics unreliable. Mapping them to a single sentinel class keeps the confusion matrix clean and focused on real categories.

### Why skip M2/M3 when anonymized_text is missing?

When `identified_text` has spans but `anonymized_text` is completely empty, it represents a pipeline failure (call 2 produced no output), not a categorisation error. Penalising M2 and M3 would double-count something that is essentially an infrastructure issue. M1 metrics are still computed normally.
