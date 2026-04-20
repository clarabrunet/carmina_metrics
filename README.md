# carmina_metrics

Evaluation pipeline for the **CARMEN-I** benchmark — a medical-text NER and anonymisation dataset covering multiple languages (Spanish, Catalan, English, …).

The pipeline evaluates how well a model can (1) identify personal entities, (2) classify them into the correct type, and (3) anonymise them end-to-end, using three complementary F1-based metrics.

---

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **M1** | Entity identification — does the model find the right text spans? |
| **M2** | Categorisation — are identified entities assigned the correct type? |
| **M3** | End-to-end anonymisation — are spans both found AND correctly typed? |

All three metrics use micro-averaged precision, recall, and F1 over token-normalised entity strings. By default a **partial-match** rule gives credit to spans that overlap but are not identical (`--no-partial-match` disables this).

---

## Repository layout

```
data/
  inputs/
    CARMEN1_mappings.tsv      ← filename <TAB> language (+ other columns)
    txt/
      identify/               ← model input texts (identify call)
      masked/                 ← ground-truth anonymised texts
      raw/                    ← original raw texts
models/
  output_<model>.json         ← one file per model (see Model JSON format below)
results/                      ← created automatically on first run
  <model>_metrics.json
  <model>_by_language.json
  <model>_by_entity_type.json
  <model>_document_results.json
  models_summary.json
  plots/
scripts/
  evaluate_model.py           ← core evaluation logic (M1/M2/M3)
  analyze_errors.py           ← per-type FN/FP/M2 error report
  identify_worst_docs.py      ← worst-performing document listing
  plots.py                    ← confusion matrices, comparison chart, summary table
run_evaluation.py             ← main entry point
```

---

## Quick start

```bash
# 1. Place model output files in models/
# 2. Run the pipeline
python run_evaluation.py

# Evaluate only specific models
python run_evaluation.py --models output_claude-sonnet.json output_gemini.json

# Disable partial-match scoring
python run_evaluation.py --no-partial-match

# Show per-language and per-type breakdowns in the console
python run_evaluation.py --verbose
```

Full CLI reference:

```
python run_evaluation.py --help
```

---

## Analysis tools

```bash
# Detailed FN / FP / M2 error breakdown for one model
python scripts/analyze_errors.py results/<model>_document_results.json

# List worst-performing documents (by M3 F1)
python scripts/identify_worst_docs.py results/<model>_document_results.json [top_n]
```

---

## Model JSON format

Each file in `models/` must be a JSON array where every element represents one document:

```json
[
  {
    "namefile": "CARMEN-I_CC_1",
    "identified_entities": ["María García", "Hospital La Paz"],
    "identified_types":    ["NOMBRE_SUJETO", "INSTITUCION"],
    "anonymized_entities": ["María García", "Hospital La Paz"],
    "anonymized_types":    ["NOMBRE_SUJETO", "INSTITUCION"],
    "anonymized_text":     "La paciente [NOMBRE_SUJETO] fue ingresada en [INSTITUCION]."
  }
]
```

`identified_*` fields come from the **identify call** (M1/M2); `anonymized_*` fields from the **anonymise call** (M3). Both are required for the full pipeline.

---

## Output files

| File | Content |
|------|---------|
| `<model>_metrics.json` | Global M1/M2/M3 precision, recall, F1, TP/FP/FN, faithfulness |
| `<model>_by_language.json` | Same metrics split by document language |
| `<model>_by_entity_type.json` | Same metrics split by entity category |
| `<model>_document_results.json` | Per-document results (used by analysis scripts) |
| `models_summary.json` | One-row-per-model comparison table |
| `plots/` | `confusion_<model>.png`, `models_comparison.png`, `summary_table.png` |

