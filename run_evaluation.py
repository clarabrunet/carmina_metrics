#!/usr/bin/env python3
"""
run_evaluation.py — Main entry point for the CARMEN-I evaluation pipeline
=========================================================================

Expected repository layout:
    data/
        inputs/
            CARMEN1_mappings.tsv   → filename <TAB> language (+ other columns)
            txt/{raw,identify,masked}/  → benchmark text files
    models/                     → output_<model>.json files
    results/                    → created automatically
    scripts/
        evaluate_model.py
        analyze_errors.py
        identify_worst_docs.py
        problematic_docs.json
    run_evaluation.py           ← this file

For each model in models/ the following files are written to results/:
    <model>_metrics.json           → global M1/M2/M3 metrics
    <model>_by_language.json       → metrics broken down by language
    <model>_by_entity_type.json    → metrics broken down by entity category
    <model>_document_results.json  → per-document results (for analysis scripts)
    models_summary.json            → comparison table across all models
    plots/                         → PNG plots (confusion matrices, comparison chart, summary table)

Usage:
    python run_evaluation.py
    python run_evaluation.py --language-file "CARMEN1_mappings.tsv"
    python run_evaluation.py --no-partial-match
"""

import json
import glob
import os
import sys
import csv
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from evaluate_model import NormalizationConfig, evaluate_all
from plots import load_confusion, plot_confusion, load_all_metrics, plot_comparison, plot_summary_table


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_excluded_docs(path: str) -> set:
    """
    Loads the list of problematic documents to exclude.

    The file is a JSON list of objects {namefile: ..., ...}.
    Plain string lists are also supported for backwards compatibility.
    """
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found. Evaluating on the full dataset.")
        return set()

    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    excluded = set()
    for item in items:
        if isinstance(item, str):
            excluded.add(item)
        elif isinstance(item, dict) and 'namefile' in item:
            excluded.add(item['namefile'])

    print(f"  [OK] {len(excluded)} documents excluded (problematic_docs.json)")
    return excluded


def load_language_map(path: str) -> dict:
    """
    Loads the {namefile: language} map from the TSV file.

    Expected TSV format:
        filename  <TAB>  language  <TAB>  ...
        CARMEN-I_IR_1    es        False

    The filename field has no .txt extension, but model JSON namefiles may
    include it — both variants are stored for maximum compatibility.

    Returns an empty dict if the file does not exist (no language breakdown).
    """
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found. No language breakdown will be available.")
        return {}

    lang_map = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            name = row.get('filename', '').strip()
            lang = row.get('language', 'unknown').strip()
            if name:
                # Store both with and without extension for maximum compatibility
                lang_map[name] = lang
                lang_map[name + '.txt'] = lang

    print(f"  [OK] {len(lang_map)//2} documents with language assigned ({path})")
    return lang_map


# ---------------------------------------------------------------------------
# Formateo de resultados
# ---------------------------------------------------------------------------

def format_metrics_table(summary_data: list):
    """Prints a comparative summary table across all models."""
    if not summary_data:
        return

    print("\n" + "=" * 110)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 110)
    header = (
        f"{'Modelo':<30} {'M1_n':>5} {'M2/3_n':>6} "
        f"{'M1_P':>7} {'M1_R':>7} {'M1_F1':>7} "
        f"{'M2_F1':>7} "
        f"{'M3_P':>7} {'M3_R':>7} {'M3_F1':>7}"
    )
    print(header)
    print("-" * 110)

    for row in sorted(summary_data, key=lambda r: -r['m3_f1']):
        print(
            f"{row['model']:<30} {row['docs_m1']:>5} {row['docs_m2_m3']:>6} "
            f"{row['m1_precision']:>7.4f} {row['m1_recall']:>7.4f} {row['m1_f1']:>7.4f} "
            f"{row['m2_f1']:>7.4f} "
            f"{row['m3_precision']:>7.4f} {row['m3_recall']:>7.4f} {row['m3_f1']:>7.4f}"
        )
    print("-" * 110)


def format_error_attribution(g: dict):
    """
    Decomposes M3 errors into two sources.

    - Identification: entities the model missed (M1 FN) or hallucinated (M1 FP).
    - Categorisation: entities correctly identified but assigned the wrong type
      (M2 FP+FN = 2 × (M1_TP − M2_TP)).

    The two terms approximate M3_FP + M3_FN.  A small discrepancy (<1%) can
    occur because M1 and M3 match independently over different sets.
    """
    m1, m2, m3 = g['m1'], g['m2'], g['m3']
    errors_id  = m1['fp'] + m1['fn']   # missed or hallucinated
    errors_cat = m2['fp'] + m2['fn']   # found but wrong type
    total_m3   = m3['fp'] + m3['fn']
    if total_m3 == 0:
        return
    pct_id  = errors_id  / total_m3 * 100
    pct_cat = errors_cat / total_m3 * 100

    print(f"\n  M3 error attribution  (M3 FP+FN = {total_m3}):")
    print(f"  {'Source':<32} {'Errors':>8}  {'% of M3 errors':>14}")
    print(f"  " + "-" * 58)
    print(f"  {'Identification  (M1 FP + FN)':<32} {errors_id:>8}  {pct_id:>13.1f}%")
    print(f"  {'Categorisation  (M2 FP + FN)':<32} {errors_cat:>8}  {pct_cat:>13.1f}%")


def format_by_type(by_entity_type: dict, min_count: int = 20):
    """
    Prints M1 and M3 F1 side-by-side for each entity category.

    Showing both metrics reveals whether a category's errors are mainly
    due to identification (low M1) or categorisation (high M1, low M3).
    Only categories with at least min_count GT entities are shown,
    sorted by M3 F1 ascending (worst first).
    """
    m1_types = by_entity_type.get('m1', {})
    m3_types = by_entity_type.get('m3', {})
    all_types = set(m1_types.keys()) | set(m3_types.keys())

    rows = []
    for etype in all_types:
        m1s = m1_types.get(etype, {'tp': 0, 'fp': 0, 'fn': 0, 'f1': 0.0})
        m3s = m3_types.get(etype, {'tp': 0, 'fp': 0, 'fn': 0, 'f1': 0.0})
        gt_count = m1s['tp'] + m1s['fn']
        if gt_count < min_count:
            continue
        rows.append((etype, m1s['f1'], m3s['f1'], m1s['tp'], m1s['fp'], m1s['fn'],
                     m3s['tp'], m3s['fp'], m3s['fn'], gt_count))

    if not rows:
        return

    rows.sort(key=lambda x: x[2])  # M3 F1 ascendente, peor primero

    print(f"\n  By category  (>={min_count} GT entities, sorted by M3 F1 asc):")
    print(f"  {'Type':<45} {'M1_F1':>7} {'M3_F1':>7}   {'M1_FP':>6} {'M1_FN':>6}   {'M3_FP':>6} {'M3_FN':>6}  {'GT':>6}")
    print(f"  " + "-" * 106)
    for etype, m1f1, m3f1, m1tp, m1fp, m1fn, m3tp, m3fp, m3fn, gt in rows:
        print(
            f"  {etype:<45} {m1f1:>7.4f} {m3f1:>7.4f}"
            f"   {m1fp:>6} {m1fn:>6}"
            f"   {m3fp:>6} {m3fn:>6}  {gt:>6}"
        )


def format_by_language(by_language: dict, model_name: str):
    """Prints metrics broken down by language."""
    if not by_language:
        return
    print(f"\n  By language ({model_name}):")
    print(f"  {'Lang':>6} {'M1_F1':>7} {'M2_F1':>7} {'M3_F1':>7}")
    print(f"  " + "-" * 30)
    for lang, metrics in sorted(by_language.items()):
        print(
            f"  {lang:>6} "
            f"{metrics['m1']['f1']:>7.4f} "
            f"{metrics['m2']['f1']:>7.4f} "
            f"{metrics['m3']['f1']:>7.4f}"
        )


def format_by_entity_type(by_entity_type: dict, model_name: str):
    """Prints M1 and M3 metrics by entity type."""
    m1_types = by_entity_type.get('m1', {})
    m3_types = by_entity_type.get('m3', {})
    all_types = sorted(set(list(m1_types.keys()) + list(m3_types.keys())))

    if not all_types:
        return

    print(f"\n  By entity type ({model_name}):")
    print(f"  {'Tipo':<40} {'M1_F1':>7} {'M3_F1':>7} {'GT_count':>9}")
    print(f"  " + "-" * 65)

    for etype in all_types:
        m1 = m1_types.get(etype, {})
        m3 = m3_types.get(etype, {})
        gt_count = m1.get('tp', 0) + m1.get('fn', 0)
        print(
            f"  {etype:<40} "
            f"{m1.get('f1', 0.0):>7.4f} "
            f"{m3.get('f1', 0.0):>7.4f} "
            f"{gt_count:>9}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Medical record anonymisation model evaluation'
    )
    parser.add_argument('--models-dir', default='models',
                        help='Directory containing model JSON files (default: models/)')
    parser.add_argument('--results-dir', default='results',
                        help='Output directory for results (default: results/)')
    parser.add_argument('--excluded-docs', default='scripts/problematic_docs.json',
                        help='JSON file listing documents to exclude')
    parser.add_argument('--language-file', default='CARMEN1_mappings.tsv',
                        help='TSV file with per-document language metadata')
    parser.add_argument('--no-partial-match', action='store_true',
                        help='Disable partial/substring matching')
    parser.add_argument('--partial-match-threshold', type=float, default=0.6,
                        help='SequenceMatcher threshold for partial matching [0-1] (default: 0.6)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show per-language and per-entity-type breakdown')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CARMEN-I ANONYMISATION EVALUATION")
    print("=" * 60)

    # Load metadata
    print("\nLoading metadata...")
    excluded_docs = load_excluded_docs(args.excluded_docs)
    language_map = load_language_map(args.language_file)

    # Matching configuration
    config = NormalizationConfig(
        normalize_case=True,
        partial_match=not args.no_partial_match,
        partial_match_threshold=args.partial_match_threshold,
    )

    print(f"\nMatching configuration:")
    print(f"  normalize_case  : True (strip + lowercase)")
    print(f"  partial_match   : {config.partial_match}")
    if not args.no_partial_match:
        print(f"  partial_match_threshold: {config.partial_match_threshold}")

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    # Find model files
    model_files = sorted(glob.glob(os.path.join(args.models_dir, '*.json')))
    if not model_files:
        print(f"\n[WARN] No JSON files found in '{args.models_dir}/'")
        print("  Place output_<model>.json files in that folder.")
        return

    print(f"\n{len(model_files)} model(s) found in '{args.models_dir}/'")

    summary_data = []

    for model_path in model_files:
        filename = os.path.basename(model_path)
        # Extraer nombre del modelo quitando prefijo 'output_' y sufijo '.json'
        model_name = filename
        if model_name.startswith('output_'):
            model_name = model_name[7:]
        if model_name.endswith('.json'):
            model_name = model_name[:-5]

        print(f"\n{'-' * 60}")
        print(f"Processing: {model_name}")

        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [ERROR] Failed to load {filename}: {e}")
            continue

        # Evaluar
        results = evaluate_all(
            data=data,
            config=config,
            excluded_docs=excluded_docs,
            language_map=language_map,
        )

        counts = results['counts']
        print(f"  Total documents       : {counts['total']}")
        print(f"  Excluded              : {counts['excluded']}")
        print(f"  No ground truth       : {counts['no_gt']}")
        print(f"  Evaluated             : {counts['evaluated']}")
        if counts['length_mismatches'] > 0:
            print(f"  [WARN] LLM span mismatches: {counts['length_mismatches']}")
            ap = results['global'].get('align_phases', {})
            total = ap.get('total_resolved', 0)
            unk   = ap.get('phase3_unknown', 0)
            print(f"         Context-anchor alignment — spans in mismatched docs:")
            print(f"           phase 0 (empty anchor/order)  : {ap.get('phase0_empty_anchor', 0):>5}")
            print(f"           phase 1 (direct match)        : {ap.get('phase1_direct', 0):>5}")
            print(f"           phase 2 (skip extra anon)     : {ap.get('phase2_skip', 0):>5}")
            print(f"           phase 3 (UNKNOWN — dropped)   : {unk:>5}")
            if total + unk > 0:
                pct = unk / (total + unk) * 100
                print(f"           → {pct:.1f}% of spans unresolved (UNKNOWN)")

        g = results['global']
        print(f"\n  {'Metric':<6} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
        print(f"  " + "-" * 48)
        for metric_name, key in [('M1', 'm1'), ('M2', 'm2'), ('M3', 'm3')]:
            m = g[key]
            print(
                f"  {metric_name:<6} "
                f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
                f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}"
            )

        fid  = g.get('faithfulness_identify_mean')
        fanon = g.get('faithfulness_anonymize_mean')
        if fid is not None or fanon is not None:
            print(f"\n  Faithfulness (cosine over inter-span text):")
            fid_str  = f'{fid:.4f}'  if fid  is not None else 'N/A'
            fanon_str = f'{fanon:.4f}' if fanon is not None else 'N/A'
            print(f"  {'Call 1 identify':<28}: {fid_str}")
            print(f"  {'Call 2 anonymize':<28}: {fanon_str}")

        # Option A: M3 error attribution
        format_error_attribution(g)

        # Option B: M1 and M3 by entity type
        format_by_type(results['by_entity_type'])

        if args.verbose:
            format_by_language(results['by_language'], model_name)
            format_by_entity_type(results['by_entity_type'], model_name)

        # ── Save results ────────────────────────────────────────────────

        # 1. Global metrics
        metrics_path = os.path.join(args.results_dir, f'{model_name}_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': model_name,
                'matching_config': {
                    'normalize_case': config.normalize_case,
                    'partial_match': config.partial_match,
                },
                'counts': counts,
                'global': g,
            }, f, indent=2, ensure_ascii=False)

        # 2. Metrics by language
        lang_path = os.path.join(args.results_dir, f'{model_name}_by_language.json')
        with open(lang_path, 'w', encoding='utf-8') as f:
            json.dump(results['by_language'], f, indent=2, ensure_ascii=False)

        # 3. Metrics by entity type
        type_path = os.path.join(args.results_dir, f'{model_name}_by_entity_type.json')
        with open(type_path, 'w', encoding='utf-8') as f:
            json.dump(results['by_entity_type'], f, indent=2, ensure_ascii=False)

        # 4. Per-document results (for analyze_errors and identify_worst_docs)
        doc_path = os.path.join(args.results_dir, f'{model_name}_document_results.json')
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(results['document_results'], f, indent=2, ensure_ascii=False)

        # 5. Partial match pairs (only when present)
        partial_pairs = results.get('global_partial_pairs', [])
        if partial_pairs:
            pp_path = os.path.join(args.results_dir, f'{model_name}_partial_pairs.json')
            with open(pp_path, 'w', encoding='utf-8') as f:
                json.dump(partial_pairs, f, indent=2, ensure_ascii=False)
            print(f"  Partial pairs saved ({len(partial_pairs)}) to '{pp_path}'")

        print(f"\n  Results saved to '{args.results_dir}'")

        # 6. Confusion matrix plot
        plots_dir = os.path.join(args.results_dir, 'plots')
        _, cm = load_confusion(metrics_path)
        if cm is not None:
            plot_confusion(model_name, cm, plots_dir)

        # Add to global summary
        m3_total_errors = g['m3']['fp'] + g['m3']['fn']
        errors_id  = g['m1']['fp'] + g['m1']['fn']
        errors_cat = g['m2']['fp'] + g['m2']['fn']
        summary_data.append({
            'model': model_name,
            'docs_m1': counts['evaluated'],
            'docs_m2_m3': counts['evaluated'] - counts.get('anon_text_missing', 0),
            'm1_precision': g['m1']['precision'],
            'm1_recall': g['m1']['recall'],
            'm1_f1': g['m1']['f1'],
            'm2_f1': g['m2']['f1'],
            'm3_precision': g['m3']['precision'],
            'm3_recall': g['m3']['recall'],
            'm3_f1': g['m3']['f1'],
            # M3 error attribution
            'm3_errors_total': m3_total_errors,
            'm3_errors_identification': errors_id,
            'm3_errors_categorization': errors_cat,
            'm3_pct_identification': round(errors_id  / m3_total_errors * 100, 1) if m3_total_errors else 0.0,
            'm3_pct_categorization':  round(errors_cat / m3_total_errors * 100, 1) if m3_total_errors else 0.0,
            # Faithfulness
            'faithfulness_identify_mean':  g.get('faithfulness_identify_mean'),
            'faithfulness_anonymize_mean': g.get('faithfulness_anonymize_mean'),
        })

    # Save and display comparative summary
    if summary_data:
        summary_path = os.path.join(args.results_dir, 'models_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        format_metrics_table(summary_data)
        print(f"\nSummary saved to '{args.results_dir}/models_summary.json'")

        # Plots
        plots_dir = os.path.join(args.results_dir, 'plots')
        rows = load_all_metrics(args.results_dir)
        if rows:
            plot_comparison(rows, os.path.join(plots_dir, 'models_comparison.png'))
        plot_summary_table(summary_data, plots_dir)

    print("\n[OK] Evaluation complete.\n")


if __name__ == '__main__':
    main()
