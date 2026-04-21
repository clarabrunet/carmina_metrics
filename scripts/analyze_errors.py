#!/usr/bin/env python3
"""
analyze_errors.py — Detailed per-model error analysis
======================================================

Reads the _document_results.json file produced by run_evaluation.py
and prints a structured error report:

  1. M1 false negatives: entities the model missed
     — broken down by entity type
     — with text examples to help spot patterns

  2. M1 false positives: entities the model hallucinated
     — broken down by predicted type

  3. M2 categorisation errors: entities found but assigned the wrong type
     — broken down as GT_TYPE → PRED_TYPE patterns

  4. M3 end-to-end summary: which entity types have the worst combined F1

Usage:
    python analyze_errors.py <model_document_results.json>
    python analyze_errors.py results/claude-haiku_document_results.json
"""

import json
import sys
import os
from collections import defaultdict
from typing import Dict, List


def analyze_errors(doc_results: List[Dict]) -> Dict:
    """
    Aggregate all errors across all documents.

    For each document we reconstruct errors from:
    - gt_entities / gt_types: ground truth
    - pred_entities / pred_types: model predictions
    - m1.fn / m1.fp: identification counters
    - m2_type_errors: categorisation errors already computed by evaluate_model.py

    M1 false negatives are reconstructed by comparing GT against entities that
    were not matched.  Since document_results does not store explicit matches,
    we use a multiset approximation: GT texts not found in pred_entities are
    treated as false negatives.
    """
    fn_by_type: Dict[str, List[Dict]] = defaultdict(list)
    fp_by_pred_type: Dict[str, List[Dict]] = defaultdict(list)
    category_error_patterns: Dict[str, List[str]] = defaultdict(list)

    for doc in doc_results:
        namefile = doc.get('namefile', '')
        gt_entities = [e.lower().strip() for e in doc.get('gt_entities', [])]
        gt_types = doc.get('gt_types', [])
        pred_entities = [e.lower().strip() for e in doc.get('pred_entities', [])]
        pred_types = doc.get('pred_types', [])

        # Reconstruct which GT entities were not matched (M1 false negatives)
        # Multiset approach: consume predictions one by one
        pred_available = list(pred_entities)  # mutable copy
        for i, gt_text in enumerate(gt_entities):
            gt_type = gt_types[i] if i < len(gt_types) else 'UNKNOWN'
            if gt_text in pred_available:
                pred_available.remove(gt_text)  # consumir match
            else:
                # No encontrado → FN
                fn_by_type[gt_type].append({
                    'text': doc.get('gt_entities', [])[i] if i < len(doc.get('gt_entities', [])) else gt_text,
                    'doc': namefile,
                })

        # Reconstruct M1 false positives: predictions that matched no GT entity
        gt_available = list(gt_entities)
        for i, pred_text in enumerate(pred_entities):
            pred_type = pred_types[i] if i < len(pred_types) else 'UNKNOWN'
            if pred_text in gt_available:
                gt_available.remove(pred_text)
            else:
                fp_by_pred_type[pred_type].append({
                    'text': doc.get('pred_entities', [])[i] if i < len(doc.get('pred_entities', [])) else pred_text,
                    'doc': namefile,
                })

        # Categorisation errors (already computed in evaluate_document)
        for err in doc.get('m2_type_errors', []):
            pattern = f"{err['gt_type']} → {err['pred_type']}"
            category_error_patterns[pattern].append(err['entity'])

    return {
        'fn_by_type': fn_by_type,
        'fp_by_pred_type': fp_by_pred_type,
        'category_error_patterns': category_error_patterns,
    }


def print_report(analysis: Dict, model_name: str, doc_results: List[Dict]):
    """Print the formatted error report."""

    total_gt = sum(d.get('gt_count', 0) for d in doc_results)
    total_pred = sum(d.get('pred_count', 0) for d in doc_results)
    total_fn = sum(len(v) for v in analysis['fn_by_type'].values())
    total_fp = sum(len(v) for v in analysis['fp_by_pred_type'].values())
    total_cat_errors = sum(len(v) for v in analysis['category_error_patterns'].values())

    print("=" * 72)
    print(f"ERROR ANALYSIS — {model_name}")
    print("=" * 72)
    print(f"\nDocuments analysed    : {len(doc_results)}")
    print(f"GT entities (total)   : {total_gt:,}")
    print(f"Predicted entities    : {total_pred:,}")

    # ------------------------------------------------------------------
    # 1. FALSE NEGATIVES BY TYPE (M1: undetected entities)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"M1 — FALSE NEGATIVES: undetected entities ({total_fn} total)")
    print("-" * 72)

    fn_by_type = analysis['fn_by_type']
    for etype, items in sorted(fn_by_type.items(), key=lambda x: -len(x[1])):
        unique_texts = list(dict.fromkeys(e['text'] for e in items))  # preserve order
        print(f"\n  {etype}: {len(items)} occurrences ({len(unique_texts)} unique)")
        for text in unique_texts[:8]:
            count = sum(1 for e in items if e['text'] == text)
            suffix = f" (×{count})" if count > 1 else ""
            print(f"    · {text!r}{suffix}")
        if len(unique_texts) > 8:
            print(f"    ... and {len(unique_texts) - 8} more")

    # 2. FALSE POSITIVES BY PREDICTED TYPE (M1: hallucinated entities)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"M1 — FALSE POSITIVES: hallucinated entities ({total_fp} total)")
    print("-" * 72)

    fp_by_type = analysis['fp_by_pred_type']
    for ptype, items in sorted(fp_by_type.items(), key=lambda x: -len(x[1])):
        unique_texts = list(dict.fromkeys(e['text'] for e in items))
        print(f"\n  {ptype}: {len(items)} occurrences ({len(unique_texts)} unique)")
        for text in unique_texts[:5]:
            print(f"    · {text!r}")
        if len(unique_texts) > 5:
            print(f"    ... and {len(unique_texts) - 5} more")

    # 3. CATEGORISATION ERRORS (M2: wrong type)
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"M2 — CATEGORISATION ERRORS: wrong type ({total_cat_errors} total)")
    print("-" * 72)

    patterns = analysis['category_error_patterns']
    for pattern, entities in sorted(patterns.items(), key=lambda x: -len(x[1])):
        unique = list(dict.fromkeys(entities))
        print(f"\n  {pattern}: {len(entities)} errors")
        for e in unique[:4]:
            print(f"    · {e!r}")
        if len(unique) > 4:
            print(f"    ... and {len(unique) - 4} more")

    # 4. DOCUMENTS WITH SPAN MISMATCHES
    # ------------------------------------------------------------------
    mismatches = [d for d in doc_results if d.get('length_mismatch')]
    if mismatches:
        print("\n" + "-" * 72)
        print(f"WARNINGS: span mismatch between call 1 and call 2 ({len(mismatches)} docs)")
        print("-" * 72)
        for d in mismatches[:10]:
            print(f"  {d['namefile']}: {d['mismatch_detail']}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")

    print("\n" + "=" * 72)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <model_document_results.json>")
        print("Example: python analyze_errors.py results/claude-haiku_document_results.json")
        sys.exit(1)

    path = sys.argv[1]
    model_name = os.path.basename(path).replace('_document_results.json', '')

    with open(path, 'r', encoding='utf-8') as f:
        doc_results = json.load(f)

    analysis = analyze_errors(doc_results)
    print_report(analysis, model_name, doc_results)
