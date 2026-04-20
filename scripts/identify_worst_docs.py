#!/usr/bin/env python3
"""
identify_worst_docs.py — Documents with worst performance
==========================================================

Reads the _document_results.json file produced by run_evaluation.py
and lists the documents with the lowest F1 across the three metrics.

Usage:
    python identify_worst_docs.py <model_document_results.json> [top_n]
    python identify_worst_docs.py results/claude-haiku_document_results.json 20
"""

import json
import sys
import os


def show_worst(doc_results, top_n=20):
    model_name = "modelo"

    # Sort by M3 F1 (end-to-end, most demanding)
    docs_sorted = sorted(doc_results, key=lambda d: d['m3']['f1'])

    total = len(doc_results)
    f1_zero = sum(1 for d in doc_results if d['m3']['f1'] == 0.0)
    f1_low  = sum(1 for d in doc_results if 0 < d['m3']['f1'] < 0.5)
    f1_mid  = sum(1 for d in doc_results if 0.5 <= d['m3']['f1'] < 0.8)
    f1_high = sum(1 for d in doc_results if d['m3']['f1'] >= 0.8)

    print("=" * 80)
    print(f"WORST-PERFORMING DOCUMENTS (M3 end-to-end F1)")
    print("=" * 80)
    print(f"\nTotal documents evaluated: {total}")
    print(f"\nM3 F1 distribution:")
    print(f"  F1 = 0.0    : {f1_zero:4d} ({100*f1_zero/total:.1f}%)")
    print(f"  F1 < 0.5    : {f1_low:4d} ({100*f1_low/total:.1f}%)")
    print(f"  F1 0.5–0.8  : {f1_mid:4d} ({100*f1_mid/total:.1f}%)")
    print(f"  F1 ≥ 0.8    : {f1_high:4d} ({100*f1_high/total:.1f}%)")

    print(f"\n{'Document':<45} {'Lang':>4} {'GT':>4} {'Pred':>4} {'M1_F1':>7} {'M2_F1':>7} {'M3_F1':>7}")
    print("-" * 80)

    for doc in docs_sorted[:top_n]:
        print(
            f"{doc['namefile'][:44]:<45} "
            f"{doc.get('language','?'):>4} "
            f"{doc['gt_count']:>4} "
            f"{doc['pred_count']:>4} "
            f"{doc['m1']['f1']:>7.3f} "
            f"{doc['m2']['f1']:>7.3f} "
            f"{doc['m3']['f1']:>7.3f}"
        )

    print("\n" + "=" * 80)
    print("DETAIL — 10 WORST DOCUMENTS")
    print("=" * 80)

    for i, doc in enumerate(docs_sorted[:10]):
        print(f"\n{i+1}. {doc['namefile']}  [{doc.get('language','?')}]")
        print(f"   GT    ({doc['gt_count']}): {doc.get('gt_entities', [])[:5]}")
        print(f"   Pred  ({doc['pred_count']}): {doc.get('pred_entities', [])[:5]}")
        print(f"   M1 → TP={doc['m1']['tp']}  FP={doc['m1']['fp']}  FN={doc['m1']['fn']}  F1={doc['m1']['f1']:.3f}")
        print(f"   M2 → TP={doc['m2']['tp']}  FP={doc['m2']['fp']}  FN={doc['m2']['fn']}  F1={doc['m2']['f1']:.3f}")
        print(f"   M3 → TP={doc['m3']['tp']}  FP={doc['m3']['fp']}  FN={doc['m3']['fn']}  F1={doc['m3']['f1']:.3f}")
        if doc.get('length_mismatch'):
            print(f"   ⚠  Span mismatch: {doc['mismatch_detail']}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python identify_worst_docs.py <model_document_results.json> [top_n]")
        sys.exit(1)

    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    with open(path, 'r', encoding='utf-8') as f:
        doc_results = json.load(f)

    show_worst(doc_results, top_n)
