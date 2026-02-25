#!/usr/bin/env python3
"""
Identify documents with low performance metrics.
Shows distribution and lists worst performing documents.
"""

import json
import sys
import os

def analyze_worst_docs(analysis_path: str, top_n: int = 30):
    """Analyze and display worst performing documents."""
    
    with open(analysis_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    model_name = os.path.basename(analysis_path).replace('_document_analysis.json', '')
    
    print('=' * 80)
    print(f'DOCUMENTOS CON PEOR RENDIMIENTO - {model_name}')
    print('=' * 80)
    
    # Sort by combined F1
    docs_sorted = sorted(docs, key=lambda x: x['combined_f1'])
    
    # Count by F1 threshold
    f1_0 = sum(1 for d in docs if d['combined_f1'] == 0.0)
    f1_low = sum(1 for d in docs if 0 < d['combined_f1'] < 0.5)
    f1_mid = sum(1 for d in docs if 0.5 <= d['combined_f1'] < 0.8)
    f1_high = sum(1 for d in docs if d['combined_f1'] >= 0.8)
    
    print(f'\nTotal documentos evaluados: {len(docs)}')
    print()
    print('Distribucion por Combined F1:')
    print(f'  F1 = 0.0:     {f1_0:4} documentos ({100*f1_0/len(docs):.1f}%)')
    print(f'  F1 < 0.5:     {f1_low:4} documentos ({100*f1_low/len(docs):.1f}%)')
    print(f'  F1 0.5-0.8:   {f1_mid:4} documentos ({100*f1_mid/len(docs):.1f}%)')
    print(f'  F1 >= 0.8:    {f1_high:4} documentos ({100*f1_high/len(docs):.1f}%)')
    
    print()
    print('=' * 80)
    print(f'TOP {top_n} PEORES DOCUMENTOS')
    print('=' * 80)
    
    header = f"{'Documento':<45} {'GT':>4} {'Pred':>4} {'IdF1':>6} {'CatF1':>6} {'CombF1':>7}"
    print(header)
    print('-' * 80)
    
    for doc in docs_sorted[:top_n]:
        name = doc['namefile'][:44]
        gt = doc['gt_entity_count']
        pred = doc['pred_entity_count']
        id_f1 = doc['identity_f1']
        cat_f1 = doc['category_f1']
        comb_f1 = doc['combined_f1']
        print(f"{name:<45} {gt:>4} {pred:>4} {id_f1:>6.3f} {cat_f1:>6.3f} {comb_f1:>7.3f}")
    
    # Show details for worst documents
    print()
    print('=' * 80)
    print('DETALLE DE LOS 10 PEORES DOCUMENTOS')
    print('=' * 80)
    
    for i, doc in enumerate(docs_sorted[:10]):
        print(f"\n{i+1}. {doc['namefile']}")
        print(f"   GT entities ({doc['gt_entity_count']}): {doc['gt_entities']}")
        print(f"   Predictions ({doc['pred_entity_count']}): {doc['pred_entities']}")
        print(f"   Identity: TP={doc['identity_tp']}, FP={doc['identity_fp']}, FN={doc['identity_fn']}")
        print(f"   Category: TP={doc['category_tp']}, FP={doc['category_fp']}")
    
    # Export worst docs to separate file
    output_path = analysis_path.replace('_document_analysis.json', '_worst_docs.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(docs_sorted[:top_n], f, indent=2, ensure_ascii=False)
    
    print(f"\n\nLista completa guardada en: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python identify_worst_docs.py <model_document_analysis.json> [top_n]")
        print("Example: python identify_worst_docs.py results/claude-4.5-sonnet_document_analysis.json 30")
        sys.exit(1)
    
    analysis_path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    analyze_worst_docs(analysis_path, top_n)
