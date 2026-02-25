#!/usr/bin/env python3
"""
Detailed error analysis for NER evaluation.
Analyzes missed entities, false positives, and category errors.
"""

import json
from collections import defaultdict
import sys

def analyze_errors(model_path: str):
    """Analyze errors in detail for a model."""
    
    with open(model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Counters
    fn_by_type = defaultdict(list)  # Missed entities by GT type
    fp_entities = []  # False positive entities
    category_errors = []  # Detected but wrong type
    
    total_gt = 0
    total_pred = 0
    
    for doc in data:
        labels = doc.get('labels', [])
        if not labels:
            continue
        
        chunks = doc.get('chunks', {})
        if isinstance(chunks, list):
            chunks = chunks[0] if chunks else {}
        
        pred_entities = chunks.get('entities_identified', []) if isinstance(chunks, dict) else []
        pred_types = chunks.get('entities_anonymized', []) if isinstance(chunks, dict) else []
        
        if isinstance(pred_entities, dict):
            pred_entities = list(pred_entities.values())
        if isinstance(pred_types, dict):
            pred_types = list(pred_types.values())
        
        pred_entities = pred_entities or []
        pred_types = pred_types or []
        
        total_gt += len(labels)
        total_pred += len(pred_entities)
        
        # Normalize for matching
        pred_lower = [p.lower().strip() for p in pred_entities]
        
        # Track matched indices
        gt_matched = [False] * len(labels)
        pred_matched = [False] * len(pred_entities)
        
        # Match and analyze
        for i, label in enumerate(labels):
            gt_text = label.get('text', '').lower().strip()
            gt_type = label.get('type', 'UNKNOWN')
            
            matched = False
            for j, pred in enumerate(pred_lower):
                if pred_matched[j]:
                    continue
                    
                # Exact match or partial match
                if pred == gt_text or pred in gt_text or gt_text in pred:
                    gt_matched[i] = True
                    pred_matched[j] = True
                    matched = True
                    
                    # Check type
                    pred_type = pred_types[j] if j < len(pred_types) else 'UNKNOWN'
                    if pred_type != gt_type:
                        category_errors.append({
                            'entity': pred_entities[j],
                            'gt_type': gt_type,
                            'pred_type': pred_type,
                            'doc': doc.get('namefile', 'unknown')
                        })
                    break
            
            if not matched:
                fn_by_type[gt_type].append({
                    'text': label.get('text', ''),
                    'doc': doc.get('namefile', 'unknown')
                })
        
        # Track FP (unmatched predictions)
        for j, pred in enumerate(pred_entities):
            if not pred_matched[j]:
                fp_entities.append({
                    'text': pred,
                    'type': pred_types[j] if j < len(pred_types) else 'UNKNOWN',
                    'doc': doc.get('namefile', 'unknown')
                })
    
    return {
        'total_gt': total_gt,
        'total_pred': total_pred,
        'fn_by_type': fn_by_type,
        'fp_entities': fp_entities,
        'category_errors': category_errors
    }


def print_report(analysis: dict, model_name: str):
    """Print detailed error report."""
    
    print("=" * 70)
    print(f"DETAILED ERROR ANALYSIS: {model_name}")
    print("=" * 70)
    print(f"\nTotal GT entities: {analysis['total_gt']:,}")
    print(f"Total Predictions: {analysis['total_pred']:,}")
    
    # 1. FALSE NEGATIVES BY TYPE
    print("\n" + "-" * 70)
    print("FALSE NEGATIVES BY TYPE (Missed Entities)")
    print("-" * 70)
    
    fn_by_type = analysis['fn_by_type']
    total_fn = sum(len(v) for v in fn_by_type.values())
    print(f"Total missed: {total_fn}")
    print()
    
    for etype, entities in sorted(fn_by_type.items(), key=lambda x: -len(x[1])):
        unique_texts = list(set(e['text'] for e in entities))
        print(f"{etype}: {len(entities)} missed ({len(unique_texts)} unique)")
        for text in unique_texts[:5]:
            print(f"  - {text}")
        if len(unique_texts) > 5:
            print(f"  ... and {len(unique_texts) - 5} more")
        print()
    
    # 2. FALSE POSITIVES
    print("-" * 70)
    print("FALSE POSITIVES (Over-detection)")
    print("-" * 70)
    
    fp = analysis['fp_entities']
    print(f"Total FP: {len(fp)}")
    
    fp_by_type = defaultdict(list)
    for f in fp:
        fp_by_type[f['type']].append(f['text'])
    
    print("\nBy predicted type:")
    for ftype, entities in sorted(fp_by_type.items(), key=lambda x: -len(x[1])):
        unique = list(set(entities))
        print(f"  {ftype}: {len(entities)} ({len(unique)} unique)")
        for e in unique[:3]:
            print(f"    - {e}")
        print()
    
    # 3. CATEGORY ERRORS
    print("-" * 70)
    print("CATEGORY ERRORS (Detected but Wrong Type)")
    print("-" * 70)
    
    cat_errors = analysis['category_errors']
    print(f"Total: {len(cat_errors)}")
    
    error_patterns = defaultdict(list)
    for err in cat_errors:
        pattern = f"{err['gt_type']} -> {err['pred_type']}"
        error_patterns[pattern].append(err['entity'])
    
    print("\nError patterns (GT -> Predicted):")
    for pattern, entities in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
        unique = list(set(entities))
        print(f"  {pattern}: {len(entities)} errors")
        for e in unique[:3]:
            print(f"    - {e}")
        print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_errors.py <model_output.json>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_name = model_path.split('/')[-1].replace('output_', '').replace('.json', '')
    
    analysis = analyze_errors(model_path)
    print_report(analysis, model_name)
