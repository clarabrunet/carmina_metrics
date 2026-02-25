#!/usr/bin/env python3
"""
Unified Model Evaluation Script
Evaluates anonymization models with configurable normalization techniques.

Three evaluation levels:
1. Identity: Entity detection (text match)
2. Category: Entity classification (type match for detected entities)
3. Combined: Entity detection + classification (text AND type match)

Usage:
    python evaluate_model.py <model_output.json> [options]
    
Options:
    --output-dir DIR         Output directory (default: results/)
    --merge-dates           Enable date fragment merging
    --standardize-seps      Standardize date separators (-, . → /)
    --partial-match         Enable partial/substring matching
    --all-normalization     Enable all normalization techniques
"""

import json
import re
import argparse
import os
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path


@dataclass
class NormalizationConfig:
    """Configuration for normalization techniques."""
    normalize_case: bool = True
    remove_accents: bool = False
    merge_dates: bool = False
    standardize_separators: bool = False
    partial_match: bool = False


@dataclass
class MetricsResult:
    """Metrics for a single evaluation level."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    df: int = 0  # Partial matches (Different/Partial)
    
    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)
    
    @property
    def accuracy(self) -> float:
        """Accuracy = TP / (TP + FP + FN)"""
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'df': self.df,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'accuracy': round(self.accuracy, 4)
        }


class Normalizer:
    """Handles all text normalization techniques."""
    
    DATE_PATTERNS = [
        r'^\d{1,2}$',              # "12", "5"
        r'^\d{2}/\d{2}$',          # "12/03"
        r'^\d{4}$',                # "2024"
        r'^\d{2}/\d{4}$',          # "03/2024"
        r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # "12/03/24"
        r'^\d{1,2}-\d{1,2}$',      # "12-03"
        r'^\d{1,2}\.\d{1,2}$',     # "12.03"
        r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$',  # Various date formats
    ]
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization: strip, lowercase, and optionally remove accents."""
        if not text:
            return ""
        result = text.strip()
        if self.config.normalize_case:
            result = result.lower()
        if self.config.remove_accents:
            # Normalize to NFKD and remove combining characters (accents)
            nfkd = unicodedata.normalize('NFKD', result)
            result = ''.join(c for c in nfkd if not unicodedata.combining(c))
        return result
    
    def is_date_component(self, text: str) -> bool:
        """Check if text looks like a date component."""
        if not text:
            return False
        return any(re.match(pattern, text) for pattern in self.DATE_PATTERNS)
    
    def standardize_separators(self, text: str) -> str:
        """Standardize date separators (-, .) to /"""
        if self.is_date_component(text):
            return text.replace('-', '/').replace('.', '/')
        return text
    
    def try_merge_date_fragments(self, e1: str, e2: str) -> Optional[str]:
        """Try to merge two consecutive date fragments."""
        if self.is_date_component(e1) and self.is_date_component(e2):
            for sep in ['/', '-', '.']:
                merged = f"{e1}{sep}{e2}"
                if self.is_date_component(merged):
                    return merged
        return None
    
    def normalize_entity_list(self, entities: List[str]) -> List[str]:
        """Apply all normalization to a list of entities."""
        if not entities:
            return []
        
        # Step 1: Basic text normalization
        normalized = [self.normalize_text(e) for e in entities if e and e.strip()]
        
        # Step 2: Standardize separators (if enabled)
        if self.config.standardize_separators:
            normalized = [self.standardize_separators(e) for e in normalized]
        
        # Step 3: Merge date fragments (if enabled)
        if self.config.merge_dates:
            merged = []
            i = 0
            while i < len(normalized):
                current = normalized[i]
                if i + 1 < len(normalized):
                    next_entity = normalized[i + 1]
                    merged_result = self.try_merge_date_fragments(current, next_entity)
                    if merged_result:
                        merged.append(merged_result)
                        i += 2
                        continue
                merged.append(current)
                i += 1
            normalized = merged
        
        return normalized


class EntityMatcher:
    """Handles entity matching logic."""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
    
    def exact_match(self, pred: str, gt: str) -> bool:
        """Check for exact text match."""
        return pred == gt
    
    def partial_match(self, pred: str, gt: str) -> bool:
        """Check for partial/substring match."""
        if not self.config.partial_match:
            return False
        return pred in gt or gt in pred
    
    def match_entities(
        self, 
        pred_entities: List[str], 
        gt_entities: List[str]
    ) -> Tuple[int, int, int, int, List[Tuple[int, int]]]:
        """
        Match predicted entities against ground truth.
        
        Returns:
            (tp, fp, fn, df, matches) where matches is list of (pred_idx, gt_idx) pairs
        """
        tp = 0
        df = 0
        
        gt_matched = [False] * len(gt_entities)
        pred_matched = [False] * len(pred_entities)
        matches = []
        
        # Pass 1: Exact matches
        for i, pred in enumerate(pred_entities):
            if pred_matched[i]:
                continue
            for j, gt in enumerate(gt_entities):
                if gt_matched[j]:
                    continue
                if self.exact_match(pred, gt):
                    tp += 1
                    pred_matched[i] = True
                    gt_matched[j] = True
                    matches.append((i, j))
                    break
        
        # Pass 2: Partial matches (if enabled)
        if self.config.partial_match:
            for i, pred in enumerate(pred_entities):
                if pred_matched[i]:
                    continue
                for j, gt in enumerate(gt_entities):
                    if gt_matched[j]:
                        continue
                    if self.partial_match(pred, gt):
                        df += 1
                        pred_matched[i] = True
                        gt_matched[j] = True
                        matches.append((i, j))
                        break
        
        # Count unmatched
        fp = sum(1 for m in pred_matched if not m)
        fn = sum(1 for m in gt_matched if not m)
        
        return tp, fp, fn, df, matches


class ModelEvaluator:
    """Main evaluator class."""
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.normalizer = Normalizer(config)
        self.matcher = EntityMatcher(config)
    
    def extract_ground_truth(self, doc: Dict) -> Tuple[List[str], List[str]]:
        """Extract GT entities and types from document labels."""
        labels = doc.get('labels', [])
        entities = []
        types = []
        for label in labels:
            text = label.get('text', '').strip()
            entity_type = label.get('type', 'UNKNOWN')
            if text:
                entities.append(text)
                types.append(entity_type)
        return entities, types
    
    def extract_predictions(self, doc: Dict) -> Tuple[List[str], List[str]]:
        """Extract predicted entities and types from document chunks."""
        chunks = doc.get('chunks', {})
        
        # Handle both dict and list formats
        if isinstance(chunks, list) and len(chunks) > 0:
            chunks = chunks[0]
        
        if isinstance(chunks, dict):
            entities = chunks.get('entities_identified', [])
            types = chunks.get('entities_anonymized', [])
        else:
            entities = []
            types = []
        
        # Handle case where entities is a dict instead of list
        if isinstance(entities, dict):
            entities = list(entities.values()) if entities else []
        if isinstance(types, dict):
            types = list(types.values()) if types else []
        
        # Ensure lists
        if not isinstance(entities, list):
            entities = []
        if not isinstance(types, list):
            types = []
        
        return entities, types
    
    def evaluate_document(self, doc: Dict) -> Dict[str, MetricsResult]:
        """Evaluate a single document."""
        gt_entities, gt_types = self.extract_ground_truth(doc)
        pred_entities, pred_types = self.extract_predictions(doc)
        
        # Normalize entities
        gt_norm = self.normalizer.normalize_entity_list(gt_entities)
        pred_norm = self.normalizer.normalize_entity_list(pred_entities)
        
        results = {
            'identity': MetricsResult(),
            'category': MetricsResult(),
            'combined': MetricsResult()
        }
        
        # === IDENTITY METRICS (entity detection) ===
        tp, fp, fn, df, matches = self.matcher.match_entities(pred_norm, gt_norm)
        results['identity'].tp = tp + df  # Partial matches count as TP for identity
        results['identity'].fp = fp
        results['identity'].fn = fn
        results['identity'].df = df
        
        # === CATEGORY METRICS (for matched entities only) ===
        # For each matched pair, check if types match
        category_tp = 0
        category_fp = 0
        
        for pred_idx, gt_idx in matches:
            # Get types for matched entities
            pred_type = pred_types[pred_idx] if pred_idx < len(pred_types) else 'UNKNOWN'
            gt_type = gt_types[gt_idx] if gt_idx < len(gt_types) else 'UNKNOWN'
            
            if pred_type == gt_type:
                category_tp += 1
            else:
                category_fp += 1
        
        results['category'].tp = category_tp
        results['category'].fp = category_fp
        results['category'].fn = len(matches) - category_tp  # Matched but wrong type
        
        # === COMBINED METRICS (entity + type must both match) ===
        combined_tp = 0
        combined_fp = 0
        
        for pred_idx, gt_idx in matches:
            pred_type = pred_types[pred_idx] if pred_idx < len(pred_types) else 'UNKNOWN'
            gt_type = gt_types[gt_idx] if gt_idx < len(gt_types) else 'UNKNOWN'
            
            if pred_type == gt_type:
                combined_tp += 1
            else:
                combined_fp += 1
        
        # Add unmatched predictions as FP
        combined_fp += fp
        
        # FN = GT entities not found OR found but with wrong type
        combined_fn = fn + (len(matches) - combined_tp)
        
        results['combined'].tp = combined_tp
        results['combined'].fp = combined_fp
        results['combined'].fn = combined_fn
        
        return results
    
    def evaluate_all(self, data: List[Dict], include_empty_gt: bool = False, excluded_docs: set = None) -> Dict:
        """Evaluate all documents and aggregate metrics.
        
        Args:
            data: List of documents to evaluate
            include_empty_gt: If True, include documents without GT (predictions count as FP)
            excluded_docs: Set of document names/IDs to skip
        """
        total_metrics = {
            'identity': MetricsResult(),
            'category': MetricsResult(),
            'combined': MetricsResult()
        }
        
        by_type = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        docs_evaluated = 0
        docs_empty_gt = 0
        docs_excluded = 0
        
        for doc in data:
            doc_id = doc.get('namefile', doc.get('id', ''))
            if excluded_docs and doc_id in excluded_docs:
                docs_excluded += 1
                continue

            gt_entities, gt_types = self.extract_ground_truth(doc)
            pred_entities, pred_types = self.extract_predictions(doc)
            
            # Handle documents without ground truth
            if not gt_entities:
                docs_empty_gt += 1
                if not include_empty_gt:
                    continue
                # Count predictions as FP when GT is empty
                num_preds = len(pred_entities) if pred_entities else 0
                if num_preds > 0:
                    docs_evaluated += 1
                    total_metrics['identity'].fp += num_preds
                    total_metrics['combined'].fp += num_preds
                continue
            
            docs_evaluated += 1
            doc_metrics = self.evaluate_document(doc)
            
            # Aggregate metrics
            for level in ['identity', 'category', 'combined']:
                total_metrics[level].tp += doc_metrics[level].tp
                total_metrics[level].fp += doc_metrics[level].fp
                total_metrics[level].fn += doc_metrics[level].fn
                total_metrics[level].df += doc_metrics[level].df
        
        return {
            'documents_evaluated': docs_evaluated,
            'documents_without_gt': docs_empty_gt,
            'include_empty_gt': include_empty_gt,
            'identity': total_metrics['identity'].to_dict(),
            'category': total_metrics['category'].to_dict(),
            'combined': total_metrics['combined'].to_dict()
        }


def extract_model_name(json_path: str) -> str:
    """Extract model name from JSON filename."""
    filename = os.path.basename(json_path)
    if filename.startswith('output_'):
        filename = filename[7:]
    if filename.endswith('.json'):
        filename = filename[:-5]
    return filename


def print_results(results: Dict, model_name: str, config: NormalizationConfig):
    """Print formatted results to console."""
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    print(f"Documents evaluated: {results['documents_evaluated']}")
    
    print(f"\nNormalization config:")
    print(f"  - Case normalization: {config.normalize_case}")
    print(f"  - Date fragment merging: {config.merge_dates}")
    print(f"  - Separator standardization: {config.standardize_separators}")
    print(f"  - Partial matching: {config.partial_match}")
    
    for level in ['identity', 'category', 'combined']:
        m = results[level]
        print(f"\n--- {level.upper()} METRICS ---")
        print(f"  TP: {m['tp']:,}  |  FP: {m['fp']:,}  |  FN: {m['fn']:,}  |  DF: {m['df']:,}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall:    {m['recall']:.4f}")
        print(f"  F1-Score:  {m['f1']:.4f}")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate anonymization model with configurable normalization'
    )
    parser.add_argument('input_json', help='Path to model output JSON file')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--merge-dates', action='store_true', 
                        help='Enable date fragment merging')
    parser.add_argument('--standardize-seps', action='store_true',
                        help='Standardize date separators (-, . → /)')
    parser.add_argument('--partial-match', action='store_true',
                        help='Enable partial/substring matching')
    parser.add_argument('--all-normalization', action='store_true',
                        help='Enable all normalization techniques')
    parser.add_argument('--analyze-docs', action='store_true',
                        help='Generate per-document analysis to identify problematic documents')
    parser.add_argument('--top-errors', type=int, default=20,
                        help='Number of worst documents to show (default: 20)')
    parser.add_argument('--include-empty-gt', action='store_true',
                        help='Include documents without GT (predictions count as FP)')
    parser.add_argument('--normalize-accents', action='store_true',
                        help='Remove accents/diacritics for matching (día → dia)')
    parser.add_argument('--exclude-docs', help='Path to JSON file with list of documents to exclude')
    
    args = parser.parse_args()
    
    # Load exclusions if provided
    excluded_docs = set()
    if args.exclude_docs:
        try:
            with open(args.exclude_docs, 'r', encoding='utf-8') as f:
                exclusions = json.load(f)
                # Handle both list of strings or list of objects with 'namefile'
                for item in exclusions:
                    if isinstance(item, str):
                        excluded_docs.add(item)
                    elif isinstance(item, dict) and 'namefile' in item:
                        excluded_docs.add(item['namefile'])
            print(f"Loaded {len(excluded_docs)} documents to exclude.")
        except Exception as e:
            print(f"Error loading exclusion file: {e}")
            return

    # Configure normalization
    config = NormalizationConfig(
        normalize_case=True,
        remove_accents=args.normalize_accents or args.all_normalization,
        merge_dates=args.merge_dates or args.all_normalization,
        standardize_separators=args.standardize_seps or args.all_normalization,
        partial_match=args.partial_match or args.all_normalization
    )
    
    # Load data
    print(f"Loading: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Evaluate
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_all(
        data, 
        include_empty_gt=args.include_empty_gt,
        excluded_docs=excluded_docs
    )
    
    # Add metadata
    model_name = extract_model_name(args.input_json)
    output = {
        'model_name': model_name,
        'normalization_config': asdict(config),
        'metrics': results
    }
    
    # Print results
    print_results(results, model_name, config)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{model_name}_metrics.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Per-document analysis
    if args.analyze_docs:
        print("\n" + "=" * 70)
        print("DOCUMENT-LEVEL ANALYSIS")
        print("=" * 70)
        
        doc_results = analyze_documents(data, evaluator)
        
        # Sort by combined F1 (ascending - worst first)
        doc_results.sort(key=lambda x: x['combined_f1'])
        
        # Save full analysis
        analysis_path = os.path.join(args.output_dir, f'{model_name}_document_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(doc_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nFull analysis saved to: {analysis_path}")
        
        # Show worst documents
        print(f"\n--- TOP {args.top_errors} WORST PERFORMING DOCUMENTS ---\n")
        print(f"{'Rank':<5} {'Document':<40} {'Id F1':<8} {'Cat F1':<8} {'Comb F1':<8} {'Errors':<8}")
        print("-" * 80)
        
        for i, doc in enumerate(doc_results[:args.top_errors]):
            errors = doc['identity_fp'] + doc['identity_fn'] + doc['category_fp']
            print(f"{i+1:<5} {doc['namefile'][:38]:<40} {doc['identity_f1']:<8.3f} {doc['category_f1']:<8.3f} {doc['combined_f1']:<8.3f} {errors:<8}")
        
        # Show error type breakdown
        print(f"\n--- ERROR TYPE SUMMARY (worst {args.top_errors} docs) ---\n")
        
        total_fp = sum(d['identity_fp'] for d in doc_results[:args.top_errors])
        total_fn = sum(d['identity_fn'] for d in doc_results[:args.top_errors])
        total_cat_err = sum(d['category_fp'] for d in doc_results[:args.top_errors])
        
        print(f"Identity False Positives (over-detection): {total_fp}")
        print(f"Identity False Negatives (missed entities): {total_fn}")
        print(f"Category Errors (wrong type assigned): {total_cat_err}")


def analyze_documents(data: List[Dict], evaluator: ModelEvaluator) -> List[Dict]:
    """Analyze each document individually and return detailed metrics."""
    doc_results = []
    
    for doc in data:
        gt_entities, gt_types = evaluator.extract_ground_truth(doc)
        
        # Skip documents without ground truth
        if not gt_entities:
            continue
        
        metrics = evaluator.evaluate_document(doc)
        
        # Calculate F1 scores
        def calc_f1(m):
            if m.precision + m.recall == 0:
                return 0.0
            return 2 * m.precision * m.recall / (m.precision + m.recall)
        
        # Get predicted entities for error details
        pred_entities, pred_types = evaluator.extract_predictions(doc)
        
        doc_results.append({
            'namefile': doc.get('namefile', doc.get('id', 'unknown')),
            'doc_id': doc.get('id', 'unknown'),
            # Identity metrics
            'identity_tp': metrics['identity'].tp,
            'identity_fp': metrics['identity'].fp,
            'identity_fn': metrics['identity'].fn,
            'identity_f1': calc_f1(metrics['identity']),
            # Category metrics
            'category_tp': metrics['category'].tp,
            'category_fp': metrics['category'].fp,
            'category_f1': calc_f1(metrics['category']),
            # Combined metrics
            'combined_tp': metrics['combined'].tp,
            'combined_fp': metrics['combined'].fp,
            'combined_fn': metrics['combined'].fn,
            'combined_f1': calc_f1(metrics['combined']),
            # Entity counts
            'gt_entity_count': len(gt_entities),
            'pred_entity_count': len(pred_entities),
            # Error details
            'gt_entities': gt_entities[:10],  # First 10 for reference
            'pred_entities': pred_entities[:10] if pred_entities else []
        })
    
    return doc_results


if __name__ == '__main__':
    main()
