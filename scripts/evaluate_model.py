#!/usr/bin/env python3
"""
evaluate_model.py — Evaluation for medical record anonymisation models
=======================================================================

THREE METRICS
--------------

  M1 — IDENTIFICATION  (LLM call 1)
       Did the model find the right entities, regardless of category?
       GT   : labels[].text
       Pred : [**...**] spans extracted from chunks.identified_text
       Match: multiset, normalised text, exact-first then substring fallback

  M2 — CATEGORISATION  (LLM call 2, conditioned on M1)
       Of the M1 true-positives, was the right category assigned?
       Only M1-matched pairs are evaluated — M1 errors are not re-penalised.
       GT   : labels[].type  (for spans matched in M1)
       Pred : [**CATEGORY**] spans from chunks.anonymized_text

  M3 — END-TO-END  (full pipeline, call 2 output)
       Did the model identify AND categorise each entity correctly?
       A TP requires text AND type to match simultaneously.
       GT   : (labels[].text, labels[].type) pairs
       Pred : (id_span, anon_span) pairs

DESIGN NOTES
-------------
  - GT is read from labels[] in the JSON — no external .txt files needed.
  - Predictions are parsed from identified_text / anonymized_text via the
    [**...**] regex pattern. Positional alignment is guaranteed because
    anonymized_text is built by replacing each [**entity**] in identified_text
    with [**CATEGORY**] in order.
  - When span counts differ between the two calls, a context-anchor aligner
    recovers correct correspondence before falling back to UNKNOWN.
  - Matching uses two passes: exact first, then substring (optional).
    This prevents a partial match from stealing a later exact match.
  - Text normalisation: strip + collapse whitespace + lowercase + remove
    accents. No date or separator transforms.
  - Predicted types outside the labeling.xml vocabulary are mapped to
    INVALID_CATEGORY to prevent spurious confusion-matrix columns.
  - Documents where anonymized_text is empty despite identified_text having
    spans (anon_text_missing) are counted in M1 but excluded from M2/M3
    to avoid penalising pipeline-level failures.
"""

import json
import math
import re
import os
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from difflib import SequenceMatcher

try:
    from sklearn.metrics import multilabel_confusion_matrix as _sklearn_mcm
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Global regex for extracting [**...**] markers (compiled once).
# The negative lookahead (?!\[\*\*|\*\*\]) prevents the capture group from
# crossing marker boundaries. Without it, a malformed unclosed marker would
# consume everything up to the next valid **], swallowing adjacent markers.
# With the lookahead, malformed markers simply produce no match and are
# silently ignored.
_PAT = re.compile(r'\[\*\*((?:(?!\[\*\*|\*\*\]).)*?)\*\*\]', re.DOTALL)

# Valid category set as defined in labeling.xml.
# Any predicted type not in this set is normalised to INVALID_CATEGORY
# so that model-invented labels do not pollute the confusion matrix.
_VALID_TYPES: set = {
    'NOMBRE_SUJETO_ASISTENCIA', 'EDAD_SUJETO_ASISTENCIA', 'SEXO_SUJETO_ASISTENCIA',
    'FAMILIARES_SUJETO_ASISTENCIA', 'NOMBRE_PERSONAL_SANITARIO', 'FECHAS',
    'PROFESION', 'HOSPITAL', 'CENTRO_SALUD', 'INSTITUCION', 'CALLE', 'TERRITORIO',
    'PAIS', 'NUMERO_TELEFONO', 'NUMERO_FAX', 'CORREO_ELECTRONICO',
    'ID_SUJETO_ASISTENCIA', 'ID_CONTACTO_ASISTENCIAL', 'ID_ASEGURAMIENTO',
    'ID_TITULACION_PERSONAL_SANITARIO', 'ID_EMPLEO_PERSONAL_SANITARIO',
    'IDENTIF_VEHICULOS_NRSERIE_PLACAS', 'IDENTIF_DISPOSITIVOS_NRSERIE',
    'IDENTIF_BIOMETRICOS', 'DIREC_PROT_INTERNET', 'URL_WEB',
    'NUMERO_IDENTIF', 'OTROS_SUJETO_ASISTENCIA',
    'UNKNOWN',  # placeholder for spans unresolved by the aligner
}


def _normalize_pred_type(t: str) -> str:
    """Maps invalid predicted types to INVALID_CATEGORY."""
    t_up = t.upper()
    return t_up if t_up in _VALID_TYPES else 'INVALID_CATEGORY'


def _flatten_nested_markers(text: str) -> str:
    """
    Flattens nested [**...**] markers by removing inner delimiters while
    preserving their text content.

    Real example (llama, dates)::
        [**02/[**07**]/2024**]  →  [**02/07/2024**]

    Uses an O(n) stack automaton: tracks nesting depth and only emits the
    delimiters of the outermost marker. Content of inner markers is kept as
    plain text. Unclosed markers (depth never reaches 0) produce text without
    a closing **], which _PAT ignores silently.
    """
    result = []
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if text[i:i + 3] == '[**':
            depth += 1
            if depth == 1:
                result.append('[**')   # outer delimiter: keep
            # depth > 1: inner [** → keep content only, not delimiter
            i += 3
        elif text[i:i + 3] == '**]':
            if depth >= 1:
                depth -= 1
                if depth == 0:
                    result.append('**]')  # outer closing: keep
                # else: inner closing → content already emitted
            # depth == 0: orphan **] → ignore
            i += 3
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Configuration and data structures
# ---------------------------------------------------------------------------

@dataclass
class NormalizationConfig:
    """Matching configuration: minimal text normalisation + optional partial match."""
    normalize_case: bool = True       # Always on: strip + lowercase
    partial_match: bool = False       # Enable second-pass substring matching
    partial_match_strategy: str = 'substring'  # 'substring' | 'sequence_matcher'
    partial_match_threshold: float = 0.6   # Minimum ratio for sequence_matcher [0, 1]
    partial_match_min_length: int = 3      # Min length on both sides for partial match


@dataclass
class MetricsResult:
    """TP/FP/FN counters for one metric. P/R/F1 are computed on-the-fly."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    partial: int = 0  # Substring matches (only when partial_match=True)

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
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'partial': self.partial,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
        }


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Prompt-leakage patterns: the model sometimes echoes the system/task prompt
# inside identified_text, adding a spurious [**value**] placeholder span.
# We strip any XML-style wrapper blocks (<task>, <user_prompt>, <input_text>, etc.)
# and anything before the first real content line.
_PROMPT_TAG_RE = re.compile(
    r'<(?:task|user_prompt|input_text|system)[^>]*>.*?</(?:task|user_prompt|input_text|system)>',
    re.DOTALL | re.IGNORECASE,
)


def _strip_prompt_leakage(text: str) -> str:
    """
    Removes XML prompt blocks (<task>, <user_prompt>, <input_text>, …)
    that the model sometimes prepends to identified_text. This prevents
    the [**value**] instruction placeholder from being counted as a real span.
    """
    cleaned = _PROMPT_TAG_RE.sub('', text)
    # Collapse any leading blank lines left after removal
    return cleaned.lstrip()


def _normalize_whitespace(s: str) -> str:
    """Collapses any sequence of spaces/tabs into a single space."""
    return ' '.join(s.split())


def _remove_accents(s: str) -> str:
    """Strips diacritic marks (tildes, umlauts, …) from the string."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def _normalize_token(s: str) -> str:
    """
    Full normalisation for matching:
      1. strip + collapse internal whitespace  (fixes whitespace typos in GT/pred)
      2. lowercase
      3. remove accents                        (fixes accentuation typos in GT/pred)
    """
    return _remove_accents(_normalize_whitespace(s).lower())


def normalize_entity_list(
    entities: List[str],
    config: NormalizationConfig
) -> List[str]:
    """
    Normalises entities for matching: whitespace, lowercase, and accent removal.
    Accents are removed because the GT occasionally has accentuation typos and
    the model should not be penalised for producing the correct spelling.
    Internal whitespace is collapsed to absorb typos like '22- 11-2018'.
    """
    if not entities:
        return []
    return [_normalize_token(e) for e in entities if e and e.strip()]


# ---------------------------------------------------------------------------
# Matching multiset
# ---------------------------------------------------------------------------

def _cosine_sim(a: str, b: str) -> float:
    """Similitud de coseno entre bigrams de caracteres de dos cadenas."""
    def bigrams(s: str) -> Counter:
        return Counter(s[i:i+2] for i in range(len(s) - 1))
    bg_a = bigrams(a)
    bg_b = bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    dot = sum(bg_a[k] * bg_b[k] for k in bg_a if k in bg_b)
    norm_a = math.sqrt(sum(v * v for v in bg_a.values()))
    norm_b = math.sqrt(sum(v * v for v in bg_b.values()))
    return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0


def _strip_markers_keep_content(text: str) -> str:
    """Removes [**...**] markers, keeping the span content as plain text."""
    return _PAT.sub(r'\1', text)


def _interspan_text(text: str) -> str:
    """Returns only the text segments between [**...**] spans, joined with a space."""
    return ' '.join(_PAT.split(text)[0::2])


def match_entities(
    pred_entities: List[str],
    gt_entities: List[str],
    config: NormalizationConfig
) -> Tuple[int, int, int, int, List[Tuple[int, int]], List[Dict]]:
    """
    Matches predictions against ground truth respecting multiset semantics.

    Two-pass strategy:
    - Pass 1: Exact match. Resolves the clearest cases first, ensuring a
      partial match cannot steal a later exact match.
    - Pass 2: Partial/substring match (only if config.partial_match=True).

    Matching is greedy and irrevocable: once a GT or pred is matched it is
    no longer available. This correctly handles duplicates (6 occurrences of
    '8/9/21' in GT require 6 predictions to score 6 TPs).

    Returns:
        tp: exact matches
        fp: unmatched predictions
        fn: unmatched GT spans
        partial: substring matches (counted as TP for M1)
        matches: list of (pred_idx, gt_idx) pairs — used for M2
        partial_pairs: list of {pred, gt} dicts for logging
    """
    tp = 0
    partial = 0
    gt_matched = [False] * len(gt_entities)
    pred_matched = [False] * len(pred_entities)
    matches: List[Tuple[int, int]] = []
    partial_pairs: List[Dict] = []

    # Pass 1: Exact match
    for i, pred in enumerate(pred_entities):
        for j, gt in enumerate(gt_entities):
            if pred_matched[i] or gt_matched[j]:
                continue
            if pred == gt:
                tp += 1
                pred_matched[i] = True
                gt_matched[j] = True
                matches.append((i, j))
                break

    # Pass 2: Partial/substring match.
    # Configurable minimum length on both sides to avoid spurious matches
    # on very short GT entities (e.g. "2", "M").
    if config.partial_match:
        min_len = config.partial_match_min_length
        for i, pred in enumerate(pred_entities):
            if pred_matched[i]:
                continue
            if len(pred) < min_len:
                continue
            for j, gt in enumerate(gt_entities):
                if gt_matched[j]:
                    continue
                if len(gt) < min_len:
                    continue
                strategy = config.partial_match_strategy
                if strategy == 'sequence_matcher':
                    score = SequenceMatcher(None, pred, gt).ratio()
                    hit = score >= config.partial_match_threshold
                    method = 'sequence_matcher'
                else:  # 'substring'
                    hit = pred in gt or gt in pred
                    score = 1.0 if hit else 0.0
                    method = 'substring'
                if hit:
                    partial += 1
                    pred_matched[i] = True
                    gt_matched[j] = True
                    matches.append((i, j))
                    partial_pairs.append({'pred': pred, 'gt': gt,
                                          'pred_idx': i, 'gt_idx': j,
                                          'score': round(score, 4),
                                          'method': method})
                    break

    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)

    return tp, fp, fn, partial, matches, partial_pairs


# ---------------------------------------------------------------------------
# M2 confusion matrix (multi-class)
# ---------------------------------------------------------------------------

def _compute_m2_confusion(pairs: List[Tuple[str, str]]) -> Dict:
    """
    Builds an N×N confusion matrix from (gt_type, pred_type) pairs collected
    across all M1 matches, and computes per-class metrics (macro and weighted)
    using a one-vs-rest perspective.

    Uses sklearn.metrics.multilabel_confusion_matrix when available;
    falls back to manual one-vs-rest TP/FP/FN otherwise.

    Returns dict with:
      classes   : sorted list of observed classes
      matrix    : N×N dict {gt_type: {pred_type: count}}
      per_class : {class: {tp, fp, fn, precision, recall, f1, support}}
      micro     : micro-average (P=R=F1, equivalent to overall M2 accuracy)
      macro     : simple average over classes (equal weight to rare classes)
      weighted  : support-weighted average
    """
    if not pairs:
        return {}

    gt_list   = [p[0] for p in pairs]
    pred_list = [p[1] for p in pairs]
    classes   = sorted(set(gt_list) | set(pred_list))

    # N×N matrix: matrix[gt_type][pred_type] = count
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gt, pred in pairs:
        matrix[gt][pred] += 1

    # Per-class TP/FP/FN
    per_class: Dict[str, Dict] = {}

    if _SKLEARN_AVAILABLE:
        mcm = _sklearn_mcm(gt_list, pred_list, labels=classes)
        for idx, cls in enumerate(classes):
            tn, fp, fn, tp = mcm[idx].ravel()
            tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class[cls] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': round(p, 4),
                'recall':    round(r, 4),
                'f1':        round(f1, 4),
                'support':   tp + fn,
            }
    else:
        # Manual one-vs-rest
        for cls in classes:
            tp = matrix[cls].get(cls, 0)
            # FP: other GT classes predicted as cls
            fp = sum(matrix[gt].get(cls, 0) for gt in classes if gt != cls)
            # FN: cls GT predicted as other class
            fn = sum(matrix[cls].get(pred, 0) for pred in classes if pred != cls)
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_class[cls] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': round(p, 4),
                'recall':    round(r, 4),
                'f1':        round(f1, 4),
                'support':   tp + fn,
            }

    n = len(classes)
    total_support = sum(per_class[c]['support'] for c in classes)

    macro_p  = sum(per_class[c]['precision'] for c in classes) / n if n else 0.0
    macro_r  = sum(per_class[c]['recall']    for c in classes) / n if n else 0.0
    macro_f1 = sum(per_class[c]['f1']        for c in classes) / n if n else 0.0

    if total_support > 0:
        weighted_p  = sum(per_class[c]['precision'] * per_class[c]['support'] for c in classes) / total_support
        weighted_r  = sum(per_class[c]['recall']    * per_class[c]['support'] for c in classes) / total_support
        weighted_f1 = sum(per_class[c]['f1']        * per_class[c]['support'] for c in classes) / total_support
    else:
        weighted_p = weighted_r = weighted_f1 = 0.0

    # Micro: sum all TP/FP/FN → P = R = F1 always (= existing M2 accuracy)
    total_tp = sum(per_class[c]['tp'] for c in classes)
    total_fp = sum(per_class[c]['fp'] for c in classes)
    total_fn = sum(per_class[c]['fn'] for c in classes)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        'classes': classes,
        'matrix':  {gt: dict(row) for gt, row in matrix.items()},
        'per_class': per_class,
        'micro': {
            'precision': round(micro_p,  4),
            'recall':    round(micro_r,  4),
            'f1':        round(micro_f1, 4),
        },
        'macro': {
            'precision': round(macro_p,  4),
            'recall':    round(macro_r,  4),
            'f1':        round(macro_f1, 4),
        },
        'weighted': {
            'precision': round(weighted_p,  4),
            'recall':    round(weighted_r,  4),
            'f1':        round(weighted_f1, 4),
        },
        'sklearn_used': _SKLEARN_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Document data extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(doc: Dict) -> Tuple[List[str], List[str]]:
    """
    Extracts entities and types from the benchmark (labels[] field).

    labels[] is the authoritative source for all three metrics:
    - .text → what must be identified (M1)
    - .type → what category must be assigned (M2, M3)

    Edge cases:
    - label with no 'text' field: skipped
    - label with empty or whitespace-only text: skipped
    - label with no 'type' field: assigned 'UNKNOWN'
    """
    labels = doc.get('labels', [])
    entities, types = [], []
    for label in labels:
        text = label.get('text', '').strip()
        if not text:
            continue
        entities.append(text)
        types.append(label.get('type', 'UNKNOWN'))
    return entities, types


def _align_markers_by_context(
    identified_text: str,
    anonymized_text: str,
) -> Tuple[List[str], int]:
    """
    Aligns [**...**] markers between identified_text and anonymized_text
    using the inter-span text as a positional anchor.

    When the two LLM calls produce different span counts (the second call
    added or dropped markers), direct positional assignment produces wrong
    or UNKNOWN types. This function recovers correct alignment by comparing
    the text segment following each marker: since the model only replaces
    [**entity**] → [**CATEGORY**] without changing surrounding text, the
    segment after a marker is identical in both texts when markers are aligned.

    Two-phase greedy strategy per marker i in identified_text:
      - Phase 1 (direct): cosine(anchor_id, anchor_anon[j]) >= 0.75
        → pair (i, j), advance both pointers.
      - Phase 2 (skip-ahead): if Phase 1 fails, search anon[j+1..j+5];
        skip over extra markers the second call inserted.
      - No match: UNKNOWN for i, j does not advance (anon dropped this span).

    The anchor combines the last 10 + first 3 words of the following segment
    (no punctuation, lowercase). Segments shorter than 1 word are matched by
    order; fuzzy cosine comparison (>= 0.75) tolerates minor text rewrites.

    Returns:
        types     : list of types parallel to ALL markers in identified_text,
                    UNKNOWN where alignment failed.
        n_resolved: number of correctly aligned markers.
    """
    # re.split with a capturing group returns [seg0, ent0, seg1, ent1, ..., segN]
    id_parts   = _PAT.split(identified_text)
    anon_parts = _PAT.split(anonymized_text)

    id_segs    = id_parts[0::2]                               # n_id + 1 segmentos
    id_ents    = id_parts[1::2]                               # n_id contenidos
    anon_segs  = anon_parts[0::2]                             # n_anon + 1 segmentos
    anon_types = [t.strip() or 'UNKNOWN' for t in anon_parts[1::2]]

    n_id   = len(id_ents)
    n_anon = len(anon_types)
    result_types = ['UNKNOWN'] * n_id
    n_resolved = 0
    n_phase0 = 0  # empty-anchor fallback (order-based)
    n_phase1 = 0  # direct anchor match
    n_phase2 = 0  # skip-ahead match (anon added extra spans)
    n_phase3 = 0  # no match → UNKNOWN (anon dropped span)

    def _anchor(seg: str, tail_words: int = 10, head_words: int = 3) -> Optional[str]:
        """
        Combined anchor: last tail_words + first head_words of the segment,
        stripped of punctuation, lowercased.

        Including leading words helps when the segment begins with informative
        content (e.g. proper names) before becoming more generic.
        Minimum threshold is 1 word: only purely empty segments are excluded.
        """
        words = re.sub(r'[^\w\s]', ' ', seg.lower()).split()
        if not words:
            return None
        tail = words[-tail_words:]
        head = words[:head_words]
        # Avoid duplicating words when the segment is very short
        combined = tail + [w for w in head if w not in tail]
        return ' '.join(combined)

    def _anon_seg(idx: int) -> str:
        return anon_segs[idx] if idx < len(anon_segs) else ''

    j = 0  # pointer into anon sequence
    for i in range(n_id):
        if j > n_anon:
            break

        anchor_id = _anchor(id_segs[i + 1])  # segment AFTER marker i

        # Empty segment → fall back to order-based pairing
        if anchor_id is None:
            if j < n_anon:
                result_types[i] = anon_types[j]
                j += 1
                n_resolved += 1
                n_phase0 += 1
            else:
                n_phase3 += 1
            continue

        # Phase 1: direct alignment (fuzzy cosine >= 0.75)
        anchor_anon_j = _anchor(_anon_seg(j + 1))
        if _cosine_sim(anchor_anon_j or '', anchor_id) >= 0.75:
            if j < n_anon:
                result_types[i] = anon_types[j]
                j += 1
                n_resolved += 1
                n_phase1 += 1
            continue

        # Phase 2: anon has extra markers → search j+1..j+5
        found = False
        for skip in range(1, 6):
            if j + skip > n_anon:
                break
            anchor_anon_skip = _anchor(_anon_seg(j + skip + 1))
            if _cosine_sim(anchor_anon_skip or '', anchor_id) >= 0.75:
                j += skip                  # skip over extra anon markers
                result_types[i] = anon_types[j]
                j += 1
                n_resolved += 1
                n_phase2 += 1
                found = True
                break

        # Phase 3 (no match): anon dropped this marker → UNKNOWN, j stays
        if not found:
            n_phase3 += 1

    return result_types, n_resolved, n_phase0, n_phase1, n_phase2, n_phase3


def extract_predictions(doc: Dict) -> Tuple[List[str], List[str], Dict]:
    """
    Extracts predicted entities and types from the model output (chunks field).

    Primary strategy — text-based extraction (positionally guaranteed):
      Parses [**...**] directly from identified_text and anonymized_text.
      id_spans[i] is the entity; anon_spans[i] is its category.
      Positional alignment is guaranteed because anonymized_text is produced
      by replacing each [**entity**] in identified_text with [**CATEGORY**].
      Empty/malformed spans are discarded.

    When span counts differ, context-anchor alignment (_align_markers_by_context)
    recovers the correct correspondence instead of padding with UNKNOWN.

    Fallback strategy — parallel lists (when identified_text is empty):
      Uses entities_identified and entities_anonymized. Length mismatches
      are padded with 'UNKNOWN' on the shorter side.

    Edge cases:
    - chunks is a list → take the first element
    - chunks missing or not a dict → empty lists
    - Different span counts → context-anchor alignment

    Returns:
        entities: list of identified entities
        types   : list of assigned types (same length as entities)
        meta    : anomaly metadata for logging
    """
    meta = {'length_mismatch': False, 'mismatch_detail': '', 'anon_text_missing': False}

    chunks = doc.get('chunks', {})
    if isinstance(chunks, list):
        chunks = chunks[0] if chunks else {}
    if not isinstance(chunks, dict):
        return [], [], meta

    # --- Primary extraction from full text --------------------------------
    identified_text = _strip_prompt_leakage(
        _flatten_nested_markers(chunks.get('identified_text', '') or '')
    )
    anonymized_text = _strip_prompt_leakage(
        _flatten_nested_markers(chunks.get('anonymized_text', '') or '')
    )

    # Extract ALL markers without filtering to compare raw counts first:
    # filtering before comparing would cause an empty [****] in one text but
    # not the other to silently shift all subsequent positions.
    id_raw   = [s.strip() for s in _PAT.findall(identified_text)]
    anon_raw = [s.strip() for s in _PAT.findall(anonymized_text)]

    # Detect pipeline failure: identified_text has spans but anonymized_text is
    # completely empty. Affected documents count in M1 but are excluded from
    # M2/M3 to avoid penalising categorisation that was never attempted.
    if id_raw and not anonymized_text.strip():
        meta['anon_text_missing'] = True

    if id_raw:
        if len(id_raw) == len(anon_raw):
            # Equal counts → pair by position.
            # In the rare edge case where the model drops one span and adds
            # another keeping the same total, this yields a wrong type;
            # but context-anchor alignment performs worse on short inter-span
            # segments, so positional pairing is the most robust option.
            pairs = [(e, t if t else 'UNKNOWN')
                     for e, t in zip(id_raw, anon_raw) if e]
            entities = [e for e, t in pairs]
            types    = [t for e, t in pairs]
        else:
            # Different span counts between the two calls → real mismatch.
            # Use context-anchor alignment to recover correct correspondence
            # instead of blindly padding with UNKNOWN.
            meta['length_mismatch'] = True
            meta['mismatch_detail'] = (
                f"id_raw={len(id_raw)}, anon_raw={len(anon_raw)} "
                f"(from identified_text/anonymized_text)"
            )
            all_types, n_resolved, n_p0, n_p1, n_p2, n_p3 = _align_markers_by_context(
                identified_text, anonymized_text
            )
            meta['aligned_by_context'] = n_resolved
            meta['align_phase0'] = n_p0
            meta['align_phase1'] = n_p1
            meta['align_phase2'] = n_p2
            meta['align_phase3_unknown'] = n_p3
            # all_types is aligned to id_raw (including empty spans); filter in pairs
            pairs = [(e, t) for e, t in zip(id_raw, all_types) if e]
            entities = [e for e, t in pairs]
            types    = [t for e, t in pairs]

        if entities:
            return entities, types, meta

    # --- Fallback: explicit JSON lists ----------------------------------
    entities = chunks.get('entities_identified', []) or []
    types = chunks.get('entities_anonymized', []) or []

    # Convert dict to list if the LLM returned the wrong format
    if isinstance(entities, dict):
        entities = list(entities.values())
    if isinstance(types, dict):
        types = list(types.values())

    if not isinstance(entities, list):
        entities = []
    if not isinstance(types, list):
        types = []

    # Strip empty strings
    entities = [str(e).strip() for e in entities if e and str(e).strip()]
    types = [str(t).strip() for t in types if t and str(t).strip()]

    # Detect and handle length mismatch
    if len(entities) != len(types):
        meta['length_mismatch'] = True
        meta['mismatch_detail'] = (
            f"entities_identified={len(entities)}, "
            f"entities_anonymized={len(types)} (fallback list fields)"
        )
        if len(types) < len(entities):
            types = types + ['UNKNOWN'] * (len(entities) - len(types))
        else:
            types = types[:len(entities)]

    return entities, types, meta


# ---------------------------------------------------------------------------
# Single-document evaluation
# ---------------------------------------------------------------------------

def evaluate_document(
    doc: Dict,
    config: NormalizationConfig
) -> Dict:
    """
    Evaluates all three metrics for a single document.

    Returns a dict with:
    - m1, m2, m3: MetricsResult for each metric
    - m1_matches: (pred_idx, gt_idx) pairs from M1 — used for error analysis
    - gt_entities, gt_types, pred_entities, pred_types: raw values
    - meta: detected anomalies (mismatches, missing anonymized_text, etc.)
    """
    gt_entities, gt_types = extract_ground_truth(doc)
    pred_entities, pred_types, meta = extract_predictions(doc)

    # Normalise predicted types: anything outside labeling.xml vocabulary
    # is grouped into INVALID_CATEGORY.
    pred_types = [_normalize_pred_type(t) for t in pred_types]

    # Minimal text normalisation for matching (strip + lowercase)
    gt_norm = normalize_entity_list(gt_entities, config)
    pred_norm = normalize_entity_list(pred_entities, config)

    # -----------------------------------------------------------------------
    # M1 — IDENTIFICATION
    # -----------------------------------------------------------------------
    m1_tp, m1_fp, m1_fn, m1_partial, m1_matches, m1_partial_pairs_raw = match_entities(
        pred_norm, gt_norm, config
    )

    # Enrich partial pairs with original texts, entity types, and similarity score
    m1_partial_pairs = [
        {
            'pred_norm':     p['pred'],
            'gt_norm':       p['gt'],
            'pred_original': pred_entities[p['pred_idx']] if p['pred_idx'] < len(pred_entities) else p['pred'],
            'gt_original':   gt_entities[p['gt_idx']]    if p['gt_idx']  < len(gt_entities)  else p['gt'],
            'pred_type':     (pred_types[p['pred_idx']] if p['pred_idx'] < len(pred_types) else 'UNKNOWN').upper(),
            'gt_type':       (gt_types[p['gt_idx']]    if p['gt_idx']  < len(gt_types)  else 'UNKNOWN').upper(),
            'score':         p.get('score'),
            'method':        p.get('method'),
        }
        for p in m1_partial_pairs_raw
    ]
    m1 = MetricsResult(
        tp=m1_tp + m1_partial,  # Partial counts as TP for identification
        fp=m1_fp,
        fn=m1_fn,
        partial=m1_partial,
    )

    # -----------------------------------------------------------------------
    # M2 — CATEGORISATION (only over M1 TPs)
    #
    # For each matched pair (pred_idx, gt_idx) from M1:
    # - gt_type   = gt_types[gt_idx]
    # - pred_type = pred_types[pred_idx]
    # Match → M2 TP; mismatch → M2 FP (wrong category)
    #
    # Denominator for both Precision and Recall = len(m1_matches).
    # M2 does NOT penalise unidentified entities; that is already M1's job.
    # Spans where pred_type is UNKNOWN (aligner failure, not a real
    # categorisation error) are excluded to avoid double-penalisation with M3.
    # -----------------------------------------------------------------------
    m2_tp = 0
    m2_fp = 0
    m2_type_errors = []  # For analysis: (entity, gt_type, pred_type)
    m2_cm_pairs: List[Tuple[str, str]] = []  # (gt_type, pred_type) for all matches

    for pred_idx, gt_idx in m1_matches:
        gt_type = (gt_types[gt_idx] if gt_idx < len(gt_types) else 'UNKNOWN').upper()
        pred_type = (pred_types[pred_idx] if pred_idx < len(pred_types) else 'UNKNOWN').upper()
        # Spans with pred_type UNKNOWN are excluded from M2:
        # the alignment failure is already penalised in M3.
        if pred_type == 'UNKNOWN':
            continue
        m2_cm_pairs.append((gt_type, pred_type))
        if pred_type == gt_type:
            m2_tp += 1
        else:
            m2_fp += 1
            m2_type_errors.append({
                'entity': pred_entities[pred_idx] if pred_idx < len(pred_entities) else '',
                'gt_type': gt_type,
                'pred_type': pred_type,
            })

    # M2 FN: matched entities with a wrong type
    # (unmatched ones are M1 FNs, not M2 FNs)
    m2_fn = m2_fp  # every type error is simultaneously an FP and an FN in M2
    m2 = MetricsResult(tp=m2_tp, fp=m2_fp, fn=m2_fn)

    # -----------------------------------------------------------------------
    # M3 — END-TO-END
    #
    # Matching over (normalised_text, type) pairs.
    # A TP requires both text and type to match.
    # M1 FPs and FNs are also M3 FPs and FNs.
    # -----------------------------------------------------------------------
    gt_pairs = list(zip(gt_norm, [t.upper() for t in gt_types]))
    pred_pairs = list(zip(pred_norm, [t.upper() for t in pred_types]))

    gt_pair_matched = [False] * len(gt_pairs)
    pred_pair_matched = [False] * len(pred_pairs)
    m3_tp = 0

    # Pass 1: exact match on both text AND type
    for i, (p_text, p_type) in enumerate(pred_pairs):
        for j, (g_text, g_type) in enumerate(gt_pairs):
            if pred_pair_matched[i] or gt_pair_matched[j]:
                continue
            if p_text == g_text and p_type == g_type:
                m3_tp += 1
                pred_pair_matched[i] = True
                gt_pair_matched[j] = True
                break

    # Pass 2: substring match on text, exact match on type (if partial_match enabled)
    if config.partial_match:
        for i, (p_text, p_type) in enumerate(pred_pairs):
            if pred_pair_matched[i]:
                continue
            if len(p_text) < 3:
                continue
            for j, (g_text, g_type) in enumerate(gt_pairs):
                if gt_pair_matched[j]:
                    continue
                if len(g_text) < 3:
                    continue
                if (p_text in g_text or g_text in p_text) and p_type == g_type:
                    m3_tp += 1
                    pred_pair_matched[i] = True
                    gt_pair_matched[j] = True
                    break

    m3_fp = sum(1 for m in pred_pair_matched if not m)
    m3_fn = sum(1 for m in gt_pair_matched if not m)
    m3 = MetricsResult(tp=m3_tp, fp=m3_fp, fn=m3_fn)

    # -----------------------------------------------------------------------
    # Per-entity-type breakdown (for M1 and M3)
    # -----------------------------------------------------------------------
    # M1 by type: for each GT type, was the entity identified?
    m1_by_type: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    gt_matched_m1 = [False] * len(gt_norm)
    for pred_idx, gt_idx in m1_matches:
        gt_matched_m1[gt_idx] = True
    for j, gt_type in enumerate(gt_types):
        if gt_matched_m1[j]:
            m1_by_type[gt_type].tp += 1
        else:
            m1_by_type[gt_type].fn += 1
    # M1 FP by predicted type
    pred_matched_flags = [False] * len(pred_norm)
    for pred_idx, _ in m1_matches:
        pred_matched_flags[pred_idx] = True
    for i, matched in enumerate(pred_matched_flags):
        if not matched:
            pred_type = pred_types[i] if i < len(pred_types) else 'UNKNOWN'
            m1_by_type[pred_type].fp += 1

    # M3 by type: for each GT type, was the entity identified AND categorised correctly?
    m3_by_type: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    for j, (g_text, g_type) in enumerate(gt_pairs):
        if gt_pair_matched[j]:
            m3_by_type[g_type].tp += 1
        else:
            m3_by_type[g_type].fn += 1
    for i, (p_text, p_type) in enumerate(pred_pairs):
        if not pred_pair_matched[i]:
            m3_by_type[p_type].fp += 1

    # -----------------------------------------------------------------------
    # FAITHFULNESS — text preservation outside annotation spans
    # -----------------------------------------------------------------------
    _chunks = doc.get('chunks', {})
    if isinstance(_chunks, list):
        _chunks = _chunks[0] if _chunks else {}
    _orig = _normalize_whitespace(_chunks.get('original_text', '') or '')
    _id_text = _flatten_nested_markers(_chunks.get('identified_text', '') or '')
    _anon_text = _flatten_nested_markers(_chunks.get('anonymized_text', '') or '')

    # Call 1: identified_text with markers stripped should reproduce original_text
    _id_stripped = _normalize_whitespace(_strip_markers_keep_content(_id_text))
    faithfulness_identify = (
        round(_cosine_sim(_id_stripped, _orig), 4)
        if (_id_stripped and _orig) else None
    )

    # Call 2: text outside spans should be identical in both calls
    _id_outside = _normalize_whitespace(_interspan_text(_id_text))
    _anon_outside = _normalize_whitespace(_interspan_text(_anon_text))
    faithfulness_anonymize = (
        round(_cosine_sim(_id_outside, _anon_outside), 4)
        if (_id_outside and _anon_outside) else None
    )

    return {
        'm1': m1,
        'm2': m2,
        'm3': m3,
        'm1_by_type': dict(m1_by_type),
        'm3_by_type': dict(m3_by_type),
        'm2_type_errors': m2_type_errors,
        'gt_entities': gt_entities,
        'gt_types': gt_types,
        'pred_entities': pred_entities,
        'pred_types': pred_types,
        'm1_matches': m1_matches,
        'm1_partial_pairs': m1_partial_pairs,
        'm2_cm_pairs': m2_cm_pairs,
        'meta': meta,
        'faithfulness_identify': faithfulness_identify,
        'faithfulness_anonymize': faithfulness_anonymize,
    }


# ---------------------------------------------------------------------------
# Aggregated evaluation over all documents
# ---------------------------------------------------------------------------

def evaluate_all(
    data: List[Dict],
    config: NormalizationConfig,
    excluded_docs: Set[str] = None,
    language_map: Dict[str, str] = None,
) -> Dict:
    """
    Evaluates all documents and aggregates global, per-language, and
    per-entity-type metrics.

    Args:
        data         : list of model output documents
        config       : normalisation/matching configuration
        excluded_docs: namefiles to skip (from problematic_docs.json)
        language_map : {namefile: language} from the TSV file

    Documents with empty labels (no real PHI): if the model predicts entities
    on them, they count as M1/M3 FPs (anonymising clean text is a real error).

    Returns dict with:
        global           : M1/M2/M3 metrics over all documents
        by_language      : same metrics disaggregated by language
        by_entity_type   : M1 and M3 broken down by entity category
        document_results : per-document results (for analysis scripts)
        counts           : document processing statistics
    """
    excluded_docs = excluded_docs or set()
    language_map = language_map or {}

    # Global accumulators
    global_m1 = MetricsResult()
    global_m2 = MetricsResult()
    global_m3 = MetricsResult()
    global_m1_by_type: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    global_m3_by_type: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    global_m2_cm_pairs: List[Tuple[str, str]] = []  # for the M2 confusion matrix
    global_partial_pairs: List[Dict] = []             # for the partial match report

    # Per-language accumulators
    lang_m1: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    lang_m2: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    lang_m3: Dict[str, MetricsResult] = defaultdict(MetricsResult)
    lang_m2_cm_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    document_results = []
    counts = {
        'total': len(data),
        'excluded': 0,
        'no_gt': 0,
        'evaluated': 0,
        'length_mismatches': 0,
        'anon_text_missing': 0,
    }

    _faith_id_sum = 0.0
    _faith_anon_sum = 0.0
    _faith_count_id = 0
    _faith_count_anon = 0
    _align_phase0 = 0
    _align_phase1 = 0
    _align_phase2 = 0
    _align_phase3 = 0

    for doc in data:
        namefile = doc.get('namefile', doc.get('id', ''))

        # Skip problematic documents
        if namefile in excluded_docs:
            counts['excluded'] += 1
            continue

        gt_entities, gt_types = extract_ground_truth(doc)

        # Documents with no ground truth (empty labels = clean text, no real PHI).
        # If the model annotated something, those are real FPs.
        if not gt_entities:
            counts['no_gt'] += 1
            pred_entities, pred_types, _ = extract_predictions(doc)
            lang = language_map.get(namefile, 'unknown')
            if pred_entities:
                n = len(pred_entities)
                global_m1.fp += n
                global_m3.fp += n
                lang_m1[lang].fp += n
                lang_m3[lang].fp += n
            counts['evaluated'] += 1
            n_fp = len(pred_entities)
            document_results.append({
                'namefile': namefile,
                'language': lang,
                'gt_count': 0,
                'pred_count': n_fp,
                'm1': MetricsResult(tp=0, fp=n_fp, fn=0, partial=0).to_dict(),
                'm2': MetricsResult(tp=0, fp=0, fn=0, partial=0).to_dict(),
                'm3': MetricsResult(tp=0, fp=n_fp, fn=0, partial=0).to_dict(),
                'length_mismatch': False,
                'mismatch_detail': '',
                'align_phase0': 0,
                'align_phase1': 0,
                'align_phase2': 0,
                'align_phase3_unknown': 0,
                'anon_text_missing': False,
                'faithfulness_identify': None,
                'faithfulness_anonymize': None,
                'gt_entities': [],
                'gt_types': [],
                'pred_entities': pred_entities,
                'pred_types': pred_types,
                'm2_type_errors': [],
                'm1_partial_pairs': [],
            })
            continue

        counts['evaluated'] += 1
        result = evaluate_document(doc, config)

        anon_missing = result['meta'].get('anon_text_missing', False)
        if anon_missing:
            counts['anon_text_missing'] += 1

        if result['meta']['length_mismatch']:
            counts['length_mismatches'] += 1
            _align_phase0 += result['meta'].get('align_phase0', 0)
            _align_phase1 += result['meta'].get('align_phase1', 0)
            _align_phase2 += result['meta'].get('align_phase2', 0)
            _align_phase3 += result['meta'].get('align_phase3_unknown', 0)

        _fi = result.get('faithfulness_identify')
        _fa = result.get('faithfulness_anonymize')
        if _fi is not None:
            _faith_id_sum += _fi
            _faith_count_id += 1
        if _fa is not None:
            _faith_anon_sum += _fa
            _faith_count_anon += 1

        if not anon_missing:
            global_m2_cm_pairs.extend(result['m2_cm_pairs'])

        # Aggregate partial pairs with document name for traceability
        for pp in result['m1_partial_pairs']:
            global_partial_pairs.append({'document': namefile, **pp})

        # Aggregate global metrics.
        # M2 and M3 are skipped for documents with missing anonymized_text
        for attr in ('tp', 'fp', 'fn', 'partial'):
            setattr(global_m1, attr, getattr(global_m1, attr) + getattr(result['m1'], attr))
        if not anon_missing:
            for attr in ('tp', 'fp', 'fn', 'partial'):
                setattr(global_m2, attr, getattr(global_m2, attr) + getattr(result['m2'], attr))
                setattr(global_m3, attr, getattr(global_m3, attr) + getattr(result['m3'], attr))

        # Aggregate per-type breakdown
        for etype, mr in result['m1_by_type'].items():
            for attr in ('tp', 'fp', 'fn'):
                setattr(global_m1_by_type[etype], attr,
                        getattr(global_m1_by_type[etype], attr) + getattr(mr, attr))
        if not anon_missing:
            for etype, mr in result['m3_by_type'].items():
                for attr in ('tp', 'fp', 'fn'):
                    setattr(global_m3_by_type[etype], attr,
                            getattr(global_m3_by_type[etype], attr) + getattr(mr, attr))

        # Aggregate per-language metrics
        lang = language_map.get(namefile, 'unknown')
        if not anon_missing:
            lang_m2_cm_pairs[lang].extend(result['m2_cm_pairs'])
        for attr in ('tp', 'fp', 'fn', 'partial'):
            setattr(lang_m1[lang], attr, getattr(lang_m1[lang], attr) + getattr(result['m1'], attr))
        if not anon_missing:
            for attr in ('tp', 'fp', 'fn', 'partial'):
                setattr(lang_m2[lang], attr, getattr(lang_m2[lang], attr) + getattr(result['m2'], attr))
                setattr(lang_m3[lang], attr, getattr(lang_m3[lang], attr) + getattr(result['m3'], attr))

        # Guardar resultado por documento
        document_results.append({
            'namefile': namefile,
            'language': lang,
            'gt_count': len(gt_entities),
            'pred_count': len(result['pred_entities']),
            'm1': result['m1'].to_dict(),
            'm2': result['m2'].to_dict(),
            'm3': result['m3'].to_dict(),
            'length_mismatch': result['meta']['length_mismatch'],
            'mismatch_detail': result['meta']['mismatch_detail'],
            'align_phase0': result['meta'].get('align_phase0', 0),
            'align_phase1': result['meta'].get('align_phase1', 0),
            'align_phase2': result['meta'].get('align_phase2', 0),
            'align_phase3_unknown': result['meta'].get('align_phase3_unknown', 0),
            'anon_text_missing': anon_missing,
            'faithfulness_identify': result.get('faithfulness_identify'),
            'faithfulness_anonymize': result.get('faithfulness_anonymize'),
            # Para identify_worst_docs y analyze_errors
            'gt_entities': gt_entities,
            'gt_types': gt_types,
            'pred_entities': result['pred_entities'],
            'pred_types': result['pred_types'],
            'm2_type_errors': result['m2_type_errors'],
            'm1_partial_pairs': result['m1_partial_pairs'],
        })

    return {
        'counts': counts,
        'global': {
            'm1': global_m1.to_dict(),
            'm2': global_m2.to_dict(),
            'm3': global_m3.to_dict(),
            'm2_confusion': _compute_m2_confusion(global_m2_cm_pairs),
            'faithfulness_identify_mean': round(_faith_id_sum / _faith_count_id, 4) if _faith_count_id else None,
            'faithfulness_anonymize_mean': round(_faith_anon_sum / _faith_count_anon, 4) if _faith_count_anon else None,
            'align_phases': {
                'docs_with_mismatch': counts['length_mismatches'],
                'phase0_empty_anchor': _align_phase0,
                'phase1_direct': _align_phase1,
                'phase2_skip': _align_phase2,
                'phase3_unknown': _align_phase3,
                'total_resolved': _align_phase0 + _align_phase1 + _align_phase2,
            },
        },
        'global_partial_pairs': global_partial_pairs,
        'by_language': {
            lang: {
                'm1': lang_m1[lang].to_dict(),
                'm2': lang_m2[lang].to_dict(),
                'm3': lang_m3[lang].to_dict(),
                'm2_confusion': _compute_m2_confusion(lang_m2_cm_pairs[lang]),
            }
            for lang in sorted(set(list(lang_m1.keys()) + list(lang_m2.keys()) + list(lang_m3.keys())))
        },
        'by_entity_type': {
            'm1': {etype: mr.to_dict() for etype, mr in sorted(global_m1_by_type.items())},
            'm3': {etype: mr.to_dict() for etype, mr in sorted(global_m3_by_type.items())},
        },
        'document_results': document_results,
    }
