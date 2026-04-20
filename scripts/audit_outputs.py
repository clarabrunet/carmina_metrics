"""
audit_outputs.py — Diagnostic tool for model output files.

Scans all JSON files in models/ and detects structural problems in model
responses BEFORE evaluation. Useful for understanding inference pipeline
quality and deciding whether a model needs to be re-run.

CHECKS IMPLEMENTED
------------------
A. prompt_leakage      System instructions (<task>, <user_prompt>,
                       <input_text>, <system>) embedded in identified_text.
                       Produces the spurious [**value**] span.

B. thinking_leak       Reasoning traces (chain-of-thought) in
                       anonymized_text: backtick blocks, phrases such as
                       "Okay, this is", "I need to", etc.

C. empty_anon_text     identified_text has spans but anonymized_text is
                       empty or has no spans → model did not complete call 2.

D. empty_id_text       Both text fields are empty. Only the explicit list
                       fallback is available.

E. fallback_only       identified_text is empty but entities_identified has
                       entries. Model responded with lists instead of marked text.

F. span_count_mismatch identified_text and anonymized_text have a different
                       number of spans (neither is 0). Model added or removed
                       markers between the two calls.

G. category_as_entity  Spans in identified_text have the CATEGORY_TYPE format
                       (uppercase + underscore). Model confused the two calls.

H. placeholder_value   Span [**value**] in identified_text after removing
                       prompt blocks (partial leakage not caught).

I. empty_span          Empty span [****] or whitespace-only span in any text.

J. unclosed_marker     '[**' appears in the text without a matching '**]'
                       (malformed / truncated marker).

K. nested_marker       '[**' inside a span's content (un-flattened nested markers).

L. anon_entity_text    Spans in anonymized_text that do not look like category
                       labels (contain mixed lowercase letters that are not
                       expected type names).

OUTPUT
------
Per model: count table per check + list of affected documents.
At the end: comparative summary table across models.
"""

import json
import re
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
_PAT = re.compile(r'\[\*\*((?:(?!\[\*\*|\*\*\]).)*?)\*\*\]', re.DOTALL)

_PROMPT_TAG_RE = re.compile(
    r'<(?:task|user_prompt|input_text|system)[^>]*>.*?</(?:task|user_prompt|input_text|system)>',
    re.DOTALL | re.IGNORECASE,
)

_THINKING_PATTERNS = [
    re.compile(r'```', re.IGNORECASE),
    re.compile(r'\bOkay,\s+this\s+is\b', re.IGNORECASE),
    re.compile(r'\bI\s+need\s+to\b', re.IGNORECASE),
    re.compile(r'\bLet\s+me\b', re.IGNORECASE),
    re.compile(r'\bthe\s+user\s+(wants|asks|said|prompt)\b', re.IGNORECASE),
    re.compile(r'<input_text>', re.IGNORECASE),
    re.compile(r'</?user_prompt>', re.IGNORECASE),
    re.compile(r'\bexpected\s+output\b', re.IGNORECASE),
    re.compile(r'This\s+looks\s+like\s+the\s+\*expected', re.IGNORECASE),
]

# Valid CARMEN category names (uppercase, underscores, up to ~35 chars)
_VALID_CATEGORY_RE = re.compile(
    r'^(NOMBRE_PERSONAL_SANITARIO|FAMILIARES_SUJETO_ASISTENCIA|'
    r'OTROS_SUJETO_ASISTENCIA|SEXO_SUJETO_ASISTENCIA|EDAD_SUJETO_ASISTENCIA|'
    r'FECHAS|HOSPITAL|CENTRO_SALUD|TERRITORIO|PAIS|CALLE|INSTITUCION|'
    r'PROFESION|NUMERO_IDENTIF|UNKNOWN)$'
)

# Heuristic for detecting "looks like entity text instead of a category label"
# A span in anonymized_text is suspicious if:
#  - it contains lowercase letters AND is not all-uppercase/underscores
#  - it has more than 4 words (no known category has that)
_ANON_ENTITY_RE = re.compile(r'[a-záéíóúñüà-ÿ]')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_chunks(doc: dict) -> dict:
    chunks = doc.get('chunks', {})
    if isinstance(chunks, list):
        chunks = chunks[0] if chunks else {}
    return chunks if isinstance(chunks, dict) else {}


def _strip_prompt(text: str) -> str:
    return _PROMPT_TAG_RE.sub('', text).lstrip()


def _spans(text: str) -> List[str]:
    return [s.strip() for s in _PAT.findall(text)]


def _count_unclosed(text: str) -> int:
    """Count '[**' occurrences that are NOT followed by '**]' before the next '[**'."""
    opens  = len(re.findall(r'\[\*\*', text))
    closes = len(re.findall(r'\*\*\]', text))
    return max(0, opens - closes)


# ---------------------------------------------------------------------------
# Per-document checks
# ---------------------------------------------------------------------------
CHECK_KEYS = [
    'prompt_leakage',
    'thinking_leak',
    'empty_anon_text',
    'empty_id_text',
    'fallback_only',
    'span_count_mismatch',
    'category_as_entity',
    'placeholder_value',
    'empty_span',
    'unclosed_marker',
    'nested_marker',
    'anon_entity_text',
]

CHECK_LABELS = {
    'prompt_leakage':      'A. prompt_leakage      (<task>/<user_prompt> in identified_text)',
    'thinking_leak':       'B. thinking_leak        (CoT/reasoning in anonymized_text)',
    'empty_anon_text':     'C. empty_anon_text      (identified OK, anon empty)',
    'empty_id_text':       'D. empty_id_text        (both texts empty)',
    'fallback_only':       'E. fallback_only        (lists only, no marked text)',
    'span_count_mismatch': 'F. span_count_mismatch  (id≠anon spans, neither=0)',
    'category_as_entity':  'G. category_as_entity   (type label in identified_text)',
    'placeholder_value':   'H. placeholder_value    ([**value**] residual)',
    'empty_span':          'I. empty_span           (empty span [****])',
    'unclosed_marker':     'J. unclosed_marker      ([** without closing)',
    'nested_marker':       'K. nested_marker        ([** inside span)',
    'anon_entity_text':    'L. anon_entity_text     (entity text in anon)',
}


def audit_doc(doc: dict) -> Dict[str, bool]:
    """Returns dict of check_key -> True if problem detected."""
    issues: Dict[str, bool] = {k: False for k in CHECK_KEYS}
    chunks = _get_chunks(doc)

    id_raw   = chunks.get('identified_text', '') or ''
    anon_raw = chunks.get('anonymized_text', '') or ''
    ents_id  = chunks.get('entities_identified', []) or []
    ents_anon= chunks.get('entities_anonymized', []) or []

    # A. Prompt leakage
    if _PROMPT_TAG_RE.search(id_raw):
        issues['prompt_leakage'] = True

    # B. Thinking leak in anonymized_text
    for pat in _THINKING_PATTERNS:
        if pat.search(anon_raw):
            issues['thinking_leak'] = True
            break

    # Work with cleaned text for subsequent checks
    id_clean   = _strip_prompt(id_raw)
    id_spans   = _spans(id_clean)
    anon_spans = _spans(anon_raw)

    # C. Empty anon text (identified has spans, anon has none)
    if id_spans and not anon_spans:
        issues['empty_anon_text'] = True

    # D. Both text fields empty
    if not id_raw.strip() and not anon_raw.strip():
        issues['empty_id_text'] = True

    # E. Fallback only (no identified_text but list has entries)
    if not id_raw.strip() and (
        (isinstance(ents_id,  list) and ents_id)  or
        (isinstance(ents_anon, list) and ents_anon)
    ):
        issues['fallback_only'] = True

    # F. Span count mismatch (both non-zero)
    if id_spans and anon_spans and len(id_spans) != len(anon_spans):
        issues['span_count_mismatch'] = True

    # G. Category name appearing as entity in identified_text
    for sp in id_spans:
        if _VALID_CATEGORY_RE.match(sp.upper()) and sp == sp.upper():
            issues['category_as_entity'] = True
            break

    # H. Placeholder [**value**] after prompt strip
    if 'value' in id_spans:
        issues['placeholder_value'] = True

    # I. Empty spans in either text
    for sp in id_spans + anon_spans:
        if not sp:
            issues['empty_span'] = True
            break

    # J. Unclosed markers
    if _count_unclosed(id_raw) > 0 or _count_unclosed(anon_raw) > 0:
        issues['unclosed_marker'] = True

    # K. Nested markers: '[**' inside a captured span content
    for sp in _PAT.findall(id_raw) + _PAT.findall(anon_raw):
        if '[**' in sp or '**]' in sp:
            issues['nested_marker'] = True
            break

    # L. Anon spans look like entity text (not a valid category)
    for sp in anon_spans:
        if not _VALID_CATEGORY_RE.match(sp.upper().replace(' ', '_')):
            # Has lowercase letters → likely entity text, not a category label
            if _ANON_ENTITY_RE.search(sp) and len(sp.split()) <= 8:
                issues['anon_entity_text'] = True
                break

    return issues


# ---------------------------------------------------------------------------
# Audit a single model file
# ---------------------------------------------------------------------------

def _load_excluded(problematic_path: str) -> set:
    """Load the set of namefiles to exclude from the audit."""
    if not os.path.exists(problematic_path):
        return set()
    with open(problematic_path, encoding='utf-8') as f:
        data = json.load(f)
    return {entry['namefile'] for entry in data if 'namefile' in entry}


def audit_model(path: str, excluded: set = None) -> Dict:
    with open(path, encoding='utf-8') as f:
        docs = json.load(f)

    if excluded is None:
        excluded = set()

    model_name = os.path.basename(path).replace('output_', '').replace('.json', '')

    counts: Dict[str, int] = defaultdict(int)
    affected: Dict[str, List[str]] = defaultdict(list)
    total_docs = 0
    any_issue  = 0

    for doc in docs:
        namefile = doc.get('namefile', doc.get('id', '?'))
        if namefile in excluded:
            continue
        total_docs += 1
        issues = audit_doc(doc)
        has_issue = any(issues.values())
        if has_issue:
            any_issue += 1
        for key, hit in issues.items():
            if hit:
                counts[key] += 1
                affected[key].append(namefile)

    return {
        'model':     model_name,
        'total':     total_docs,
        'any_issue': any_issue,
        'counts':    dict(counts),
        'affected':  dict(affected),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_model_report(result: Dict, verbose: bool = False) -> None:
    name    = result['model']
    total   = result['total']
    any_iss = result['any_issue']
    counts  = result['counts']
    affected= result['affected']

    print(f"\n{'='*70}")
    print(f"  Model: {name}")
    print(f"  Documents: {total}   With any issue: {any_iss} "
          f"({100*any_iss/total:.1f}%)")
    print(f"{'='*70}")

    if not counts:
        print("  [OK] No problems detected.")
        return

    for key in CHECK_KEYS:
        n = counts.get(key, 0)
        if n == 0:
            continue
        label = CHECK_LABELS[key]
        print(f"  {label}")
        print(f"      -> {n} doc(s)")
        if verbose:
            for fn in affected.get(key, [])[:10]:
                print(f"         {fn}")
            if len(affected.get(key, [])) > 10:
                print(f"         ... and {len(affected[key])-10} more")


def print_summary_table(results: List[Dict]) -> None:
    col_w = 28
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY — ISSUES PER MODEL")
    print(f"{'='*70}")

    # Header
    header = f"{'Model':<{col_w}}"
    for key in CHECK_KEYS:
        header += f" {key[0].upper():>4}"
    header += f"  {'ANY':>5}  {'TOTAL':>6}"
    print(header)
    print('-' * len(header))

    for r in results:
        row = f"{r['model']:<{col_w}}"
        for key in CHECK_KEYS:
            n = r['counts'].get(key, 0)
            row += f" {n:>4}" if n else f" {'':>4}"
        row += f"  {r['any_issue']:>5}  {r['total']:>6}"
        print(row)

    print('-' * len(header))
    # Check key legend
    print()
    for key in CHECK_KEYS:
        print(f"  {key[0].upper()}: {CHECK_LABELS[key]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description='Audit model anonymisation output files for structural problems.'
    )
    parser.add_argument(
        'models_dir', nargs='?', default='models',
        help='Directory with output_*.json files (default: models/)'
    )
    parser.add_argument(
        '--model', '-m', default=None,
        help='Audit only this model (substring match on filename)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show the list of affected documents for each check'
    )
    parser.add_argument(
        '--check', '-c', default=None,
        help='Show only documents affected by this check (e.g. thinking_leak)'
    )
    parser.add_argument(
        '--exclude', '-e', default=None,
        help='Path to problematic_docs.json to exclude from audit '
             '(default: scripts/problematic_docs.json if present)'
    )
    args = parser.parse_args()

    # Resolve exclusion file: explicit arg → default path → empty set
    if args.exclude:
        excl_path = args.exclude
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excl_path  = os.path.join(script_dir, 'problematic_docs.json')
    excluded = _load_excluded(excl_path)
    if excluded:
        print(f"[INFO] Excluding {len(excluded)} problematic documents ({os.path.basename(excl_path)})")

    pattern = os.path.join(args.models_dir, 'output_*.json')
    paths   = sorted(glob.glob(pattern))

    if not paths:
        print(f"[ERROR] No files found matching '{pattern}'")
        sys.exit(1)

    if args.model:
        paths = [p for p in paths if args.model.lower() in p.lower()]
        if not paths:
            print(f"[ERROR] No model matching '{args.model}'")
            sys.exit(1)

    results = []
    for path in paths:
        name = os.path.basename(path)
        print(f"Auditing {name} ...", end=' ', flush=True)
        r = audit_model(path, excluded=excluded)
        results.append(r)
        n = r['any_issue']
        print(f"{n} issue(s)" if n else "OK")

    # Per-model detail
    verbose_flag = args.verbose or (args.check is not None)
    for r in results:
        if args.check:
            # Override: only show the requested check
            r2 = dict(r)
            key = args.check
            if key not in CHECK_KEYS:
                print(f"[ERROR] Unknown check: {key}. Options: {CHECK_KEYS}")
                sys.exit(1)
            r2['counts']  = {key: r['counts'].get(key, 0)}
            r2['affected']= {key: r['affected'].get(key, [])}
            print_model_report(r2, verbose=True)
        else:
            print_model_report(r, verbose=verbose_flag)

    if len(results) > 1:
        print_summary_table(results)


if __name__ == '__main__':
    main()
