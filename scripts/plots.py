#!/usr/bin/env python3
"""
plots.py — All visualisations for the CARMEN-I evaluation pipeline.

Functions called by run_evaluation.py:
    load_confusion(metrics_path)          → (model_name, cm_dict | None)
    plot_confusion(model_name, cm, dir)   → M2 confusion matrix heatmap
    load_all_metrics(results_dir)         → list of metric dicts
    plot_comparison(rows, out_path)       → grouped-bar M1/M2/M3 comparison
    plot_summary_table(summary_data, dir) → styled table PNG

Each function can also be called standalone via __main__ for debugging.
"""

import json
import glob
import os
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False

# Okabe-Ito palette — colorblind-safe, print-friendly
METRIC_COLORS = {
    'm1': '#0072B2',   # blue
    'm2': '#E69F00',   # amber
    'm3': '#009E73',   # green
}
METRIC_LABELS = {
    'm1': 'M1 – Identification',
    'm2': 'M2 – Categorization',
    'm3': 'M3 – End-to-end',
}
BAR_W = 0.22


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# M2 confusion matrix
# ---------------------------------------------------------------------------

def load_confusion(metrics_path: str):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cm = data.get('global', {}).get('m2_confusion', None)
    if not cm:
        return None, None
    return data.get('model_name', os.path.basename(metrics_path)), cm


def plot_confusion(model_name: str, cm: dict, out_dir: str):
    matrix = cm.get('matrix', {})
    pc     = cm.get('per_class', {})

    if not matrix:
        print(f"    [SKIP confusion] {model_name}: empty matrix")
        return

    gt_classes = [c for c in sorted(pc, key=lambda x: -pc[x]['support'])
                  if c in matrix]
    if not gt_classes:
        return

    all_preds = set()
    for row in matrix.values():
        all_preds.update(row.keys())
    all_cols = sorted(
        set(gt_classes) | all_preds,
        key=lambda x: -pc.get(x, {'support': 0})['support']
    )

    n_rows  = len(gt_classes)
    n_cols  = len(all_cols)
    col_idx = {c: i for i, c in enumerate(all_cols)}

    A = np.zeros((n_rows, n_cols), dtype=float)
    for r, gt_cls in enumerate(gt_classes):
        for pred_cls, count in matrix[gt_cls].items():
            c = col_idx.get(pred_cls)
            if c is not None:
                A[r, c] += count

    row_sums = A.sum(axis=1, keepdims=True)
    A_norm   = np.divide(A, row_sums, where=row_sums > 0)

    cell_fs  = max(6, 10 - n_rows // 5)
    fig_w    = max(8, n_cols * 0.65 + 2.5)
    fig_h    = max(6, n_rows * 0.55 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(A_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

    for i in range(n_rows):
        for j in range(n_cols):
            if A[i, j] > 0:
                txt_col = 'white' if A_norm[i, j] > 0.60 else 'black'
                ax.text(j, i, f'{A_norm[i, j]:.2f}',
                        ha='center', va='center',
                        fontsize=cell_fs, color=txt_col)

    tick_fs = max(7, 10 - max(n_rows, n_cols) // 5)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(all_cols, rotation=45, ha='right', fontsize=tick_fs)
    ax.set_yticklabels(gt_classes, fontsize=tick_fs)
    ax.set_xlabel('Predicted type', fontsize=11)
    ax.set_ylabel('True type (GT)', fontsize=11)
    ax.set_title(f'M2  Confusion matrix (row-normalised)  —  {model_name}',
                 fontsize=11, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Proportion', fontsize=9)

    plt.tight_layout()
    safe = model_name.replace('/', '_').replace(' ', '_')
    _save(fig, out_dir, f'{safe}_m2_confusion.png')


# ---------------------------------------------------------------------------
# Models comparison bar chart
# ---------------------------------------------------------------------------

def load_all_metrics(results_dir: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*_metrics.json'))):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        g = data.get('global', {})
        if not all(k in g for k in ('m1', 'm2', 'm3')):
            continue
        rows.append({
            'model': data.get('model_name', os.path.basename(path).replace('_metrics.json', '')),
            'm1': g['m1'],
            'm2': g['m2'],
            'm3': g['m3'],
        })
    return rows


def plot_comparison(rows: list, out_path: str, sort_by: str = 'm3'):
    if not rows:
        print("No data to plot.")
        return

    rows = sorted(rows, key=lambda r: r[sort_by]['f1'])
    model_names = [r['model'] for r in rows]
    n = len(model_names)
    x = np.arange(n)

    all_f1s = [r[m]['f1'] for r in rows for m in ('m1', 'm2', 'm3')]
    y_min = max(0.0, min(all_f1s) - 0.05)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.78 + 1.5), 5.5))

    for metric, off in zip(('m1', 'm2', 'm3'), (-BAR_W, 0, BAR_W)):
        f1s  = np.array([r[metric]['f1'] for r in rows])
        bars = ax.bar(x + off, f1s - y_min, BAR_W,
                      bottom=y_min, color=METRIC_COLORS[metric],
                      alpha=0.88, label=METRIC_LABELS[metric], zorder=2)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.004, f'{val:.3f}',
                    ha='center', va='bottom',
                    fontsize=6.5, color='#333333', rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=9)
    ax.set_ylim(y_min, 1.0 + 0.085)
    ax.set_ylabel('F1 score', fontsize=11)
    ax.set_title(f'M1 / M2 / M3 F1  —  all models  (sorted by {sort_by.upper()} F1)',
                 fontsize=12, pad=14)
    ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc='upper left')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table PNG
# ---------------------------------------------------------------------------

def plot_summary_table(summary_data: list, out_dir: str):
    if not summary_data:
        return

    data = sorted(summary_data, key=lambda x: -x['m3_f1'])

    cols = ['Model', 'M1_n', 'M2/3_n',
            'M1_P', 'M1_R', 'M1_F1',
            'M2_F1',
            'M3_P', 'M3_R', 'M3_F1',
            'Faith.id', 'Faith.anon',
            '%err.id', '%err.cat']

    rows = []
    for d in data:
        fi = d.get('faithfulness_identify_mean')
        fa = d.get('faithfulness_anonymize_mean')
        rows.append([
            d['model'],
            str(d.get('docs_m1', d.get('docs_evaluated', ''))),
            str(d.get('docs_m2_m3', d.get('docs_evaluated', ''))),
            f"{d['m1_precision']:.4f}",
            f"{d['m1_recall']:.4f}",
            f"{d['m1_f1']:.4f}",
            f"{d['m2_f1']:.4f}",
            f"{d['m3_precision']:.4f}",
            f"{d['m3_recall']:.4f}",
            f"{d['m3_f1']:.4f}",
            f"{fi:.4f}" if fi is not None else 'N/A',
            f"{fa:.4f}" if fa is not None else 'N/A',
            f"{d['m3_pct_identification']:.1f}%",
            f"{d['m3_pct_categorization']:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(22, max(3.5, len(data) * 0.7 + 1.5)))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Header
    for j in range(len(cols)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Row styles + M3_F1 heatmap
    best_m3 = max(d['m3_f1'] for d in data)
    m3_col  = cols.index('M3_F1')
    for i, d in enumerate(data):
        bg = '#f0f4f8' if i % 2 == 0 else '#ffffff'
        for j in range(len(cols)):
            table[i + 1, j].set_facecolor(bg)
        v = d['m3_f1']
        if v == best_m3:
            table[i + 1, m3_col].set_facecolor('#27ae60')
            table[i + 1, m3_col].set_text_props(color='white', fontweight='bold')
        elif v >= 0.93:
            table[i + 1, m3_col].set_facecolor('#a8d5a2')
        elif v >= 0.90:
            table[i + 1, m3_col].set_facecolor('#f9e79f')
        elif v < 0.80:
            table[i + 1, m3_col].set_facecolor('#f1948a')

    ax.set_title('CARMEN-I evaluation summary', fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    _save(fig, out_dir, 'models_summary_table.png')


# ---------------------------------------------------------------------------
# Standalone CLI (regenerate all plots from existing results/)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Regenerate all evaluation plots')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out-dir',     default='results/plots')
    parser.add_argument('--sort-by', default='m3', choices=['m1', 'm2', 'm3'])
    args = parser.parse_args()

    # Confusion matrices
    for path in sorted(glob.glob(os.path.join(args.results_dir, '*_metrics.json'))):
        model_name, cm = load_confusion(path)
        if cm is not None:
            print(f"Confusion: {model_name}")
            plot_confusion(model_name, cm, args.out_dir)

    # Comparison bar chart
    rows = load_all_metrics(args.results_dir)
    if rows:
        print(f"Comparison: {len(rows)} models")
        plot_comparison(rows, os.path.join(args.out_dir, 'models_comparison.png'),
                        sort_by=args.sort_by)

    # Summary table
    summary_path = os.path.join(args.results_dir, 'models_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            summary_data = json.load(f)
        print("Summary table")
        plot_summary_table(summary_data, args.out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
