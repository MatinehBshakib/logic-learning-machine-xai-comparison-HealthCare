"""
Visualization.py
----------------
All plots for the XAI thesis.
Called from main_analysis.py after the comparative analysis is complete.

Usage (in main_analysis.py):
    from Visualization import generate_all_plots
    generate_all_plots(
        summary_csv   = 'outputs/final_xai_summary.csv',
        performance_csv = 'outputs/classifier_performance.csv',
        output_folder = 'outputs/figures'
    )
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Shared style ──────────────────────────────────────────────────────────────
DARK_BLUE   = '#1F3864'
MID_BLUE    = '#2E75B6'
LIGHT_BLUE  = '#BDD7EE'
ACCENT_RED  = '#C00000'     # used to highlight the SHAP-LIME baseline
GREY        = '#7F7F7F'
FONT_FAMILY = 'DejaVu Serif'

plt.rcParams.update({
    'font.family'       : FONT_FAMILY,
    'axes.titlesize'    : 13,
    'axes.labelsize'    : 11,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

METHOD_PAIRS = [
    'Rulex-SHAP', 'Rulex-LIME', 'SHAP-LIME',
    'Rulex-Ablat', 'SHAP-Ablat', 'LIME-Ablat'
]

PAIR_COLOURS = {
    'Rulex-SHAP' : DARK_BLUE,
    'Rulex-LIME' : MID_BLUE,
    'SHAP-LIME'  : ACCENT_RED,   # baseline — always highlighted in red
    'Rulex-Ablat': '#4472C4',
    'SHAP-Ablat' : '#70AD47',
    'LIME-Ablat' : '#ED7D31',
}

def _clean_dataset_label(name: str) -> str:
    """Remove file extensions and long suffixes for clean axis labels."""
    name = os.path.basename(name).replace('_final_explanation_results.csv', '')
    name = name.replace('_', ' ').replace('.csv', '').strip()
    return name

def _load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].apply(_clean_dataset_label)
    return df

def _ensure_folder(folder: str):
    os.makedirs(folder, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Spearman Heatmap (6 pairs × 9 datasets)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spearman_heatmap(summary_csv: str, output_folder: str):
    """
    Heatmap of Spearman correlations: rows = datasets, columns = method pairs.
    The SHAP-LIME column is framed in red to show the baseline.
    Saved as: figures/spearman_heatmap.png
    """
    df = _load_summary(summary_csv)
    # Keep only data rows (drop TOTAL AVERAGE)
    data = df[df['Dataset'] != 'TOTAL AVERAGE'].copy()

    cols = [f'Spearman ({p})' for p in METHOD_PAIRS]
    # Some CSVs use 'Rulex-SHAP', some 'Rulex–SHAP' — normalise
    col_map = {}
    for c in data.columns:
        for p in METHOD_PAIRS:
            if p.replace('-', '') in c.replace('-', '').replace('–', ''):
                col_map[c] = f'Spearman ({p})'
    data = data.rename(columns=col_map)

    # Build matrix: rows = datasets, cols = pairs
    matrix = data.set_index('Dataset')[[f'Spearman ({p})' for p in METHOD_PAIRS]].astype(float)

    # Custom diverging colormap: red (−1) → white (0) → dark blue (1)
    cmap = LinearSegmentedColormap.from_list(
        'rw_blue', ['#C00000', '#FFFFFF', DARK_BLUE], N=256
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=-0.2, vmax=1.0, aspect='auto')

    # Axis labels
    ax.set_xticks(range(len(METHOD_PAIRS)))
    ax.set_xticklabels(
        [p.replace('-', '–') for p in METHOD_PAIRS],
        rotation=30, ha='right', fontsize=9
    )
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=9)

    # Cell values
    for i in range(len(matrix)):
        for j in range(len(METHOD_PAIRS)):
            val = matrix.values[i, j]
            colour = 'white' if abs(val) > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=colour, fontweight='bold')

    # Highlight SHAP-LIME column with red border (it is column index 2)
    baseline_col = METHOD_PAIRS.index('SHAP-LIME')
    rect = plt.Rectangle(
        (baseline_col - 0.5, -0.5),
        1, len(matrix),
        linewidth=2.5, edgecolor=ACCENT_RED, facecolor='none', zorder=5
    )
    ax.add_patch(rect)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Spearman ρ', fontsize=9)

    # Annotations
    ax.set_title(
        'Spearman Rank Correlation between XAI Methods\n'
        '(red border = SHAP–LIME baseline)',
        fontsize=13, fontweight='bold', pad=12
    )

    out = os.path.join(output_folder, 'spearman_heatmap.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"[Visualization] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Three-metric Grand Average Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_grand_average_bars(summary_csv: str, output_folder: str):
    """
    Grouped bar chart: for each metric (Spearman, Kendall, Jaccard),
    shows the grand average for each of the 6 method pairs.
    The SHAP-LIME bar is always coloured red as the reference baseline.
    Saved as: figures/grand_average_bars.png
    """
    df = _load_summary(summary_csv)
    total = df[df['Dataset'] == 'TOTAL AVERAGE'].iloc[0]

    metrics = ['Spearman', 'Kendall', 'Jaccard']
    metric_labels = ['Spearman ρ', "Kendall's τ", 'Jaccard J (top-5)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        values, colours, labels = [], [], []
        for pair in METHOD_PAIRS:
            # Try several column name patterns
            for sep in ['-', '–', ' - ']:
                col = f'{metric} (Rulex{sep}SHAP)' if pair == 'Rulex-SHAP' else None
                col = f'{metric} ({pair.replace("-", sep)})'
                if col in total.index:
                    break
            val = float(total.get(col, 0))
            values.append(val)
            colours.append(ACCENT_RED if pair == 'SHAP-LIME' else PAIR_COLOURS[pair])
            labels.append(pair.replace('-', '–'))

        x = np.arange(len(METHOD_PAIRS))
        bars = ax.bar(x, values, color=colours, edgecolor='white',
                      linewidth=0.8, width=0.65, zorder=3)

        # Value labels on bars
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        # Horizontal dashed line at SHAP-LIME value
        baseline_val = values[METHOD_PAIRS.index('SHAP-LIME')]
        ax.axhline(baseline_val, color=ACCENT_RED, linestyle='--',
                   linewidth=1.4, alpha=0.7, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f'{mlabel}\n(grand average, N=9 datasets)',
                     fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=1)
        ax.set_axisbelow(True)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=ACCENT_RED, label='SHAP–LIME (baseline)'),
        mpatches.Patch(color=DARK_BLUE,  label='Rulex–SHAP (main result)'),
        mpatches.Patch(color=GREY,       label='Other pairs'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        'Grand Average Alignment across All Nine Medical Datasets\n'
        '— red dashed line marks the SHAP–LIME reference level —',
        fontsize=12, fontweight='bold', y=1.02
    )

    out = os.path.join(output_folder, 'grand_average_bars.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"[Visualization] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Per-Dataset Spearman Profile (Rulex-SHAP vs SHAP-LIME)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_parity_profile(summary_csv: str, output_folder: str):
    """
    Line plot: per dataset, shows Spearman ρ for Rulex-SHAP (dark blue)
    vs SHAP-LIME (red). The gap between the two lines is the parity evidence.
    Saved as: figures/parity_profile.png
    """
    df = _load_summary(summary_csv)
    data = df[df['Dataset'] != 'TOTAL AVERAGE'].copy()

    # Normalise column names
    rs_col = next((c for c in data.columns if 'Rulex' in c and 'SHAP' in c
                   and 'Spearman' in c and 'LIME' not in c and 'Ablat' not in c), None)
    sl_col = next((c for c in data.columns if 'SHAP' in c and 'LIME' in c
                   and 'Spearman' in c and 'Rulex' not in c and 'Ablat' not in c), None)

    if rs_col is None or sl_col is None:
        print("[Visualization] Warning: Could not find required columns for parity_profile.")
        return

    datasets   = data['Dataset'].tolist()
    rulex_shap = data[rs_col].astype(float).tolist()
    shap_lime  = data[sl_col].astype(float).tolist()

    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(x, rulex_shap, 'o-', color=DARK_BLUE, linewidth=2,
            markersize=8, label='Rulex–SHAP (main result)', zorder=4)
    ax.plot(x, shap_lime, 's--', color=ACCENT_RED, linewidth=2,
            markersize=8, label='SHAP–LIME (baseline)', zorder=4)

    # Shade the gap between the two lines
    ax.fill_between(x, rulex_shap, shap_lime,
                    where=[rs >= sl for rs, sl in zip(rulex_shap, shap_lime)],
                    alpha=0.15, color=DARK_BLUE, label='Rulex–SHAP ≥ baseline')
    ax.fill_between(x, rulex_shap, shap_lime,
                    where=[rs < sl for rs, sl in zip(rulex_shap, shap_lime)],
                    alpha=0.15, color=ACCENT_RED, label='Rulex–SHAP < baseline')

    # Grand averages as horizontal lines
    ax.axhline(np.mean(rulex_shap), color=DARK_BLUE, linestyle=':',
               linewidth=1.2, alpha=0.7)
    ax.axhline(np.mean(shap_lime),  color=ACCENT_RED, linestyle=':',
               linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_ylim(-0.2, 1.05)
    ax.set_title(
        'Per-Dataset Spearman Correlation: Rulex–SHAP vs. SHAP–LIME Baseline\n'
        '(dotted lines = grand averages)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    out = os.path.join(output_folder, 'parity_profile.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"[Visualization] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Jaccard Heatmap (top-5 overlap)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_jaccard_heatmap(summary_csv: str, output_folder: str):
    """
    Same layout as the Spearman heatmap but for Jaccard (top-5).
    Saved as: figures/jaccard_heatmap.png
    """
    df = _load_summary(summary_csv)
    data = df[df['Dataset'] != 'TOTAL AVERAGE'].copy()

    col_map = {}
    for c in data.columns:
        for p in METHOD_PAIRS:
            if 'Jaccard' in c and p.replace('-', '') in c.replace('-', '').replace('–', ''):
                col_map[c] = f'Jaccard ({p})'
    data = data.rename(columns=col_map)

    jac_cols = [f'Jaccard ({p})' for p in METHOD_PAIRS]
    matrix = data.set_index('Dataset')[jac_cols].astype(float)

    cmap = LinearSegmentedColormap.from_list(
        'white_blue', ['#FFFFFF', LIGHT_BLUE, DARK_BLUE], N=256
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(METHOD_PAIRS)))
    ax.set_xticklabels(
        [p.replace('-', '–') for p in METHOD_PAIRS],
        rotation=30, ha='right', fontsize=9
    )
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=9)

    for i in range(len(matrix)):
        for j in range(len(METHOD_PAIRS)):
            val = matrix.values[i, j]
            colour = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=colour, fontweight='bold')

    baseline_col = METHOD_PAIRS.index('SHAP-LIME')
    rect = plt.Rectangle(
        (baseline_col - 0.5, -0.5), 1, len(matrix),
        linewidth=2.5, edgecolor=ACCENT_RED, facecolor='none', zorder=5
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Jaccard J (top-5)', fontsize=9)

    ax.set_title(
        'Jaccard Similarity (Top-5 Features) between XAI Methods\n'
        '(red border = SHAP–LIME baseline)',
        fontsize=13, fontweight='bold', pad=12
    )

    out = os.path.join(output_folder, 'jaccard_heatmap.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"[Visualization] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Classifier Performance Table (Accuracy, F1, AUROC)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_performance_table(performance_csv: str, output_folder: str):
    """
    Renders the classifier performance CSV (one row per dataset) as a clean
    formatted table image. Saved as: figures/classifier_performance.png
    """
    if not os.path.exists(performance_csv):
        print(f"[Visualization] Warning: {performance_csv} not found. "
              "Run main_single.py first to generate it.")
        return

    df = pd.read_csv(performance_csv)
    df['Dataset'] = df['Dataset'].apply(_clean_dataset_label)

    # Round numeric columns
    num_cols = [c for c in df.columns if c != 'Dataset']
    df[num_cols] = df[num_cols].round(4)

    fig, ax = plt.subplots(figsize=(10, 0.55 * (len(df) + 2)))
    ax.axis('off')

    col_labels = df.columns.tolist()
    cell_text  = df.values.tolist()

    # Colour alternating rows
    row_colours = [
        ['#EBF3FB' if i % 2 == 0 else '#FFFFFF'] * len(col_labels)
        for i in range(len(df))
    ]

    tbl = ax.table(
        cellText   = cell_text,
        colLabels  = col_labels,
        cellColours= row_colours,
        loc        = 'center',
        cellLoc    = 'center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(DARK_BLUE)
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title(
        'Classifier Performance per Dataset\n'
        '(RF = Random Forest, XGB = XGBoost; 70/30 stratified split)',
        fontsize=12, fontweight='bold', pad=16
    )

    out = os.path.join(output_folder, 'classifier_performance.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"[Visualization] Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Cumulative Ablation Degradation Curve  (one plot per dataset)
#           Called directly from Analysis.py after each dataset is processed.
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cumulative_ablation_per_dataset(
        explanation_csv: str,
        dataset_name:    str,
        output_folder:   str = 'outputs/figures',
        max_display:     int = 20,
):
    """
    Reads the merged explanation CSV produced by PostProcessor
    (e.g. 'outputs/Breast_Cancer_final_explanation_results.csv')
    and produces a two-panel figure showing:

    TOP PANEL  — Step-down line: mean predicted probability as features are
                 progressively masked (least → most important, left → right).
                 The steeper the drop at a step, the more important that feature.

    BOTTOM PANEL — Horizontal bar chart: per-feature drop magnitude
                   (mean cum_ablation_value).  Bars are coloured by intensity;
                   the top-3 drops are labelled and highlighted in red.

    Parameters
    ----------
    explanation_csv : path to *_final_explanation_results.csv
    dataset_name    : human-readable label used in the title and filename
    output_folder   : where to save the figure
    max_display     : maximum number of features to display (right-most = most
                      important).  If d > max_display, the least-important
                      features are collapsed to one "... N features" group.

    Output file
    -----------
    outputs/figures/cum_ablation_<dataset_name>.png
    """
    if not os.path.exists(explanation_csv):
        print(f"[Visualization] cum_ablation — file not found: {explanation_csv}")
        return

    df = pd.read_csv(explanation_csv)

    # ── Required columns ─────────────────────────────────────────────────────
    needed = {'feature', 'ablation_value', 'cum_ablation_value',
              'current_prediction', 'original_prediction'}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        print(f"[Visualization] cum_ablation — missing columns {missing} in "
              f"{explanation_csv}")
        return

    # ── Aggregate to per-feature level ───────────────────────────────────────
    agg = (
        df.groupby('feature')
          .agg(
              mean_ablation      = ('ablation_value',      'mean'),
              mean_cum_jump      = ('cum_ablation_value',  'mean'),
              mean_current_pred  = ('current_prediction',  'mean'),
          )
          .reset_index()
    )

    # Masking order = ascending standard ablation (least important first)
    agg = agg.sort_values('mean_ablation', ascending=True).reset_index(drop=True)

    orig_pred = df['original_prediction'].mean()

    # ── Limit display to max_display features ────────────────────────────────
    total_features = len(agg)
    if total_features > max_display:
        # Keep the max_display MOST important features (rightmost = biggest drops)
        hidden_n  = total_features - max_display
        agg = agg.tail(max_display).reset_index(drop=True)
        truncated = True
    else:
        hidden_n  = 0
        truncated = False

    n = len(agg)
    features   = agg['feature'].tolist()
    jumps      = agg['mean_cum_jump'].tolist()
    cur_preds  = agg['mean_current_pred'].tolist()

    # Prediction curve: starts at orig_pred before any masking,
    # then steps down to cur_preds[0], cur_preds[1], ...
    # (cur_preds[i] = prediction AFTER masking features 0..i)
    curve_x = list(range(-1, n))           # -1 = before any masking
    curve_y = [orig_pred] + cur_preds

    # ── Identify top-3 largest drops ────────────────────────────────────────
    sorted_jumps = sorted(enumerate(jumps), key=lambda x: x[1], reverse=True)
    top3_idx = {idx for idx, _ in sorted_jumps[:3]}

    # ── Colours: intensity-mapped, top-3 in red ─────────────────────────────
    max_jump = max(jumps) if max(jumps) > 0 else 1.0
    bar_colours = []
    for i, j in enumerate(jumps):
        if i in top3_idx:
            bar_colours.append(ACCENT_RED)
        else:
            intensity = 0.2 + 0.6 * (j / max_jump)
            bar_colours.append(
                (1 - intensity * 0.6,
                 1 - intensity * 0.6,
                 1 - intensity * 0.0 + intensity * 0.14)
            )

    # ── Figure layout ────────────────────────────────────────────────────────
    fig_h = max(7, 4 + n * 0.25)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(10, n * 0.55), fig_h),
        gridspec_kw={'height_ratios': [2, 1.2]},
        sharex=False
    )

    x_ticks = list(range(n))

    # ────────────────────────────────────────────────────────────────────────
    # TOP PANEL — prediction degradation curve
    # ────────────────────────────────────────────────────────────────────────
    # Draw step-down line
    ax_top.step(
        curve_x, curve_y,
        where='post', color=DARK_BLUE, linewidth=2.2, zorder=4
    )
    ax_top.fill_between(
        curve_x, curve_y, orig_pred,
        step='post', alpha=0.08, color=DARK_BLUE
    )

    # Original prediction baseline
    ax_top.axhline(
        orig_pred, color=GREY, linestyle='--', linewidth=1.2,
        label=f'Original prediction = {orig_pred:.3f}', zorder=3
    )

    # Annotate top-3 drops on the line
    for rank, (idx, jump_val) in enumerate(sorted_jumps[:3]):
        x_pos = idx
        y_pos = cur_preds[idx]
        ax_top.annotate(
            f'  Δ={jump_val:.3f}
  {features[idx]}',
            xy=(x_pos, y_pos),
            xytext=(x_pos + 0.3, y_pos + 0.03 * (3 - rank)),
            fontsize=7.5,
            color=ACCENT_RED,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=1.2),
            zorder=6,
        )

    ax_top.set_ylabel('Mean predicted probability
(positive class)', fontsize=10)
    ax_top.set_ylim(
        max(0, min(cur_preds) - 0.08),
        orig_pred + 0.08
    )
    ax_top.legend(fontsize=9, frameon=False)
    ax_top.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax_top.set_axisbelow(True)
    ax_top.set_xticks([])          # x-labels shown on bottom panel only

    trunc_note = f' (showing {n} most important of {total_features})' if truncated else ''
    ax_top.set_title(
        f'Cumulative Ablation Degradation — {dataset_name}{trunc_note}
'
        f'Features masked left → right (least → most important).  '
        f'Red = top-3 biggest drops.',
        fontsize=11, fontweight='bold', pad=10
    )

    # ────────────────────────────────────────────────────────────────────────
    # BOTTOM PANEL — per-feature drop magnitude (horizontal bars)
    # ────────────────────────────────────────────────────────────────────────
    bars = ax_bot.bar(
        x_ticks, jumps,
        color=bar_colours,
        edgecolor='white', linewidth=0.6,
        width=0.75, zorder=3
    )

    # Value labels on bars that are in top-3
    for idx, (bar, j) in enumerate(zip(bars, jumps)):
        if idx in top3_idx:
            ax_bot.text(
                bar.get_x() + bar.get_width() / 2,
                j + max_jump * 0.01,
                f'{j:.3f}',
                ha='center', va='bottom',
                fontsize=7.5, color=ACCENT_RED, fontweight='bold'
            )

    # X-axis feature labels
    ax_bot.set_xticks(x_ticks)
    ax_bot.set_xticklabels(
        features, rotation=45, ha='right',
        fontsize=max(6, 9 - n // 5)
    )
    ax_bot.set_ylabel('Drop magnitude
(Δ mean prediction)', fontsize=9)
    ax_bot.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax_bot.set_axisbelow(True)

    # Label x-axis direction
    if truncated:
        ax_bot.set_xlabel(
            f'← {hidden_n} less-important features not shown  |  '
            f'Most important features →',
            fontsize=9
        )
    else:
        ax_bot.set_xlabel(
            'Features in masking order  (least important → most important)',
            fontsize=9
        )

    plt.tight_layout(h_pad=1.0)
    _ensure_folder(output_folder)
    safe_name = dataset_name.replace(' ', '_').replace('/', '_')
    out = os.path.join(output_folder, f'cum_ablation_{safe_name}.png')
    fig.savefig(out)
    plt.close(fig)
    print(f"[Visualization] Saved: {out}")

# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION — call this from main_analysis.py
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(summary_csv:     str = 'outputs/final_xai_summary.csv',
                       performance_csv: str = 'outputs/classifier_performance.csv',
                       output_folder:   str = 'outputs/figures'):
    """
    Generates the summary-level thesis figures from pipeline outputs.
    Called once from main_analysis.py AFTER save_final_table().

    NOTE: Cumulative ablation figures (one per dataset) are generated
    automatically by plot_cumulative_ablation_per_dataset(), which is
    called from Analysis.execute_analysis() after each dataset — NOT here.

    Parameters
    ----------
    summary_csv     : path to final_xai_summary.csv
    performance_csv : path to classifier_performance.csv
    output_folder   : all figures saved here
    """
    _ensure_folder(output_folder)
    print(f"\n>>> Generating thesis summary figures in: {output_folder}")

    if os.path.exists(summary_csv):
        plot_spearman_heatmap(summary_csv, output_folder)
        plot_grand_average_bars(summary_csv, output_folder)
        plot_parity_profile(summary_csv, output_folder)
        plot_jaccard_heatmap(summary_csv, output_folder)
    else:
        print(f"[Visualization] Warning: {summary_csv} not found — skipping XAI plots.")

    plot_performance_table(performance_csv, output_folder)

    print(f"\n>>> Summary figures saved to '{output_folder}/'")
