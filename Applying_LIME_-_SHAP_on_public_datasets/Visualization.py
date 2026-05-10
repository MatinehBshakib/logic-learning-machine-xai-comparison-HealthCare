"""
Visualization.py  — XAI Thesis Figures
---------------------------------------
Generates all thesis figures from the pipeline output CSVs.

Called from main_analysis.py:
    from Visualization import generate_all_plots
    generate_all_plots(summary_csv, output_folder)

Cumulative ablation plots (one per dataset) are called automatically from
Analysis.execute_analysis() via plot_cumulative_ablation_per_dataset().
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BLUE  = '#1F3864'
MID_BLUE   = '#2E75B6'
LIGHT_BLUE = '#BDD7EE'
ACCENT_RED = '#C00000'
GREY       = '#7F7F7F'

plt.rcParams.update({
    'font.family'     : 'DejaVu Serif',
    'axes.titlesize'  : 13,
    'axes.labelsize'  : 11,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'figure.dpi'      : 150,
    'savefig.dpi'     : 300,
    'savefig.bbox'    : 'tight',
    'axes.spines.top' : False,
    'axes.spines.right': False,
})

# ── Constants ─────────────────────────────────────────────────────────────────
METHOD_PAIRS = [
    'Rulex-SHAP', 'Rulex-LIME', 'SHAP-LIME',
    'Rulex-Ablat', 'SHAP-Ablat', 'LIME-Ablat'
]
PAIR_COLOURS = {
    'Rulex-SHAP' : DARK_BLUE,
    'Rulex-LIME' : MID_BLUE,
    'SHAP-LIME'  : ACCENT_RED,
    'Rulex-Ablat': '#4472C4',
    'SHAP-Ablat' : '#70AD47',
    'LIME-Ablat' : '#ED7D31',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_label(raw):
    """Strip .csv suffix and underscores for readable axis labels."""
    return raw.replace('.csv', '').replace('_', ' ').strip()

def _load_summary(csv_path):
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].apply(_clean_label)
    return df

def _ensure(folder):
    os.makedirs(folder, exist_ok=True)

def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"[Visualization] Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Spearman Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spearman_heatmap(summary_csv, output_folder):
    df   = _load_summary(summary_csv)
    data = df[df['Dataset'] != 'TOTAL AVERAGE'].copy()

    cols   = [f'Spearman ({p})' for p in METHOD_PAIRS]
    matrix = data.set_index('Dataset')[cols].astype(float)

    cmap = LinearSegmentedColormap.from_list(
        'rw_blue', ['#C00000', '#FFFFFF', DARK_BLUE], N=256)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=-0.2, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(METHOD_PAIRS)))
    ax.set_xticklabels([p.replace('-', '–') for p in METHOD_PAIRS],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=9)

    for i in range(len(matrix)):
        for j in range(len(METHOD_PAIRS)):
            val = matrix.values[i, j]
            col = 'white' if abs(val) > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=col, fontweight='bold')

    # Red border on SHAP-LIME baseline column
    bi = METHOD_PAIRS.index('SHAP-LIME')
    ax.add_patch(plt.Rectangle((bi - 0.5, -0.5), 1, len(matrix),
                                lw=2.5, edgecolor=ACCENT_RED,
                                facecolor='none', zorder=5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Spearman ρ', fontsize=9)
    ax.set_title('Spearman Rank Correlation between XAI Methods\n'
                 '(red border = SHAP–LIME baseline)',
                 fontsize=13, fontweight='bold', pad=12)

    _ensure(output_folder)
    _save(fig, os.path.join(output_folder, 'spearman_heatmap.png'))


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Grand Average Bar Chart (all 3 metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_grand_average_bars(summary_csv, output_folder):
    df    = _load_summary(summary_csv)
    total = df[df['Dataset'] == 'TOTAL AVERAGE'].iloc[0]

    metrics = ['Spearman', 'Kendall', 'Jaccard']
    mlabels = ['Spearman ρ', "Kendall's τ", 'Jaccard J (top-5)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, metric, mlabel in zip(axes, metrics, mlabels):
        values  = [float(total[f'{metric} ({p})']) for p in METHOD_PAIRS]
        colours = [ACCENT_RED if p == 'SHAP-LIME' else PAIR_COLOURS[p]
                   for p in METHOD_PAIRS]

        bars = ax.bar(range(len(METHOD_PAIRS)), values,
                      color=colours, edgecolor='white', linewidth=0.8,
                      width=0.65, zorder=3)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        baseline = values[METHOD_PAIRS.index('SHAP-LIME')]
        ax.axhline(baseline, color=ACCENT_RED, linestyle='--',
                   linewidth=1.4, alpha=0.7, zorder=2)

        ax.set_xticks(range(len(METHOD_PAIRS)))
        ax.set_xticklabels([p.replace('-', '–') for p in METHOD_PAIRS],
                           rotation=35, ha='right', fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f'{mlabel}\n(grand average, 9 datasets)',
                     fontsize=10, fontweight='bold')
        ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=1)
        ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=ACCENT_RED, label='SHAP–LIME (baseline)'),
        mpatches.Patch(color=DARK_BLUE,  label='Rulex–SHAP (main result)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Grand Average Alignment — red dashed line = SHAP–LIME reference',
                 fontsize=12, fontweight='bold', y=1.02)

    _ensure(output_folder)
    _save(fig, os.path.join(output_folder, 'grand_average_bars.png'))


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Per-Dataset Parity Profile (Rulex-SHAP vs SHAP-LIME)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_parity_profile(summary_csv, output_folder):
    df   = _load_summary(summary_csv)
    data = df[df['Dataset'] != 'TOTAL AVERAGE'].copy()

    datasets   = data['Dataset'].tolist()
    rulex_shap = data['Spearman (Rulex-SHAP)'].astype(float).tolist()
    shap_lime  = data['Spearman (SHAP-LIME)'].astype(float).tolist()
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(x, rulex_shap, 'o-', color=DARK_BLUE, linewidth=2,
            markersize=8, label='Rulex–SHAP (main result)', zorder=4)
    ax.plot(x, shap_lime, 's--', color=ACCENT_RED, linewidth=2,
            markersize=8, label='SHAP–LIME (baseline)', zorder=4)

    ax.fill_between(x, rulex_shap, shap_lime,
                    where=[rs >= sl for rs, sl in zip(rulex_shap, shap_lime)],
                    alpha=0.12, color=DARK_BLUE)
    ax.fill_between(x, rulex_shap, shap_lime,
                    where=[rs < sl for rs, sl in zip(rulex_shap, shap_lime)],
                    alpha=0.12, color=ACCENT_RED)

    ax.axhline(np.mean(rulex_shap), color=DARK_BLUE,
               linestyle=':', linewidth=1.2, alpha=0.7)
    ax.axhline(np.mean(shap_lime),  color=ACCENT_RED,
               linestyle=':', linewidth=1.2, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Spearman ρ', fontsize=11)
    ax.set_ylim(-0.2, 1.05)
    ax.set_title('Per-Dataset Spearman: Rulex–SHAP vs. SHAP–LIME Baseline\n'
                 '(dotted lines = grand averages)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.yaxis.grid(True, linestyle=':', alpha=0.4)
    ax.set_axisbelow(True)

    _ensure(output_folder)
    _save(fig, os.path.join(output_folder, 'parity_profile.png'))


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Jaccard Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_jaccard_heatmap(summary_csv, output_folder):
    df   = _load_summary(summary_csv)
    data = df[df['Dataset'] != 'TOTAL AVERAGE'].copy()

    cols   = [f'Jaccard ({p})' for p in METHOD_PAIRS]
    matrix = data.set_index('Dataset')[cols].astype(float)

    cmap = LinearSegmentedColormap.from_list(
        'white_blue', ['#FFFFFF', LIGHT_BLUE, DARK_BLUE], N=256)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(METHOD_PAIRS)))
    ax.set_xticklabels([p.replace('-', '–') for p in METHOD_PAIRS],
                       rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index.tolist(), fontsize=9)

    for i in range(len(matrix)):
        for j in range(len(METHOD_PAIRS)):
            val = matrix.values[i, j]
            col = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=col, fontweight='bold')

    bi = METHOD_PAIRS.index('SHAP-LIME')
    ax.add_patch(plt.Rectangle((bi - 0.5, -0.5), 1, len(matrix),
                                lw=2.5, edgecolor=ACCENT_RED,
                                facecolor='none', zorder=5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Jaccard J (top-5)', fontsize=9)
    ax.set_title('Jaccard Similarity (Top-5 Features) between XAI Methods\n'
                 '(red border = SHAP–LIME baseline)',
                 fontsize=13, fontweight='bold', pad=12)

    _ensure(output_folder)
    _save(fig, os.path.join(output_folder, 'jaccard_heatmap.png'))

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Cumulative Ablation Curve  (one per dataset, called from Analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cumulative_ablation_per_dataset(
        explanation_csv : str,
        dataset_name    : str,
        output_folder   : str = 'outputs/figures',
        max_features    : int = 20,
        top_k_annotate  : int = 5,
):
    """
    Step curve showing signed prediction change as features are masked
    cumulatively (least → most important).

    Annotation strategy: numbered circles on vlines + fixed legend box.
    This is overlap-proof regardless of how the top-k features cluster.
    """
    if not os.path.exists(explanation_csv):
        print(f"[cum_ablation] File not found: {explanation_csv}")
        return

    df = pd.read_csv(explanation_csv)

    needed = {'feature', 'ablation_value', 'cum_ablation_value',
              'current_prediction', 'original_prediction'}
    missing = needed - set(df.columns)
    if missing:
        print(f"[cum_ablation] Missing columns: {missing}")
        return

    # ── Per-feature means ─────────────────────────────────────────────────────
    agg = (
        df.groupby('feature')
          .agg(
              mean_ablation = ('ablation_value',     'mean'),
              mean_drop     = ('cum_ablation_value', 'mean'),
              mean_pred     = ('current_prediction', 'mean'),
          )
          .reset_index()
          .sort_values('mean_ablation', ascending=True)
          .reset_index(drop=True)
    )

    orig_pred   = float(df['original_prediction'].mean())
    total_feats = len(agg)

    truncated = total_feats > max_features
    if truncated:
        agg = agg.tail(max_features).reset_index(drop=True)

    n         = len(agg)
    features  = agg['feature'].astype(str).tolist()
    drops     = agg['mean_drop'].tolist()
    run_preds = agg['mean_pred'].tolist()

    x_vals = list(range(n + 1))
    y_vals = [orig_pred] + run_preds

    # Rank by absolute magnitude of change (biggest impact first)
    ranked = sorted(range(n), key=lambda i: drops[i], reverse=True)
    top_k  = set(ranked[:top_k_annotate])

    # ── Y-axis limits: span the full actual curve range ───────────────────────
    y_min = max(0.0, min(y_vals) - 0.06)
    y_max = max(y_vals) + 0.06

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_w = max(10, n * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    # Shading under curve (use full actual range as floor)
    y_floor = max(0.0, min(y_vals) - 0.02)
    ax.fill_between(x_vals, y_vals, y_floor,
                    step='post', color=DARK_BLUE, alpha=0.07, zorder=1)

    # Step curve
    ax.step(x_vals, y_vals, where='post',
            color=DARK_BLUE, linewidth=2.5, zorder=4, solid_capstyle='round')

    # Baseline dashed line
    ax.axhline(orig_pred, color=GREY, linestyle='--', linewidth=1.0, zorder=2,
               label=f'Original prediction  ({orig_pred:.3f})')

    # ── Grey vlines for non-top-k impactful steps ─────────────────────────────
    for i in range(n):
        if i in top_k or drops[i] < 0.001:
            continue
        y_after  = run_preds[i]
        y_before = run_preds[i - 1] if i > 0 else orig_pred
        ax.vlines(i + 1, y_after, y_before,
                  color=GREY, linewidth=1.0, alpha=0.45, zorder=3)

    # ── Top-k: red vlines + numbered circles ─────────────────────────────────
    # Legend entries built in rank order (rank 1 = biggest impact)
    legend_lines = []

    for rank_pos, i in enumerate(ranked[:top_k_annotate]):
        x_step   = i + 1
        y_after  = run_preds[i]
        y_before = run_preds[i - 1] if i > 0 else orig_pred
        mid_y    = (y_before + y_after) / 2

        # Red vertical line marking the step
        ax.vlines(x_step, y_after, y_before,
                  color=ACCENT_RED, linewidth=2.5, alpha=1.0, zorder=5)

        # Small numbered circle at the midpoint of the vline
        rank_num = str(rank_pos + 1)
        ax.text(
            x_step, mid_y, rank_num,
            ha='center', va='center',
            fontsize=6.5, fontweight='bold', color='white',
            zorder=8,
            bbox=dict(boxstyle='circle,pad=0.20',
                      facecolor=ACCENT_RED, edgecolor='white', linewidth=0.6),
        )

        # Accumulate legend row  (direction arrow + magnitude + feature name)
        direction = '\u2193' if y_after <= y_before else '\u2191'
        legend_lines.append(
            f'  {rank_num}. {features[i]}  {direction} {drops[i]:.3f}'
        )

    # ── Legend box in upper-left (always clear of right-side cluster) ─────────
    legend_text = 'Top impactful:\n' + '\n'.join(legend_lines)
    ax.text(
        0.01, 0.98, legend_text,
        transform=ax.transAxes,
        fontsize=8.5, color=ACCENT_RED, fontweight='bold',
        va='top', ha='left', linespacing=1.6,
        bbox=dict(boxstyle='round,pad=0.45',
                  facecolor='white', edgecolor=ACCENT_RED,
                  alpha=0.90, linewidth=0.9),
        zorder=9,
    )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(1, n + 1))
    tick_fs = max(6.5, 9.5 - n // 4)
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=tick_fs)
    ax.set_xlim(-0.2, n + 0.5)

    ax.set_ylim(y_min, y_max)
    ax.set_ylabel('Mean predicted probability\n(positive class)', fontsize=11)
    ax.yaxis.grid(True, linestyle=':', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(fontsize=9, frameon=False, loc='upper right')

    trunc_note = (f'  ({n} most important of {total_feats} features shown)'
                  if truncated else '')
    ax.set_title(
        f'Cumulative Ablation \u2014 {dataset_name}{trunc_note}\n'
        f'Each step = prediction change after masking that feature.  '
        f'\u2193\u2191 = direction.  Numbered = top {top_k_annotate} most impactful.',
        fontsize=12, fontweight='bold', pad=12, loc='left'
    )
    ax.set_xlabel(
        ('← less important features hidden  |  ' if truncated else '') +
        'Features: least important (left) → most important (right)',
        fontsize=9
    )

    plt.tight_layout()
    _ensure(output_folder)
    safe = dataset_name.replace(' ', '_').replace('/', '_')
    out  = os.path.join(output_folder, f'cum_ablation_{safe}.png')
    _save(fig, out)

# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION — called from main_analysis.py
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(summary_csv     = 'outputs/final_xai_summary.csv',
                       output_folder   = 'outputs/figures'):
    """
    Generates all summary-level thesis figures.
    Cumulative ablation plots are generated per-dataset inside Analysis.py.
    """
    _ensure(output_folder)
    print(f'\n>>> Generating thesis figures in: {output_folder}')

    if os.path.exists(summary_csv):
        plot_spearman_heatmap(summary_csv, output_folder)
        plot_grand_average_bars(summary_csv, output_folder)
        plot_parity_profile(summary_csv, output_folder)
        plot_jaccard_heatmap(summary_csv, output_folder)
    else:
        print(f'[Visualization] {summary_csv} not found — skipping XAI plots.')

    print(f'\n>>> Done. Figures in: {output_folder}')