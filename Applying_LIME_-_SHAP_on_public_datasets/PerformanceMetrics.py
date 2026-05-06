"""
PerformanceMetrics.py
---------------------
Computes Accuracy, F1 (weighted), and AUROC for trained classifiers
and appends rows to outputs/classifier_performance.csv.

Three functions, one per strategy type:
  save_performance_metrics()              -> SingleOutput       (1 row)
  save_performance_metrics_multilabel()   -> MultiLabelStrategy (1 or N+1 rows)
  save_performance_metrics_hierarchical() -> HierarchicalStrategy(1 or N+1 rows)

Set  average_only=True  (default) to write ONE averaged row per dataset.
Set  average_only=False           to write one row per target + one average row.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_row(clf, x_test, y_true_1d, dataset_name, target_name,
                 n_train, n_test, clf_name=None):
    """Compute metrics for one binary target → return one dict (= one CSV row)."""
    # Ensure we only use the features the model was trained on
    if hasattr(clf, 'feature_names_in_') and isinstance(x_test, pd.DataFrame):
        x_test = x_test[clf.feature_names_in_]

    y_true = np.array(y_true_1d).ravel()
    y_pred = clf.predict(x_test)

    try:
        y_prob = clf.predict_proba(x_test)[:, 1]
    except AttributeError:
        try:
            y_prob = clf.decision_function(x_test)
        except AttributeError:
            y_prob = y_pred.astype(float)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')

    return {
        'Dataset'       : dataset_name,
        'Target'        : target_name,
        'Classifier'    : clf_name or type(clf).__name__,
        'N_train'       : n_train,
        'N_test'        : n_test,
        'Accuracy'      : round(acc,   4),
        'F1 (weighted)' : round(f1,    4),
        'AUROC'         : round(auroc, 4) if not np.isnan(auroc) else 'N/A',
    }


def _average_row(rows, dataset_name, n_train, n_test, clf_label):
    """Build one averaged row from a list of per-target metric dicts."""
    valid_auroc = [float(r['AUROC']) for r in rows if r['AUROC'] != 'N/A']
    return {
        'Dataset'       : dataset_name,
        'Target'        : 'Averaged across targets',
        'Classifier'    : clf_label,
        'N_train'       : n_train,
        'N_test'        : n_test,
        'Accuracy'      : round(np.mean([r['Accuracy']       for r in rows]), 4),
        'F1 (weighted)' : round(np.mean([r['F1 (weighted)']  for r in rows]), 4),
        'AUROC'         : round(np.mean(valid_auroc), 4) if valid_auroc else 'N/A',
    }


def _append_rows(rows, output_folder, filename):
    """Append a list of row-dicts to the CSV, creating it if it doesn't exist."""
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, filename)
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(out_path)
    df.to_csv(out_path, mode='a', header=write_header, index=False)
    for r in rows:
        print(f"[PerformanceMetrics] {r['Dataset']}/{r['Target']} — "
              f"Acc={r['Accuracy']}  F1={r['F1 (weighted)']}  AUROC={r['AUROC']}")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def save_performance_metrics(clf, x_test, y_test,
                             dataset_name, target_name='target',
                             n_train=0, output_folder='outputs',
                             filename='classifier_performance.csv'):
    """
    SingleOutput strategy — one binary clf, one target column.
    Always writes exactly ONE row (no averaging needed).

    main_single.py:
        save_performance_metrics(clf=clf, x_test=x_test, y_test=y_test_1d,
                                 dataset_name=dataset_name,
                                 target_name=target_col,
                                 n_train=len(x_train))
    """
    row = _compute_row(clf, x_test, y_test, dataset_name, target_name,
                       n_train=n_train, n_test=len(y_test))
    _append_rows([row], output_folder, filename)
    return row


def save_performance_metrics_multilabel(clf, x_test, y_test,
                                        dataset_name, target_cols,
                                        n_train=0,
                                        average_only=True,       # ← key parameter
                                        output_folder='outputs',
                                        filename='classifier_performance.csv'):
    """
    MultiLabelStrategy — clf is a MultiOutputClassifier.
    y_test is a DataFrame with one column per target.

    average_only=True  (default) → writes ONE averaged row per dataset.
                                    Best for the thesis performance table.
    average_only=False            → writes one row per target + one average row.
                                    Best for detailed appendix analysis.

    main_multiOutput.py:
        save_performance_metrics_multilabel(
            clf=clf, x_test=x_test, y_test=y_test,
            dataset_name='Cervical_Cancer',
            target_cols=['Hinselmann','Schiller','Citology','Biopsy'],
            n_train=len(x_train),
            average_only=True)          # ← one clean row in the table
    """
    # Compute one row per target (needed for averaging even if not saved)
    per_target_rows = []
    for i, col in enumerate(target_cols):
        estimator  = clf.estimators_[i]
        y_true_col = (y_test.iloc[:, i] if isinstance(y_test, pd.DataFrame)
                      else y_test[:, i])
        per_target_rows.append(_compute_row(
            estimator, x_test, y_true_col,
            dataset_name, target_name=col,
            n_train=n_train, n_test=len(y_true_col),
            clf_name=f'MultiOutput({type(estimator).__name__})'
        ))

    avg = _average_row(
        per_target_rows, dataset_name,
        n_train=n_train, n_test=len(y_test),
        clf_label=f'MultiOutputClassifier ({len(target_cols)} targets averaged)'
    )

    if average_only:
        # Write only the averaged row
        _append_rows([avg], output_folder, filename)
        return avg
    else:
        # Write one row per target + the average row
        _append_rows(per_target_rows + [avg], output_folder, filename)
        return per_target_rows + [avg]


def save_performance_metrics_hierarchical(results, x_test, y_test,
                                          dataset_name, group_mapping,
                                          n_train=0,
                                          average_only=True,      # ← key parameter
                                          output_folder='outputs',
                                          filename='classifier_performance.csv'):
    """
    HierarchicalStrategy — results dict returned by execute():
        { category: (gate_model, spec_model) }

    average_only=True  (default) → writes ONE averaged row (specialist subtypes only).
                                    Best for the thesis performance table.
    average_only=False            → writes one gatekeeper row + one row per subtype
                                    + one average row.
                                    Best for detailed appendix analysis.

    main_hierarchical.py:
        results = strategy.execute(x_train, x_test, y_train, y_test)
        save_performance_metrics_hierarchical(
            results=results, x_test=x_test, y_test=y_test,
            dataset_name='Myocardial_Infarction',
            group_mapping=config.Hierarchy_mapping,
            n_train=len(x_train),
            average_only=True)          # ← one clean row in the table
    """
    gate_rows = []
    spec_rows = []

    for category, (gate_model, spec_model) in results.items():
        subtypes = [c for c in group_mapping.get(category, [])
                    if c in y_test.columns]
        if not subtypes:
            continue

        # ── Gatekeeper ───────────────────────────────────────────────────────
        y_gate = y_test[subtypes].max(axis=1)
        gate_rows.append(_compute_row(
            gate_model, x_test, y_gate,
            dataset_name,
            target_name=f'{category} | Gatekeeper',
            n_train=n_train, n_test=len(y_gate),
            clf_name=type(gate_model).__name__
        ))

        # ── Specialist ───────────────────────────────────────────────────────
        if spec_model is None:
            print(f"[PerformanceMetrics] {category}: no specialist model, skipping.")
            continue

        if hasattr(gate_model, 'feature_names_in_') and isinstance(x_test, pd.DataFrame):
            gate_pred = gate_model.predict(x_test[gate_model.feature_names_in_])
        else:
            gate_pred = gate_model.predict(x_test)
        pos_mask  = gate_pred == 1
        x_pos     = x_test[pos_mask]

        if pos_mask.sum() == 0:
            print(f"[PerformanceMetrics] {category}: gatekeeper found no positives "
                  "— skipping specialist rows.")
            continue

        for i, col in enumerate(subtypes):
            est   = spec_model.estimators_[i]
            y_sub = y_test.loc[x_pos.index, col]
            spec_rows.append(_compute_row(
                est, x_pos, y_sub,
                dataset_name,
                target_name=f'{category} | {col} (specialist)',
                n_train=n_train, n_test=len(y_sub),
                clf_name=f'Specialist({type(est).__name__})'
            ))

    if not spec_rows:
        print("[PerformanceMetrics] No specialist rows computed — nothing saved.")
        return []

    # Average over specialist subtypes only (gatekeepers are auxiliary models)
    avg = _average_row(
        spec_rows, dataset_name,
        n_train=n_train, n_test=len(y_test),
        clf_label=(f'HierarchicalStrategy '
                   f'({len(spec_rows)} specialist subtypes averaged)')
    )

    if average_only:
        # Write only the averaged specialist row
        _append_rows([avg], output_folder, filename)
        return avg
    else:
        # Write gatekeeper rows + all specialist rows + average
        _append_rows(gate_rows + spec_rows + [avg], output_folder, filename)
        return gate_rows + spec_rows + [avg]