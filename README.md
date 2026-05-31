# Evaluating the Reliability of Logic Learning Machines as an Explainable AI Method in Healthcare

A reproducible pipeline for benchmarking the **Logic Learning Machine (LLM / Rulex)** — a logic-based explainable AI (XAI) method — against standard continuous explainers (**SHAP** and **LIME**) and a **perturbation-based ablation** reference, across nine medical datasets.

The central question is whether a logic-based explainer assigns feature importance in a way that agrees with established attribution methods. Agreement is quantified with **Spearman rank correlation**, **Kendall Tau-b**, and **Top-K Jaccard similarity** over global feature-importance rankings.

---

## Key Result

Across the nine datasets, the average alignment of **Rulex with SHAP** (Spearman $\rho = 0.56$) is comparable to the baseline agreement between **SHAP and LIME** (Spearman $\rho = 0.56$), indicating that the logic-based explainer is about as consistent with SHAP as the two continuous explainers are with each other. The **SHAP–Ablation** pair shows the strongest agreement overall (Spearman $\rho = 0.95$), serving as a sanity-check anchor for the perturbation reference.

---

## Repository Structure

### Orchestration entry points
The pipeline is split by target structure. Each entry point loads data, trains a classifier, runs all explainers, and aggregates per-instance attributions into a single tidy table.

- `main_single.py` — Single-target classification. Iterates over the datasets defined in `dataset_config.json`, applies dataset-specific optimizers, trains an `XGBoost` or `Random Forest` model, and runs SHAP, LIME, standard ablation, and cumulative ablation.
- `main_multiOutput.py` — Multi-label classification (Cervical Cancer; four targets averaged).
- `main_hierarchical.py` — Hierarchical multi-target classification (Myocardial Infarction; specialist subtypes grouped by mechanical, electrical, and chemical complications, as defined in `Config.py`).
- `main_analysis.py` — Post-hoc comparative analysis. Consumes the aggregated per-dataset explanation tables, computes the agreement metrics, writes `final_xai_summary.csv`, and generates all figures.

### Core modules
- `Load.py` — `LoadData`: dataset ingestion (local files, `.arff`, and OpenML by ID), leakage-safe imputation (fit on train only), correlated-feature dropping, discretization and export for Rulex.
- `Strategy.py` — Training strategies: `SingleOutput`, `MultiLabelStrategy`, and `HierarchicalStrategy`. Each extends `Explainability` and triggers the explainer runs after fitting.
- `Explainability.py` — `Explainability`: wrappers for `run_shap` (TreeExplainer), `run_lime` (LimeTabularExplainer), `run_ablation` (marginal background-sampling perturbation), and `run_cumulative_ablation`.
- `Analysis.py` — `XAIComparativeAnalysis`: computes Spearman, Kendall Tau-b, and Top-K Jaccard between every explainer pair, averaging across targets per dataset and then across datasets.
- `PostProcessor.py` — `PostProcessor`: merges the separate SHAP / LIME / ablation outputs into one `*_final_explanation_results.csv` per dataset and cleans intermediate files.
- `PerformanceMetrics.py` — Classifier performance reporting (Accuracy, weighted F1, AUROC) for single, multi-label, and hierarchical settings.
- `Visualization.py` — Plotting utilities, including cumulative-ablation curves and summary figures.
- `Config.py` — `MycordinalConfig`: the Myocardial Infarction target hierarchy (mechanical, electrical, and chemical complication groups).

### Dataset-specific optimizers
Custom feature-engineering and encoding logic per dataset:
- `HCVOptimizer.py` (Hepatitis C)
- `ObesityOptimizer.py` (Obesity Level)
- `DiabetesOptimizer.py` (Diabetes 130-US)
- `Diabetic_Retinopathy_Optimizer.py` (Diabetic Retinopathy Debrecen)

### Configuration & data
- `dataset_config.json` — Maps each dataset to its source (local filename or OpenML ID) and target column(s).

### Outputs
- `classifier_performance.csv` — Per-dataset classifier metrics.
- `final_xai_summary.csv` — The main result: pairwise explainer-agreement metrics for every dataset plus a total average row.

---

## Datasets

Nine medical datasets spanning binary, multi-class, multi-label, and hierarchical targets:

| Dataset | Target | Classifier | Accuracy | F1 (weighted) | AUROC |
|---|---|---|---|---|---|
| Breast Cancer | class | Random Forest | $0.96$ | $0.96$ | $0.99$ |
| Obesity Level | Obesity | XGBoost | $0.91$ | $0.91$ | $0.96$ |
| Glioma Grading | Grade | Random Forest | $0.83$ | $0.82$ | $0.91$ |
| Diabetes 130-US | Readmitted | XGBoost | $0.86$ | $0.84$ | $0.56$ |
| CDC Diabetes | Diabetes_binary | XGBoost | $0.83$ | $0.82$ | $0.77$ |
| Diabetic Retinopathy | Class | XGBoost | $0.66$ | $0.66$ | $0.72$ |
| Cervical Cancer | 4 targets (avg) | MultiOutputClassifier | $0.93$ | $0.92$ | $0.56$ |
| Myocardial Infarction | 11 subtypes (avg) | HierarchicalStrategy | $0.80$ | $0.81$ | $0.62$ |
| Hepatitis | Stage | XGBoost | $0.49$ | $0.49$ | $0.49$ |

Large datasets are downsampled to $2500$ rows; data is split $70/30$ (train/test) with stratification where applicable, and a fixed seed (`42`) is used throughout for reproducibility.

---

## Methodology

1. **Load & clean** — Imputation statistics are learned on the training split only to prevent leakage; highly correlated features (threshold $0.90$) are dropped.
2. **Train** — An `XGBoost` or `Random Forest` classifier is fitted, with `scale_pos_weight` set from the class-imbalance ratio for imbalanced targets.
3. **Explain** — For every test instance, four importance signals are produced: Rulex (logic-based), SHAP, LIME, and ablation.
4. **Aggregate** — Per-instance attributions are merged into one table per dataset.
5. **Compare** — Global feature rankings are derived (with deterministic tie-breaking) and compared pairwise:
   - **Spearman** $\rho$ on the rankings,
   - **Kendall Tau-b** on raw values (native tie penalties),
   - **Top-K Jaccard** ($K = 5$) on the most important features.

---

## Usage

### Requirements
Python 3.x with the following packages:

```
numpy
pandas
scipy
scikit-learn
xgboost
shap
lime
imbalanced-learn
matplotlib
```

```bash
pip install numpy pandas scipy scikit-learn xgboost shap lime imbalanced-learn matplotlib
```

### Running the pipeline

```bash
# 1. Generate explanations for single-target datasets
python main_single.py

# 2. Generate explanations for the multi-label dataset
python main_multiOutput.py

# 3. Generate explanations for the hierarchical dataset
python main_hierarchical.py

# 4. Run comparative analysis and produce figures + final_xai_summary.csv
python main_analysis.py
```

Rulex importances are produced externally and supplied as the `Rulex` column in each dataset's aggregated explanation table before the analysis step.

---

## Notes

- All randomness is seeded (`42`) for reproducibility.
- The `Rulex` importance values are obtained from the Rulex / Logic Learning Machine tool and are not generated by this codebase; the pipeline ingests them alongside the SHAP, LIME, and ablation outputs for comparison.
- This repository accompanies a graduate thesis on explainable AI and machine learning in healthcare.
