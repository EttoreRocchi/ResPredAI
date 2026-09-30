# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.9.4] - 2026-09-30

### Added

- Per-fold feature-importance CSV (`{model}_feature_importance_per_fold.csv`) with one row per outer CV fold and, for linear models, the intercept of each fold
- Documentation on reading linear coefficients: they refer to standardized and one-hot encoded features and are taken before probability calibration; the `feature-importance` page now shows how they relate to the calibrated probabilities and how to extract them from the saved models
- Configuration validation: model names are checked against the available models (a misspelt name used to be skipped at run time with a warning while the run reported success), `outer_folds` and `inner_folds` must be at least 2, `verbosity` must be 0, 1 or 2, and an empty `targets` is rejected
- Warning when a categorical feature has high cardinality (at least 20 distinct values and more than half as many as rows), since it is one-hot encoded into one column per level
- `training_metadata.json` records `calibration_bins` and the resolved `threshold_method` (`oof` or `cv` instead of `auto`); `respredai evaluate` uses the recorded bin count for ECE and MCE
- Temporal validation reports progress in the CLI
- Checkpoints store the per-fold reliability-curve data, so a resumed or re-loaded run still produces reliability curves

### Changed

- `extract_feature_importance_from_models()` also returns the per-fold intercepts
- Model names are case-insensitive and normalised to their canonical spelling (`lr` selects `LR`), duplicates are dropped; applies to `[Pipeline] models`, `run --models`, `train --models` and `feature-importance --model`
- `continuous_features` may be empty or omitted when every feature is categorical
- With imputation disabled, the missing-value check covers the feature columns only and names the offending columns; subgroup columns may contain missing values, which are reported as `Unknown`
- Feature-name sanitisation also replaces `[` and `]`, which XGBoost rejects in feature names, in addition to `<` and `>`
- With `threshold_method = cv` the saved fold model is the (calibrated) classifier itself with the tuned threshold stored beside it, as for `oof`; `TunedThresholdClassifierCV` is used only to search the threshold. With a `group_column` that search runs on group-aware inner splits, while the calibrator refit inside the search uses plain stratified folds

### Removed

- `tqdm` dependency (unused); `torch` is no longer required unless the `tabpfn` extra is installed

### Fixed

- `import respredai` failed without `torch`, which has not been a dependency of the base package since `tabpfn` became an optional extra; `torch` is now imported only when the `TabPFN` model is built
- Resuming a run from a partially completed checkpoint raised `KeyError` and wrote a corrupted checkpoint (duplicated metric rows, missing fold models, `completed_folds` equal to the total); the per-fold bookkeeping is now recorded in one step after every fallible computation
- `threshold_method = cv` (also selected by `auto` from 1000 training rows) with a `group_column` failed every fold, because sklearn rejects `groups=` on `TunedThresholdClassifierCV` without metadata routing; the group-aware splits are now passed as a precomputed list
- `respredai train` with a `group_column` and `calibrate_probabilities = true` crashed with "indices are out-of-bounds" in the conformal step
- Temporal validation swallowed per-target failures and the CLI reported success; a failed model/target now raises `RuntimeError` like the CV pipeline
- The HTML report assumed `CI95_*` columns, so confidence intervals disappeared when `confidence_level` was not 0.95 while the headers still said 95%
- Progress bars used `outer_folds` instead of folds times repeats with `outer_cv_repeats > 1`
- Loading several configurations in one process added a log file handler each time, so later runs also logged to the earlier files
- `feature-importance` dropped the sign of `LR` and `Linear_SVC` coefficients, so protective (negative) coefficients were reported as positive
- With `calibrate_probabilities = true`, feature importance came from a single calibration sub-model instead of the whole fold model; it is now averaged over all sub-models

## [1.9.3] - 2026-06-16

### Added

- Input validation: targets must be binary 0/1 with both classes present, declared `continuous_features` must exist in the data, and group/temporal columns must not contain missing values; `respredai evaluate` uses the same reader and validator as training
- `[Pipeline] calibration_bins` option (default 10) to set the number of bins for the ECE/MCE metrics (point estimates and bootstrap confidence intervals)
- Importance provenance: feature-importance CSVs include `Importance_Type` (gain, impurity, coefficient, mean_abs_shap - not comparable across model families) and `N_folds_present`; a `Neutral (~0)` direction label is reported for features with no clear direction
- Reproducibility manifest records `n_jobs`, the git commit and the full installed-package versions
- CI job running the slow (integration) test suite

### Changed

- Cross-fold feature-importance aggregation skips folds where a feature is absent instead of filling zeros, so rare one-hot categories are not diluted toward zero

### Fixed

- Empty results with the default config: `respredai run` with `calibrate_threshold = false` (the `create-config` default) raised an `UnboundLocalError` per fold that was swallowed, producing empty/NaN metrics while reporting success
- Silent fold/target failures: per-fold errors are warned and counted, and a model/target whose folds all fail raises a clear `RuntimeError` instead of writing NaN metrics as success
- `feature-importance --direction` silently returned nothing for RandomForest because the 3-D SHAP TreeExplainer output was not handled; SHAP output of any shape is now reduced to the positive class
- Calibration bin collapse: quantile bin edges are deduplicated so tied probabilities no longer collapse ECE/MCE/reliability bins
- `calibrate_probabilities = true` with a `group_column` made every fold fail ("indices are out-of-bounds") in the conformal step

## [1.9.2] - 2026-05-22

### Added

- `TABPFN_TOKEN` environment variable (PriorLabs API token) required for TabPFN v3; the pipeline validates it before starting when `TabPFN` is among the requested models
- `ensure_tabpfn_available()` helper in `respredai.core.model_builder` that checks both the installed package and the token

### Changed

- TabPFN is an optional dependency: install with `pip install respredai[tabpfn]` to enable the TabPFN model
- TabPFN bumped to `>=8.0.0,<9.0.0` and switched to the v3 model by default

## [1.9.1] - 2026-04-13

### Fixed

- Group leakage in `perform_training` conformal CV: `StratifiedGroupKFold` is used when groups are available, matching `perform_pipeline` and `perform_temporal_validation`

## [1.9.0] - 2026-04-10

### Added

- FOR (False Omission Rate) metric in all metrics CSVs and HTML reports
- CV+ Mondrian conformal prediction: per-class prediction sets `{S}`, `{R}` or `{S, R}` with distribution-free coverage guarantee `1 - 2α`, computed per fold inside nested CV; `q_hat` per class saved in model bundles; dedicated HTML report section with per-model coverage tables
- `[Uncertainty] alpha` config option (miscoverage rate)
- Model category constants `SHAP_FALLBACK_MODELS` and `NO_CLASS_WEIGHT_MODELS` in `constants.py`
- Documentation on configuring the `[Metadata]` section for subgroup analysis

### Changed

- Feature importance functions (`has_feature_importance`, `get_feature_importance`, `compute_shap_importance`) require a `model_name` parameter and dispatch on category constants instead of duck-typing

### Removed

- Legacy config fallback for `group_column` in `[Data]` and `temporal_split_column` in `[Validation]` (deprecated since 1.8.0); use the `[Metadata]` section
- `calculate_uncertainty()` function, replaced by `compute_conformal_qhat()` and `conformal_prediction_sets()`
- `[Uncertainty] margin` config key, replaced by `[Uncertainty] alpha`
- `uncertainty_margin` field in model bundles, replaced by `conformal_q_hat` and `conformal_alpha`

## [1.8.0] - 2026-03-25

### Added

- Unified `[Metadata]` config section: `group_column` (moved from `[Data]`), `temporal_column` (moved from `[Validation]` as `temporal_split_column`) and the new comma-separated `subgroup_columns`
- Subgroup performance evaluation: full metric set (AUROC, F1, MCC, Precision, Recall, ECE, MCE, Brier Score) per subgroup value, with sample size and class prevalence, for both CV and temporal validation; results saved as CSV in `subgroup_analysis/` and shown in the HTML report; warns when a subgroup has fewer than 10 samples
- Signed feature importance direction (`compute_feature_direction` config flag, `respredai feature-importance --direction`): linear models use the sign of the coefficients, tree-based models use `shap.TreeExplainer`, other models fall back to `shap.KernelExplainer`; adds a `Direction` column (`Risk (+)` / `Protective (-)`) to the CSV and colors the plot by direction

### Deprecated

- `group_column` in `[Data]` and `temporal_split_column` in `[Validation]`; use the `[Metadata]` section

## [1.7.1] - 2026-03-23

### Added

- BCa (bias-corrected and accelerated) bootstrap confidence intervals via `scipy.stats.bootstrap`, replacing percentile bootstrap, for better coverage on small samples and bounded or skewed metrics
- Nadeau-Bengio corrected standard error: new `SE` column in the metrics CSV using `(1/k + n_test/n_train) * s**2`, accounting for training-set overlap in k-fold CV; the summary report `±` notation uses SE instead of raw Std
- `confidence_level` and `n_bootstrap` options in the `[Pipeline]` config section
- `respredai/core/constants.py` with centralized validation lists, directory names and defaults

### Changed

- `ConfigHandler` split into seven domain-specific dataclasses
- Config validation lists reference the centralized constants
- README quick-start config example shows the `[Validation]` section
- HTML report confidence-interval row reflects the configured `confidence_level` and `n_bootstrap`

### Fixed

- Name sanitization unified through `sanitize_name()` / `sanitize_metric_name()`
- `assert` in the temporal split replaced with a proper `ValueError`
- Empty CV folds are validated, with a warning when train or test sets are empty after splitting
- Explicit warning when all bootstrap samples fail (previously returned NaN silently)

## [1.7.0] - 2026-03-18

### Added

- Temporal (prospective-style) validation: `validation_strategy` config option (`cv`, `temporal` or `both`), `temporal_split_column`, `temporal_split_date` and `temporal_split_ratio` options, group-aware temporal splitting, `--validation-strategy` CLI override for `run`, and a temporal validation section in the HTML report
- Per-fold one-hot encoding fitted on the training part of each CV fold instead of the full dataset, preventing category leakage
- NaN-safe scaling for KNN imputation: features are pre-scaled with NaN-tolerant statistics before distance-based imputation
- `scale_pos_weight` in the XGBoost hyperparameter grid

### Changed

- `validate-config` summary table shows the validation strategy and temporal split parameters
- `create-config` template includes a commented `[Validation]` section
- `uncertainty_margin` stored in training metadata and model bundles

### Fixed

- SVC models set `probability=False` when external calibration is enabled, avoiding double Platt scaling
- `TunedThresholdClassifierCV` (CV threshold method) uses `StratifiedGroupKFold` and passes group labels when grouped CV is configured
- Repeated CV deduplication uses the threshold-aware decision boundary instead of a fixed 0.5 cutoff
- All `nanstd` calls use `ddof=1` for the unbiased sample standard deviation
- Reliability curve binning replaced with a self-contained implementation to keep `bin_counts` aligned
- Logger initialization deferred until after CLI overrides so the log file uses the correct output folder
- Feature importance name resolution improved with a multi-source fallback
- `evaluate` reuses the saved one-hot encoder from training instead of ad-hoc `pd.get_dummies`

## [1.6.2] - 2026-03-05

### Added

- Makefile for development workflows

### Fixed

- Bootstrap confidence intervals deduplicate samples when using repeated outer CV
- Threshold optimization (CV method) uses the calibrated estimator when probability calibration is enabled
- Metrics aggregation respects the repeat structure
- Reliability curve fold labels indicate the repeat number when using repeated CV
- Reliability curves use quantile binning for smoother calibration plots on imbalanced data

## [1.6.1] - 2026-02-06

### Changed

- `create-config` template includes `threshold_objective`, `vme_cost` and `me_cost`
- `validate-config` summary table shows probability calibration and threshold objective details

### Fixed

- `train` applies probability calibration (`CalibratedClassifierCV`) when `calibrate_probabilities = true`
- `train` supports the CV threshold method (`TunedThresholdClassifierCV`) in addition to OOF
- Reproducibility manifest includes the probability calibration parameters

## [1.6.0] - 2026-02-05

### Added

- Optional post-hoc probability calibration (`sigmoid` Platt scaling or `isotonic`) on the best estimator of each outer CV fold, applied after hyperparameter tuning and before threshold tuning
- Calibration diagnostics: Brier Score, ECE (Expected Calibration Error), MCE (Maximum Calibration Error) and reliability curves per outer CV fold and aggregate
- Repeated stratified cross-validation via the `outer_cv_repeats` config option (default 1)

### Changed

- `metric_dict()` includes Brier Score, ECE and MCE by default
- HTML report includes a calibration diagnostics section
- Output folder includes a `calibration/` directory with reliability curve images

## [1.5.1] - 2026-01-29

### Added

- OneHotEncoder `min_frequency` parameter to reduce noise from rare categorical values

### Changed

- `requirements.txt` pins explicit version constraints for all dependencies (`scikit-learn>=1.5.0` is required for `TunedThresholdClassifierCV`)

## [1.5.0] - 2026-01-20

### Added

- VME (Very Major Error: predicted susceptible when resistant) and ME (Major Error: predicted resistant when susceptible) in reports
- Flexible threshold objectives: `threshold_objective` config option (`youden`, `f1`, `f2`, `cost_sensitive`) with configurable `vme_cost` and `me_cost` weights
- Per-prediction uncertainty quantification: `uncertainty_margin` config option (default 0.1) flags predictions near the decision threshold, with 0-1 uncertainty scores for each prediction
- Reproducibility manifest (`reproducibility.json`) generated by `run` and `train` with environment info, data fingerprint and full configuration

### Changed

- HTML report framework summary shows the threshold objective and cost weights
- Evaluation output includes `uncertainty` and `is_uncertain` columns

## [1.4.1] - 2026-01-15

### Changed

- Documentation migrated from MkDocs to Sphinx
- Documentation and development dependencies loaded from `docs-requirements.txt` and `dev-requirements.txt`

## [1.4.0] - 2026-01-14

### Added

- K-Nearest Neighbors (KNN) classifier
- Missing data imputation: `SimpleImputer` (`mean`, `median`, `most_frequent`), `KNNImputer` and `IterativeImputer` with `BayesianRidge` or `RandomForest` estimator
- HTML report with run metadata, framework summary, results tables with 95% confidence intervals and confusion matrices
- Ruff linter in the CI workflow

### Changed

- Bootstrap confidence intervals use sample-level predictions instead of fold-level metrics
- CI workflow runs lint checks before tests and includes Python 3.13

## [1.3.1] - 2026-01-08

### Added

- `docs/` structure built with MkDocs

### Changed

- Package reorganized into sub-packages: `respredai/core/` (pipeline, metrics, models), `respredai/io/` (configuration and data handling) and `respredai/visualization/` (plots)

## [1.3.0] - 2025-12-12

### Added

- `train` command for training on the entire dataset (cross-dataset validation): GridSearchCV tuning with inner CV only, one model file per model-target combination, `training_metadata.json` for evaluation compatibility
- `evaluate` command to apply trained models to new data: validates columns against the training metadata, outputs per-sample predictions with probabilities and computes metrics against ground truth
- Automatic summary report after `run`: `summary.csv` per target and `summary_all.csv` globally with Mean±Std for all metrics across models
- SHAP-based feature importance as fallback for models without native importance (MLP, RBF_SVC, TabPFN via KernelExplainer): mean absolute SHAP values across CV test folds, `_shap` suffix on output files, `--seed` flag for reproducibility
- Documentation pages for the `train` and `evaluate` commands

### Changed

- `feature-importance` documentation describes the SHAP fallback

## [1.2.0] - 2025-12-10

### Added

- `validate-config` command to validate configuration files without running the pipeline, with an optional `--check-data` flag to verify the data file and its columns
- CLI override options for `run`: `--models`, `--targets`, `--output`, `--seed`
- `CONTRIBUTING.md` with development setup and contribution workflow
- Documentation page for the `validate-config` command

### Changed

- Bootstrap confidence intervals (10,000 resamples) replace the t-distribution CI in the metrics output
- User-friendly error messages for missing config files or data paths
- `run` documentation describes the CLI overrides

## [1.1.0] - 2025-12-04

### Added

- Threshold optimization with Youden's J statistic using either the OOF method (global optimization on concatenated out-of-fold predictions) or the CV method (per-fold optimization with threshold averaging), selected automatically by dataset size (OOF below 1000 samples, CV otherwise)
- Grouped cross-validation (`StratifiedGroupKFold`) to prevent data leakage in clinical datasets
- Command documentation for `run`, `create-config` and `feature-importance`
- This changelog

### Changed

- Expanded hyperparameter grids for XGBoost, Random Forest, CatBoost and MLP
- Enhanced CLI information display
- README with logo, quick start guide and output structure

### Fixed

- XGBoost feature naming with special characters
- Color scheme in feature importance plots

## [1.0.0] - 2025-12-01

### Added

- Nested cross-validation framework (outer loop for evaluation, inner loop for hyperparameter tuning)
- Eight machine learning models: LR, Linear SVC, RBF SVC, MLP, RF, XGBoost, CatBoost, TabPFN
- Metrics: Precision, Recall, F1, MCC, Balanced Accuracy, AUROC
- Data preprocessing: StandardScaler, one-hot encoding, multi-target support
- INI-based configuration system
- Structured output: CSV metrics, confusion matrix plots, logs
- `feature-importance` command with plot and CSV export

[Unreleased]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.9.4...HEAD
[1.9.4]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.9.3...v1.9.4
[1.9.3]: https://github.com/EttoreRocchi/ResPredAI/compare/v.1.9.2...v1.9.3
[1.9.2]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.9.1...v.1.9.2
[1.9.1]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.7.1...v1.8.0
[1.7.1]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.6.2...v1.7.0
[1.6.2]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/EttoreRocchi/ResPredAI/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/EttoreRocchi/ResPredAI/releases/tag/v1.0.0
