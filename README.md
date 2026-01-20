# ResPredAI

## Antimicrobial resistance predictions via artificial intelligence models

[![PyPI](https://img.shields.io/pypi/v/respredai)](https://pypi.org/project/respredai/)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
![CI](https://github.com/EttoreRocchi/ResPredAI/actions/workflows/ci.yaml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the pipeline described in:

> Bonazzetti, C., Rocchi, E., Toschi, A. _et al._ Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections. _npj Digit. Med._ **8**, 319 (2025). https://doi.org/10.1038/s41746-025-01696-x

<p align="center">
  <img src="docs/assets/logo_ResPredAI.png" alt="ResPredAI Logo" width="350"/>
</p>

<p align="center">
  <strong><a href="https://ettorerocchi.github.io/respredai-website/">Website</a></strong> |
  <strong><a href="https://ettorerocchi.github.io/ResPredAI/">Documentation</a></strong> |
  <strong><a href="#installation">Installation</a></strong> |
  <strong><a href="#quick-start">Quick Start</a></strong> |
  <strong><a href="#cli-commands">CLI Commands</a></strong> |
  <strong><a href="#citation">Citation</a></strong>
</p>

<p align="center">
  <em>A reproducible machine learning framework designed to accelerate clinical decision-making by predicting antimicrobial resistance patterns from patient data.</em>
</p>

## Links

- **[Project Website](https://ettorerocchi.github.io/respredai-website/)** - Overview of the project, original paper, and related work
- **[Documentation](https://ettorerocchi.github.io/ResPredAI/)** - Installation guides, CLI reference, tutorials, and API documentation

## Installation

Install from PyPI:

```bash
pip install respredai
```

Or install from source:

```bash
git clone https://github.com/EttoreRocchi/ResPredAI.git
cd ResPredAI
# For development (includes pytest)
pip install -e ".[dev]"
```

## Testing the Installation

Verify the installation:

```bash
respredai --version
```

## Quick Start

### 1. Create a configuration file

```bash
respredai create-config my_config.ini
```

### 2. Edit the configuration file

Edit `my_config.ini` with your data paths and parameters:

```ini
[Data]
data_path = ./data/my_data.csv
targets = Target1,Target2
continuous_features = Feature1,Feature2,Feature3
# group_column = PatientID  # Optional: prevents data leakage when you have multiple samples per patient

[Pipeline]
models = LR,RF,XGB,CatBoost
outer_folds = 5
inner_folds = 3
calibrate_threshold = false
threshold_method = auto
# Threshold optimization objective: youden (default), f1, f2, cost_sensitive
threshold_objective = youden
# Cost weights for cost_sensitive objective (VME = false susceptible, ME = false resistant)
vme_cost = 1.0
me_cost = 1.0

[Uncertainty]
# Margin around threshold for flagging uncertain predictions (0-0.5)
margin = 0.1

[Reproducibility]
seed = 42

[Log]
verbosity = 1
log_basename = respredai.log

[Resources]
n_jobs = -1

[ModelSaving]
enable = true
compression = 3

[Imputation]
method = none
strategy = mean
n_neighbors = 5
estimator = bayesian_ridge

[Output]
out_folder = ./output/
```

### 3. Run the pipeline

```bash
respredai run --config my_config.ini
```

## CLI Commands

### Run the pipeline

```bash
respredai run --config path/to/config.ini [--quiet]
```

Train models using nested cross-validation with the specified configuration.

📖 **[Detailed Documentation](https://ettorerocchi.github.io/ResPredAI/cli-reference/run-command/)** - Complete guide with all configuration options and workflow details.

### Train models for cross-dataset validation

```bash
respredai train --config path/to/config.ini [--models LR,RF] [--output ./trained/]
```

Train models on the entire dataset using GridSearchCV for hyperparameter tuning. Saves one model file per model-target combination for later use with `evaluate`.

📖 **[Detailed Documentation](https://ettorerocchi.github.io/ResPredAI/cli-reference/train-command/)** - Complete guide with output structure and workflow.

### Evaluate on new data

```bash
respredai evaluate --models-dir ./output/trained_models --data new_data.csv --output ./eval/
```

Apply trained models to new data with ground truth. Outputs predictions and metrics.

📖 **[Detailed Documentation](https://ettorerocchi.github.io/ResPredAI/cli-reference/evaluate-command/)** - Complete guide with data requirements and output format.

### Extract feature importance

```bash
respredai feature-importance --output <output_folder> --model <model_name> --target <target_name> [--top-n 20]
```

Extract and visualize feature importance/coefficients from trained models across all outer cross-validation iterations. Uses SHAP as fallback for models without native feature importance.

📖 **[Detailed Documentation](https://ettorerocchi.github.io/ResPredAI/cli-reference/feature-importance-command/)** - Complete guide with interpretation, examples, and statistical considerations.

### List available models

```bash
respredai list-models
```

Display all available machine learning models with descriptions.

```
Available Models:
┌────────────┬──────────────────────────┐
│ Code       │ Name                     │
├────────────┼──────────────────────────┤
│ LR         │ Logistic Regression      │
│ MLP        │ Neural Network           │
│ XGB        │ XGBoost                  │
│ RF         │ Random Forest            │
│ CatBoost   │ CatBoost                 │
│ TabPFN     │ TabPFN                   │
│ RBF_SVC    │ RBF SVM                  │
│ Linear_SVC │ Linear SVM               │
│ KNN        │ K-Nearest Neighbors      │
└────────────┴──────────────────────────┘
```

### Create a template configuration file

```bash
respredai create-config output_path.ini
```

Generate a template configuration file that you can edit for your data.

📖 **[Detailed Documentation](https://ettorerocchi.github.io/ResPredAI/cli-reference/create-config-command/)** - Complete guide to configuration file structure and customization.

### Validate a configuration file

```bash
respredai validate-config <path_to_config.ini> [--check-data]
```

Validate a configuration file without running the pipeline. It can also check that the dataset load without errors.

📖 **[Detailed Documentation](https://ettorerocchi.github.io/ResPredAI/cli-reference/validate-config-command/)** - Complete guide to configuration file validation.

### Show information

```bash
respredai info
```

Display information about ResPredAI including scientific paper citation and version details.

Or just:

```bash
respredai --version
```

to show the installed version of ResPredAI.

## Output

The pipeline generates:
- **Confusion matrices**: PNG files with heatmaps showing model performance for each target
- **Detailed metrics tables**: CSV files with comprehensive metrics (precision, recall, F1, MCC, balanced accuracy, AUROC, VME, ME) with mean, std, and 95% CI
- **Trained models**: Saved models for resumption and feature importance extraction (if model saving enabled)
- **Feature importance**: Plots and CSV files showing feature importance/coefficients (generated separately)
- **Log files**: Detailed execution logs (if verbosity > 0)

### Output Structure
```
output_folder/
├── models/                                         # Trained models (if model saving enabled)
│   └── {Model}_{Target}_models.joblib
├── trained_models/                                 # Models for cross-dataset validation (from train command)
│   ├── {Model}_{Target}.joblib
│   └── training_metadata.json
├── metrics/                                        # Detailed performance metrics
│   ├── {target_name}/
│   │   ├── {model_name}_metrics_detailed.csv
│   │   └── summary.csv                            # Summary across all models
│   └── summary_all.csv                            # Global summary
├── feature_importance/                             # Feature importance (if extracted)
│   └── {target_name}/
│       ├── {model_name}_feature_importance.csv    # Importance values
│       └── {model_name}_feature_importance.png    # Barplot visualization
├── confusion_matrices/                             # Confusion matrix heatmaps
│   └── Confusion_matrix_{model_name}_{target_name}.png
├── report.html                                     # Comprehensive HTML report
└── respredai.log                                   # Execution log (if verbosity > 0)
```

## Changelog

See the full history of changes in the [CHANGELOG.md](CHANGELOG.md) file.

## Citation

If you use `ResPredAI` in your research, please cite:

```bibtex
@article{Bonazzetti2025,
  author = {Bonazzetti, Cecilia and Rocchi, Ettore and Toschi, Alice and Derus, Nicolas Riccardo and Sala, Claudia and Pascale, Renato and Rinaldi, Matteo and Campoli, Caterina and Pasquini, Zeno Adrien Igor and Tazza, Beatrice and Amicucci, Armando and Gatti, Milo and Ambretti, Simone and Viale, Pierluigi and Castellani, Gastone and Giannella, Maddalena},
  title = {Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections},
  journal = {npj Digital Medicine},
  volume = {8},
  pages = {319},
  year = {2025},
  doi = {10.1038/s41746-025-01696-x},
  url = {https://doi.org/10.1038/s41746-025-01696-x}
}
```

## Funding

This research was supported by EU funding within the NextGenerationEU-MUR PNRR Extended Partnership initiative on Emerging Infectious Diseases (Project no. PE00000007, INF-ACT).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up a development environment, running tests, and submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.