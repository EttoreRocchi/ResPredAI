# Feature Importance Command - Detailed Documentation

The `feature-importance` command extracts and visualizes feature importance or coefficients from trained models across all outer cross-validation iterations.

## Usage

```bash
respredai feature-importance --output <output_folder> --model <model_name> --target <target_name> [options]
```

## Options

### Required

- `--output, -o` - Path to the output folder containing trained models
  - Must be the same folder used in the `run` command
  - Must contain a `models/` subdirectory with saved model files
  - Example: `./output/` or `./out_run_example/`

- `--model, -m` - Model name to extract importance from
  - Must match one of the models trained in the pipeline
  - Examples: `LR`, `RF`, `XGB`, `CatBoost`, `Linear_SVC`
  - Case-sensitive

- `--target, -t` - Target name to extract importance for
  - Must match one of the targets from the training pipeline
  - Example: `Target1`, `Ciprofloxacin_R`
  - Case-sensitive

### Optional

- `--top-n, -n` - Number of top features to display (default: 20)
  - Features are ranked by absolute importance
  - Range: 1 to total number of features
  - Example: `--top-n 30` for top 30 features

- `--no-plot` - Skip generating the barplot
  - Only CSV file will be created
  - Useful for batch processing or server environments

- `--no-csv` - Skip generating the CSV file
  - Only plot will be created
  - Useful if you only need visualizations

- `--seed, -s` - Random seed for SHAP reproducibility
  - Ensures reproducible SHAP values across runs
  - Only affects models using SHAP fallback

## Supported Models

The command uses native importance when available, with SHAP as fallback:

### Native Importance (Primary)

#### Linear Models (Coefficients)
- **LR** (Logistic Regression) - Uses coefficient values
- **Linear_SVC** (Linear SVM) - Uses coefficient values

#### Tree-Based Models (Feature Importances)
- **RF** (Random Forest) - Uses Gini importance
- **XGB** (XGBoost) - Uses gain-based importance
- **CatBoost** - Uses feature importance scores

For tree-based models importance values are always positive.

### SHAP Fallback

For models without native importance/coefficients, SHAP (SHapley Additive exPlanations) values are computed as a fallback:

- **MLP** (Multi-Layer Perceptron) - Uses KernelExplainer
- **RBF_SVC** (RBF SVM) - Uses KernelExplainer
- **TabPFN** - Uses KernelExplainer

SHAP values are computed on the test fold of each outer CV iteration and aggregated across folds. The mean absolute SHAP value represents feature importance.

Note: SHAP computation with KernelExplainer can be slow for large datasets.

## Output Files

The command generates files in the following structure:

```
output_folder/
└── feature_importance/
    └── {target}/
        ├── {model}_feature_importance.csv         # Native importance (if available)
        ├── {model}_feature_importance.png
        ├── {model}_feature_importance_shap.csv    # SHAP importance (fallback)
        └── {model}_feature_importance_shap.png
```

Files have `_shap` suffix when SHAP is used instead of native importance.

### CSV File Format (Native)

For models with native importance:

| Column | Description |
|--------|-------------|
| `Feature` | Feature name |
| `Mean_Importance` | Mean importance across folds (signed for linear models) |
| `Std_Importance` | Standard deviation across folds |
| `Abs_Mean_Importance` | Absolute mean importance (used for ranking) |
| `Mean±Std` | Formatted string with mean ± std |

### CSV File Format (SHAP)

For models using SHAP fallback:

| Column | Description |
|--------|-------------|
| `Feature` | Feature name |
| `Mean_Abs_SHAP` | Mean absolute SHAP value across folds |
| `Std_Abs_SHAP` | Standard deviation across folds |
| `Mean±Std` | Formatted string with mean ± std |

Features are **sorted by importance** (absolute mean value).

Across all folds:
- Calculate **mean importance** for each feature
- Calculate **standard deviation** (uncertainty measure)
- Rank features by importance

### Plot Color Coding

The barplot uses different colors to indicate importance type:

| Method | Color | Meaning |
|--------|-------|---------|
| SHAP | Orange | Mean absolute SHAP value |
| Native (tree-based) | Blue | Feature importance (always positive) |
| Native (linear, positive) | Red | Positive coefficient |
| Native (linear, negative) | Green | Negative coefficient |

Error bars show standard deviation across CV folds.

## Examples

### Basic Usage

Extract top 20 features for Logistic Regression on Target1:

```bash
respredai feature-importance --output ./output --model LR --target Target1
```

### Custom Number of Features

Show top 5 features:

```bash
respredai feature-importance -o ./output -m RF -t Target2 --top-n 5
```

### Multiple Models

Extract importance for multiple models (run separately):

```bash
respredai feature-importance -o ./output -m LR -t Target1
respredai feature-importance -o ./output -m RF -t Target1
respredai feature-importance -o ./output -m XGB -t Target1
```

### CSV Only (No Plot)

Generate only the CSV file for automated analysis:

```bash
respredai feature-importance -o ./output -m LR -t Target1 --no-plot
```

### Plot Only (No CSV)

Generate only the visualization:

```bash
respredai feature-importance -o ./output -m RF -t Target1 --no-csv
```

## See Also

- [Run Command](run-command.md) - Train models with nested CV and save model files
- [Train Command](train-command.md) - Train models on entire dataset for cross-dataset validation
- [Create Config Command](create-config-command.md) - How to create the configuration file
