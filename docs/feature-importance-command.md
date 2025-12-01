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

## Supported Models

Not all models provide feature importance. The following models are supported:

### Linear Models (Coefficients)
- **LR** (Logistic Regression) - Uses coefficient values
- **Linear_SVC** (Linear SVM) - Uses coefficient values

### Tree-Based Models (Feature Importances)
- **RF** (Random Forest) - Uses Gini importance
- **XGB** (XGBoost) - Uses gain-based importance
- **CatBoost** - Uses feature importance scores

For tree-based models importance values are always positive.

## Output Files

The command generates files in the following structure:

```
output_folder/
└── feature_importance/
    └── {target}/
        ├── {model}_feature_importance.csv         # Detailed importance table
        └── {model}_feature_importance.png         # Barplot visualization
```

### CSV File Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `Feature` | Feature name |
| `Mean_Importance` | Mean importance across folds (signed for linear models) |
| `Std_Importance` | Standard deviation across folds |
| `Abs_Mean_Importance` | Absolute mean importance (used for ranking) |
| `Mean±Std` | Formatted string with mean ± std |

Features are **sorted by absolute importance**, so the most important features appear first.

Across all folds:
- Calculate **mean importance** for each feature
- Calculate **standard deviation** (uncertainty measure)
- Rank features by **absolute mean importance**

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

- [Run Command Documentation](run-command.md) - Train models and save trained model files
- [Configuration Documentatin](create-config-command.md) - How to create the configuration file
