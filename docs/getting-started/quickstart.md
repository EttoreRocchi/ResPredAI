# Quick Start

This guide will help you get started with ResPredAI in just a few minutes.

## Step 1: Create a Configuration File

Generate a template configuration file:

```bash
respredai create-config my_config.ini
```

## Step 2: Edit the Configuration

Open `my_config.ini` and customize it for your data:

```ini
[Data]
data_path = ./data/my_data.csv
targets = Target1,Target2
continuous_features = Feature1,Feature2,Feature3
# group_column = PatientID  # Optional: prevents data leakage

[Pipeline]
models = LR,RF,XGB,CatBoost
outer_folds = 5
inner_folds = 3
calibrate_threshold = false
threshold_method = auto

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

### Configuration Sections

| Section | Description |
|---------|-------------|
| `[Data]` | Input data path, target columns, feature types |
| `[Pipeline]` | Models to train, CV folds, threshold calibration |
| `[Reproducibility]` | Random seed for reproducibility |
| `[Log]` | Logging verbosity and file name |
| `[Resources]` | Parallel processing settings |
| `[ModelSaving]` | Model persistence options |
| `[Imputation]` | Missing data imputation settings |
| `[Output]` | Output directory path |

## Step 3: Validate Configuration (Optional)

Check that your configuration is valid before running:

```bash
respredai validate-config my_config.ini --check-data
```

## Step 4: Run the Pipeline

Execute the nested cross-validation pipeline:

```bash
respredai run --config my_config.ini
```

## Step 5: Explore Results

After the pipeline completes, check your output folder:

- `report.html` - Comprehensive HTML report with all results
- `metrics/` - Performance metrics with 95% confidence intervals
- `confusion_matrices/` - Visualization of model performance
- `models/` - Saved models for feature importance extraction

## Step 6: Extract Feature Importance (Optional)

Analyze which features are most important:

```bash
respredai feature-importance --output ./output --model LR --target Target1
```

## Available Models

| Code | Model |
|------|-------|
| `LR` | Logistic Regression |
| `RF` | Random Forest |
| `XGB` | XGBoost |
| `CatBoost` | CatBoost |
| `MLP` | Neural Network |
| `TabPFN` | TabPFN |
| `RBF_SVC` | RBF SVM |
| `Linear_SVC` | Linear SVM |
| `KNN` | K-Nearest Neighbors |

## Next Steps

- Read the [CLI Reference](../cli-reference/index.md) for detailed command options
- Check the [run command documentation](../cli-reference/run-command.md) for advanced configuration
