# Create Config Command - Detailed Documentation

The `create-config` command generates a template configuration file that you can customize for your data.

## Usage

```bash
respredai create-config <output_path.ini>
```

## Options

### Required

- `output_path` - Path where the template configuration file will be created
  - Must end with `.ini` extension
  - Parent directory must exist or be creatable
  - File will be overwritten if it already exists

## Description

This command creates a ready-to-use configuration template with all required sections pre-populated and inline comments explaining each parameter.

The generated template follows the INI format required by the `run` command.

## Generated Template

The command creates a file with the following structure:

```ini
[Data]
data_path = ./data/my_data.csv
targets = Target1,Target2
continuous_features = Feature1,Feature2

[Pipeline]
# Available models: LR, MLP, XGB, RF, CatBoost, TabPFN, RBF_SVC, Linear_SVC
models = LR,XGB,RF
outer_folds = 5
inner_folds = 3

[Reproducibility]
seed = 42

[Log]
# Verbosity levels: 0 = no log, 1 = basic logging, 2 = detailed logging
verbosity = 1
log_basename = respredai.log

[Resources]
# Number of parallel jobs (-1 uses all available cores)
n_jobs = -1

[ModelSaving]
# Enable model saving for resuming interrupted runs
enable = true
# Compression level for saved models (1-9, higher = more compression but slower)
compression = 3

[Output]
out_folder = ./output/
```

## Customization Steps

After generating the template, customize it for your data:

### 1. Update Data Section

```ini
[Data]
data_path = ./path/to/your/data.csv
targets = AntibioticA,AntibioticB
continuous_features = Feature1,Feature3,Feature4
```

- **data_path**: Path to your CSV file
- **targets**: Comma-separated list of target columns (binary classification)
- **continuous_features**: Features to scale with StandardScaler (all others are one-hot encoded)

### 2. Select Models

```ini
[Pipeline]
models = LR,RF,XGB,CatBoost
```

Use `respredai list-models` to see all available models.

### 3. Configure Cross-Validation

```ini
outer_folds = 5  # For model evaluation
inner_folds = 3  # For hyperparameter tuning
```

- **outer_folds**: Number of folds for performance evaluation
- **inner_folds**: Number of folds for GridSearchCV hyperparameter tuning

### 4. Adjust Resources

```ini
[Resources]
n_jobs = -1  # Use all cores
```

- `-1`: Use all available CPU cores
- `1`: No parallelization (useful for debugging)
- `N`: Use N cores

### 5. Configure Model Saving

```ini
[ModelSaving]
enable = true
compression = 3
```

- **enable**: Set to `true` to save models every folds
- **compression**: 0-9 (0=no compression, 3=balanced, 9=maximum)

### 6. Set Output Location

```ini
[Output]
out_folder = ./results/
```

The folder will be created if it doesn't exist.

## See Also

- [Run Command Documentation](run-command.md) - Execute the pipeline with your configuration
- [Example Configuration](../example/config_example.ini) - Complete working example
