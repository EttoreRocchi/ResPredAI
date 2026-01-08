# CLI Reference

ResPredAI provides a command-line interface for all operations. This section documents all available commands.

## Getting Help

```bash
# Show all available commands
respredai --help

# Show help for a specific command
respredai run --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| [`run`](run-command.md) | Run nested cross-validation pipeline |
| [`train`](train-command.md) | Train models on full dataset for cross-dataset validation |
| [`evaluate`](evaluate-command.md) | Evaluate trained models on new data |
| [`feature-importance`](feature-importance-command.md) | Extract and visualize feature importance |
| [`create-config`](create-config-command.md) | Generate template configuration file |
| [`validate-config`](validate-config-command.md) | Validate configuration file |
| `list-models` | Display available ML models |
| `info` | Show version and citation information |

## Main Workflow

### 1. Nested Cross-Validation (Standard)

```bash
# Create configuration
respredai create-config my_config.ini

# Validate configuration
respredai validate-config my_config.ini --check-data

# Run pipeline
respredai run --config my_config.ini

# Extract feature importance
respredai feature-importance --output ./output --model LR --target Target1
```

### 2. Cross-Dataset Validation

```bash
# Train on dataset A
respredai train --config config_A.ini --output ./trained/

# Evaluate on dataset B
respredai evaluate --models-dir ./trained/trained_models --data dataset_B.csv --output ./eval/
```

## Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version number |

## Quick Examples

```bash
# Show available models
respredai list-models

# Show version and citation
respredai info

# Run pipeline quietly
respredai run --config config.ini --quiet

# Train specific models only
respredai train --config config.ini --models LR,RF
```
