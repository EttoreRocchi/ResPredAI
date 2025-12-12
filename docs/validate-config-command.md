# Validate Config Command - Detailed Documentation

The `validate-config` command validates a configuration file without running the pipeline. This is useful for catching configuration errors before starting a potentially long training run.

## Usage

```bash
respredai validate-config <path_to_config.ini> [--check-data]
```

## Options

### Required

- `config` - Path to the configuration file (INI format)
  - Positional argument (no flag needed)

### Optional

- `--check-data, -d` - Also validate that the data file exists and can be loaded
  - Checks that `data_path` points to a valid CSV file
  - Verifies all specified targets and features exist in the data
  - Reports dataset dimensions and group information

## What Gets Validated

### Without `--check-data`

1. **File existence** - Config file exists and is readable
2. **File format** - Config file has `.ini` extension
3. **Syntax** - Valid INI format with all required sections
4. **Required parameters** - All mandatory parameters are present
5. **Parameter values** - Values are valid (e.g., valid model names, threshold methods)

### With `--check-data`

All of the above, plus:

6. **Data file existence** - CSV file specified in `data_path` exists
7. **Data loading** - CSV can be parsed without errors
8. **Column validation** - Target columns exist in the data
9. **Missing values** - Data doesn't contain unexpected missing values
10. **Group validation** - Group column exists (if specified)

## See Also

- [Run Command](run-command.md) - Execute the full pipeline with nested CV
- [Train Command](train-command.md) - Train models on entire dataset for cross-dataset validation
- [Create Config Command](create-config-command.md) - Generate template configs
