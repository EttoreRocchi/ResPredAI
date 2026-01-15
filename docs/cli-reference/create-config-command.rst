Create Config Command
=====================

The ``create-config`` command generates a template configuration file that you can customize for your data.

Usage
-----

.. code-block:: bash

    respredai create-config <output_path.ini>

Options
-------

Required
~~~~~~~~

- ``output_path`` - Path where the template configuration file will be created

  - Must end with ``.ini`` extension
  - Parent directory must exist or be creatable
  - File will be overwritten if it already exists

Description
-----------

This command creates a ready-to-use configuration template with all required sections pre-populated and inline comments explaining each parameter.

The generated template follows the INI format required by the ``run`` command.

Generated Template
------------------

The command creates a file with the following structure:

.. code-block:: ini

    [Data]
    data_path = ./data/my_data.csv
    targets = Target1,Target2
    continuous_features = Feature1,Feature2

    [Pipeline]
    # Available models: LR, MLP, XGB, RF, CatBoost, TabPFN, RBF_SVC, Linear_SVC
    models = LR,XGB,RF
    outer_folds = 5
    inner_folds = 3
    calibrate_threshold = false
    threshold_method = auto

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

Customization Steps
-------------------

After generating the template, customize it for your data:

1. Update Data Section
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Data]
    data_path = ./path/to/your/data.csv
    targets = AntibioticA,AntibioticB
    continuous_features = Feature1,Feature3,Feature4
    # group_column = PatientID  # Optional

- **data_path**: Path to your CSV file
- **targets**: Comma-separated list of target columns (binary classification)
- **continuous_features**: Features to scale with StandardScaler (all others are one-hot encoded)
- **group_column** (optional): Column name for grouping multiple samples from the same patient/subject to prevent data leakage

2. Select Models
~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Pipeline]
    models = LR,RF,XGB,CatBoost

Use ``respredai list-models`` to see all available models.

3. Configure Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    outer_folds = 5  # For model evaluation
    inner_folds = 3  # For hyperparameter tuning

- **outer_folds**: Number of folds for performance evaluation
- **inner_folds**: Number of folds for GridSearchCV hyperparameter tuning

4. Configure Threshold Calibration (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    calibrate_threshold = true
    threshold_method = auto

- **calibrate_threshold**: Enable decision threshold optimization using Youden's J statistic

  - ``true``: Calibrate threshold to maximize Youden's J (Sensitivity + Specificity - 1)
  - ``false``: Use default threshold of 0.5

- **threshold_method**: Method for threshold calibration (only used when ``calibrate_threshold = true``)

  - ``auto``: Automatically choose based on sample size (OOF if n < 1000, CV otherwise)
  - ``oof``: Out-of-fold predictions method - aggregates predictions from all CV folds into a single set, then finds one global threshold maximizing Youden's J across all concatenated samples
  - ``cv``: TunedThresholdClassifierCV method - calculates optimal threshold separately for each CV fold, then aggregates (averages) the fold-specific thresholds
  - **Key difference**: ``oof`` finds one threshold on all concatenated OOF predictions (global optimization), while ``cv`` finds per-fold thresholds then averages them (fold-wise optimization then aggregation)

5. Adjust Resources
~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Resources]
    n_jobs = -1  # Use all cores

- ``-1``: Use all available CPU cores
- ``1``: No parallelization (useful for debugging)
- ``N``: Use N cores

6. Configure Model Saving
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [ModelSaving]
    enable = true
    compression = 3

- **enable**: Set to ``true`` to save models every folds
- **compression**: 0-9 (0=no compression, 3=balanced, 9=maximum)

7. Set Output Location
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Output]
    out_folder = ./results/

The folder will be created if it doesn't exist.

See Also
--------

- :doc:`run-command` - Execute the nested CV pipeline
- :doc:`train-command` - Train models on entire dataset for cross-dataset validation
- :doc:`validate-config-command` - Validate configuration before running
