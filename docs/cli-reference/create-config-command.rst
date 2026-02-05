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
    # Available models: LR, MLP, XGB, RF, CatBoost, TabPFN, RBF_SVC, Linear_SVC, KNN
    models = LR,XGB,RF
    outer_folds = 5
    inner_folds = 3
    outer_cv_repeats = 1
    calibrate_threshold = false
    threshold_method = auto
    calibrate_probabilities = false
    probability_calibration_method = sigmoid
    probability_calibration_cv = 5

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
    threshold_objective = youden
    vme_cost = 1.0
    me_cost = 1.0

- **calibrate_threshold**: Enable decision threshold optimization

  - ``true``: Calibrate threshold using the specified objective
  - ``false``: Use default threshold of 0.5

- **threshold_method**: Method for threshold calibration (only used when ``calibrate_threshold = true``)

  - ``auto``: Automatically choose based on sample size (OOF if n < 1000, CV otherwise)
  - ``oof``: Out-of-fold predictions method - aggregates predictions from all CV folds into a single set, then finds one global threshold across all concatenated samples
  - ``cv``: TunedThresholdClassifierCV method - calculates optimal threshold separately for each CV fold, then aggregates (averages) the fold-specific thresholds
  - **Key difference**: ``oof`` finds one threshold on all concatenated OOF predictions (global optimization), while ``cv`` finds per-fold thresholds then averages them (fold-wise optimization then aggregation)

- **threshold_objective**: Objective function for threshold optimization

  - ``youden``: Maximize Youden's J statistic (Sensitivity + Specificity - 1) - balanced approach
  - ``f1``: Maximize F1 score - balances precision and recall
  - ``f2``: Maximize F2 score - prioritizes recall over precision (reduces VME at potential cost of increased ME)
  - ``cost_sensitive``: Minimize weighted error cost using ``vme_cost`` and ``me_cost``

- **vme_cost** / **me_cost**: Cost weights for cost-sensitive threshold optimization

  - VME (Very Major Error): Predicted susceptible when actually resistant
  - ME (Major Error): Predicted resistant when actually susceptible
  - Higher ``vme_cost`` relative to ``me_cost`` will shift threshold to reduce false susceptible predictions

5. Configure Repeated Cross-Validation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    outer_cv_repeats = 3

- **outer_cv_repeats**: Number of times to repeat outer cross-validation (default: 1)

  - ``1``: Standard (non-repeated) cross-validation
  - ``>1``: Repeated CV with different random shuffles for more robust estimates
  - Total iterations = ``outer_folds`` × ``outer_cv_repeats``
  - Example: 5 folds × 3 repeats = 15 total train/test iterations

6. Configure Probability Calibration (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    calibrate_probabilities = true
    probability_calibration_method = sigmoid
    probability_calibration_cv = 5

- **calibrate_probabilities**: Enable post-hoc probability calibration

  - ``true``: Apply CalibratedClassifierCV to calibrate predicted probabilities
  - ``false``: Use uncalibrated probabilities (default)
  - Applied after Applied after hyper-parameters tuning and before threshold tuning

- **probability_calibration_method**: Calibration method

  - ``sigmoid``: Platt scaling - fits logistic regression (default, works well for most cases)
  - ``isotonic``: Isotonic regression - non-parametric (requires more data)

- **probability_calibration_cv**: CV folds for calibration (default: 5)

  - Internal cross-validation used by CalibratedClassifierCV
  - Must be at least 2

**Note**: Calibration diagnostics (Brier Score, ECE, MCE, reliability curves) are always computed regardless of this setting.

8. Configure Uncertainty Quantification (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Uncertainty]
    margin = 0.1

- **margin**: Margin around the decision threshold for flagging uncertain predictions (0-0.5)

  - Predictions with probability within ``margin`` of the threshold are flagged as uncertain
  - Default: 0.1
  - Uncertainty scores and flags are included in evaluation output

- **Uncertainty score computation**:

  .. code-block:: text

      distance = |probability - threshold|
      max_distance = max(threshold, 1 - threshold)
      uncertainty = 1 - (distance / max_distance)
      is_uncertain = distance < margin

  - Score ranges from 0 (confident, at probability extremes) to 1 (uncertain, at threshold)
  - When threshold is calibrated, uncertainty is computed relative to the calibrated threshold

9. Configure Preprocessing (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Preprocessing]
    ohe_min_frequency = 0.05

- **ohe_min_frequency**: Minimum frequency for categorical values in OneHotEncoder

  - Categories appearing below this threshold are grouped into an "infrequent" category
  - Values in (0, 1): proportion of samples (e.g., 0.05 = at least 5% of samples)
  - Values >= 1: absolute count (e.g., 10 = at least 10 occurrences)
  - Omit or comment out to disable (keep all categories)
  - Useful for reducing noise from rare categorical values and preventing overfitting

10. Adjust Resources
~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Resources]
    n_jobs = -1  # Use all cores

- ``-1``: Use all available CPU cores
- ``1``: No parallelization (useful for debugging)
- ``N``: Use N cores

11. Configure Model Saving
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [ModelSaving]
    enable = true
    compression = 3

- **enable**: Set to ``true`` to save models every folds
- **compression**: 0-9 (0=no compression, 3=balanced, 9=maximum)

12. Set Output Location
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Output]
    out_folder = ./results/

The folder will be created if it doesn't exist.

See Also
--------

- :doc:`run-command` - Execute the nested CV pipeline
- :doc:`train-command` - Train models on entire dataset for cross-dataset validation
- :doc:`validate-config-command` - Validate configuration before running
