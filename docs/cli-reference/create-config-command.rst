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
    # confidence_level = 0.95
    # n_bootstrap = 1000

    [Reproducibility]
    seed = 42

    [Log]
    # Verbosity levels: 0 = no log, 1 = basic logging, 2 = detailed logging
    verbosity = 1
    log_basename = respredai.log

    [Resources]
    # Number of parallel jobs (-1 uses all available cores)
    n_jobs = -1

    # [Uncertainty]
    # Miscoverage rate for conformal prediction (default 0.1 = 90% coverage)
    # alpha = 0.1

    [Preprocessing]
    ohe_min_frequency = 0.05

    # [Imputation]
    # method = none  # none, simple, knn, or iterative
    # strategy = mean  # For simple: mean, median, most_frequent
    # n_neighbors = 5  # For knn
    # estimator = bayesian_ridge  # For iterative: bayesian_ridge or random_forest

    [ModelSaving]
    # Enable model saving for resuming interrupted runs
    enable = true
    # Compression level for saved models (1-9, higher = more compression but slower)
    compression = 3

    [Output]
    out_folder = ./output/

    # [Metadata]
    # temporal_column = collection_date
    # group_column = PatientID
    # subgroup_columns = Ward,Specimen

    # [Validation]
    # Validation strategy: cv (default), temporal (prospective-style), or both
    # validation_strategy = cv
    # temporal_split_date = 2023-01-01
    # temporal_split_ratio = 0.8

Customization Steps
-------------------

After generating the template, customize it for your data.

.. note::

   Optional parameters can be disabled by commenting out the line with ``#``.
   Empty values (e.g., ``group_column =``) are treated as absent.

1. Update Data Section
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Data]
    data_path = ./path/to/your/data.csv
    targets = AntibioticA,AntibioticB
    continuous_features = Feature1,Feature3,Feature4

    [Metadata]
    # group_column = PatientID  # Optional
    # subgroup_columns = Ward,Specimen  # Optional

- **data_path**: Path to your CSV file
- **targets**: Comma-separated list of target columns (binary classification)
- **continuous_features**: Features to scale with StandardScaler (all others are one-hot encoded)

The ``[Metadata]`` section holds columns that describe sample context but are not used as features:

- **group_column** (optional): Column name for grouping multiple samples from the same patient/subject to prevent data leakage
- **subgroup_columns** (optional): Comma-separated column names for defining subgroups for stratified performance evaluation
- **temporal_column** (optional): Column with dates for temporal (prospective-style) validation

.. _subgroup-config:

Configuring Subgroup Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subgroup analysis evaluates model performance separately for each distinct value
of one or more categorical columns, helping identify disparities across clinical
subgroups (e.g., ward, specimen type, species).

.. code-block:: ini

    [Metadata]
    group_column = PatientID
    subgroup_columns = Ward,Specimen

- Each column listed in ``subgroup_columns`` must exist in the input CSV.
- Subgroup columns are automatically removed from the feature matrix — they are
  used for stratified evaluation only, not as predictive features.
- Multiple columns can be specified (comma-separated); each is analyzed independently.

**What subgroup analysis produces:**

For each model–target–subgroup combination, a CSV is saved under
``<out_folder>/subgroup_analysis/<target>/<model>_<subgroup_column>_subgroup.csv``
containing:

- One row per unique subgroup value
- Columns: ``Subgroup``, ``N`` (sample count), ``Prevalence`` (class 1 rate),
  plus all standard metrics (Precision, Recall, F1, MCC, AUROC, VME, ME, FOR, etc.)
- Subgroups with fewer than 10 samples are flagged with a warning

.. code-block:: text

    Subgroup,N,Prevalence,Precision (0),Precision (1),...,AUROC,VME,ME,FOR
    ICU,142,0.35,0.81,0.62,...,0.78,0.22,0.10,0.15
    General,310,0.18,0.88,0.45,...,0.72,0.40,0.05,0.08
    ER,89,0.28,0.79,0.55,...,0.74,0.30,0.12,0.11

.. note::

   ``group_column`` and ``subgroup_columns`` serve different purposes:
   ``group_column`` controls cross-validation splitting (keeping all samples from
   the same patient in the same fold to prevent data leakage), while
   ``subgroup_columns`` only affects post-hoc metric stratification. They can
   overlap — for instance, group by ``PatientID`` while analyzing performance
   by ``Ward``.

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

4. Configure Threshold Optimization (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    calibrate_threshold = true
    threshold_method = auto
    threshold_objective = youden
    vme_cost = 1.0
    me_cost = 1.0

- **calibrate_threshold**: Enable decision threshold optimization

  - ``true``: Calibrate threshold using the specified objective
  - ``false``: Use default threshold of 0.5

- **threshold_method**: Method for threshold optimization (only used when ``calibrate_threshold = true``)

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
    confidence_level = 0.95
    n_bootstrap = 1000

- **calibrate_probabilities**: Enable post-hoc probability calibration

  - ``true``: Apply CalibratedClassifierCV to calibrate predicted probabilities
  - ``false``: Use uncalibrated probabilities (default)
  - Applied after hyper-parameters tuning and before threshold tuning

- **probability_calibration_method**: Calibration method

  - ``sigmoid``: Platt scaling - fits logistic regression (default, works well for most cases)
  - ``isotonic``: Isotonic regression - non-parametric (requires more data)

- **probability_calibration_cv**: CV folds for calibration (default: 5)

  - Internal cross-validation used by CalibratedClassifierCV
  - Must be at least 2

- **confidence_level**: Confidence level for bootstrap CIs (default: 0.95)

  - Must be between 0.5 and 1.0

- **n_bootstrap**: Number of bootstrap resamples for CIs (default: 1000)

  - Must be at least 100

**Note**: Calibration diagnostics (Brier Score, ECE, MCE, reliability curves) are always computed regardless of this setting.

7. Configure Imputation (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Imputation]
    method = none
    strategy = mean
    n_neighbors = 5
    estimator = bayesian_ridge

- **method**: Imputation method

  - ``none``: No imputation (default, requires complete data)
  - ``simple``: SimpleImputer from scikit-learn
  - ``knn``: KNNImputer for k-nearest neighbors imputation
  - ``iterative``: IterativeImputer (MissForest-style)

- **strategy**: Strategy for SimpleImputer (only used when ``method = simple``)

  - ``mean``, ``median``, or ``most_frequent``

- **n_neighbors**: Number of neighbors for KNNImputer (only used when ``method = knn``, default: 5)

- **estimator**: Estimator for IterativeImputer (only used when ``method = iterative``)

  - ``bayesian_ridge`` (default) or ``random_forest``

8. Configure Conformal Prediction (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Uncertainty]
    alpha = 0.1

- **alpha**: Miscoverage rate for conformal prediction (0-0.5)

  - Default: 0.1 (90% target coverage)
  - Controls the width of prediction sets: lower alpha → wider sets, higher coverage

- **How it works**: CV+ conformal prediction with Mondrian (class-conditional) coverage

  - Nonconformity score: ``s(x, y) = 1 - p̂(y | x)``
  - Separate ``q_hat`` thresholds per class — critical for AMR class imbalance
  - A class is included in the prediction set if ``1 - p̂(class | x) <= q_hat[class]``
  - Prediction sets: ``{S}`` (susceptible only), ``{R}`` (resistant only), or ``{S, R}`` (uncertain)
  - Finite-sample, distribution-free coverage guarantees per class
  - CV+ guarantee: ``1 - 2*alpha`` worst-case (typically closer to ``1 - alpha`` in practice)

- **Output**: prediction CSV includes ``prediction_set_size`` (1 = certain, 2 = uncertain) and
  metrics CSV includes conformal diagnostics (empirical coverage, fraction uncertain, average set size)

9. Configure Preprocessing (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Preprocessing]
    ohe_min_frequency = 0.05

- **ohe_min_frequency**: Minimum frequency for categorical values in OneHotEncoder

  - Categories appearing below this threshold are grouped into an "infrequent" category
  - Values in (0, 1): proportion of samples (e.g., 0.05 = at least 5% of samples)
  - Values >= 1: absolute count (e.g., 10 = at least 10 occurrences)
  - Set to 0, omit, or comment out to disable (keep all categories)
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
- **compression**: 1-9 (1=minimal compression, 3=balanced, 9=maximum)

12. Set Output Location
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Output]
    out_folder = ./results/

The folder will be created if it doesn't exist.

13. Configure Validation Strategy (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

    [Metadata]
    # temporal_column = collection_date

    [Validation]
    validation_strategy = cv
    # temporal_split_date = 2023-01-01
    # temporal_split_ratio = 0.8

- **temporal_column** (in ``[Metadata]``): Name of the date/time column for temporal splitting

  - Required when ``validation_strategy`` is ``temporal`` or ``both``

- **validation_strategy**: Validation approach (default: ``cv``)

  - ``cv``: Standard nested cross-validation only
  - ``temporal``: Temporal (prospective-style) validation only
  - ``both``: Run both CV and temporal validation

- **temporal_split_date**: Cutoff date in ISO format (e.g., ``2023-01-01``)

  - Train set: dates before cutoff; test set: dates on or after cutoff
  - Mutually exclusive with ``temporal_split_ratio``

- **temporal_split_ratio**: Fraction of data for training by sorted date order

  - Must be between 0 and 1 (exclusive)
  - Mutually exclusive with ``temporal_split_date``

**Note:** When ``group_column`` is configured in ``[Metadata]``, temporal splitting assigns entire groups based on the group's latest date to prevent data leakage.

See Also
--------

- :doc:`run-command` - Execute the nested CV pipeline
- :doc:`train-command` - Train models on entire dataset for cross-dataset validation
- :doc:`validate-config-command` - Validate configuration before running
