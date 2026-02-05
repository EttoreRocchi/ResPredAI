Run Command
===========

The ``run`` command executes the full machine learning pipeline with nested cross-validation for antimicrobial resistance prediction.

Usage
-----

.. code-block:: bash

    respredai run --config <path_to_config.ini> [options]

Options
-------

Required
~~~~~~~~

- ``--config, -c`` - Path to the configuration file (INI format)

  - Must be a valid, readable file
  - See `Configuration File`_ section for details

Optional
~~~~~~~~

- ``--quiet, -q`` - Suppress banner and progress output

  - Does not suppress error messages or logs

CLI Overrides
~~~~~~~~~~~~~

Override configuration file parameters without editing the file:

- ``--models, -m`` - Override models (comma-separated)

  - Example: ``--models LR,RF,XGB``

- ``--targets, -t`` - Override targets (comma-separated)

  - Example: ``--targets Target1,Target2``

- ``--output, -o`` - Override output folder

  - Example: ``--output ./new_results/``

- ``--seed, -s`` - Override random seed

  - Example: ``--seed 123``

**Examples with overrides:**

.. code-block:: bash

    # Run with different models
    respredai run --config my_config.ini --models LR,RF

    # Run only specific targets with a different output folder
    respredai run --config my_config.ini --targets Target1 --output ./experiment1/

    # Quick experiment with different seed
    respredai run --config my_config.ini --seed 42 --quiet

Configuration File
------------------

The configuration file uses INI format with the following sections:

[Data] Section
~~~~~~~~~~~~~~

Defines the input data and features.

.. code-block:: ini

    [Data]
    data_path = ./data/my_data.csv
    targets = Target1,Target2,Target3
    continuous_features = Age,Weight,Temperature

**Parameters:**

- ``data_path`` - Path to CSV file containing the dataset

  - Must include all features and target columns
  - First column is assumed to be the sample ID

- ``targets`` - Comma-separated list of target column names

  - Each target will be trained separately
  - Must exist in the CSV file

- ``continuous_features`` - Comma-separated list of continuous feature names

  - These features will be scaled using StandardScaler
  - All other features are treated as categorical and one-hot encoded

- ``group_column`` (optional) - Column name for grouping related samples

  - Use when you have multiple samples from the same patient/subject
  - Prevents data leakage by keeping all samples from the same group in the same fold
  - Enables ``StratifiedGroupKFold`` for both outer and inner cross-validation (if not specified, standard ``StratifiedKFold`` is used)
  - See details in :doc:`create-config-command`

[Pipeline] Section
~~~~~~~~~~~~~~~~~~

Controls the machine learning pipeline configuration.

.. code-block:: ini

    [Pipeline]
    models = LR,RF,XGB,CatBoost
    outer_folds = 5
    inner_folds = 3
    outer_cv_repeats = 1
    calibrate_threshold = false
    threshold_method = auto
    calibrate_probabilities = false
    probability_calibration_method = sigmoid
    probability_calibration_cv = 5

**Parameters:**

- ``models`` - Comma-separated list of models to train

  - Available models: ``LR``, ``MLP``, ``XGB``, ``RF``, ``CatBoost``, ``TabPFN``, ``RBF_SVC``, ``Linear_SVC``, ``KNN``
  - Use ``respredai list-models`` to see all available models with descriptions

- ``outer_folds`` - Number of folds for outer cross-validation

  - Used for model evaluation

- ``inner_folds`` - Number of folds for inner cross-validation

  - Used for hyperparameter tuning with GridSearchCV

- ``calibrate_threshold`` - Enable decision threshold calibration (optional, default: ``false``)

  - ``true``: Calibrate threshold using Youden's J statistic (Sensitivity + Specificity - 1)
  - ``false``: Use default threshold of 0.5
  - Threshold calibration uses ``inner_folds`` for cross-validation
  - Hyperparameters are tuned first (optimizing ROC-AUC), then threshold is calibrated

- ``threshold_method`` - Method for threshold calibration (optional, default: ``auto``)

  - ``auto``: Automatically choose based on sample size (OOF if n < 1000, CV otherwise)
  - ``oof``: Out-of-fold predictions method - aggregates predictions from all CV folds into a single set, then finds one global threshold maximizing Youden's J across all concatenated samples
  - ``cv``: TunedThresholdClassifierCV method - calculates optimal threshold separately for each CV fold, then aggregates (averages) the fold-specific thresholds
  - **Key difference**: ``oof`` finds one threshold on all concatenated OOF predictions (global optimization), while ``cv`` finds per-fold thresholds then averages them (fold-wise optimization then aggregation)
  - Only used when ``calibrate_threshold = true``

- ``outer_cv_repeats`` - Number of repetitions for outer cross-validation (optional, default: ``1``)

  - ``1``: Standard (non-repeated) cross-validation
  - ``>1``: Repeated stratified cross-validation with different random shuffles
  - Provides more robust performance estimates by averaging over multiple CV runs

- ``calibrate_probabilities`` - Enable post-hoc probability calibration (optional, default: ``false``)

  - ``true``: Apply CalibratedClassifierCV to the best estimator from GridSearchCV
  - ``false``: Use uncalibrated probability predictions
  - Applied after hyper-parameters tuning and before threshold tuning

- ``probability_calibration_method`` - Method for probability calibration (optional, default: ``sigmoid``)

  - ``sigmoid``: Platt scaling - fits a logistic regression on the classifier outputs
  - ``isotonic``: Isotonic regression - non-parametric, monotonic transformation
  - Only used when ``calibrate_probabilities = true``

- ``probability_calibration_cv`` - Number of folds for probability calibration (optional, default: ``5``)

  - CV folds used internally by CalibratedClassifierCV
  - Must be at least 2
  - Only used when ``calibrate_probabilities = true``

[Reproducibility] Section
~~~~~~~~~~~~~~~~~~~~~~~~~

Ensures reproducible results.

.. code-block:: ini

    [Reproducibility]
    seed = 42

**Parameters:**

- ``seed`` - Random seed for reproducibility

  - Same seed ensures identical results across runs
  - Affects data splitting and model initialization

[Log] Section
~~~~~~~~~~~~~

Controls logging behavior.

.. code-block:: ini

    [Log]
    verbosity = 1
    log_basename = respredai.log

**Parameters:**

- ``verbosity`` - Logging level

  - ``0``: No logging to file
  - ``1``: Log major events (model start/end, target completion)
  - ``2``: Verbose logging (includes fold-level details)

- ``log_basename`` - Name of the log file

  - Created in the output folder
  - Contains detailed execution information

[Resources] Section
~~~~~~~~~~~~~~~~~~~

Controls computational resources.

.. code-block:: ini

    [Resources]
    n_jobs = -1

**Parameters:**

- ``n_jobs`` - Number of parallel jobs

  - ``-1``: Use all available CPU cores
  - ``1``: No parallelization
  - ``N``: Use N cores

[ModelSaving] Section
~~~~~~~~~~~~~~~~~~~~~

Enables saving trained models for resumption.

.. code-block:: ini

    [ModelSaving]
    enable = true
    compression = 3

**Parameters:**

- ``enable`` - Enable saving trained models

  - ``true``: Save models after each fold (enables resumption)
  - ``false``: No model saving (faster but no resumption)

- ``compression`` - Compression level for saved model files

  - Range: 0-9
  - ``0``: No compression (fastest, largest files)
  - ``3``: Balanced compression (recommended)
  - ``9``: Maximum compression (slowest, smallest files)

[Imputation] Section
~~~~~~~~~~~~~~~~~~~~

Controls missing data imputation (optional).

.. code-block:: ini

    [Imputation]
    method = none
    strategy = mean
    n_neighbors = 5
    estimator = bayesian_ridge

**Parameters:**

- ``method`` - Imputation method

  - ``none``: No imputation (default, requires complete data)
  - ``simple``: SimpleImputer from scikit-learn
  - ``knn``: KNNImputer for k-nearest neighbors imputation
  - ``iterative``: IterativeImputer (MissForest-style)

- ``strategy`` - Strategy for SimpleImputer (only used when ``method = simple``)

  - ``mean``: Replace missing values with column mean (default)
  - ``median``: Replace with column median
  - ``most_frequent``: Replace with most frequent value

- ``n_neighbors`` - Number of neighbors for KNNImputer (only used when ``method = knn``)

  - Default: ``5``

- ``estimator`` - Estimator for IterativeImputer (only used when ``method = iterative``)

  - ``bayesian_ridge``: BayesianRidge estimator (default)
  - ``random_forest``: RandomForestRegressor (MissForest-style)

[Output] Section
~~~~~~~~~~~~~~~~

Specifies output location.

.. code-block:: ini

    [Output]
    out_folder = ./output/

**Parameters:**

- ``out_folder`` - Path to output directory

  - Will be created if it doesn't exist
  - Contains all results, metrics, and saved models

Pipeline Workflow
-----------------

The ``run`` command executes the following steps:

1. **Configuration Loading** - Parse and validate the configuration file
2. **Data Loading** - Read CSV and validate features/targets
3. **Preprocessing** - One-hot encode categorical features, prepare data
4. **Nested Cross-Validation** - For each model and target:

   - **Outer CV Loop**: Split data for evaluation
   - **Inner CV Loop**: Hyperparameter tuning with GridSearchCV
   - **Training**: Train best model on outer training fold
   - **Evaluation**: Test on outer test fold
   - **Save Models**: Save trained models and metrics (if enabled)

5. **Results Aggregation** - Calculate mean and std across folds
6. **Output Generation** - Save confusion matrices, metrics, and plots

Output Files
------------

The pipeline generates the following output structure:

::

    output_folder/
    ├── models/                                       # Trained models (if model saving enabled)
    │   ├── {Model}_{Target}_models.joblib            # Saved models for resumption
    │   └── ...
    ├── metrics/                                      # Detailed metrics
    │   ├── {target}/
    │   │   ├── {model}_metrics_detailed.csv          # Comprehensive metrics with CI
    │   │   └── summary.csv                           # Summary across all models for this target
    │   └── summary_all.csv                           # Global summary across all models and targets
    ├── confusion_matrices/                           # Confusion matrix heatmaps
    │   └── Confusion_matrix_{model}_{target}.png     # One PNG per model-target combination
    ├── calibration/                                  # Calibration diagnostics
    │   └── reliability_curve_{model}_{target}.png    # Reliability curves per fold + aggregate
    ├── report.html                                   # Comprehensive HTML report
    ├── reproducibility.json                          # Reproducibility manifest
    └── respredai.log                                 # Execution log (if verbosity > 0)

Metrics Files
~~~~~~~~~~~~~

Each ``{model}_metrics_detailed.csv`` contains:

- **Metric**: Name of the metric (Precision, Recall, F1, MCC, Balanced Acc, AUROC, VME, ME, Brier Score, ECE, MCE)
- **Mean**: Mean value across folds
- **Std**: Standard deviation across folds
- **CI95_lower**: Lower bound of 95% confidence interval (bootstrap, 1,000 resamples)
- **CI95_upper**: Upper bound of 95% confidence interval (bootstrap, 1,000 resamples)

**Calibration Metrics** (always computed, independent of probability calibration setting):

- **Brier Score**: Mean squared error of probability predictions (lower is better, range 0-1)
- **ECE** (Expected Calibration Error): Weighted average of calibration error across probability bins
- **MCE** (Maximum Calibration Error): Maximum calibration error across any probability bin

Confusion Matrix Plots
~~~~~~~~~~~~~~~~~~~~~~

Each ``Confusion_matrix_{model}_{target}.png`` shows:

- Normalized confusion matrix for a single model-target combination
- Mean F1, MCC, and AUROC scores with standard deviations
- Color-coded heatmap (0.0 = poor, 1.0 = perfect)

HTML Report
~~~~~~~~~~~

The ``report.html`` file provides a comprehensive, self-contained summary:

- **Metadata**: Configuration settings, data path, timestamp
- **Framework Summary**: Pipeline parameters, models, targets, and calibration settings
- **Results Tables**: Per-target metrics with 95% confidence intervals for each model
- **Confusion Matrices**: Embedded visualizations in a responsive grid layout
- **Calibration Diagnostics**: Brier Score, ECE, MCE metrics with 95% CIs, plus reliability curve plots

The report can be opened in any web browser and shared without additional dependencies.

Model Saving System
-------------------

Each ``{Model}_{Target}_models.joblib`` file contains all data from the outer cross-validation in a single file:

- **fold_models**: A list containing one trained model per outer CV fold
- **fold_transformers**: A list containing one fitted transformer (scaler) per fold
- **metrics**: All metrics (precision, recall, F1, MCC, AUROC, confusion matrices) for every fold
- **completed_folds**: Number of completed folds
- **timestamp**: When the file was saved

For example, with ``outer_folds=5``, each joblib file will contain 5 trained models and their corresponding transformers and metrics.

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

    respredai run --config my_config.ini

Quiet Mode (for scripts)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    respredai run --config my_config.ini --quiet

See Also
--------

- :doc:`train-command` - Train models on entire dataset for cross-dataset validation
- :doc:`evaluate-command` - Apply trained models to new data
- :doc:`feature-importance-command` - Extract and visualize feature importance
- :doc:`validate-config-command` - Validate configuration files
- :doc:`create-config-command` - How to create the configuration file
