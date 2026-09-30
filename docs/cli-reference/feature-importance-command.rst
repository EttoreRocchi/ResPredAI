Feature Importance Command
==========================

The ``feature-importance`` command extracts and visualizes feature importance or coefficients from trained models across all outer cross-validation iterations.

Usage
-----

.. code-block:: bash

    respredai feature-importance --output <output_folder> --model <model_name> --target <target_name> [options]

Options
-------

Required
~~~~~~~~

- ``--output, -o`` - Path to the output folder containing trained models

  - Must be the same folder used in the ``run`` command
  - Must contain a ``models/`` subdirectory with saved model files
  - Example: ``./output/`` or ``./out_run_example/``

- ``--model, -m`` - Model name to extract importance from

  - Must match one of the models trained in the pipeline
  - Examples: ``LR``, ``RF``, ``XGB``, ``CatBoost``, ``Linear_SVC``
  - Case-sensitive

- ``--target, -t`` - Target name to extract importance for

  - Must match one of the targets from the training pipeline
  - Example: ``Target1``, ``Ciprofloxacin_R``
  - Case-sensitive

Optional
~~~~~~~~

- ``--top-n, -n`` - Number of top features to display (default: 20)

  - Features are ranked by absolute importance
  - Range: 1 to total number of features
  - Example: ``--top-n 30`` for top 30 features

- ``--no-plot`` - Skip generating the barplot

  - Only CSV file will be created
  - Useful for batch processing or server environments

- ``--no-csv`` - Skip generating the CSV file

  - Only plot will be created
  - Useful if you only need visualizations

- ``--seed, -s`` - Random seed for SHAP reproducibility

  - Ensures reproducible SHAP values across runs
  - Only affects models using SHAP fallback

- ``--direction`` - Compute signed feature direction (Risk/Protective)

  - Determines whether each feature is a risk factor or protective factor
  - Linear models: uses coefficient sign directly (no extra computation)
  - Tree-based models (RF, XGB, CatBoost): uses TreeExplainer for signed SHAP values
  - Other models: falls back to KernelExplainer for signed SHAP values
  - Adds a ``Direction`` column to the CSV output (``Risk (+)``, ``Protective (-)``, or ``Neutral (~0)``)
  - Colors plot bars by direction: firebrick (risk) / seagreen (protective) / gray (neutral)

Supported Models
----------------

The command uses native importance when available, with SHAP as fallback:

Native Importance (Primary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linear Models (Coefficients)**

- **LR** (Logistic Regression) - Uses signed coefficient values
- **Linear_SVC** (Linear SVM) - Uses signed coefficient values

Coefficients refer to the preprocessed feature space and precede probability
calibration; see :ref:`interpreting-linear-coefficients` below.

**Tree-Based Models (Feature Importances)**

- **RF** (Random Forest) - Uses Gini importance
- **XGB** (XGBoost) - Uses gain-based importance
- **CatBoost** - Uses feature importance scores

For tree-based models importance values are always positive.

SHAP Fallback
~~~~~~~~~~~~~

For models without native importance/coefficients, SHAP (SHapley Additive exPlanations) values are computed as a fallback:

- **MLP** (Multi-Layer Perceptron) - Uses KernelExplainer
- **RBF_SVC** (RBF SVM) - Uses KernelExplainer
- **TabPFN** - Uses KernelExplainer

SHAP values are computed on the test fold of each outer CV iteration and aggregated across folds. The mean absolute SHAP value represents feature importance.

Note: SHAP computation with KernelExplainer can be slow for large datasets.

Calibrated Fold Models
~~~~~~~~~~~~~~~~~~~~~~

With ``calibrate_probabilities = true`` each outer-fold model is a
``CalibratedClassifierCV``. It clones the best estimator found by the grid
search and refits it once per calibration split (``probability_calibration_cv``,
default 5), each time on a different subset of the outer training fold, and
fits a Platt sigmoid (or isotonic map) on the held-out part. The fold's
probability is the *average* of those sub-models.

Native importances (coefficients, impurity, gain) and TreeExplainer SHAP values
are therefore computed for **every** inner estimator of a fold model and
averaged, so the reported value describes the whole fold model rather than one
arbitrary sub-model. KernelExplainer SHAP values already use the fold model's
``predict_proba`` and need no extra handling.

.. _interpreting-linear-coefficients:

Interpreting Linear Coefficients
--------------------------------

For ``LR`` and ``Linear_SVC`` the reported importance is the raw model
coefficient. Three things determine what that number means:

- **Feature space.** Coefficients refer to the features *as the model sees
  them*: continuous features are standardized (z-scored) inside each fold and
  categorical features are one-hot encoded. A coefficient is the change in
  log-odds per standard deviation of a continuous feature, or per unit of a
  one-hot column. The fitted scalers and encoders are stored per fold
  (``fold_transformers`` and ``fold_ohe_transformers`` in the saved
  ``.joblib``), so a coefficient can be mapped back to the original units by
  dividing by the scaler's ``scale_``.
- **Before probability calibration.** The CSV reports the mean of the
  sub-models' raw coefficients (and, in the per-fold CSV, intercepts). With
  sigmoid calibration scikit-learn maps a decision value ``f`` to
  ``expit(-(a * f + b))``, so the slope of the calibrated logit of a sub-model
  is ``-a * coef`` and its offset is ``-(a * intercept + b)``. The Platt
  offset ``b`` also absorbs the prior shift introduced by
  ``class_weight="balanced"``. Coefficients or intercepts recovered by
  regressing the calibrated probabilities on the covariates are therefore
  **not** expected to match the raw values in the CSV, and the average of
  several sigmoids is only approximately a single logistic function.
- **The grid-search refit is not the predictive model.** When calibration is
  enabled, ``GridSearchCV.best_estimator_`` (refit on the whole outer training
  fold) only acts as a template: its hyperparameters are kept, while its
  fitted coefficients and intercept are discarded and never used for
  prediction. Reading ``intercept_`` from it does not describe the
  probabilities the pipeline outputs.

Everything needed to inspect a fold model is in the saved ``.joblib``:

.. code-block:: python

    import joblib

    data = joblib.load("output/models/LR_Target1_models.joblib")
    for position, model in enumerate(data["fold_models"]):
        if model is None:  # failed fold
            continue
        _, feature_names = data["fold_test_data"][position]
        # checkpoints written by earlier releases may wrap the calibrator in a
        # TunedThresholdClassifierCV; current ones store the calibrator itself
        calibrator = model.estimator_ if hasattr(model, "estimator_") else model
        for sub in calibrator.calibrated_classifiers_:
            lr = sub.estimator  # LogisticRegression of this calibration split
            platt = sub.calibrators[0]  # sigmoid calibrator with a_ and b_
            raw_coef, raw_intercept = lr.coef_[0], lr.intercept_[0]
            effective_coef = -platt.a_ * raw_coef
            effective_intercept = -(platt.a_ * raw_intercept + platt.b_)
        transformer = data["fold_transformers"][position]
        scaler = transformer.named_transformers_["scaler"]  # StandardScaler
        # with imputation enabled the scaler is the last step of the
        # "continuous" pipeline: named_transformers_["continuous"]["scaler"]
        # scaler.mean_ and scaler.scale_ map standardized coefficients to raw units

The same walk over inner estimators is available as
``respredai.visualization.feature_importance.iter_inner_estimators(model)``.
Without probability calibration the fold model is the grid-search estimator
itself and ``calibrated_classifiers_`` does not exist.

Output Files
------------

The command generates files in the following structure:

::

    output_folder/
    └── feature_importance/
        └── {target}/
            ├── {model}_feature_importance.csv               # Native importance (if available)
            ├── {model}_feature_importance_per_fold.csv      # One row per outer fold
            ├── {model}_feature_importance.png
            ├── {model}_feature_importance_shap.csv          # SHAP importance (fallback)
            ├── {model}_feature_importance_shap_per_fold.csv
            └── {model}_feature_importance_shap.png

Files have ``_shap`` suffix when SHAP is used instead of native importance.

All importance CSVs also include an ``Importance_Type`` column (the importance
scale, e.g. ``gain``, ``impurity (MDI)``, ``coefficient``, or ``mean_abs_shap`` -
not comparable across model families) and an ``N_folds_present`` column (the
number of CV folds in which the feature appeared). Cross-fold means skip folds
where a feature was absent rather than treating it as zero importance.

CSV File Format (Native)
~~~~~~~~~~~~~~~~~~~~~~~~

For models with native importance:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``Feature``
     - Feature name
   * - ``Mean_Importance``
     - Mean importance across folds (signed for linear models)
   * - ``Std_Importance``
     - Standard deviation across folds
   * - ``Abs_Mean_Importance``
     - Absolute mean importance (used for ranking)
   * - ``Mean±Std``
     - Formatted string with mean ± std
   * - ``Direction``
     - Feature direction: ``Risk (+)`` or ``Protective (-)`` (only when ``--direction`` is used)

CSV File Format (SHAP)
~~~~~~~~~~~~~~~~~~~~~~

For models using SHAP fallback:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``Feature``
     - Feature name
   * - ``Mean_Abs_SHAP``
     - Mean absolute SHAP value across folds
   * - ``Std_Abs_SHAP``
     - Standard deviation across folds
   * - ``Mean±Std``
     - Formatted string with mean ± std

CSV File Format (Per Fold)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``_per_fold`` file exposes the values behind the summary mean and std,
one row per successful outer fold:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``Fold``
     - 1-based outer fold number (with repeated CV, folds are numbered
       consecutively across repeats, matching the order of ``fold_models``
       in the saved ``.joblib``)
   * - ``Intercept``
     - Intercept of the fold model, averaged over its inner estimators
       (linear models only; same feature space and pre-calibration scale as
       the coefficients)
   * - one column per feature
     - Importance/coefficient (or mean absolute SHAP value) of that fold, in the same
       feature order as the summary CSV

Features are **sorted by importance** (absolute mean value).

Across all folds:

- Calculate **mean importance** for each feature
- Calculate **standard deviation** (uncertainty measure)
- Rank features by importance

Plot Color Coding
~~~~~~~~~~~~~~~~~

The barplot uses different colors to indicate importance type:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Method
     - Color
     - Meaning
   * - SHAP
     - Orange
     - Mean absolute SHAP value
   * - Native (tree-based)
     - Blue
     - Feature importance (always positive)
   * - Native (linear, positive)
     - Red
     - Positive coefficient
   * - Native (linear, negative)
     - Green
     - Negative coefficient
   * - Direction (risk)
     - Red (firebrick)
     - Risk factor (``--direction`` flag)
   * - Direction (protective)
     - Green (seagreen)
     - Protective factor (``--direction`` flag)

Error bars show standard deviation across CV folds.

Examples
--------

Basic Usage
~~~~~~~~~~~

Extract top 20 features for Logistic Regression on Target1:

.. code-block:: bash

    respredai feature-importance --output ./output --model LR --target Target1

Custom Number of Features
~~~~~~~~~~~~~~~~~~~~~~~~~

Show top 5 features:

.. code-block:: bash

    respredai feature-importance -o ./output -m RF -t Target2 --top-n 5

Multiple Models
~~~~~~~~~~~~~~~

Extract importance for multiple models (run separately):

.. code-block:: bash

    respredai feature-importance -o ./output -m LR -t Target1
    respredai feature-importance -o ./output -m RF -t Target1
    respredai feature-importance -o ./output -m XGB -t Target1

CSV Only (No Plot)
~~~~~~~~~~~~~~~~~~

Generate only the CSV file for automated analysis:

.. code-block:: bash

    respredai feature-importance -o ./output -m LR -t Target1 --no-plot

Plot Only (No CSV)
~~~~~~~~~~~~~~~~~~

Generate only the visualization:

.. code-block:: bash

    respredai feature-importance -o ./output -m RF -t Target1 --no-csv

See Also
--------

- :doc:`run-command` - Train models with nested CV and save model files
- :doc:`train-command` - Train models on entire dataset for cross-dataset validation
- :doc:`create-config-command` - How to create the configuration file
