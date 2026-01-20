Train Command
=============

The ``train`` command trains models on the entire dataset using GridSearchCV for hyperparameter tuning, then saves the best model for each model-target combination.

Usage
-----

.. code-block:: bash

    respredai train --config <path_to_config.ini> [options]

Options
-------

Required
~~~~~~~~

- ``--config, -c`` - Path to the configuration file (INI format)

Optional
~~~~~~~~

- ``--quiet, -q`` - Suppress banner and progress output
- ``--models, -m`` - Override models (comma-separated)
- ``--targets, -t`` - Override targets (comma-separated)
- ``--output, -o`` - Override output folder
- ``--seed, -s`` - Override random seed

How It Differs from ``run``
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - ``run`` Command
     - ``train`` Command
   * - Purpose
     - Evaluate model performance
     - Train models for cross-dataset validation
   * - CV Strategy
     - Nested CV (outer + inner)
     - Single CV (only for HP tuning)
   * - Data Split
     - Multiple train/test splits
     - Uses entire dataset
   * - Output
     - Metrics, confusion matrices
     - Trained model files, ready for evaluation on another dataset
   * - Model Files
     - Per-fold models (optional)
     - Single model per target

Configuration Parameters
------------------------

The ``train`` command uses the same configuration file as ``run``, but some parameters are ignored:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Used
     - Notes
   * - ``data_path``
     - Yes
     - Path to training data
   * - ``targets``
     - Yes
     - Target columns to train
   * - ``continuous_features``
     - Yes
     - Features to scale
   * - ``group_column``
     - Yes
     - Used for grouped CV during HP tuning
   * - ``models``
     - Yes
     - Models to train
   * - ``inner_folds``
     - Yes
     - CV folds for hyperparameter tuning
   * - ``outer_folds``
     - **No**
     - Only used by ``run`` for nested CV
   * - ``calibrate_threshold``
     - Yes
     - Enables threshold calibration
   * - ``threshold_method``
     - Yes
     - Method for threshold calibration
   * - ``seed``
     - Yes
     - Random seed for reproducibility
   * - ``n_jobs``
     - Yes
     - Parallel jobs
   * - ``out_folder``
     - Yes
     - Output directory
   * - ``[ModelSaving] enable``
     - **No**
     - Always ``true`` for ``train`` (models are always saved)
   * - ``[ModelSaving] compression``
     - Yes
     - Compression level for saved models

Output Structure
----------------

::

    output_folder/
    ├── trained_models/
    │   ├── LR_Target1.joblib
    │   ├── LR_Target2.joblib
    │   ├── RF_Target1.joblib
    │   ├── ...
    │   └── training_metadata.json
    └── reproducibility.json                  # Reproducibility manifest

Model Bundle Contents
~~~~~~~~~~~~~~~~~~~~~

Each ``.joblib`` file contains:

.. code-block:: python

    {
        "model": <fitted_classifier>,
        "transformer": <fitted_scaler>,
        "threshold": 0.42,                  # Calibrated threshold
        "hyperparams": {"C": 0.1, ...},
        "feature_names": [...],             # Original feature names
        "feature_names_transformed": [...], # After one-hot encoding
        "target_name": "Target1",
        "model_name": "LR",
        "training_timestamp": "2025-12-11T..."
    }

training_metadata.json
~~~~~~~~~~~~~~~~~~~~~~

Contains information needed for evaluation on new data:

.. code-block:: json

    {
        "features": ["age", "sex", "category_col"],
        "continuous_features": ["age"],
        "categorical_features": ["sex", "category_col"],
        "targets": ["Target1", "Target2"],
        "feature_names_transformed": ["age", "sex_M", "category_col_A", "..."],
        "feature_dtypes": {"age": "float64", "sex": "object"},
        "training_data_path": "data.csv",
        "training_timestamp": "2025-12-11T..."
    }

Example Workflow
----------------

.. code-block:: bash

    # 1. Train models
    respredai train --config config.ini --output ./trained_output/

    # 2. Later, evaluate on new data
    respredai evaluate \
        --models-dir ./trained_output/trained_models \
        --data new_patients.csv \
        --output ./evaluation_results/

Threshold Calibration
---------------------

If ``calibrate_threshold = true`` in the config, the train command:

1. Runs GridSearchCV to find best HP
2. Gets out-of-fold predictions using the best estimator
3. Calculates optimal threshold using Youden's J statistic
4. Saves the calibrated threshold with the model

This threshold is automatically applied during evaluation.

See Also
--------

- :doc:`evaluate-command` - Apply trained models to new data
- :doc:`run-command` - Full nested CV pipeline for model evaluation
- :doc:`create-config-command` - Configuration file setup
- :doc:`validate-config-command` - Validate configuration before training
