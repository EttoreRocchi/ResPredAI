Evaluate Command
================

The ``evaluate`` command applies trained models to new data and computes metrics against ground truth.

Usage
-----

.. code-block:: bash

    respredai evaluate --models-dir <path> --data <csv_file> --output <output_dir>

Options
-------

Required
~~~~~~~~

- ``--models-dir, -m`` - Directory containing trained models and ``training_metadata.json``

  - Must be output from ``respredai train`` command

- ``--data, -d`` - Path to new data CSV file

  - Must contain all feature columns from training
  - Must contain target columns (ground truth required)

- ``--output, -o`` - Output directory for evaluation results

Optional
~~~~~~~~

- ``--quiet, -q`` - Suppress progress output

Data Requirements
-----------------

The new data CSV must have:

1. **All feature columns** from the training data (same names)
2. **All target columns** for ground truth comparison
3. **Same categorical values** (new categories will be ignored)

The command validates columns before evaluation and provides clear error messages for missing columns.

Output Structure
----------------

::

    output_dir/
    ├── metrics/
    │   ├── Target1/
    │   │   ├── LR_metrics.csv
    │   │   └── RF_metrics.csv
    │   └── Target2/
    │       └── ...
    ├── predictions/
    │   ├── LR_Target1_predictions.csv
    │   ├── LR_Target2_predictions.csv
    │   └── ...
    └── evaluation_summary.csv

predictions CSV Format
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    sample_id,y_true,y_pred,y_prob
    0,1,1,0.73
    1,0,0,0.21
    2,1,0,0.38

metrics CSV Format
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Metric,Value
    Precision (0),0.82
    Precision (1),0.51
    Recall (0),0.91
    Recall (1),0.33
    F1 (0),0.86
    F1 (1),0.40
    F1 (weighted),0.79
    MCC,0.28
    Balanced Acc,0.62
    AUROC,0.71

evaluation_summary.csv
~~~~~~~~~~~~~~~~~~~~~~

Aggregated view of all model-target combinations:

.. code-block:: text

    Model,Target,Precision (0),Precision (1),...,AUROC
    LR,Target1,0.82,0.51,...,0.71
    RF,Target1,0.85,0.48,...,0.73
    LR,Target2,0.79,0.55,...,0.69

Example Usage
-------------

.. code-block:: bash

    # Basic evaluation
    respredai evaluate \
        --models-dir ./output/trained_models \
        --data ./new_data.csv \
        --output ./evaluation/

    # Quiet mode
    respredai evaluate \
        --models-dir ./output/trained_models \
        --data ./new_data.csv \
        --output ./evaluation/ \
        --quiet

Handling Different Data
-----------------------

Missing Categorical Values
~~~~~~~~~~~~~~~~~~~~~~~~~~

If new data has different categorical values than training:

- Missing dummy columns are added with value 0
- Extra categories in new data are ignored (encoded as all zeros)

Column Order
~~~~~~~~~~~~

Column order doesn't matter - the command reorders columns to match training.

Error Scenarios
---------------

**Missing features:**

::

    Validation Error: Missing feature columns: {'age', 'bmi'}

**Missing targets:**

::

    Validation Error: Missing target columns (ground truth required): {'Target1'}

**Invalid models directory:**

::

    Error: Training metadata not found: ./invalid/training_metadata.json
    Ensure this directory was created by 'respredai train'

See Also
--------

- :doc:`train-command` - Train models (required before evaluate)
- :doc:`run-command` - Full nested CV pipeline for model evaluation
- :doc:`feature-importance-command` - Extract feature importance from trained models
