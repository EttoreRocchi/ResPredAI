CLI Reference
=============

ResPredAI provides a command-line interface for all operations. This section documents all available commands.

Getting Help
------------

.. code-block:: bash

    # Show all available commands
    respredai --help

    # Show help for a specific command
    respredai run --help

Commands Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - :doc:`run <run-command>`
     - Run nested cross-validation pipeline
   * - :doc:`train <train-command>`
     - Train models on full dataset for cross-dataset validation
   * - :doc:`evaluate <evaluate-command>`
     - Evaluate trained models on new data
   * - :doc:`feature-importance <feature-importance-command>`
     - Extract and visualize feature importance
   * - :doc:`create-config <create-config-command>`
     - Generate template configuration file
   * - :doc:`validate-config <validate-config-command>`
     - Validate configuration file
   * - ``list-models``
     - Display available ML models
   * - ``info``
     - Show version and citation information

Main Workflow
-------------

1. Nested Cross-Validation (Standard)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create configuration
    respredai create-config my_config.ini

    # Validate configuration
    respredai validate-config my_config.ini --check-data

    # Run pipeline
    respredai run --config my_config.ini

    # Extract feature importance
    respredai feature-importance --output ./output --model LR --target Target1

2. Cross-Dataset Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Train on dataset A
    respredai train --config config_A.ini --output ./trained/

    # Evaluate on dataset B
    respredai evaluate --models-dir ./trained/trained_models --data dataset_B.csv --output ./eval/

Global Options
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Option
     - Description
   * - ``--help``
     - Show help message
   * - ``--version``
     - Show version number

Quick Examples
--------------

.. code-block:: bash

    # Show available models
    respredai list-models

    # Show version and citation
    respredai info

    # Run pipeline quietly
    respredai run --config config.ini --quiet

    # Train specific models only
    respredai train --config config.ini --models LR,RF

.. toctree::
   :maxdepth: 1
   :caption: Commands
   :hidden:

   run-command
   train-command
   evaluate-command
   feature-importance-command
   create-config-command
   validate-config-command
