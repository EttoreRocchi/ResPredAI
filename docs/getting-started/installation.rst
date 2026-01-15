Installation
============

Requirements
------------

- Python 3.9 or higher
- pip (Python package manager)

Install from Source
-------------------

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/EttoreRocchi/ResPredAI.git
    cd ResPredAI

    # Install the package
    pip install .

Install for Development
-----------------------

If you want to contribute to ResPredAI:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/EttoreRocchi/ResPredAI.git
    cd ResPredAI

    # Install in editable mode with development dependencies
    pip install -e ".[dev]"

Verify Installation
-------------------

Test that the installation was successful:

.. code-block:: bash

    respredai --version

You should see the version number displayed.

Test with Example Data
----------------------

Run the pipeline on the included synthetic dataset:

.. code-block:: bash

    respredai run --config example/config_example.ini

Results will be generated in ``./out_run_example/``.

Dependencies
------------

ResPredAI requires the following main dependencies:

- **scikit-learn**: Core ML functionality
- **xgboost**: XGBoost classifier
- **catboost**: CatBoost classifier
- **tabpfn**: TabPFN classifier
- **shap**: Feature importance (SHAP fallback)
- **typer**: CLI framework
- **rich**: Terminal output formatting
- **matplotlib/seaborn**: Visualization

All dependencies are automatically installed when you run ``pip install .``
