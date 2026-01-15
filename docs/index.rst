ResPredAI
=========

Antimicrobial Resistance Prediction via AI
-------------------------------------------

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://github.com/EttoreRocchi/ResPredAI/actions/workflows/ci.yaml/badge.svg
   :alt: CI

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: assets/logo_ResPredAI.png
   :alt: ResPredAI Logo
   :width: 350
   :align: center

ResPredAI is a machine learning pipeline for predicting antimicrobial resistance. It implements the methodology described in:

    Bonazzetti, C., Rocchi, E., Toschi, A. *et al.* Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections. *npj Digit. Med.* **8**, 319 (2025). https://doi.org/10.1038/s41746-025-01696-x

Features
--------

- **Nested Cross-Validation**: Rigorous evaluation with inner CV for hyperparameter tuning and outer CV for performance estimation
- **8 ML Models**: Support for Logistic Regression, Random Forest, XGBoost, CatBoost, MLP, TabPFN, and SVM variants
- **Threshold Calibration**: Optional calibration using Youden's J statistic
- **Group-Aware CV**: Prevent data leakage with stratified group k-fold
- **Feature Importance**: Native importance extraction with SHAP fallback
- **Model Persistence**: Save and resume training, cross-dataset validation

Quick Links
-----------

- :doc:`getting-started/installation`
- :doc:`getting-started/quickstart`
- :doc:`cli-reference/index`
- :doc:`changelog`

Output Structure
----------------

The pipeline generates:

- **Confusion matrices**: PNG files with heatmaps showing model performance
- **Detailed metrics**: CSV files with precision, recall, F1, MCC, balanced accuracy, AUROC
- **Trained models**: Saved models for resumption and feature importance extraction
- **Feature importance**: Plots and CSV files showing feature importance/coefficients

::

    output_folder/
    ├── models/                          # Trained models (if enabled)
    ├── trained_models/                  # Models for cross-dataset validation
    ├── metrics/                         # Detailed performance metrics
    ├── feature_importance/              # Feature importance outputs
    ├── confusion_matrices/              # Confusion matrix heatmaps
    └── respredai.log                    # Execution log

Citation
--------

If you use ResPredAI in your research, please cite:

.. code-block:: bibtex

    @article{Bonazzetti2025,
      author = {Bonazzetti, Cecilia and Rocchi, Ettore and Toschi, Alice and others},
      title = {Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections},
      journal = {npj Digital Medicine},
      volume = {8},
      pages = {319},
      year = {2025},
      doi = {10.1038/s41746-025-01696-x}
    }

Funding
-------

This research was supported by EU funding within the NextGenerationEU-MUR PNRR Extended Partnership initiative on Emerging Infectious Diseases (Project no. PE00000007, INF-ACT).

License
-------

This project is licensed under the MIT License.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   getting-started/installation
   getting-started/quickstart
   cli-reference/index
   changelog
   contributing
