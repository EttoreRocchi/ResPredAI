# Changelog

All changes to ResPredAI are documented in this file.

## [1.1.0] - 2025-12-04

### Added
- **Threshold calibration** with dual methods (OOF and CV) using Youden's J statistic
  - OOF method: Global optimization on concatenated out-of-fold predictions
  - CV method: Per-fold optimization with threshold averaging
  - Auto selection based on dataset size (n < 1000: OOF, otherwise: CV)
- **Grouped cross-validation** (`StratifiedGroupKFold`) to prevent data leakage in clinical datasets

### Changed
- Expanded hyperparameter grids for XGBoost, Random Forest, CatBoost, and MLP
- Enhanced CLI information display

### Fixed
- XGBoost feature naming issue with special characters
- Color scheme in feature importance plots

### Documentation
- Added comprehensive command documentation (`docs/run-command.md`, `docs/create-config-command.md`, `docs/feature-importance-command.md`)
- Updated README with logo, quick start guide, and output structure
- Add CHANGELOG.md


## [1.0.0] - Initial Release

### Core Features
- Nested cross-validation framework (outer: evaluation, inner: hyperparameter tuning)
- Eight machine learning models: LR, Linear SVC, RBF SVC, MLP, RF, XGBoost, CatBoost, TabPFN
- Comprehensive metrics: Precision, Recall, F1, MCC, Balanced Accuracy, AUROC
- Data preprocessing: StandardScaler, one-hot encoding, multi-target support
- INI-based configuration system
- Structured output: CSV metrics, confusion matrix plots, logs
- Feature importance extraction command with visualization and CSV export


## Citation

If you use ResPredAI in your research, please cite:

> Bonazzetti, C., Rocchi, E., Toschi, A. _et al._ Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections. _npj Digit. Med._ **8**, 319 (2025). https://doi.org/10.1038/s41746-025-01696-x


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Funding

This research was supported by EU funding within the NextGenerationEU-MUR PNRR Extended Partnership initiative on Emerging Infectious Diseases (Project no. PE00000007, INF-ACT).
