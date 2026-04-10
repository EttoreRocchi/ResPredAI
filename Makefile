.DEFAULT_GOAL := help
.PHONY: help install install-docs lint format format-check pre-commit \
        test docs docs-clean docs-serve clean clean-out publish publish-test \
        run-example run-temporal run-imputation \
        run-feature-importance run-feature-direction \
        run-train run-evaluate run-validate run-all

# Colours
BOLD  := \033[1m
RESET := \033[0m
CYAN  := \033[36m

# Paths
SRC      := respredai
TESTS    := tests
OUT_DIRS := out_run_example out_run_temporal out_run_imputation

help:  ## Show this help message
	@printf "$(BOLD)ResPredAI - available targets$(RESET)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

# Installation

install:  ## Install package with development dependencies
	@pip install -e ".[dev]"

install-docs:  ## Install package with documentation dependencies
	@pip install -e ".[docs]"

# Code quality

lint:  ## Run ruff linter
	@ruff check --fix $(SRC)/ $(TESTS)/

format:  ## Run ruff formatter (applies changes)
	@ruff format $(SRC)/ $(TESTS)/

format-check:  ## Run ruff formatter in check-only mode (no changes)
	@ruff format --check $(SRC)/ $(TESTS)/

pre-commit:  ## Run the full pre-commit suite on all files
	@pre-commit run --all-files

# Tests

test:  ## Run tests
	@pytest $(TESTS)/

test-fast:  ## Run tests but the ones marked as 'slow'
	@pytest -m "not slow" $(TESTS)/

test-slow:  ## Run tests marked as 'slow'
	@pytest -m "slow" $(TESTS)/

# Example runs

run-example:  ## Run basic CV pipeline (threshold + calibration)
	respredai run --config example/config_example.ini

run-temporal:  ## Run temporal validation pipeline
	respredai run --config example/config_example_temporal.ini

run-imputation:  ## Run imputation pipeline (iterative, repeated CV)
	respredai run --config example/config_example_imputation.ini

run-feature-importance:  ## Extract feature importance (requires run-example)
	respredai feature-importance --output out_run_example --model LR --target Target1

run-feature-direction:  ## Extract feature importance with direction (requires run-example)
	respredai feature-importance --output out_run_example --model LR --target Target1 --direction

run-train:  ## Train models on full dataset (requires run-example)
	respredai train --config example/config_example.ini --output out_run_example/train_output

run-evaluate:  ## Evaluate trained models on example data (requires run-train)
	respredai evaluate --models-dir out_run_example/train_output/trained_models \
	  --data example/data_example.csv --output out_run_example/evaluate

run-validate:  ## Validate all example config files
	respredai validate-config example/config_example.ini --check-data
	respredai validate-config example/config_example_temporal.ini --check-data
	respredai validate-config example/config_example_imputation.ini --check-data

run-all: clean-out  ## Run full end-to-end smoke test (all examples + CLI commands)
	@$(MAKE) run-example
	@$(MAKE) run-feature-importance
	@$(MAKE) run-feature-direction
	@$(MAKE) run-train
	@$(MAKE) run-evaluate
	@$(MAKE) run-temporal
	@$(MAKE) run-imputation
	@$(MAKE) run-validate

# Documentation

docs:  ## Build Sphinx HTML documentation
	@$(MAKE) -C docs html

docs-clean:  ## Remove Sphinx build artefacts
	@$(MAKE) -C docs clean

docs-serve:  ## Serve built docs locally on http://localhost:8080
	@python -m http.server --directory docs/_build/html 8080

# Housekeeping

clean-out:  ## Remove example pipeline output folders
	@rm -rf $(OUT_DIRS)

clean: clean-out  ## Remove build artefacts, caches, and output folders
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	@rm -rf dist/ build/

# Release

publish: clean  ## Build and publish package to PyPI
	@read -p "Publish to PyPI? [y/N] " ans && [ "$$ans" = "y" ]
	@python -m build
	@twine upload dist/*

publish-test: clean  ## Build and publish package to TestPyPI
	@python -m build
	@twine upload --repository testpypi dist/*
