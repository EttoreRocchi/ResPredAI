"""Reproducibility manifest generation."""

import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import respredai
from respredai.core.constants import FILE_REPRODUCIBILITY


def get_package_versions() -> dict:
    """Get versions of key packages.

    Returns
    -------
    dict
        Mapping of package name to version string for installed dependencies.
    """
    packages = {}
    # Mapping of package names to their import names
    pkg_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "shap": "shap",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "joblib": "joblib",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "tabpfn": "tabpfn",
    }
    for pkg_name, import_name in pkg_map.items():
        try:
            mod = __import__(import_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return packages


def get_installed_packages() -> dict:
    """Return name -> version for every installed distribution (pip freeze equivalent).

    Captures the full environment so a run can be reproduced exactly, including
    transitive dependencies. For bit-for-bit reproduction, recreate the
    environment from these recorded versions and run with n_jobs=1.

    Returns
    -------
    dict
        Sorted mapping of distribution name to version string.
    """
    import importlib.metadata as importlib_metadata

    packages: dict = {}
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata["Name"]
            if name:
                packages[name] = dist.version
        except Exception:
            continue
    return dict(sorted(packages.items(), key=lambda kv: kv[0].lower()))


def get_git_commit() -> Optional[str]:
    """Return the current git commit SHA, or None if unavailable.

    Returns
    -------
    str or None
        Full commit hash, or None if not in a git repository or git is missing.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file.

    Parameters
    ----------
    path : Path
        Path to the file to hash.

    Returns
    -------
    str
        Hex-encoded SHA256 digest.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def create_reproducibility_manifest(config_handler, datasetter) -> dict:
    """Create reproducibility manifest.

    Parameters
    ----------
    config_handler : ConfigHandler
        Configuration handler with pipeline settings.
    datasetter : DataSetter
        Data setter with loaded data.

    Returns
    -------
    dict
        Reproducibility manifest dictionary.
    """
    return {
        "respredai_version": respredai.__version__,
        "git_commit": get_git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "packages": get_package_versions(),
            "installed_packages": get_installed_packages(),
        },
        "data": {
            "path": str(config_handler.data_cfg.data_path),
            "sha256": hash_file(Path(config_handler.data_cfg.data_path)),
            "shape": list(datasetter.data.shape),
            "features": list(datasetter.X.columns),
            "targets": config_handler.data_cfg.targets,
            "class_distribution": {
                t: datasetter.data[t].value_counts().to_dict()
                for t in config_handler.data_cfg.targets
            },
        },
        "config": {
            "seed": config_handler.reproducibility_cfg.seed,
            "n_jobs": config_handler.reproducibility_cfg.n_jobs,
            "outer_folds": config_handler.pipeline.outer_folds,
            "inner_folds": config_handler.pipeline.inner_folds,
            "models": config_handler.pipeline.models,
            "calibration_bins": config_handler.pipeline.calibration_bins,
            "calibrate_threshold": config_handler.pipeline.calibrate_threshold,
            "threshold_method": config_handler.pipeline.threshold_method,
            "threshold_objective": config_handler.pipeline.threshold_objective,
            "calibrate_probabilities": config_handler.pipeline.calibrate_probabilities,
            "probability_calibration_method": config_handler.pipeline.probability_calibration_method,
            "probability_calibration_cv": config_handler.pipeline.probability_calibration_cv,
            "imputation_method": config_handler.imputation.method,
        },
    }


def save_reproducibility_manifest(manifest: dict, output_dir: Path) -> Path:
    """Save manifest to JSON file.

    Parameters
    ----------
    manifest : dict
        Reproducibility manifest dictionary.
    output_dir : Path
        Output directory path.

    Returns
    -------
    Path
        Path to saved manifest file.
    """
    output_path = output_dir / FILE_REPRODUCIBILITY
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return output_path
