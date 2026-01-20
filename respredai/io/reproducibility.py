"""Reproducibility manifest generation."""

import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import respredai


def get_package_versions() -> dict:
    """Get versions of key packages."""
    packages = {}
    # Mapping of package names to their import names
    pkg_map = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
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


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "packages": get_package_versions(),
        },
        "data": {
            "path": str(config_handler.data_path),
            "sha256": hash_file(Path(config_handler.data_path)),
            "shape": list(datasetter.data.shape),
            "features": datasetter.feature_columns,
            "targets": config_handler.targets,
            "class_distribution": {
                t: datasetter.data[t].value_counts().to_dict() for t in config_handler.targets
            },
        },
        "config": {
            "seed": config_handler.seed,
            "outer_folds": config_handler.outer_folds,
            "inner_folds": config_handler.inner_folds,
            "models": config_handler.models,
            "calibrate_threshold": config_handler.calibrate_threshold,
            "threshold_method": config_handler.threshold_method,
            "threshold_objective": config_handler.threshold_objective,
            "imputation_method": config_handler.imputation_method,
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
    output_path = output_dir / "reproducibility.json"
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return output_path
