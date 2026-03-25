"""HTML report generation for ResPredAI results."""

import base64
import html as html_mod
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respredai import __version__
from respredai.core.constants import (
    DIR_CALIBRATION,
    DIR_CONFUSION_MATRICES,
    DIR_METRICS,
    DIR_SUBGROUP,
    sanitize_metric_name,
    sanitize_name,
)

logger = logging.getLogger("respredai")


def _esc(value: object) -> str:
    """HTML-escape a value for safe interpolation into HTML."""
    return html_mod.escape(str(value))


def _fmt_metric(data: dict, metric_base: str, decimals: int = 3) -> str:
    """Format a metric value with optional CI bracket.

    Parameters
    ----------
    data : dict
        Metrics dictionary with keys like ``{metric_base}_mean``,
        ``{metric_base}_ci_lower``, ``{metric_base}_ci_upper``.
    metric_base : str
        Base name of the metric (e.g. ``"AUROC"``, ``"Brier_Score"``).
    decimals : int
        Number of decimal places (3 for classification metrics, 4 for calibration).
    """
    mean = data.get(f"{metric_base}_mean", np.nan)
    ci_lower = data.get(f"{metric_base}_ci_lower", np.nan)
    ci_upper = data.get(f"{metric_base}_ci_upper", np.nan)
    if np.isnan(mean):
        return "N/A"
    fmt = f".{decimals}f"
    ci_str = ""
    if not np.isnan(ci_lower) and not np.isnan(ci_upper):
        ci_str = f' <span class="ci-bracket">[{ci_lower:{fmt}}-{ci_upper:{fmt}}]</span>'
    return f"{mean:{fmt}}{ci_str}"


def _get_css_styles() -> str:
    """Return CSS styles for academic-style report."""
    return """
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --background-color: #ffffff;
            --text-color: #333333;
            --border-color: #dee2e6;
            --table-stripe: #f8f9fa;
        }

        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background-color: var(--background-color);
        }

        h1, h2, h3 {
            color: var(--primary-color);
            margin-top: 2em;
            margin-bottom: 0.5em;
        }

        h1 {
            font-size: 2.2em;
            border-bottom: 3px solid var(--secondary-color);
            padding-bottom: 0.3em;
        }

        h2 {
            font-size: 1.6em;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.2em;
        }

        h3 {
            font-size: 1.3em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }

        th, td {
            border: 1px solid var(--border-color);
            padding: 12px;
            text-align: left;
        }

        th {
            background-color: var(--primary-color);
            color: white;
            font-weight: bold;
        }

        tr:nth-child(even) {
            background-color: var(--table-stripe);
        }

        tr:hover {
            background-color: #e8f4f8;
        }

        .metric-value {
            font-family: 'Courier New', monospace;
        }

        .best-value {
            font-weight: bold;
            color: var(--secondary-color);
        }

        .ci-bracket {
            color: #666;
            font-size: 0.85em;
        }

        .figure-container {
            text-align: center;
            margin: 30px 0;
        }

        .figure-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
        }

        .figure-caption {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }

        .cm-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin: 20px 0;
        }

        .cm-item {
            text-align: center;
            background: #fafafa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .cm-item img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        .cm-item .figure-caption {
            margin-top: 8px;
            font-size: 0.9em;
        }

        .config-table {
            width: auto;
            min-width: 50%;
        }

        .config-table th {
            width: 40%;
        }

        .toc {
            background-color: #f8f9fa;
            border: 1px solid var(--border-color);
            padding: 20px;
            margin: 20px 0;
        }

        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }

        .toc a {
            text-decoration: none;
            color: var(--secondary-color);
        }

        .toc a:hover {
            text-decoration: underline;
        }

        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.85em;
            color: #666;
            text-align: center;
        }

        @media print {
            body {
                max-width: none;
                padding: 20px;
            }

            .figure-container {
                page-break-inside: avoid;
            }

            table {
                page-break-inside: avoid;
            }
        }
    </style>
    """


def _generate_header(config_handler: Any) -> str:
    """Generate report header."""
    data_path = getattr(config_handler.data_cfg, "data_path", "N/A")
    return f"""
    <header>
        <h1>ResPredAI Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Data Source:</strong> {_esc(data_path)}</p>
    </header>
    """


def _generate_toc(targets: list[str], has_subgroup: bool = False) -> str:
    """Generate table of contents."""
    toc_items = [
        '<li><a href="#metadata">1. Run Metadata</a></li>',
        '<li><a href="#framework-summary">2. Framework Summary</a></li>',
        '<li><a href="#results">3. Results</a>',
        "<ul>",
    ]
    for i, target in enumerate(targets, 1):
        safe_id = sanitize_name(target)
        toc_items.append(f'<li><a href="#results-{safe_id}">3.{i}. {_esc(target)}</a></li>')
    toc_items.append("</ul></li>")

    next_num = 4
    if has_subgroup:
        toc_items.append(f'<li><a href="#subgroup-analysis">{next_num}. Subgroup Analysis</a></li>')
        next_num += 1
    toc_items.extend(
        [
            f'<li><a href="#confusion-matrices">{next_num}. Confusion Matrices</a></li>',
            f'<li><a href="#calibration-diagnostics">{next_num + 1}. Calibration Diagnostics</a></li>',
        ]
    )

    return f"""
    <nav class="toc">
        <h2>Table of Contents</h2>
        <ul>
            {"".join(toc_items)}
        </ul>
    </nav>
    """


def _generate_metadata_section(config_handler: Any) -> str:
    """Generate run metadata section."""
    config_path = getattr(config_handler, "config_path", "N/A")
    data_path = getattr(config_handler.data_cfg, "data_path", "N/A")
    out_folder = getattr(config_handler.output, "out_folder", "N/A")
    seed = getattr(config_handler.reproducibility_cfg, "seed", "N/A")
    n_jobs = getattr(config_handler.reproducibility_cfg, "n_jobs", "N/A")

    return f"""
    <section id="metadata">
        <h2>1. Run Metadata</h2>
        <table class="config-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Configuration File</td><td>{_esc(config_path)}</td></tr>
            <tr><td>Data Path</td><td>{_esc(data_path)}</td></tr>
            <tr><td>Output Folder</td><td>{_esc(out_folder)}</td></tr>
            <tr><td>Random Seed</td><td>{_esc(seed)}</td></tr>
            <tr><td>Parallel Jobs</td><td>{_esc(n_jobs)}</td></tr>
            <tr><td>Report Generated</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
            <tr><td>ResPredAI Version</td><td>{__version__}</td></tr>
        </table>
    </section>
    """


def _generate_framework_summary_section(config_handler: Any) -> str:
    """Generate framework summary section with configuration table."""
    outer_folds = getattr(config_handler.pipeline, "outer_folds", "N/A")
    inner_folds = getattr(config_handler.pipeline, "inner_folds", "N/A")
    outer_cv_repeats = getattr(config_handler.pipeline, "outer_cv_repeats", 1)
    models = getattr(config_handler.pipeline, "models", [])
    targets = getattr(config_handler.data_cfg, "targets", [])
    calibrate_threshold = getattr(config_handler.pipeline, "calibrate_threshold", False)
    threshold_method = getattr(config_handler.pipeline, "threshold_method", "N/A")
    imputation_method = getattr(config_handler.imputation, "method", "none")
    imputation_strategy = getattr(config_handler.imputation, "strategy", "mean")
    imputation_n_neighbors = getattr(config_handler.imputation, "n_neighbors", 5)
    imputation_estimator = getattr(config_handler.imputation, "estimator", "bayesian_ridge")

    # Probability calibration settings
    calibrate_probabilities = getattr(config_handler.pipeline, "calibrate_probabilities", False)
    prob_calibration_method = getattr(
        config_handler.pipeline, "probability_calibration_method", "sigmoid"
    )
    prob_calibration_cv = getattr(config_handler.pipeline, "probability_calibration_cv", 5)

    models_str = ", ".join(models) if models else "N/A"
    targets_str = ", ".join(targets) if targets else "N/A"

    threshold_objective = getattr(config_handler.pipeline, "threshold_objective", "youden")
    vme_cost = getattr(config_handler.pipeline, "vme_cost", 1.0)
    me_cost = getattr(config_handler.pipeline, "me_cost", 1.0)

    # Build imputation details
    if imputation_method == "none":
        imputation_details = "Disabled"
    elif imputation_method == "simple":
        imputation_details = f"SimpleImputer (strategy: {imputation_strategy})"
    elif imputation_method == "knn":
        imputation_details = f"KNNImputer (n_neighbors: {imputation_n_neighbors})"
    elif imputation_method == "iterative":
        imputation_details = f"IterativeImputer (estimator: {imputation_estimator})"
    else:
        imputation_details = imputation_method

    # Threshold optimization details
    if calibrate_threshold:
        threshold_details = (
            f"Enabled ({threshold_method.upper()}, objective: {threshold_objective})"
        )
        if threshold_objective == "cost_sensitive":
            threshold_details += f" [VME cost: {vme_cost}, ME cost: {me_cost}]"
    else:
        threshold_details = "Disabled"

    # Probability calibration details
    if calibrate_probabilities:
        prob_calib_details = f"Enabled ({prob_calibration_method}, {prob_calibration_cv}-fold CV)"
    else:
        prob_calib_details = "Disabled"

    # Outer CV details (include repeats if > 1)
    if outer_cv_repeats > 1:
        outer_cv_details = f"{outer_folds} folds x {outer_cv_repeats} repeats = {outer_folds * outer_cv_repeats} iterations"
    else:
        outer_cv_details = str(outer_folds)

    return f"""
    <section id="framework-summary">
        <h2>2. Framework Summary</h2>
        <table class="config-table">
            <tr><th>Setting</th><th>Value</th></tr>
            <tr><td>Targets</td><td>{_esc(targets_str)}</td></tr>
            <tr><td>Models</td><td>{_esc(models_str)}</td></tr>
            <tr><td>Outer CV Folds</td><td>{_esc(outer_cv_details)}</td></tr>
            <tr><td>Inner CV Folds</td><td>{_esc(inner_folds)}</td></tr>
            <tr><td>Probability Calibration</td><td>{_esc(prob_calib_details)}</td></tr>
            <tr><td>Threshold Optimization</td><td>{_esc(threshold_details)}</td></tr>
            <tr><td>Missing Data Imputation</td><td>{_esc(imputation_details)}</td></tr>
            <tr><td>Confidence Intervals</td><td>{int(getattr(config_handler.pipeline, "confidence_level", 0.95) * 100)}% ({getattr(config_handler.pipeline, "n_bootstrap", 1000):,} bootstrap samples)</td></tr>
        </table>
    </section>
    """


def _generate_results_section(
    metrics_data: dict, models: list[str], targets: list[str], output_path: Path
) -> str:
    """Generate detailed results section with tables."""
    sections = ['<section id="results">', "<h2>3. Results</h2>"]

    for idx, target in enumerate(targets, 1):
        safe_id = sanitize_name(target)
        sections.append(f'<h3 id="results-{safe_id}">3.{idx}. {_esc(target)}</h3>')

        # Build results table
        table_rows = []
        for model in models:
            key = f"{model}_{target}"
            if key not in metrics_data:
                continue

            data = metrics_data[key]

            row = f"""
            <tr>
                <td>{_esc(model)}</td>
                <td class="metric-value">{_fmt_metric(data, "AUROC")}</td>
                <td class="metric-value">{_fmt_metric(data, "F1_weighted")}</td>
                <td class="metric-value">{_fmt_metric(data, "MCC")}</td>
                <td class="metric-value">{_fmt_metric(data, "Balanced_Acc")}</td>
                <td class="metric-value">{_fmt_metric(data, "VME")}</td>
                <td class="metric-value">{_fmt_metric(data, "ME")}</td>
            </tr>
            """
            table_rows.append(row)

        if table_rows:
            sections.append(f"""
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>AUROC [95% CI]</th>
                        <th>F1 (weighted) [95% CI]</th>
                        <th>MCC [95% CI]</th>
                        <th>Balanced Acc [95% CI]</th>
                        <th>VME [95% CI]</th>
                        <th>ME [95% CI]</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
            """)
        else:
            sections.append("<p>No results available for this target.</p>")

    sections.append("</section>")
    return "\n".join(sections)


def _generate_subgroup_section(
    output_path: Path, models: list[str], targets: list[str], section_num: int = 4
) -> str:
    """Generate subgroup analysis section from saved CSV files."""
    sg_dir = output_path / DIR_SUBGROUP
    if not sg_dir.exists():
        return ""

    sections = [
        '<section id="subgroup-analysis">',
        f"<h2>{section_num}. Subgroup Analysis</h2>",
        "<p>Performance metrics broken down by subgroup column values. "
        "Subgroups with fewer than 10 samples may have unreliable metrics.</p>",
    ]

    key_metrics = [
        "AUROC",
        "F1 (weighted)",
        "MCC",
        "Precision (1)",
        "Recall (1)",
        "Brier Score",
    ]

    found_any = False
    for target in targets:
        target_safe = sanitize_name(target)
        target_dir = sg_dir / target_safe
        if not target_dir.exists():
            continue

        csv_files = sorted(target_dir.glob("*_subgroup.csv"))
        if not csv_files:
            continue

        sections.append(f"<h3>Target: {_esc(target)}</h3>")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue

            if df.empty:
                continue

            found_any = True

            # Extract subgroup column name from filename: {model}_{sg_col}_subgroup.csv
            stem = csv_file.stem  # e.g. "LR_ward_subgroup"
            parts = stem.rsplit("_subgroup", 1)[0]  # "LR_ward"
            # Find model prefix by checking known models
            sg_label = parts
            for m in models:
                m_safe = sanitize_name(m)
                if parts.startswith(f"{m_safe}_"):
                    sg_label = parts[len(f"{m_safe}_") :]
                    sections.append(f"<h4>{_esc(m)} &mdash; {_esc(sg_label)}</h4>")
                    break
            else:
                sections.append(f"<h4>{_esc(sg_label)}</h4>")

            # Build table
            sections.append("<table><thead><tr>")
            sections.append("<th>Subgroup</th><th>N</th><th>Prevalence</th>")
            for metric in key_metrics:
                if metric in df.columns:
                    sections.append(f"<th>{_esc(metric)}</th>")
            sections.append("</tr></thead><tbody>")

            for _, row in df.iterrows():
                sections.append("<tr>")
                sections.append(f"<td><strong>{_esc(str(row.get('Subgroup', '')))}</strong></td>")
                sections.append(f"<td>{int(row.get('N', 0))}</td>")
                prev = row.get("Prevalence", np.nan)
                sections.append(f"<td>{prev:.3f}</td>" if not np.isnan(prev) else "<td>N/A</td>")
                for metric in key_metrics:
                    val = row.get(metric, np.nan)
                    if not np.isnan(val):
                        sections.append(f'<td class="metric-value">{val:.3f}</td>')
                    else:
                        sections.append("<td>N/A</td>")
                sections.append("</tr>")

            sections.append("</tbody></table>")

    if not found_any:
        return ""

    sections.append("</section>")
    return "\n".join(sections)


def _generate_confusion_matrices_section(
    output_path: Path, models: list[str], targets: list[str], section_num: int = 4
) -> str:
    """Generate confusion matrices section with responsive grid layout."""
    cm_dir = output_path / DIR_CONFUSION_MATRICES
    sections = ['<section id="confusion-matrices">', f"<h2>{section_num}. Confusion Matrices</h2>"]

    if not cm_dir.exists():
        sections.append("<p>No confusion matrix visualizations available.</p>")
        sections.append("</section>")
        return "\n".join(sections)

    found_any = False
    for model in models:
        model_safe = sanitize_name(model)
        sections.append(f"<h3>{_esc(model)}</h3>")
        sections.append('<div class="cm-grid">')

        for target in targets:
            target_safe = sanitize_name(target)
            cm_path = cm_dir / f"Confusion_matrix_{model_safe}_{target_safe}.png"

            if cm_path.exists():
                found_any = True
                with open(cm_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                sections.append(f"""
                <div class="cm-item">
                    <img src="data:image/png;base64,{img_base64}" alt="Confusion Matrix - {_esc(model)} - {_esc(target)}">
                    <p class="figure-caption">{_esc(target)}</p>
                </div>
                """)

        sections.append("</div>")  # Close cm-grid

    if not found_any:
        sections.append("<p>No confusion matrix visualizations available.</p>")

    sections.append("</section>")
    return "\n".join(sections)


def _generate_calibration_section(
    output_path: Path,
    metrics_data: dict,
    models: list[str],
    targets: list[str],
    section_num: int = 5,
) -> str:
    """Generate calibration diagnostics section with metrics and reliability curves."""
    sections = [
        '<section id="calibration-diagnostics">',
        f"<h2>{section_num}. Calibration Diagnostics</h2>",
        "<p>Calibration metrics measure how well the predicted probabilities "
        "match the observed frequencies. Lower Brier Score, ECE, and MCE indicate "
        "better calibration.</p>",
    ]

    calibration_dir = output_path / DIR_CALIBRATION

    for model in models:
        model_safe = sanitize_name(model)
        sections.append(f"<h3>{_esc(model)}</h3>")

        # Build calibration metrics table
        table_rows = []
        for target in targets:
            key = f"{model}_{target}"
            if key not in metrics_data:
                continue

            data = metrics_data[key]

            row = f"""
            <tr>
                <td>{_esc(target)}</td>
                <td class="metric-value">{_fmt_metric(data, "Brier_Score", decimals=4)}</td>
                <td class="metric-value">{_fmt_metric(data, "ECE", decimals=4)}</td>
                <td class="metric-value">{_fmt_metric(data, "MCE", decimals=4)}</td>
            </tr>
            """
            table_rows.append(row)

        if table_rows:
            sections.append("""
            <table>
                <thead>
                    <tr>
                        <th>Target</th>
                        <th>Brier Score [95% CI]</th>
                        <th>ECE [95% CI]</th>
                        <th>MCE [95% CI]</th>
                    </tr>
                </thead>
                <tbody>
            """)
            sections.append("".join(table_rows))
            sections.append("</tbody></table>")

        # Reliability curves
        sections.append("<h4>Reliability Curves</h4>")
        sections.append('<div class="cm-grid">')

        found_curves = False
        for target in targets:
            target_safe = sanitize_name(target)
            curve_path = calibration_dir / f"reliability_curve_{model_safe}_{target_safe}.png"

            if curve_path.exists():
                found_curves = True
                with open(curve_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                sections.append(f"""
                <div class="cm-item">
                    <img src="data:image/png;base64,{img_base64}"
                         alt="Reliability Curve - {_esc(model)} - {_esc(target)}">
                    <p class="figure-caption">{_esc(target)}</p>
                </div>
                """)

        if not found_curves:
            sections.append("<p>No reliability curves available.</p>")

        sections.append("</div>")  # Close cm-grid

    sections.append("</section>")
    return "\n".join(sections)


def _generate_footer() -> str:
    """Generate report footer."""
    return f"""
    <footer>
        <p>Generated by ResPredAI v{__version__}</p>
        <p>Citation: Bonazzetti, C., Rocchi, E., Toschi, A. et al.
        Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections.
        npj Digit. Med. 8, 319 (2025).</p>
    </footer>
    """


def _collect_metrics_data(
    output_path: Path,
    models: list[str],
    targets: list[str],
    filename_pattern: str = "{model}_metrics_detailed.csv",
) -> dict:
    """Collect metrics data from CSV files.

    Parameters
    ----------
    output_path : Path
        Root output directory.
    models : list[str]
        Model names.
    targets : list[str]
        Target names.
    filename_pattern : str
        CSV filename pattern with ``{model}`` placeholder.
        Use ``"{model}_metrics_detailed.csv"`` for CV results or
        ``"{model}_temporal_metrics.csv"`` for temporal validation.
    """
    metrics_data: dict = {}

    for target in targets:
        target_safe = sanitize_name(target)
        metrics_dir = output_path / DIR_METRICS / target_safe

        for model in models:
            model_safe = sanitize_name(model)
            metrics_file = metrics_dir / filename_pattern.format(model=model_safe)

            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    key = f"{model}_{target}"
                    metrics_data[key] = {}

                    for _, row in df.iterrows():
                        metric_name = sanitize_metric_name(row["Metric"])
                        metrics_data[key][f"{metric_name}_mean"] = row["Mean"]
                        if "Std" in df.columns:
                            metrics_data[key][f"{metric_name}_std"] = row["Std"]
                        if "SE" in df.columns:
                            metrics_data[key][f"{metric_name}_se"] = row["SE"]
                        if "CI95_lower" in df.columns:
                            metrics_data[key][f"{metric_name}_ci_lower"] = row["CI95_lower"]
                            metrics_data[key][f"{metric_name}_ci_upper"] = row["CI95_upper"]
                except Exception as exc:
                    logger.debug("Failed to parse metrics file %s: %s", metrics_file, exc)
                    continue

    return metrics_data


def _generate_temporal_section(temporal_data: dict, models: list[str], targets: list[str]) -> str:
    """Generate HTML section for temporal validation results."""
    if not temporal_data:
        return ""

    html = ['<div class="section" id="temporal-validation">']
    html.append("<h2>Temporal (Prospective-Style) Validation</h2>")
    html.append(
        "<p>Models were trained on historical data and evaluated on prospective data "
        "using a temporal cutoff. This simulates real-world deployment conditions.</p>"
    )

    key_metrics = [
        ("AUROC_mean", "AUROC"),
        ("F1_weighted_mean", "F1 (weighted)"),
        ("MCC_mean", "MCC"),
        ("Balanced_Acc_mean", "Balanced Acc"),
        ("Brier_Score_mean", "Brier Score"),
    ]

    for target in targets:
        html.append(f"<h3>Target: {_esc(target)}</h3>")
        html.append("<table><thead><tr><th>Model</th>")
        for _, display_name in key_metrics:
            html.append(f"<th>{_esc(display_name)}</th>")
        html.append("</tr></thead><tbody>")

        for model in models:
            key = f"{model}_{target}"
            if key not in temporal_data:
                continue

            data = temporal_data[key]
            html.append(f"<tr><td><strong>{_esc(model)}</strong></td>")

            for metric_key, _ in key_metrics:
                val = data.get(metric_key)
                if val is not None and not np.isnan(val):
                    ci_lower = data.get(metric_key.replace("_mean", "_ci_lower"))
                    ci_upper = data.get(metric_key.replace("_mean", "_ci_upper"))
                    if ci_lower is not None and ci_upper is not None:
                        html.append(f"<td>{val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]</td>")
                    else:
                        html.append(f"<td>{val:.3f}</td>")
                else:
                    html.append("<td>-</td>")

            html.append("</tr>")

        html.append("</tbody></table>")

    html.append("</div>")
    return "\n".join(html)


def generate_html_report(
    output_folder: str,
    models: list[str],
    targets: list[str],
    config_handler: Any,
    output_filename: str = "report.html",
) -> Path:
    """
    Generate comprehensive HTML report in academic style.

    Parameters
    ----------
    output_folder : str
        Path to output folder containing results
    models : List[str]
        List of model names
    targets : List[str]
        List of target names
    config_handler : Any
        Configuration handler with run parameters
    output_filename : str
        Name of output HTML file

    Returns
    -------
    Path
        Path to generated HTML report
    """
    output_path = Path(output_folder)

    # Collect all data
    metrics_data = _collect_metrics_data(output_path, models, targets)
    temporal_data = _collect_metrics_data(
        output_path, models, targets, filename_pattern="{model}_temporal_metrics.csv"
    )

    # Generate subgroup section (may be empty if no subgroup data)
    has_subgroup = (output_path / DIR_SUBGROUP).exists()
    subgroup_html = _generate_subgroup_section(output_path, models, targets, section_num=4)
    if not subgroup_html:
        has_subgroup = False
    cm_num = 5 if has_subgroup else 4
    calib_num = cm_num + 1

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>ResPredAI Analysis Report - {datetime.now().strftime('%Y-%m-%d')}</title>",
        _get_css_styles(),
        "</head>",
        "<body>",
        _generate_header(config_handler),
        _generate_toc(targets, has_subgroup=has_subgroup),
        _generate_metadata_section(config_handler),
        _generate_framework_summary_section(config_handler),
        _generate_results_section(metrics_data, models, targets, output_path),
        _generate_temporal_section(temporal_data, models, targets),
        subgroup_html,
        _generate_confusion_matrices_section(output_path, models, targets, section_num=cm_num),
        _generate_calibration_section(
            output_path, metrics_data, models, targets, section_num=calib_num
        ),
        _generate_footer(),
        "</body>",
        "</html>",
    ]

    html_content = "\n".join(html_parts)

    # Write report
    report_path = output_path / output_filename
    report_path.write_text(html_content, encoding="utf-8")

    return report_path
