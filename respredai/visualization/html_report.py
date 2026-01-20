"""HTML report generation for ResPredAI results."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from respredai import __version__


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
    data_path = getattr(config_handler, "data_path", "N/A")
    return f"""
    <header>
        <h1>ResPredAI Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Data Source:</strong> {data_path}</p>
    </header>
    """


def _generate_toc(targets: List[str]) -> str:
    """Generate table of contents."""
    toc_items = [
        '<li><a href="#metadata">1. Run Metadata</a></li>',
        '<li><a href="#framework-summary">2. Framework Summary</a></li>',
        '<li><a href="#results">3. Results</a>',
        "<ul>",
    ]
    for i, target in enumerate(targets, 1):
        safe_id = target.replace(" ", "_")
        toc_items.append(f'<li><a href="#results-{safe_id}">3.{i}. {target}</a></li>')
    toc_items.extend(
        [
            "</ul></li>",
            '<li><a href="#confusion-matrices">4. Confusion Matrices</a></li>',
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
    data_path = getattr(config_handler, "data_path", "N/A")
    out_folder = getattr(config_handler, "out_folder", "N/A")
    seed = getattr(config_handler, "seed", "N/A")
    n_jobs = getattr(config_handler, "n_jobs", "N/A")

    return f"""
    <section id="metadata">
        <h2>1. Run Metadata</h2>
        <table class="config-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Configuration File</td><td>{config_path}</td></tr>
            <tr><td>Data Path</td><td>{data_path}</td></tr>
            <tr><td>Output Folder</td><td>{out_folder}</td></tr>
            <tr><td>Random Seed</td><td>{seed}</td></tr>
            <tr><td>Parallel Jobs</td><td>{n_jobs}</td></tr>
            <tr><td>Report Generated</td><td>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
            <tr><td>ResPredAI Version</td><td>{__version__}</td></tr>
        </table>
    </section>
    """


def _generate_framework_summary_section(config_handler: Any) -> str:
    """Generate framework summary section with configuration table."""
    outer_folds = getattr(config_handler, "outer_folds", "N/A")
    inner_folds = getattr(config_handler, "inner_folds", "N/A")
    models = getattr(config_handler, "models", [])
    targets = getattr(config_handler, "targets", [])
    calibrate_threshold = getattr(config_handler, "calibrate_threshold", False)
    threshold_method = getattr(config_handler, "threshold_method", "N/A")
    imputation_method = getattr(config_handler, "imputation_method", "none")
    imputation_strategy = getattr(config_handler, "imputation_strategy", "mean")
    imputation_n_neighbors = getattr(config_handler, "imputation_n_neighbors", 5)
    imputation_estimator = getattr(config_handler, "imputation_estimator", "bayesian_ridge")

    models_str = ", ".join(models) if models else "N/A"
    targets_str = ", ".join(targets) if targets else "N/A"

    # Get threshold objective settings
    threshold_objective = getattr(config_handler, "threshold_objective", "youden")
    vme_cost = getattr(config_handler, "vme_cost", 1.0)
    me_cost = getattr(config_handler, "me_cost", 1.0)

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

    # Threshold calibration details
    if calibrate_threshold:
        threshold_details = (
            f"Enabled ({threshold_method.upper()}, objective: {threshold_objective})"
        )
        if threshold_objective == "cost_sensitive":
            threshold_details += f" [VME cost: {vme_cost}, ME cost: {me_cost}]"
    else:
        threshold_details = "Disabled"

    return f"""
    <section id="framework-summary">
        <h2>2. Framework Summary</h2>
        <table class="config-table">
            <tr><th>Setting</th><th>Value</th></tr>
            <tr><td>Targets</td><td>{targets_str}</td></tr>
            <tr><td>Models</td><td>{models_str}</td></tr>
            <tr><td>Outer CV Folds</td><td>{outer_folds}</td></tr>
            <tr><td>Inner CV Folds</td><td>{inner_folds}</td></tr>
            <tr><td>Threshold Calibration</td><td>{threshold_details}</td></tr>
            <tr><td>Missing Data Imputation</td><td>{imputation_details}</td></tr>
            <tr><td>Confidence Intervals</td><td>95% (1,000 bootstrap samples)</td></tr>
        </table>
    </section>
    """


def _generate_results_section(
    metrics_data: Dict, models: List[str], targets: List[str], output_path: Path
) -> str:
    """Generate detailed results section with tables."""
    sections = ['<section id="results">', "<h2>3. Results</h2>"]

    for idx, target in enumerate(targets, 1):
        safe_id = target.replace(" ", "_")
        sections.append(f'<h3 id="results-{safe_id}">3.{idx}. {target}</h3>')

        # Build results table
        table_rows = []
        for model in models:
            key = f"{model}_{target}"
            if key not in metrics_data:
                continue

            data = metrics_data[key]

            def fmt_val(mean_key, ci_lower_key, ci_upper_key, std_key=None):
                mean = data.get(mean_key, np.nan)
                ci_lower = data.get(ci_lower_key, np.nan)
                ci_upper = data.get(ci_upper_key, np.nan)
                if np.isnan(mean):
                    return "N/A"
                ci_str = ""
                if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                    ci_str = f' <span class="ci-bracket">[{ci_lower:.3f}-{ci_upper:.3f}]</span>'
                return f"{mean:.3f}{ci_str}"

            row = f"""
            <tr>
                <td>{model}</td>
                <td class="metric-value">{fmt_val("AUROC_mean", "AUROC_ci_lower", "AUROC_ci_upper")}</td>
                <td class="metric-value">{fmt_val("F1_weighted_mean", "F1_weighted_ci_lower", "F1_weighted_ci_upper")}</td>
                <td class="metric-value">{fmt_val("MCC_mean", "MCC_ci_lower", "MCC_ci_upper")}</td>
                <td class="metric-value">{fmt_val("Balanced_Acc_mean", "Balanced_Acc_ci_lower", "Balanced_Acc_ci_upper")}</td>
                <td class="metric-value">{fmt_val("VME_mean", "VME_ci_lower", "VME_ci_upper")}</td>
                <td class="metric-value">{fmt_val("ME_mean", "ME_ci_lower", "ME_ci_upper")}</td>
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


def _generate_confusion_matrices_section(
    output_path: Path, models: List[str], targets: List[str]
) -> str:
    """Generate confusion matrices section with responsive grid layout."""
    cm_dir = output_path / "confusion_matrices"
    sections = ['<section id="confusion-matrices">', "<h2>4. Confusion Matrices</h2>"]

    if not cm_dir.exists():
        sections.append("<p>No confusion matrix visualizations available.</p>")
        sections.append("</section>")
        return "\n".join(sections)

    found_any = False
    for model in models:
        model_safe = model.replace(" ", "_")
        sections.append(f"<h3>{model}</h3>")
        sections.append('<div class="cm-grid">')

        for target in targets:
            target_safe = target.replace(" ", "_")
            cm_path = cm_dir / f"Confusion_matrix_{model_safe}_{target_safe}.png"

            if cm_path.exists():
                found_any = True
                with open(cm_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                sections.append(f"""
                <div class="cm-item">
                    <img src="data:image/png;base64,{img_base64}" alt="Confusion Matrix - {model} - {target}">
                    <p class="figure-caption">{target}</p>
                </div>
                """)

        sections.append("</div>")  # Close cm-grid

    if not found_any:
        sections.append("<p>No confusion matrix visualizations available.</p>")

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


def _collect_metrics_data(output_path: Path, models: List[str], targets: List[str]) -> Dict:
    """Collect all metrics data from CSV files."""
    metrics_data = {}

    for target in targets:
        target_safe = target.replace(" ", "_")
        metrics_dir = output_path / "metrics" / target_safe

        for model in models:
            model_safe = model.replace(" ", "_")
            metrics_file = metrics_dir / f"{model_safe}_metrics_detailed.csv"

            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    key = f"{model}_{target}"
                    metrics_data[key] = {}

                    for _, row in df.iterrows():
                        metric_name = (
                            row["Metric"].replace(" ", "_").replace("(", "").replace(")", "")
                        )
                        metrics_data[key][f"{metric_name}_mean"] = row["Mean"]
                        metrics_data[key][f"{metric_name}_std"] = row["Std"]
                        if "CI95_lower" in df.columns:
                            metrics_data[key][f"{metric_name}_ci_lower"] = row["CI95_lower"]
                            metrics_data[key][f"{metric_name}_ci_upper"] = row["CI95_upper"]
                except Exception:
                    continue

    return metrics_data


def generate_html_report(
    output_folder: str,
    models: List[str],
    targets: List[str],
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
        _generate_toc(targets),
        _generate_metadata_section(config_handler),
        _generate_framework_summary_section(config_handler),
        _generate_results_section(metrics_data, models, targets, output_path),
        _generate_confusion_matrices_section(output_path, models, targets),
        _generate_footer(),
        "</body>",
        "</html>",
    ]

    html_content = "\n".join(html_parts)

    # Write report
    report_path = output_path / output_filename
    report_path.write_text(html_content, encoding="utf-8")

    return report_path
