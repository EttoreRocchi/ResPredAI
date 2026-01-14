"""Command-line interface for ResPredAI."""

from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from respredai import __version__
from respredai.core.pipeline import perform_evaluation, perform_pipeline, perform_training
from respredai.io.config import ConfigHandler, DataSetter
from respredai.visualization.feature_importance import process_feature_importance

app = typer.Typer(
    name="respredai",
    help="ResPredAI - Antimicrobial Resistance Prediction via AI",
    add_completion=False,
)
console = Console()


def print_banner() -> None:
    """Print the ResPredAI banner."""
    banner_text = f"""
    ResPredAI
    Version: {__version__}
    Antimicrobial Resistance predictions via AI models
    """
    console.print(Panel(banner_text, title="ResPredAI", border_style="cyan"))


def print_config_info(config_handler: ConfigHandler) -> None:
    """Print configuration information in a table."""
    table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("Data Path", str(config_handler.data_path))
    table.add_row("Targets", ", ".join(config_handler.targets))
    table.add_row(
        "Group Column", str(config_handler.group_column) if config_handler.group_column else "None"
    )
    table.add_row("Models", ", ".join(config_handler.models))
    table.add_row("Outer Folds", str(config_handler.outer_folds))
    table.add_row("Inner Folds", str(config_handler.inner_folds))
    table.add_row("Threshold Calibration", str(config_handler.calibrate_threshold))
    if config_handler.calibrate_threshold:
        table.add_row("Threshold Method", config_handler.threshold_method.upper())
    table.add_row("Random Seed", str(config_handler.seed))
    table.add_row("Parallel Jobs", str(config_handler.n_jobs))
    table.add_row("Output Folder", str(config_handler.out_folder))
    table.add_row("Model Saving Enabled", str(config_handler.save_models_enable))

    console.print(table)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"ResPredAI version {__version__}", style="bold cyan")
        raise typer.Exit()


@app.callback()
def callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    ResPredAI - Antimicrobial Resistance Prediction via AI

    A machine learning pipeline for predicting antimicrobial resistance
    in Gram-negative bloodstream infections.
    """
    pass


class TrainingProgressCallback:
    """Callback class to handle training progress updates for nested CV."""

    def __init__(self, console: Console, quiet: bool = False) -> None:
        self.console: Console = console
        self.quiet: bool = quiet
        self.progress: Optional[Progress] = None
        self.overall_task: Optional[int] = None
        self.model_task: Optional[int] = None
        self.target_task: Optional[int] = None
        self.fold_task: Optional[int] = None

    def start(self, total_work: int) -> None:
        """Start the progress tracking."""
        if self.quiet:
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.progress.start()
        self.overall_task = self.progress.add_task("[cyan]Overall Progress", total=total_work)

    def start_model(self, model_name: str, total_work: int) -> None:
        """Start tracking a new model."""
        if self.quiet or not self.progress:
            return

        if self.model_task is not None:
            self.progress.remove_task(self.model_task)

        self.model_task = self.progress.add_task(f"[green]Model: {model_name}", total=total_work)

    def start_target(self, target_name: str, total_folds: int, resumed_from: int = 0) -> None:
        """Start tracking a new target."""
        if self.quiet or not self.progress:
            return

        if self.target_task is not None:
            self.progress.remove_task(self.target_task)

        status = f"[yellow]Target: {target_name}"
        if resumed_from > 0:
            status += f" (resumed from fold {resumed_from + 1})"

        self.target_task = self.progress.add_task(status, total=total_folds, completed=resumed_from)

    def start_fold(self, fold_num: int, total_folds: int) -> None:
        """Start tracking a new fold."""
        if self.quiet or not self.progress:
            return

        if self.fold_task is not None:
            self.progress.remove_task(self.fold_task)

        self.fold_task = self.progress.add_task(
            f"[blue]  Fold {fold_num}/{total_folds}: Training...", total=100
        )

    def update_fold_status(self, status: str, progress: Optional[int] = None) -> None:
        """Update fold status."""
        if self.quiet or not self.progress or self.fold_task is None:
            return

        self.progress.update(self.fold_task, description=f"[blue]  {status}")
        if progress is not None:
            self.progress.update(self.fold_task, completed=progress)

    def complete_fold(self, fold_num: int, metrics: Dict[str, Any]) -> None:
        """Complete a fold and display metrics."""
        if self.quiet or not self.progress:
            return

        if self.fold_task is not None:
            self.progress.remove_task(self.fold_task)
            self.fold_task = None

        # Advance all progress levels
        if self.target_task is not None:
            self.progress.advance(self.target_task)
        if self.model_task is not None:
            self.progress.advance(self.model_task)
        if self.overall_task is not None:
            self.progress.advance(self.overall_task)

        # Print fold metrics
        metrics_str = f"Fold {fold_num}: "
        metrics_str += f"F1={metrics.get('F1 (weighted)', 0):.3f}, "
        metrics_str += f"MCC={metrics.get('MCC', 0):.3f}, "
        metrics_str += f"AUROC={metrics.get('AUROC', 0):.3f}"
        self.console.print(f"    [dim]{metrics_str}[/dim]")

    def complete_target(self, target_name: str, summary_metrics: Dict[str, Any]) -> None:
        """Complete a target and display summary."""
        if self.quiet or not self.progress:
            return

        if self.target_task is not None:
            self.progress.remove_task(self.target_task)
            self.target_task = None

        # Print target summary
        self.console.print(f"\n  [bold green]✓[/bold green] Completed {target_name}")
        summary_str = "    "
        summary_str += f"F1={summary_metrics.get('F1 (weighted)', 0):.3f}±{summary_metrics.get('F1_std', 0):.3f}, "
        summary_str += (
            f"MCC={summary_metrics.get('MCC', 0):.3f}±{summary_metrics.get('MCC_std', 0):.3f}, "
        )
        summary_str += (
            f"AUROC={summary_metrics.get('AUROC', 0):.3f}±{summary_metrics.get('AUROC_std', 0):.3f}"
        )
        self.console.print(f"[cyan]{summary_str}[/cyan]\n")

    def complete_model(self, model_name: str) -> None:
        """Complete a model."""
        if self.quiet or not self.progress:
            return

        if self.model_task is not None:
            self.progress.remove_task(self.model_task)
            self.model_task = None

        self.console.print(f"\n[bold green]✓ Completed model: {model_name}[/bold green]\n\n")

    def skip_target(self, target_name: str, num_folds: int, reason: str = "saved models") -> None:
        """Skip a target (loaded from saved models)."""
        if self.quiet or not self.progress:
            return

        # Advance model and overall progress by the number of folds that were skipped
        if self.model_task is not None:
            self.progress.advance(self.model_task, advance=num_folds)
        if self.overall_task is not None:
            self.progress.advance(self.overall_task, advance=num_folds)

        self.console.print(f"  [dim]✓ Skipped {target_name} (loaded from {reason})[/dim]")

    def skip_model(self, model_name: str, total_work_skipped: int, reason: str = "error") -> None:
        """Skip a model (failed to initialize)."""
        if self.quiet or not self.progress:
            return

        # Remove model task if it exists
        if self.model_task is not None:
            self.progress.remove_task(self.model_task)
            self.model_task = None

        # Advance overall progress by all the work that would have been done
        if self.overall_task is not None:
            self.progress.advance(self.overall_task, advance=total_work_skipped)

        self.console.print(f"[dim]✓ Skipped model: {model_name} ({reason})[/dim]")

    def stop(self) -> None:
        """Stop the progress tracking."""
        if self.quiet or not self.progress:
            return

        self.progress.stop()


class SimpleTrainingProgressCallback:
    """Callback for train command (no fold-level tracking needed)."""

    def __init__(self, console: Console, quiet: bool = False) -> None:
        self.console: Console = console
        self.quiet: bool = quiet
        self.progress: Optional[Progress] = None
        self.overall_task: Optional[int] = None
        self.model_task: Optional[int] = None
        self.target_task: Optional[int] = None

    def start(self, total_models: int, total_targets: int, total_folds: int = 1) -> None:
        """Start the progress tracking."""
        if self.quiet:
            return

        total_work = total_models * total_targets
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.progress.start()
        self.overall_task = self.progress.add_task("[cyan]Overall Progress", total=total_work)

    def start_model(self, model_name: str) -> None:
        """Start tracking a new model."""
        if self.quiet or not self.progress:
            return

        if self.model_task is not None:
            self.progress.remove_task(self.model_task)

        self.model_task = self.progress.add_task(f"[green]Model: {model_name}", total=None)

    def start_target(self, target_name: str) -> None:
        """Start tracking a new target."""
        if self.quiet or not self.progress:
            return

        if self.target_task is not None:
            self.progress.remove_task(self.target_task)

        self.target_task = self.progress.add_task(f"[yellow]  Target: {target_name}", total=None)

    def complete_target(self, target_name: str, info: Dict[str, Any]) -> None:
        """Complete a target."""
        if self.quiet or not self.progress:
            return

        if self.target_task is not None:
            self.progress.remove_task(self.target_task)
            self.target_task = None

        if self.overall_task is not None:
            self.progress.advance(self.overall_task)

        threshold = info.get("threshold", 0.5)
        self.console.print(
            f"  [bold green]✓[/bold green] {target_name} [dim](threshold: {threshold:.3f})[/dim]"
        )

    def complete_model(self, model_name: str) -> None:
        """Complete a model."""
        if self.quiet or not self.progress:
            return

        if self.model_task is not None:
            self.progress.remove_task(self.model_task)
            self.model_task = None

        self.console.print(f"[bold green]✓ Completed: {model_name}[/bold green]\n")

    def stop(self) -> None:
        """Stop the progress tracking."""
        if self.quiet or not self.progress:
            return

        self.progress.stop()


def _load_config_with_error_handling(config_path: Path) -> ConfigHandler:
    """Load configuration with user-friendly error handling."""
    if not config_path.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Configuration file not found: [cyan]{config_path}[/cyan]\n\n"
            f"[dim]Hint: Create a config file with:[/dim] respredai create-config {config_path}",
            style="red",
        )
        raise typer.Exit(code=1)

    if config_path.suffix.lower() != ".ini":
        console.print(
            f"\n[bold red]Error:[/bold red] Configuration file must have .ini extension, "
            f"got: [cyan]{config_path.suffix}[/cyan]",
            style="red",
        )
        raise typer.Exit(code=1)

    try:
        return ConfigHandler(str(config_path))
    except FileNotFoundError as e:
        console.print(
            f"\n[bold red]Error:[/bold red] {str(e)}\n\n"
            f"[dim]Check that your data_path in the config file points to an existing file.[/dim]",
            style="red",
        )
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"\n[bold red]Configuration Error:[/bold red] {str(e)}", style="red")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Error loading configuration:[/bold red] {str(e)}", style="red")
        raise typer.Exit(code=1)


def _load_data_with_error_handling(config_handler: ConfigHandler) -> DataSetter:
    """Load data with user-friendly error handling."""
    data_path = Path(config_handler.data_path)

    if not data_path.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Data file not found: [cyan]{data_path}[/cyan]\n\n"
            f"[dim]Check that 'data_path' in your config file points to an existing CSV file.[/dim]",
            style="red",
        )
        raise typer.Exit(code=1)

    try:
        return DataSetter(config_handler)
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}", style="red")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"\n[bold red]Data Error:[/bold red] {str(e)}", style="red")
        raise typer.Exit(code=1)
    except AssertionError as e:
        console.print(
            f"\n[bold red]Data Validation Error:[/bold red] {str(e)}\n\n"
            f"[dim]Check your data for missing values or invalid entries.[/dim]",
            style="red",
        )
        raise typer.Exit(code=1)


@app.command(rich_help_panel="Configuration")
def validate_config(
    config: Path = typer.Argument(
        ...,
        help="Path to the configuration file (.ini format)",
    ),
    check_data: bool = typer.Option(
        False,
        "--check-data",
        "-d",
        help="Also validate that the data file exists and can be loaded",
    ),
):
    """Validate a configuration file without running the pipeline."""

    console.print(f"\n[bold cyan]Validating configuration: {config}[/bold cyan]\n")

    config_handler = _load_config_with_error_handling(config)
    console.print("[bold green]✓[/bold green] Configuration file syntax is valid")
    console.print("[bold green]✓[/bold green] All required parameters present")
    console.print("[bold green]✓[/bold green] Parameter values are valid")

    print_config_info(config_handler)

    if check_data:
        console.print("\n[bold cyan]Checking data file...[/bold cyan]")
        datasetter = _load_data_with_error_handling(config_handler)
        console.print(
            f"[bold green]✓[/bold green] Data loaded successfully: "
            f"{datasetter.X.shape[0]} samples, {datasetter.X.shape[1]} features"
        )
        console.print(f"[bold green]✓[/bold green] Targets: {', '.join(datasetter.targets)}")
        if config_handler.group_column:
            n_groups = len(set(datasetter.groups))
            console.print(f"[bold green]✓[/bold green] Groups: {n_groups} unique groups")

    console.print(
        Panel(
            "[bold green]✓ Configuration is valid![/bold green]",
            title="Validation Passed",
            border_style="green",
        )
    )


@app.command(rich_help_panel="Pipeline")
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration file (.ini format)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress banner and progress output"),
    # CLI overrides
    models: Optional[str] = typer.Option(
        None, "--models", "-m", help="Override models (comma-separated, e.g., 'LR,RF,XGB')"
    ),
    targets: Optional[str] = typer.Option(
        None, "--targets", "-t", help="Override targets (comma-separated)"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output folder"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override random seed"),
):
    """
    Run the machine learning pipeline with the specified configuration.

    Configuration options can be overridden via CLI flags without editing the file.

    Example:
        respredai run --config my_config.ini
        respredai run --config my_config.ini --models LR,RF --output ./new_output/
        respredai run --config my_config.ini --seed 123
    """
    print_banner()

    # Load configuration with error handling
    config_handler = _load_config_with_error_handling(config)

    # Apply CLI overrides
    if models:
        config_handler.models = [m.strip() for m in models.split(",")]
        console.print(f"[dim]Override: models = {', '.join(config_handler.models)}[/dim]")
    if targets:
        config_handler.targets = [t.strip() for t in targets.split(",")]
        console.print(f"[dim]Override: targets = {', '.join(config_handler.targets)}[/dim]")
    if output:
        config_handler.out_folder = str(output)
        console.print(f"[dim]Override: output = {config_handler.out_folder}[/dim]")
    if seed is not None:
        config_handler.seed = seed
        console.print(f"[dim]Override: seed = {config_handler.seed}[/dim]")

    if not quiet:
        console.print("\n[bold green]✓[/bold green] Configuration loaded successfully\n")
        print_config_info(config_handler)

    # Load data with error handling
    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    datasetter = _load_data_with_error_handling(config_handler)
    console.print(
        f"[bold green]✓[/bold green] Data loaded: {datasetter.X.shape[0]} samples, "
        f"{datasetter.X.shape[1]} features"
    )

    # Create output directory
    Path(config_handler.out_folder).mkdir(parents=True, exist_ok=True)

    # Create progress callback
    progress_callback = TrainingProgressCallback(console, quiet)

    # Run pipeline
    console.print("\n[bold cyan]Starting model training pipeline...[/bold cyan]\n")
    try:
        perform_pipeline(
            datasetter=datasetter,
            models=config_handler.models,
            config_handler=config_handler,
            progress_callback=progress_callback,
        )
    except Exception as e:
        console.print(f"\n[bold red]Pipeline Error:[/bold red] {str(e)}")
        if not quiet:
            console.print_exception()
        raise typer.Exit(code=1)

    # Success message
    success_panel = Panel(
        f"[bold green]✓ Pipeline completed successfully![/bold green]\n\n"
        f"Results saved to: [cyan]{config_handler.out_folder}[/cyan]",
        title="Success",
        border_style="green",
    )
    console.print("\n", success_panel)


@app.command(rich_help_panel="Cross-datasets")
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration file (.ini format)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress banner and progress output"),
    models: Optional[str] = typer.Option(
        None, "--models", "-m", help="Override models (comma-separated, e.g., 'LR,RF,XGB')"
    ),
    targets: Optional[str] = typer.Option(
        None, "--targets", "-t", help="Override targets (comma-separated)"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output folder"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override random seed"),
):
    """
    Train models on entire dataset using GridSearchCV for hyperparameter tuning.

    Saves one model file per model-target combination to trained_models/ directory.
    Use the 'evaluate' command to apply trained models to new data.

    Example:
        respredai train --config my_config.ini
        respredai train --config my_config.ini --models LR,RF
    """
    print_banner()

    config_handler = _load_config_with_error_handling(config)

    # Apply CLI overrides
    if models:
        config_handler.models = [m.strip() for m in models.split(",")]
        console.print(f"[dim]Override: models = {', '.join(config_handler.models)}[/dim]")
    if targets:
        config_handler.targets = [t.strip() for t in targets.split(",")]
        console.print(f"[dim]Override: targets = {', '.join(config_handler.targets)}[/dim]")
    if output:
        config_handler.out_folder = str(output)
        console.print(f"[dim]Override: output = {config_handler.out_folder}[/dim]")
    if seed is not None:
        config_handler.seed = seed
        console.print(f"[dim]Override: seed = {config_handler.seed}[/dim]")

    if not quiet:
        console.print("\n[bold green]✓[/bold green] Configuration loaded successfully\n")
        print_config_info(config_handler)

    console.print("\n[bold cyan]Loading data...[/bold cyan]")
    datasetter = _load_data_with_error_handling(config_handler)
    console.print(
        f"[bold green]✓[/bold green] Data loaded: {datasetter.X.shape[0]} samples, "
        f"{datasetter.X.shape[1]} features"
    )

    Path(config_handler.out_folder).mkdir(parents=True, exist_ok=True)

    progress_callback = SimpleTrainingProgressCallback(console, quiet) if not quiet else None

    console.print("\n[bold cyan]Starting model training...[/bold cyan]\n")
    try:
        perform_training(
            datasetter=datasetter,
            models=config_handler.models,
            config_handler=config_handler,
            progress_callback=progress_callback,
        )
    except Exception as e:
        console.print(f"\n[bold red]Training Error:[/bold red] {str(e)}")
        if not quiet:
            console.print_exception()
        raise typer.Exit(code=1)

    trained_models_dir = Path(config_handler.out_folder) / "trained_models"
    success_panel = Panel(
        f"[bold green]✓ Training completed successfully![/bold green]\n\n"
        f"Models saved to: [cyan]{trained_models_dir}[/cyan]\n"
        f"Metadata saved to: [cyan]{trained_models_dir / 'training_metadata.json'}[/cyan]",
        title="Success",
        border_style="green",
    )
    console.print("\n", success_panel)


@app.command(rich_help_panel="Cross-datasets")
def evaluate(
    models_dir: Path = typer.Option(
        ...,
        "--models-dir",
        "-m",
        help="Directory containing trained models and training_metadata.json",
    ),
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to new data CSV file (must include target columns for ground truth)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory for evaluation results",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output"),
):
    """
    Evaluate trained models on new data with ground truth.

    Requires models trained with 'train' command. New data must have the same
    features and target columns as training data.

    Example:
        respredai evaluate --models-dir ./output/trained_models --data new_data.csv --output ./eval_output
    """
    print_banner()

    # Validate inputs
    if not models_dir.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Models directory not found: [cyan]{models_dir}[/cyan]",
            style="red",
        )
        raise typer.Exit(code=1)

    metadata_path = models_dir / "training_metadata.json"
    if not metadata_path.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Training metadata not found: [cyan]{metadata_path}[/cyan]\n\n"
            f"[dim]Ensure this directory was created by 'respredai train'[/dim]",
            style="red",
        )
        raise typer.Exit(code=1)

    if not data.exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Data file not found: [cyan]{data}[/cyan]", style="red"
        )
        raise typer.Exit(code=1)

    console.print(f"\n[bold cyan]Loading models from:[/bold cyan] {models_dir}")
    console.print(f"[bold cyan]Evaluating on data:[/bold cyan] {data}")
    console.print(f"[bold cyan]Output directory:[/bold cyan] {output}\n")

    try:
        results = perform_evaluation(
            models_dir=models_dir, data_path=data, output_dir=output, verbose=not quiet
        )
    except ValueError as e:
        console.print(f"\n[bold red]Validation Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Evaluation Error:[/bold red] {str(e)}")
        if not quiet:
            console.print_exception()
        raise typer.Exit(code=1)

    n_models = len(results)
    output_path = Path(output)
    success_panel = Panel(
        f"[bold green]✓ Evaluation completed successfully![/bold green]\n\n"
        f"Evaluated [cyan]{n_models}[/cyan] model-target combinations\n\n"
        f"Results saved to:\n"
        f"  • Metrics: [cyan]{output_path / 'metrics'}[/cyan]\n"
        f"  • Predictions: [cyan]{output_path / 'predictions'}[/cyan]\n"
        f"  • Summary: [cyan]{output_path / 'evaluation_summary.csv'}[/cyan]",
        title="Success",
        border_style="green",
    )
    console.print("\n", success_panel)


@app.command(rich_help_panel="Info")
def list_models():
    """
    List all available machine learning models.

    Example:
        respredai list-models
    """

    models_info = [
        ("LR", "Logistic Regression"),
        ("MLP", "Multi-Layer Perceptron"),
        ("XGB", "XGBoost"),
        ("RF", "Random Forest"),
        ("CatBoost", "CatBoost"),
        ("TabPFN", "TabPFN"),
        ("RBF_SVC", "RBF SVM"),
        ("Linear_SVC", "Linear SVM"),
        ("KNN", "K-Nearest Neighbors"),
    ]

    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Code", style="cyan", width=15)
    table.add_column("Name", style="green", width=25)

    for code, name in models_info:
        table.add_row(code, name)

    console.print("\n")
    console.print(table)
    console.print(
        "\n[dim]Use these codes in your config file under the 'models' parameter.[/dim]\n"
    )


@app.command(rich_help_panel="Configuration")
def create_config(
    output_path: Path = typer.Argument(
        ..., help="Path where the template configuration file will be created"
    ),
):
    """
    Create a template configuration file.

    Example:
        respredai create-config my_config.ini
    """

    # Validate file extension
    if output_path.suffix.lower() != ".ini":
        console.print(
            "[bold red]Error:[/bold red] Configuration file must have .ini extension", style="red"
        )
        raise typer.Exit(code=1)

    template = """[Data]
data_path = ./data/your_data.csv
targets = Target1,Target2
continuous_features = Feature1,Feature2,Feature3
# group_column = PatientID  # Optional: column for grouping samples to prevent data leakage

[Pipeline]
models = LR,RF,XGB
outer_folds = 5
inner_folds = 3
calibrate_threshold = false
threshold_method = auto

[Reproducibility]
seed = 42

[Log]
verbosity = 1
log_basename = respredai.log

[Resources]
n_jobs = -1

[Output]
out_folder = ./output/

[ModelSaving]
enable = true
compression = 3
"""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template)
        console.print(
            f"[bold green]✓[/bold green] Template configuration created: [cyan]{output_path}[/cyan]"
        )
        console.print("\n[dim]Edit this file with your data paths and parameters.[/dim]\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Could not create config file: {str(e)}")
        raise typer.Exit(code=1)


@app.command(rich_help_panel="Info")
def info():
    """
    Display information about ResPredAI.

    Example:
        respredai info
    """

    info_text = f"""
    [bold cyan]ResPredAI v{__version__}[/bold cyan]

    [bold]Description:[/bold]
    Antimicrobial Resistance predictions via AI models

    [bold]Citation:[/bold]
    Bonazzetti, C., Rocchi, E., Toschi, A. et al.
    Artificial Intelligence model to predict resistances in
    Gram-negative bloodstream infections.
    npj Digit. Med. 8, 319 (2025).
    https://doi.org/10.1038/s41746-025-01696-x

    [bold]Funding:[/bold]
    EU NextGenerationEU-MUR PNRR Extended Partnership
    on Emerging Infectious Diseases (Project no. PE00000007, INF-ACT)
    """

    console.print(Panel(info_text, title="About", border_style="cyan"))


@app.command(rich_help_panel="Pipeline")
def feature_importance(
    output_folder: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to the output folder containing trained models",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    model: str = typer.Option(..., "--model", "-m", help="Model name (e.g., LR, RF, XGB)"),
    target: str = typer.Option(..., "--target", "-t", help="Target name"),
    top_n: int = typer.Option(20, "--top-n", "-n", help="Number of top features to display"),
    no_plot: bool = typer.Option(False, "--no-plot", help="Skip generating the plot"),
    no_csv: bool = typer.Option(False, "--no-csv", help="Skip generating the CSV file"),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Random seed for SHAP reproducibility"
    ),
):
    """
    Extract and visualize feature importance/coefficients for a trained model.

    This command extracts feature importance or coefficients from all outer iterations
    and creates a barplot showing mean values with standard deviation as error bars.

    For models without native importance (MLP, RBF_SVC, TabPFN), SHAP values are
    computed on test fold data as a fallback.

    Supported models:
    - LR, Linear_SVC: Uses coefficient values (native)
    - XGB, RF, CatBoost: Uses feature importance (native)
    - MLP, RBF_SVC, TabPFN: Uses SHAP values (fallback)

    Example:
        respredai feature-importance --output ./output --model RF --target Target1 --top-n 30
    """
    print_banner()

    console.print(
        f"\n[bold cyan]Extracting feature importance for {model} - {target}...[/bold cyan]\n"
    )

    try:
        result = process_feature_importance(
            output_folder=str(output_folder),
            model=model,
            target=target,
            top_n=top_n,
            save_plot=not no_plot,
            save_csv=not no_csv,
            seed=seed,
        )

        if result is None:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Feature importance not available for "
                f"{model} - {target}.\n"
                f"This may occur if:\n"
                f"  • The model doesn't support feature importance (e.g., MLP, TabPFN, RBF_SVC)\n"
                f"  • The saved model file doesn't exist\n"
                f"  • The model training failed\n"
                f"  • SHAP computation failed (check test data availability)"
            )
            raise typer.Exit(code=1)

        result_df, method = result
        method_label = "SHAP values" if method == "shap" else "native importance"
        console.print(f"[dim]Using {method_label}[/dim]\n")

        # Display summary
        mean_importance = result_df.mean(axis=0)
        std_importance = result_df.std(axis=0)

        # Select top N by ABSOLUTE value
        abs_mean = mean_importance.abs()
        top_feature_names = abs_mean.nlargest(top_n).index

        # Get signed values for display
        top_mean = mean_importance[top_feature_names]
        top_std = std_importance[top_feature_names]

        method_suffix = " (SHAP)" if method == "shap" else ""
        table = Table(
            title=f"Top {top_n} Features{method_suffix}: {model} - {target}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Feature", style="green")
        importance_col = "Mean |SHAP|" if method == "shap" else "Importance"
        table.add_column(importance_col, style="yellow", justify="right")

        for rank, feature in enumerate(top_feature_names, 1):
            importance = top_mean[feature]
            std = top_std[feature]
            table.add_row(str(rank), feature, f"{importance:.4f} ± {std:.4f}")

        console.print("\n")
        console.print(table)

        # Show output paths
        model_safe = model.replace(" ", "_")
        target_safe = target.replace(" ", "_")
        suffix = "_shap" if method == "shap" else ""

        output_messages = []
        if not no_csv:
            csv_path = (
                output_folder
                / "feature_importance"
                / target_safe
                / f"{model_safe}_feature_importance{suffix}.csv"
            )
            output_messages.append(f"CSV: [cyan]{csv_path}[/cyan]")
        if not no_plot:
            plot_path = (
                output_folder
                / "feature_importance"
                / target_safe
                / f"{model_safe}_feature_importance{suffix}.png"
            )
            output_messages.append(f"Plot: [cyan]{plot_path}[/cyan]")

        if output_messages:
            success_text = (
                "[bold green]✓ Feature importance extracted successfully![/bold green]\n\n"
            )
            success_text += "Output files:\n" + "\n".join(f"  • {msg}" for msg in output_messages)

            success_panel = Panel(success_text, title="Success", border_style="green")
            console.print("\n", success_panel)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        console.print_exception()
        raise typer.Exit(code=1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
