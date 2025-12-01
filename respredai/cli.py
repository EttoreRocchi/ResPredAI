"""Command-line interface for ResPredAI."""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from pathlib import Path
from typing import Optional
from .__init__ import __version__

from .main import perform_pipeline
from .utils import ConfigHandler, DataSetter
from .feature_importance import process_feature_importance

app = typer.Typer(
    name="respredai",
    help="ResPredAI - Antimicrobial Resistance Prediction via AI",
    add_completion=False,
)
console = Console()


def print_banner():
    """Print the ResPredAI banner."""
    banner_text = f"""
    ResPredAI
    Version: {__version__}
    Antimicrobial Resistance predictions via AI models
    """
    console.print(Panel(banner_text, title="ResPredAI", border_style="cyan"))

def print_config_info(config_handler: ConfigHandler):
    """Print configuration information in a nice table."""
    table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=25)
    table.add_column("Value", style="green")

    table.add_row("Data Path", str(config_handler.data_path))
    table.add_row("Targets", ", ".join(config_handler.targets))
    table.add_row("Models", ", ".join(config_handler.models))
    table.add_row("Outer Folds", str(config_handler.outer_folds))
    table.add_row("Inner Folds", str(config_handler.inner_folds))
    table.add_row("Random Seed", str(config_handler.seed))
    table.add_row("Parallel Jobs", str(config_handler.n_jobs))
    table.add_row("Output Folder", str(config_handler.out_folder))
    table.add_row("Model Saving Enabled", str(config_handler.save_models_enable))

    console.print(table)


def version_callback(value: bool):
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
    )
):
    """
    ResPredAI - Antimicrobial Resistance Prediction via AI

    A machine learning pipeline for predicting antimicrobial resistance
    in Gram-negative bloodstream infections.
    """
    pass


class TrainingProgressCallback:
    """Callback class to handle training progress updates."""

    def __init__(self, console: Console, quiet: bool = False):
        self.console = console
        self.quiet = quiet
        self.progress = None
        self.overall_task = None
        self.model_task = None
        self.target_task = None
        self.fold_task = None

    def start(self, total_work: int):
        """Start the progress tracking."""
        if self.quiet:
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        self.progress.start()
        self.overall_task = self.progress.add_task(
            "[cyan]Overall Progress", total=total_work
        )

    def start_model(self, model_name: str, total_work: int):
        """Start tracking a new model."""
        if self.quiet or not self.progress:
            return

        if self.model_task is not None:
            self.progress.remove_task(self.model_task)

        self.model_task = self.progress.add_task(
            f"[green]Model: {model_name}", total=total_work
        )

    def start_target(self, target_name: str, total_folds: int, resumed_from: int = 0):
        """Start tracking a new target."""
        if self.quiet or not self.progress:
            return

        if self.target_task is not None:
            self.progress.remove_task(self.target_task)

        status = f"[yellow]Target: {target_name}"
        if resumed_from > 0:
            status += f" (resumed from fold {resumed_from + 1})"

        self.target_task = self.progress.add_task(status, total=total_folds, completed=resumed_from)

    def start_fold(self, fold_num: int, total_folds: int):
        """Start tracking a new fold."""
        if self.quiet or not self.progress:
            return

        if self.fold_task is not None:
            self.progress.remove_task(self.fold_task)

        self.fold_task = self.progress.add_task(
            f"[blue]  Fold {fold_num}/{total_folds}: Training...", total=100
        )

    def update_fold_status(self, status: str, progress: int = None):
        """Update fold status."""
        if self.quiet or not self.progress or self.fold_task is None:
            return

        self.progress.update(self.fold_task, description=f"[blue]  {status}")
        if progress is not None:
            self.progress.update(self.fold_task, completed=progress)

    def complete_fold(self, fold_num: int, metrics: dict):
        """Complete a fold and show metrics."""
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

    def complete_target(self, target_name: str, summary_metrics: dict):
        """Complete a target and show summary."""
        if self.quiet or not self.progress:
            return

        if self.target_task is not None:
            self.progress.remove_task(self.target_task)
            self.target_task = None

        # Print target summary
        self.console.print(f"\n  [bold green]✓[/bold green] Completed {target_name}")
        summary_str = "    "
        summary_str += f"F1={summary_metrics.get('F1 (weighted)', 0):.3f}±{summary_metrics.get('F1_std', 0):.3f}, "
        summary_str += f"MCC={summary_metrics.get('MCC', 0):.3f}±{summary_metrics.get('MCC_std', 0):.3f}, "
        summary_str += f"AUROC={summary_metrics.get('AUROC', 0):.3f}±{summary_metrics.get('AUROC_std', 0):.3f}"
        self.console.print(f"[cyan]{summary_str}[/cyan]\n")

    def complete_model(self, model_name: str):
        """Complete a model."""
        if self.quiet or not self.progress:
            return

        if self.model_task is not None:
            self.progress.remove_task(self.model_task)
            self.model_task = None

        self.console.print(f"\n[bold green]✓ Completed model: {model_name}[/bold green]\n\n")

    def skip_target(self, target_name: str, num_folds: int, reason: str = "saved models"):
        """Skip a target (loaded from saved models)."""
        if self.quiet or not self.progress:
            return

        # Advance model and overall progress by the number of folds that were skipped
        if self.model_task is not None:
            self.progress.advance(self.model_task, advance=num_folds)
        if self.overall_task is not None:
            self.progress.advance(self.overall_task, advance=num_folds)

        self.console.print(f"  [dim]⊙ Skipped {target_name} (loaded from {reason})[/dim]")

    def skip_model(self, model_name: str, total_work_skipped: int, reason: str = "error"):
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

        self.console.print(f"[dim]⊙ Skipped model: {model_name} ({reason})[/dim]")

    def stop(self):
        """Stop the progress tracking."""
        if self.quiet or not self.progress:
            return

        self.progress.stop()


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the configuration file (.ini format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress banner and progress output"
    )
):
    """
    Run the machine learning pipeline with the specified configuration.

    Example:
        respredai run --config my_config.ini
    """
    print_banner()

    try:
        # Load configuration
        with console.status("[bold green]Loading configuration..."):
            config_handler = ConfigHandler(str(config))

        if not quiet:
            console.print("\n[bold green]✓[/bold green] Configuration loaded successfully\n")
            print_config_info(config_handler)

        # Load data
        console.print("\n[bold cyan]Loading data...[/bold cyan]")
        with console.status("[bold green]Reading and validating data..."):
            datasetter = DataSetter(config_handler)
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
        perform_pipeline(
            datasetter=datasetter,
            models=config_handler.models,
            config_handler=config_handler,
            progress_callback=progress_callback
        )

        # Success message
        success_panel = Panel(
            f"[bold green]✓ Pipeline completed successfully![/bold green]\n\n"
            f"Results saved to: [cyan]{config_handler.out_folder}[/cyan]",
            title="Success",
            border_style="green"
        )
        console.print("\n", success_panel)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if not quiet:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
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
    ]

    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Code", style="cyan", width=15)
    table.add_column("Name", style="green", width=25)

    for code, name in models_info:
        table.add_row(code, name)

    console.print("\n")
    console.print(table)
    console.print("\n[dim]Use these codes in your config file under the 'models' parameter.[/dim]\n")


@app.command()
def create_config(
    output_path: Path = typer.Argument(
        ...,
        help="Path where the template configuration file will be created"
    )
):
    """
    Create a template configuration file.

    Example:
        respredai create-config my_config.ini
    """

    template = """[Data]
data_path = ./data/your_data.csv
targets = Target1,Target2
continuous_features = Feature1,Feature2,Feature3

[Pipeline]
models = LR,RF,XGB
outer_folds = 5
inner_folds = 3

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
            f"[bold green]✓[/bold green] Template configuration created: "
            f"[cyan]{output_path}[/cyan]"
        )
        console.print("\n[dim]Edit this file with your data paths and parameters.[/dim]\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Could not create config file: {str(e)}")
        raise typer.Exit(code=1)


@app.command()
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


@app.command()
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
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model name (e.g., LR, RF, XGB)"
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target name"
    ),
    top_n: int = typer.Option(
        20,
        "--top-n",
        "-n",
        help="Number of top features to display"
    ),
    no_plot: bool = typer.Option(
        False,
        "--no-plot",
        help="Skip generating the plot"
    ),
    no_csv: bool = typer.Option(
        False,
        "--no-csv",
        help="Skip generating the CSV file"
    )
):
    """
    Extract and visualize feature importance/coefficients for a trained model.

    This command extracts feature importance or coefficients from all outer iterations
    and creates a barplot showing mean values with standard deviation as error bars.

    Supported models:
    - LR, Linear_SVC: Uses absolute coefficient values
    - XGB, RF, CatBoost: Uses feature importance

    Example:
        respredai feature-importance --output ./output --model RF --target Target1 --top-n 30
    """
    print_banner()

    console.print(f"\n[bold cyan]Extracting feature importance for {model} - {target}...[/bold cyan]\n")

    try:
        result_df = process_feature_importance(
            output_folder=str(output_folder),
            model=model,
            target=target,
            top_n=top_n,
            save_plot=not no_plot,
            save_csv=not no_csv
        )

        if result_df is None:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Feature importance not available for "
                f"{model} - {target}.\n"
                f"This may occur if:\n"
                f"  • The model doesn't support feature importance (e.g., MLP, TabPFN, RBF_SVC)\n"
                f"  • The saved model file doesn't exist\n"
                f"  • The model training failed"
            )
            raise typer.Exit(code=1)

        # Display summary
        mean_importance = result_df.mean(axis=0)
        std_importance = result_df.std(axis=0)

        # Select top N by ABSOLUTE value
        abs_mean = mean_importance.abs()
        top_feature_names = abs_mean.nlargest(top_n).index

        # Get signed values for display
        top_mean = mean_importance[top_feature_names]
        top_std = std_importance[top_feature_names]

        table = Table(
            title=f"Top {top_n} Features (by |importance|): {model} - {target}",
            show_header=True,
            header_style="bold magenta"
        )
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Feature", style="green")
        table.add_column("Importance", style="yellow", justify="right")

        for rank, feature in enumerate(top_feature_names, 1):
            importance = top_mean[feature]
            std = top_std[feature]
            table.add_row(
                str(rank),
                feature,
                f"{importance:.4f} ± {std:.4f}"
            )

        console.print("\n")
        console.print(table)

        # Show output paths
        model_safe = model.replace(" ", "_")
        target_safe = target.replace(" ", "_")

        output_messages = []
        if not no_csv:
            csv_path = output_folder / "feature_importance" / target_safe / f"{model_safe}_feature_importance.csv"
            output_messages.append(f"CSV: [cyan]{csv_path}[/cyan]")
        if not no_plot:
            plot_path = output_folder / "feature_importance" / target_safe / f"{model_safe}_feature_importance.png"
            output_messages.append(f"Plot: [cyan]{plot_path}[/cyan]")

        if output_messages:
            success_text = "[bold green]✓ Feature importance extracted successfully![/bold green]\n\n"
            success_text += "Output files:\n" + "\n".join(f"  • {msg}" for msg in output_messages)

            success_panel = Panel(
                success_text,
                title="Success",
                border_style="green"
            )
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