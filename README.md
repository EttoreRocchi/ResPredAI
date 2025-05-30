# ResPredAI

## Antimicrobial **Res**istance **pre**dictions via **AI** models

Implementation of the pipeline described in the work Bonazzetti, C., Rocchi, E., Toschi, A. _et al._ Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections. _npj Digit. Med._ **8**, 319 (2025). https://doi.org/10.1038/s41746-025-01696-x

### Installation

Download the project by cloning the repository:
```bash
git clone https://github.com/EttoreRocchi/ResPredAI.git
cd ResPredAI
```
Install the required packages:
```bash
pip install -r ./requirements.txt
```
To test the installation, simply run:
```bash
python main.py -c ./example/config_example.ini 
```
This will run the pipeline on a fictitious dataset. The results will be reported in `./out_run_example/`.

## Usage

To run the pipeline you may use this command:

```bash
python main.py -c <path/to/config.ini>
```

where `<path/to/config.ini>` is the path to the configuration file, structured as below:
```ini
[Data]
data_path = # path to data file
targets = # list of targets in the DataFrame
continuous_features = # list of continuous features in the DataFrame

[Pipeline]
models = # accepted models are: LR (for Logistic Regression), MLP (for Multi-Layer Perceptron) and XGB (for eXtreme Gradient Boosting classifier)
outer_folds = # number of folds for outer cross-validation
inner_folds = # number of folds for inner cross-validation

[Reproducibility]
seed = # an integer for reproducibility

[Log]
verbosity = # Verbosity level (possible values: 0, 1, 2):
            # 0 = no log file will be created;
            # 1 = the log file will record just the start and end times of model's training; 
            # 2 = the log file will also record the start time of each iteration.
log_basename = # name of the log file (if verbosity is not 0)

[Resources]
n_jobs = # number of jobs to run in parallel 

[Output]
out_folder = # path to the folder in which the outputs will be saved
```

### Citation

```BibTex
@article{Bonazzetti2025,
  author = {Bonazzetti, Cecilia and Rocchi, Ettore and Toschi, Alice and Derus, Nicolas Riccardo and Sala, Claudia and Pascale, Renato and Rinaldi, Matteo and Campoli, Caterina and Pasquini, Zeno Adrien Igor and Tazza, Beatrice and Amicucci, Armando and Gatti, Milo and Ambretti, Simone and Viale, Pierluigi and Castellani, Gastone and Giannella, Maddalena},
  title = {Artificial Intelligence model to predict resistances in Gram-negative bloodstream infections},
  journal = {npj Digital Medicine},
  volume = {8},
  pages = {319},
  year = {2025},
  doi = {10.1038/s41746-025-01696-x},
  url = {https://doi.org/10.1038/s41746-025-01696-x}
}
```

### Funding
This research was supported by EU funding within the NextGenerationEU-MUR PNRR Extended Partnership initiative on Emerging Infectious Diseases (Project no. PE00000007, INF-ACT)