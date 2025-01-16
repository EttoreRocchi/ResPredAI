# ResPredAI

## Antimicrobial **Res**istance **pre**dictions via **AI** models

Implementation of the pipeline described in the work "**Artificial intelligence model to predict resistances in Gram-negative bloodstream infections**" by _Bonazzetti et al._

### Installation

Download the project by cloning the repository:
```
git clone https://github.com/EttoreRocchi/ResPredAI.git
cd ResPredAI
```
Install the required packages:
```
pip install -r ./requirements.txt
```
To test the installation, simply run:
```
python main.py -c ./example/config_example.ini 
```
This will run the pipeline on a fictitious dataset. The results will be reported in `./out_run_example/`.

## Usage

To run the pipeline you may use this command:

```
python main.py -c {path/to/config.ini}
```

where `{path/to/config.ini}` is the path to the configuration file, structured as below:
```
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

### Funding
This research was supported by EU funding within the NextGenerationEU-MUR PNRR Extended Partnership initiative on Emerging Infectious Diseases (Project no. PE00000007, INF-ACT)