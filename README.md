# DSAA Machine Learning Practicals

This repository provides the notebooks and supporting files for the DSAA
Machine Learning practical classes.

## Repository materials

| Folder | Purpose |
|---|---|
| [notebooks/](notebooks/) | Practical notebooks organised by week |
| [solutions/](solutions/) | Worked solutions published as separate notebooks |
| [data/](data/) | Datasets opened by the notebooks |
| [logs/](logs/) | Cleaning logs used in later practicals |
| [assets/](assets/) | Diagrams used by the notebooks |
| [results/](results/) | Shared scores, fitted models and saved selections |
| [project/](project/) | Course project materials |

Keep your own work in a separate copy of a notebook so repository updates can
bring in new materials without overwriting your answers.

The notebooks share the supplied data, cleaning logs, diagrams and saved model
results. Keep the complete repository together: a later practical can read
results from an earlier one even if you have not run that notebook to its end.
Running a notebook can update files in `data/`, `logs/` and `results/`.

## Set up the course environment

Complete this setup once on the computer you will use for the practicals.

### 1. Install Conda

Install [Miniconda or Anaconda](https://www.anaconda.com/download). On Windows,
open Anaconda Prompt after installation. On macOS or Linux, open a terminal.

Check that Conda is available:

```text
conda --version
```

### 2. Download the repository

For the simplest start, select **Code**, then **Download ZIP** on the repository
page and extract the archive. To receive later notebooks and solutions more
easily, clone the repository with GitHub Desktop or Git:

```text
git clone https://github.com/RicardoSantos0/ML_DSAA_Practicals.git
cd ML_DSAA_Practicals
```

Run the remaining commands from the folder containing `environment.yml`.

### 3. Create and activate the environment

```text
conda env create -f environment.yml
conda activate machine_learning
```

If the environment already exists and `environment.yml` has changed, update it:

```text
conda env update -f environment.yml --prune
```

### 4. Start Jupyter Notebook

```text
jupyter notebook
```

Open the notebook for the current week under `notebooks/`. Keep the repository
folder structure unchanged because the notebooks load files from `data/`,
`logs/` and `assets/` by relative path.

## Get new notebooks and solutions

If you cloned the repository, close any running notebooks and pull the latest
version. Save your work outside the tracked notebook files first:

```text
git pull
```

GitHub Desktop users can select **Fetch origin** and then **Pull origin**.
Students using ZIP downloads should download and extract a fresh copy, then
copy their own working notebooks into it.

## Common setup problems

| Problem | Resolution |
|---|---|
| `conda` is not recognised on Windows | Use Anaconda Prompt, or run `conda init powershell` once and reopen PowerShell. |
| `EnvironmentFileNotFound` appears | Change into the repository folder that contains `environment.yml`. |
| The environment already exists | Run `conda env update -f environment.yml --prune`. |
| Imports fail in Jupyter | Stop Jupyter, activate `machine_learning`, and start Jupyter again. |
| A notebook cannot find a data or image file | Start Jupyter from the repository root and keep the supplied folder structure unchanged. |

If a notebook still fails after these checks, record the complete error message
and ask an instructor for help.
