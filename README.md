# DSAA Machine Learning Practicals

This repository contains the notebooks and supporting files used in the DSAA
Machine Learning practical classes. Unsolved notebooks are available under
`notebooks/`. A solved notebook is added under `solutions/` after the
corresponding practical class.

## Available materials

| Week | Topic | Unsolved notebook | Solution |
|---:|---|---|---|
| Week 1 | What is supervised machine learning? | [Open](notebooks/week_01/week_01_supervised_ml_and_environment.ipynb) | Not yet released |
| Week 2 | The machine-learning process | [Open](notebooks/week_02/week_02_ml_pipeline.ipynb) | Not yet released |

Solutions are separate files, so an update will not replace the notebook in
which you have been working.

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
conda activate dsaa-ml-practicals
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
folder structure unchanged because the notebooks load files from `data/` and
`assets/` by relative path.

## Get new notebooks and solutions

If you cloned the repository, close any running notebooks and pull the latest
version:

```text
git pull
```

GitHub Desktop users can select **Fetch origin** and then **Pull origin**.
Students using ZIP downloads should download and extract a fresh copy.

## Repository structure

```text
notebooks/      Unsolved practical notebooks
solutions/      Solutions released after each practical
data/           Data opened by the notebooks
assets/         Diagrams used by the notebooks
project/        Course project materials
environment.yml Conda environment specification
```

## Common setup problems

| Problem | Resolution |
|---|---|
| `conda` is not recognised on Windows | Use Anaconda Prompt, or run `conda init powershell` once and reopen PowerShell. |
| `EnvironmentFileNotFound` appears | Change into the repository folder that contains `environment.yml`. |
| The environment already exists | Run `conda env update -f environment.yml --prune`. |
| Imports fail in Jupyter | Stop Jupyter, activate `dsaa-ml-practicals`, and start Jupyter again. |
| A notebook cannot find a data or image file | Start Jupyter from the repository root and keep the supplied folder structure unchanged. |

If a notebook still fails after these checks, record the complete error message
and ask an instructor for help.
