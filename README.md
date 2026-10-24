# DSAA Machine Learning Practicals

This repository contains the notebooks and supporting files used in the DSAA
Machine Learning practical classes. Unsolved notebooks are available under
`notebooks/`. A solved notebook is added under `solutions/` after the
corresponding practical class.

## Available materials

| Week | Topic | Unsolved notebook | Solution |
|---:|---|---|---|
| Week 1 | What is supervised machine learning? | [Open](notebooks/week_01/week_01_supervised_ml_and_environment.ipynb) | [Open](solutions/week_01/week_01_supervised_ml_and_environment_solution.ipynb) |
| Week 2 | The machine-learning process | [Open](notebooks/week_02/week_02_ml_pipeline.ipynb) | [Open](solutions/week_02/week_02_ml_pipeline_solution.ipynb) |
| Week 3 | Deepening exploration: quality, missing values and outliers | [Open](notebooks/week_03/week_03_deepen_exploration.ipynb) | [Open](solutions/week_03/week_03_deepen_exploration_solution.ipynb) |
| Week 3 | Deepening exploration: a price, not a class | [Open](notebooks/week_03/week_03_deepen_exploration_regression.ipynb) | [Open](solutions/week_03/week_03_deepen_exploration_regression_solution.ipynb) |
| Week 4 | Feature work and dimensionality reduction before selection | [Open](notebooks/week_04/week_04_feature_work_classification.ipynb) | [Open](solutions/week_04/week_04_feature_work_classification_solution.ipynb) |
| Week 4 | Feature work on a regression target | [Open](notebooks/week_04/week_04_feature_work_regression.ipynb) | [Open](solutions/week_04/week_04_feature_work_regression_solution.ipynb) |
| Week 5 | Performance measures | [Open](notebooks/week_05/week_05_performance_measures.ipynb) | [Open](solutions/week_05/week_05_performance_measures_solution.ipynb) |
| Week 5 | Model selection | [Open](notebooks/week_05/week_05_model_selection.ipynb) | [Open](solutions/week_05/week_05_model_selection_solution.ipynb) |
| Week 6 | Linear and logistic models | [Open](notebooks/week_06/week_06_linear_logistic.ipynb) | [Open](solutions/week_06/week_06_linear_logistic_solution.ipynb) |
| Week 7 | Probability and Naive Bayes | [Open](notebooks/week_07/week_07_naive_bayes.ipynb) | [Open](solutions/week_07/week_07_naive_bayes_solution.ipynb) |
| Week 7 | Bayesian regression | [Open](notebooks/week_07/week_07_bayesian_regression.ipynb) | [Open](solutions/week_07/week_07_bayesian_regression_solution.ipynb) |
| Week 8 | Similarity and KNN | [Open](notebooks/week_08/week_08_similarity_knn.ipynb) | Not yet released |
| Week 8 | KNN for a numeric target | [Open](notebooks/week_08/week_08_knn_regression.ipynb) | Not yet released |
| Week 9 | Decision trees | [Open](notebooks/week_09/week_09_decision_trees.ipynb) | Not yet released |
| Week 9 | Decision trees for a numeric target | [Open](notebooks/week_09/week_09_regression_trees.ipynb) | Not yet released |
| Week 10 | Bagging and random forests | [Open](notebooks/week_10/week_10_ensembles_bagging.ipynb) | Not yet released |
| Week 10 | Bagging and random forests for a numeric target | [Open](notebooks/week_10/week_10_ensembles_bagging_regression.ipynb) | Not yet released |

Solutions are separate files, so an update will not replace the notebook in
which you have been working.

The `results/` folder holds the scores and the fitted models of every released
week. Each notebook, whether you open it from `notebooks/` or from `solutions/`,
reads the earlier weeks from that folder and writes its own week there, so a
later week still works when a class ended before you reached the end of a
notebook. Running a notebook to its end in the course environment writes the
same files again.

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

Week 3 cleans the raw datasets, writing the cleaned files to `data/interim/` and
its cleaning logs to `logs/`, and later weeks read both. Copies of those files
are supplied, so each week runs on its own; running Week 3 replaces them.

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
logs/           Cleaning logs that later notebooks read
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
| Imports fail in Jupyter | Stop Jupyter, activate `machine_learning`, and start Jupyter again. |
| A notebook cannot find a data or image file | Start Jupyter from the repository root and keep the supplied folder structure unchanged. |

If a notebook still fails after these checks, record the complete error message
and ask an instructor for help.
