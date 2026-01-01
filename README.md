# Credit Default Prediction: A Comparative Analysis of ML Models

## Research Question
Which Machine Learning algorithm performs best for predicting credit default risk: Logistic Regression, KNN, Decision Trees, Random Forest, XGBoost, or CatBoost?

## Project Overview
This project implements a reproducible Machine Learning pipeline to predict credit default risk using the German Credit dataset. It processes the raw data, applies feature scaling and encoding, and evaluates multiple classification models to minimize financial risk for lenders.

**Key Finding:** CatBoost achieved the best performance with **78.5% Accuracy** and **0.817 ROC-AUC**, outperforming traditional models like Decision Trees.

## Project Structure
```text
MLproject/
├── data/
│   └── raw/               # Original dataset
├── report/
│   ├── project_report.pdf # Final detailed report
│   └── figures/           # Images for the report
├── results/               # Generated plots and metrics (Confusion Matrices)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data ingestion and preprocessing
│   ├── models.py          # Model definitions (CatBoost, XGBoost, etc.)
│   └── evaluation.py      # Metrics and visualization logic
├── notebooks/             # Jupyter notebooks
├── AI_LOG.md              # Log of AI tools used
├── environment.yml        # Conda environment configuration
├── main.py                # Main execution script
└── README.md              # This file

## Setup instructions


1. Clone/Download the repository and navigate to the project folder:
```bash
cd MLproject
```

2. Create the Conda environment (ensures reproducibility):
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate mlproject
```

## Usage

Run the main script to train and evaluate models:
```bash
python main.py
```

The script will:
1. Load the dataset from `data/raw/dataset.data`.
2. Clean and preprocess the data.
3. Train multiple models (Logistic Regression, Random Forest, XGBoost, etc.).
4. Evaluate the models and print metrics.
5. Save confusion matrix plots and a summary CSV to the `results/` directory.


## Expected Output
The script will:
1. Load the dataset from data/raw/dataset.data.

2. Clean and preprocess the data (scaling, encoding).

3. Train multiple models sequentially (Logistic Regression, Random Forest, XGBoost, etc.).

4. Print the Accuracy and classification report for each model in the terminal.

5. Display the winner (Best Model).

6. Save confusion matrix plots and a summary CSV to the results/ directory.

## Models Evaluated
The following models are trained and evaluated in this study:
- Logistic Regression
- Random Forest
- AdaBoost
- KNN
- Gaussian Naive Bayes
- Decision Tree
- XGBoost
- LightGBM
- CatBoost (Winner)
- Explainable Boosting Machine (EBM)

## Requirements 
- Python 3.11
- Libraries: pandas, scikit-learn, matplotlib, seaborn, catboost, xgboost, lightgbm.
- Dependencies are automatically handled by environment.yml.
  
## AI usage
I utilized AI tools (Antigravity, Gemini 3.0, ChatGPT) to assist with the technical development of this project.

Specifically, AI assistance was used for:
- Debugging & Error Resolution: Solving complex compatibility issues (e.g., the scikit-learn vs. CatBoost version conflict) and fixing execution errors.
- Code Refactoring: Optimizing the project structure into modular components (src/ folder).
For more details, see AI_LOG.md.

> **Important (project root):** The runnable code is inside the `MLproject/` subfolder of this repository.
> After cloning, you must `cd` into `MLproject/` **before** creating the conda environment or running any scripts.
Note: If you already see environment.yml and main.py in the current directory, you are already at the runnable project root and you should NOT cd MLproject.

### Step 0 — Move into the correct folder

# After cloning, enter the repository:
cd Asia-Di-Girolamo-DataScience-MachineLearning-FinalProject

# Sanity check (project root):
# You must see BOTH "environment.yml" and "main.py" here.
# If you only see a subfolder called "MLproject/", run:
#   cd MLproject

# Only run `cd MLproject` if you see a subfolder named `MLproject/`. If you already see `environment.yml` and `main.py` in the current directory, you are already at the project root—do NOT `cd MLproject`.


# Only run this if you do NOT see environment.yml and main.py in the current directory.
# Copy/paste quick start (only if the runnable root is MLproject/)

git clone https://github.com/Asiadigi/Asia-Di-Girolamo-DataScience-MachineLearning-FinalProject.git
cd Asia-Di-Girolamo-DataScience-MachineLearning-FinalProject


# create the environment
conda env create -f environment.yml
conda activate mlproject

# update sklearn to a version compatible with interpret-core
pip install -U "scikit-learn>=1.6.0"

# update CatBoost (fix the __sklearn_tags__ problem)
pip install -U catboost

python main.py



# Sanity check: you should now see environment.yml and main.py
# Windows (PowerShell):
dir
# macOS/Linux:
ls

### Step 1 — Create the conda environment
# --- Create / update the conda environment ---
conda env create -f environment.yml
conda activate mlproject

# If the environment already exists, update it with:
conda env update -f environment.yml --prune
conda activate mlproject

# --- Run the project ---
python main.py


> If the environment already exists, update it with:
> ```bash
> conda env update -f environment.yml --prune
conda activate mlproject
> ```
> **Windows note (PowerShell):** if `conda activate mlproject` does not work, open **Anaconda Prompt** (recommended) and re-run the commands.
> Alternatively, run once:
> ```powershell
> conda init powershell
> ```
> then **restart PowerShell** and try again.

### Step 2 - Run the project
After running, the project will train/evaluate the models and generate evaluation artifacts (metrics + plots).
Check these folders for outputs:

report/project_report.pdf (final report PDF)

report/project_report.md (markdown source)

notebooks/credit_default_prediction.ipynb (full notebook)
