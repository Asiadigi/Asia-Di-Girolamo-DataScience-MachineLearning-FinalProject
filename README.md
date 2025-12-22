# Credit Default Prediction: A Comparative Analysis of ML Models

## Research Question
Which Machine Learning algorithm performs best for predicting credit default risk: Logistic Regression, KNN, Decision Trees, Random Forest, XGBoost, or CatBoost?

## Project Overview
This project implements a reproducible Machine Learning pipeline to predict credit default risk using the German Credit dataset. It processes the raw data, applies feature scaling and encoding, and evaluates multiple classification models to minimize financial risk for lenders.

**Key Finding:** CatBoost achieved the best performance with **77.5% Accuracy** and **0.814 ROC-AUC**, outperforming traditional models like Decision Trees.

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
