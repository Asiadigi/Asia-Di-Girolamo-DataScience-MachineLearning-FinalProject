# Credit Default Prediction Project

This project implements a machine learning pipeline to predict credit default risk using the German Credit dataset.

## Project Structure

```
mlproject/
 ├── README.md              # Project documentation
 ├── environment.yml        # Conda environment file
 ├── main.py                # Main execution script
 ├── src/                   # Source code
 │   ├── data_loader.py     # Data loading and cleaning
 │   ├── models.py          # Model definitions and preprocessing
 │   └── evaluation.py      # Evaluation metrics and plotting
 ├── data/
 │   └── raw/               # Raw data files
 ├── results/               # Evaluation results and plots
 └── notebooks/             # Jupyter notebooks
```

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
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

## Models

The following models are trained and evaluated:
- Logistic Regression
- Random Forest
- AdaBoost
- KNN
- Gaussian Naive Bayes
- Decision Tree
- XGBoost
- LightGBM
- CatBoost
- Explainable Boosting Machine (EBM)
