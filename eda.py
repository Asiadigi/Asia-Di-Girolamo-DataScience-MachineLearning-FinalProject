import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define column names based on dataset-info.txt
columns = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk"
]

def load_data():
    """Loads the dataset from dataset.data"""
    filepath = "dataset.data"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    
    # The dataset is space-separated
    df = pd.read_csv(filepath, sep=" ", names=columns, header=None)
    return df

def plot_numerical_by_target(df):
    """Generates Violin and Box plots for numerical features split by target."""
    numerical_cols = ["duration", "credit_amount", "age", "installment_rate", "residence_since"]
    
    for col in numerical_cols:
        # Violin Plot
        plt.figure(figsize=(8, 5))
        sns.violinplot(x='credit_risk', y=col, data=df, palette="muted", split=True)
        plt.title(f'Violin Plot of {col} by Credit Risk')
        plt.xlabel('Credit Risk (1=Good, 2=Bad)')
        plt.savefig(f'plots/violin_{col}.png')
        plt.close()
        
        # KDE Plot
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x=col, hue="credit_risk", fill=True, common_norm=False, palette="muted")
        plt.title(f'KDE Plot of {col} by Credit Risk')
        plt.savefig(f'plots/kde_{col}.png')
        plt.close()

def plot_categorical_by_target(df):
    """Generates Stacked Bar Charts for categorical features."""
    categorical_cols = ["status", "credit_history", "purpose", "savings", "employment_duration", "housing", "job"]
    
    for col in categorical_cols:
        # Calculate proportions
        crosstab = pd.crosstab(df[col], df['credit_risk'], normalize='index')
        
        # Plot
        crosstab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', alpha=0.8)
        plt.title(f'Proportion of Credit Risk by {col}')
        plt.xlabel(col)
        plt.ylabel('Proportion')
        plt.legend(title='Credit Risk', loc='upper right', labels=['Good', 'Bad'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/stacked_bar_{col}.png')
        plt.close()

def plot_interactions(df):
    """Generates Scatter plots and Pairplots."""
    # Scatter: Amount vs Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='duration', y='credit_amount', hue='credit_risk', palette='deep', alpha=0.6)
    plt.title('Credit Amount vs Duration (Colored by Risk)')
    plt.savefig('plots/scatter_amount_duration.png')
    plt.close()
    
    # Pairplot for key features
    subset_cols = ["duration", "credit_amount", "age", "credit_risk"]
    sns.pairplot(df[subset_cols], hue='credit_risk', palette='deep', diag_kind='kde')
    plt.savefig('plots/pairplot_key_features.png')
    plt.close()

def perform_eda(df):
    """Performs basic EDA and saves plots"""
    if df is None:
        return

    print("Dataset Info:")
    print(df.info())
    
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Set style
    sns.set_style("whitegrid")

    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='credit_risk', data=df, palette='pastel')
    plt.title('Distribution of Credit Risk (1=Good, 2=Bad)')
    plt.savefig('plots/credit_risk_distribution.png')
    plt.close()

    # 2. Numerical Analysis
    plot_numerical_by_target(df)
    
    # 3. Categorical Analysis
    plot_categorical_by_target(df)
    
    # 4. Interactions
    plot_interactions(df)

    # 5. Correlation Matrix (Numerical only)
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 8))
    corr = numerical_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    print("\nEnhanced EDA complete. Plots saved to 'plots/' directory.")

if __name__ == "__main__":
    df = load_data()
    perform_eda(df)
