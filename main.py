import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data_loader import load_data, clean_data
from src.models import get_models, get_preprocessor
from src.evaluation import evaluate_model

def main():
    # 1. Load Data
    print("Loading data...")
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Clean Data
    print("Cleaning data...")
    df = clean_data(df)

    # 3. Prepare Data
    target = "credit_risk"
    X = df.drop(columns=[target])
    y = df[target]

    # Define feature types
    numeric_features = [
        "duration", "credit_amount", "installment_rate", "residence_since", 
        "age", "existing_credits", "people_liable"
    ]
    categorical_features = [
        "status", "credit_history", "purpose", "savings", "employment_duration", 
        "personal_status_sex", "other_debtors", "property", "other_installment_plans", 
        "housing", "job", "telephone", "foreign_worker"
    ]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Get Preprocessor
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # 5. Get Models
    models = get_models()

    # 6. Train and Evaluate
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Train
        clf.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(clf, X_test, y_test, model_name=name)
        results[name] = metrics

    print("\nSummary of Results:")
    results_df = pd.DataFrame(results).T
    print(results_df.sort_values(by="f1", ascending=False))
    results_df.to_csv("results/model_comparison.csv")

if __name__ == "__main__":
    main()
