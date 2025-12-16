import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
import os

# Define column names
columns = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk"
]

def load_data():
    filepath = "dataset.data"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    df = pd.read_csv(filepath, sep=" ", names=columns, header=None)
    return df

def perform_pca_analysis(X_train, y_train, X_test, y_test):
    """
    Performs PCA analysis and visualizes the results.
    """
    print("\nPerforming PCA Analysis...")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio (First 2 components): {explained_variance}")
    
    # Plot 2D PCA for Train set
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    plt.title('PCA: First 2 Principal Components (Training Set)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Target (0=Good, 1=Bad)')
    plt.savefig('plots/pca_2d_scatter_train.png')
    plt.close()
    print("PCA plots saved.")
    return X_train_pca, X_test_pca

def perform_clustering(X_train, y_train, X_test, y_test):
    """
    Performs K-Means and GMM clustering and compares with actual labels.
    """
    print("\nPerforming Clustering Analysis (Unsupervised)...")
    
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train)
    train_clusters_km = kmeans.predict(X_train)
    test_clusters_km = kmeans.predict(X_test)
    
    # GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X_train)
    train_clusters_gmm = gmm.predict(X_train)
    test_clusters_gmm = gmm.predict(X_test)
    
    # Evaluate alignment with target (Out-of-Sample)
    # Note: Cluster labels (0, 1) might not match Target labels (0, 1) directly (could be swapped).
    # We check accuracy for both permutations and take the max to see "purity".
    
    def get_cluster_accuracy(y_true, y_cluster):
        acc_direct = accuracy_score(y_true, y_cluster)
        acc_swapped = accuracy_score(y_true, 1 - y_cluster)
        return max(acc_direct, acc_swapped)

    km_acc = get_cluster_accuracy(y_test, test_clusters_km)
    gmm_acc = get_cluster_accuracy(y_test, test_clusters_gmm)
    
    print(f"K-Means Test Set Alignment (Purity): {km_acc:.4f}")
    print(f"GMM Test Set Alignment (Purity):     {gmm_acc:.4f}")
    
    return test_clusters_km, test_clusters_gmm

def train_and_evaluate(df):
    # Encode target: 1 (Good) -> 0, 2 (Bad) -> 1
    df['target'] = df['credit_risk'].apply(lambda x: 0 if x == 1 else 1)
    
    X = df.drop(['credit_risk', 'target'], axis=1)
    y = df['target']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    # Split data (Out-of-Sample Evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Set Size: {X_train.shape[0]}")
    print(f"Test Set Size:     {X_test.shape[0]}")
    
    # Fit preprocessor
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # PCA
    perform_pca_analysis(X_train_preprocessed, y_train, X_test_preprocessed, y_test)
    
    # Clustering
    perform_clustering(X_train_preprocessed, y_train, X_test_preprocessed, y_test)
    
    # Supervised Models
    models = {
        "Logistic Regression (L2)": LogisticRegression(penalty='l2', max_iter=1000, class_weight='balanced', random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    print(f"\n{'Model':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC AUC':<10}")
    print("-" * 65)
    
    results = {}
    
    for name, model in models.items():
        # Train on Training Set
        model.fit(X_train_preprocessed, y_train)
        
        # Evaluate on Test Set (Out-of-Sample)
        y_pred = model.predict(X_test_preprocessed)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_preprocessed)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = 0.0
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {"acc": acc, "f1": f1, "auc": auc}
        
        print(f"{name:<30} | {acc:.4f}     | {f1:.4f}     | {auc:.4f}")
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred Good', 'Pred Bad'], 
                    yticklabels=['Actual Good', 'Actual Bad'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'plots/cm_{name.replace(" ", "_").replace("(", "").replace(")", "").lower()}.png')
        plt.close()

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        train_and_evaluate(df)
