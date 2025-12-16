from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, model_name="Model", save_results=True):
    """
    Evaluates the model and prints metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.4f}")
    
    print(classification_report(y_test, y_pred))
    
    if save_results:
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(results_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
        plt.close()

    return {"accuracy": acc, "f1": f1, "auc": auc}
