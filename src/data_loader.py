import pandas as pd
import os

def load_data(filepath="data/raw/dataset.data"):
    """
    Loads the German Credit dataset.
    """
    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment_duration", "installment_rate", "personal_status_sex",
        "other_debtors", "residence_since", "property", "age", "other_installment_plans",
        "housing", "existing_credits", "job", "people_liable", "telephone", "foreign_worker",
        "credit_risk"
    ]
    
    # Check if file exists at the given path, if not try to find it relative to the project root
    if not os.path.exists(filepath):
        # Assuming this script might be called from root or src, try to resolve absolute path
        # This assumes the structure is project_root/data/raw/dataset.data
        # and this file is in project_root/src/data_loader.py
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_path, "data", "raw", "dataset.data")
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, sep=" ", names=columns, header=None)
    return df

def clean_data(df):
    """
    Maps cryptic codes to meaningful labels.
    """
    mappings = {
        "status": {"A11": "< 0 DM", "A12": "0 <= ... < 200 DM", "A13": ">= 200 DM", "A14": "no checking"},
        "credit_history": {"A30": "no credits/paid", "A31": "all paid at this bank", "A32": "existing paid", "A33": "delay", "A34": "critical/other"},
        "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/tv", "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining", "A49": "business", "A410": "others"},
        "savings": {"A61": "< 100 DM", "A62": "100 <= ... < 500 DM", "A63": "500 <= ... < 1000 DM", "A64": ">= 1000 DM", "A65": "unknown/none"},
        "employment_duration": {"A71": "unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years", "A74": "4 <= ... < 7 years", "A75": ">= 7 years"},
        "personal_status_sex": {"A91": "male: divorced/separated", "A92": "female: div/dep/mar", "A93": "male: single", "A94": "male: mar/wid", "A95": "female: single"},
        "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
        "property": {"A121": "real estate", "A122": "building society/life ins", "A123": "car/other", "A124": "unknown/none"},
        "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
        "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
        "job": {"A171": "unemployed/unskilled non-res", "A172": "unskilled res", "A173": "skilled", "A174": "management/self-employed"},
        "telephone": {"A191": "none", "A192": "yes"},
        "foreign_worker": {"A201": "yes", "A202": "no"}
    }

    df_clean = df.copy()
    for col, mapping in mappings.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping).fillna(df_clean[col])
            
    # Encode target: 1 (Good) -> 0, 2 (Bad) -> 1
    if "credit_risk" in df_clean.columns:
        df_clean["credit_risk"] = df_clean["credit_risk"].map({1: 0, 2: 1})

    return df_clean
